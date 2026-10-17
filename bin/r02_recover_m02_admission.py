#!/usr/bin/env python3
"""Small, read-only admission for M02 recovery from an existing M01 product.

Whole-object staging checks remain the launcher's responsibility. This validates
the pinned inventory, local storage, payload size and the actual VCF header; it
does not claim that a header proves every genotype or the complete VCF stream.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from preprocess_storage_guard import inspect_storage


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(path):
    lines = []
    size = 0
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            size += len(line.encode())
            require(size <= 8 * 1024**2, 'VCF header exceeds the bounded admission budget')
            require(line.startswith('#'), 'Missing VCF column header')
            lines.append(line)
            if line.startswith('#CHROM\t'):
                return lines
    raise ValueError('Incomplete VCF header')


def index_counts(index, chrom):
    """Use established tabix/bcftools index readers; no fabricated RAW header.

    index -s requires the companion VCF, but tabix -l and index -n do not.
    This behavior is covered by the real-container synthetic integration.
    """
    source_name = str(index)[:-4]
    contigs = subprocess.run(['tabix', '-l', source_name], check=True, capture_output=True,
                            text=True, timeout=30).stdout.splitlines()
    require(len(contigs) == 1 and contigs[0].removeprefix('chr') == str(chrom),
            'Index contigs do not match the requested autosome')
    count = subprocess.run(['bcftools', 'index', '-n', source_name], check=True,
                           capture_output=True, text=True, timeout=30).stdout.strip()
    require(count.isdigit() and int(count) > 0, 'Index must contain a positive record count')
    return int(count)


def recover_raw_count(index, chrom, source, evidence_path=None, evidence_sha=None):
    """A missing optional TBI statistics bin is not a zero-variant source.

    Reuse the project's existing bounded TBI inspector. The fallback must be a
    pinned log of the exact normalized source, authenticated by the launcher.
    The M01 original-allele annotator preserves record count; bcftools' input
    count is therefore a count before normalization, not the split output count.
    """
    if evidence_path is None:
        return index_counts(index, chrom), 'AUTHENTIC_RAW_TBI_ONLY', None
    from preprocess_count_validation import tabix_record_statistics
    require(evidence_sha and sha(evidence_path) == evidence_sha, 'Raw count evidence hash differs')
    evidence = json.loads(Path(evidence_path).read_text())
    require(evidence['schema'] == 'r02_m02_raw_count_evidence_v1' and
            str(evidence['chromosome']) == str(chrom), 'Wrong RAW count evidence assignment')
    require(evidence['normalized_source_uri'] == source['uri'] and
            evidence['normalized_source_generation'] == source['generation'] and
            evidence['raw_index_sha256'] == sha(index), 'Count evidence is not bound to these inputs')
    log = evidence['normalization_log']
    require(log['uri'] == source['uri'].removesuffix('.vcf.gz') + '.log' and
            log['generation'] and len(log['text'].encode()) == int(log['bytes']) and
            base64.b64encode(hashlib.md5(log['text'].encode()).digest()).decode() == log['md5Hash'] and
            hashlib.sha256(log['text'].encode()).hexdigest() == log['sha256'],
            'Invalid authenticated normalization log')
    match = re.fullmatch(r'Lines\s+total/split/realigned/skipped:\s*(\d+)/(\d+)/(\d+)/(\d+)\s*', log['text'])
    require(match is not None and int(match[1]) > 0 and int(match[4]) == 0,
            'Normalization input count missing or skipped records present')
    count = int(match[1])
    require(evidence['raw_records'] == count, 'RAW count differs from normalization input count')
    stats = tabix_record_statistics(index, chrom)
    if stats['record_statistics_present']:
        require(index_counts(index, chrom) == count, 'TBI and normalization input counts disagree')
        return count, 'AUTHENTIC_RAW_TBI_AND_M01_NORMALIZATION_LOG', evidence
    require(stats['unplaced_records'] == 0 and stats['data_bins'] > 0 and
            stats['data_chunks'] > 0 and stats['linear_intervals'] > 0,
            'Missing TBI count is not explained by an otherwise populated legacy index')
    return count, 'M01_NORMALIZATION_INPUT_LOG_WITH_MISSING_TBI_STATISTICS', evidence


def admit(inventory_path, expected_sha, chrom, vcf, index, raw_index, scratch,
          required_bytes, minimum_free_gib, count_evidence=None, count_evidence_sha=None,
          expected_raw_count=None):
    require(re.fullmatch(r'[0-9a-f]{64}', expected_sha), 'Invalid inventory SHA256')
    require(sha(inventory_path) == expected_sha, 'Source inventory hash mismatch')
    inventory = json.loads(Path(inventory_path).read_text())
    require(inventory.get('schema') == 'r02_m02_recovery_staged_v1', 'Unexpected staged source receipt')
    require(str(inventory.get('chromosome')) == str(chrom), 'Chromosome differs from staged receipt')
    require(re.fullmatch(r'[0-9a-f]{64}', inventory.get('source_inventory_sha256', '')), 'Missing source inventory binding')
    entry = inventory
    vcf, index, raw_index = (Path(x).resolve(strict=True) for x in (vcf, index, raw_index))
    require(index == Path(str(vcf) + '.tbi'), 'Normalized VCF/index must be adjacent companions')
    require(raw_index.name.endswith('.tbi') and raw_index != index, 'RAW tabix index must be distinct')
    before = {}
    for key, path in (('source', vcf), ('index', index), ('raw_index', raw_index)):
        require(path.is_file(), 'An input is not a regular file')
        record = entry[key]
        stat = path.stat()
        require(path == Path(record['path']).resolve(strict=True), 'Input path differs from authenticated staging')
        require(stat.st_size == int(record['bytes']) and stat.st_mtime_ns == record['mtime_ns'] and stat.st_ctime_ns == record['ctime_ns'],
                'Staged input size/mtime/ctime changed; authenticate it again before recovery')
        require(re.fullmatch(r'[0-9a-f]{64}', record.get('sha256', '')) and record.get('generation') and
                record.get('md5Hash') == record.get('md5_base64'), 'Incomplete staging checksum/generation receipt')
        before[key] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        # For read-only inputs only the filesystem type is inspected below;
        # inspect_storage requires writable output space, not genomic inputs.
    from preprocess_storage_guard import DISK_FILESYSTEMS, find_mount
    mountinfo = Path('/proc/self/mountinfo').read_text()
    for path in (vcf, index, raw_index):
        require(find_mount(path, mountinfo)['filesystem'] in DISK_FILESYSTEMS,
                'Recovery inputs must be staged on local/block disk, not FUSE')
    require(vcf.stat().st_size == int(entry['source']['bytes']), 'Normalized VCF size differs from source inventory')
    require(index.stat().st_size == int(entry['index']['bytes']), 'Normalized index size differs from source inventory')
    for path, record in ((index, entry['index']), (raw_index, entry['raw_index'])):
        require(path.stat().st_size == int(record['bytes']) and
                base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode() == record['md5Hash'] and sha(path) == record['sha256'],
                'Index checksum/size differs from pinned object')
    normalized_count = index_counts(index, chrom)
    raw_count, count_source, count_record = recover_raw_count(raw_index, chrom, entry['source'],
                                                            count_evidence, count_evidence_sha)
    require(expected_raw_count is None or expected_raw_count == raw_count,
            'Configured M02 RAW count differs from authenticated admission')
    lines = header(vcf)
    samples = lines[-1].rstrip('\n').split('\t')[9:]
    require(samples and len(set(samples)) == len(samples), 'Missing or duplicate samples')
    sample_sha = hashlib.sha256(('\n'.join(samples) + '\n').encode()).hexdigest()
    expected = entry['source_header_evidence']
    require(len(samples) == expected['sample_count'] and sample_sha == expected['sample_order_sha256_lf_terminated'],
            'Actual sample identities/order differ from the authenticated header probe')
    require('##dnabr_original_alleles=v1\n' in lines and
            any(x.startswith('##INFO=<ID=ORIG_NALLELES,') for x in lines) and
            any(x.startswith('##INFO=<ID=ORIG_SITE,') for x in lines), 'Missing original-site provenance')
    formats = sorted(re.match(r'##FORMAT=<ID=([^,>]+)', x).group(1) for x in lines if x.startswith('##FORMAT=<ID='))
    require(formats == sorted(expected['format_ids']) and 'GT' in formats, 'FORMAT declarations changed')
    refs = [x.rstrip('\n') for x in lines if x.startswith('##reference=')]
    require(refs == expected['reference_header_lines'] and refs, 'Declared reference changed or absent')
    # This verifies the declaration, not a new FASTA-based build validation.
    require(any('assembly38' in x or 'GRCh38' in x or 'hg38' in x for x in refs), 'Expected GRCh38 declaration')
    storage = inspect_storage(Path(scratch), required_bytes, minimum_free_gib)
    require(sha(inventory_path) == expected_sha, 'Inventory changed during admission')
    for key, path in (('source', vcf), ('index', index), ('raw_index', raw_index)):
        stat = path.stat()
        require(before[key] == (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns), 'Input changed during admission')
    return dict(schema='r02_m02_recovery_admission_v1', status='LOCAL_INPUTS_AND_HEADER_CHECKED_NOT_FULL_PAYLOAD_VALIDATION',
                chromosome=str(chrom), input_stage='normalized', source_inventory_sha256=expected_sha,
                normalized_sha256_from_authenticated_staging=entry['source']['sha256'],
                source_generation=entry['source']['generation'], sample_count=len(samples),
                sample_order_sha256=sample_sha, format_ids=formats, raw_count_source=count_source,
                raw_count_evidence=count_record,
                raw_index_sha256=sha(raw_index), normalized_index_sha256=sha(index),
                normalized_bytes=vcf.stat().st_size, storage=storage,
                raw_index_records=raw_count, normalized_index_records=normalized_count,
                limitations=['Launcher must authenticate complete staged VCF against its pinned remote object.',
                             'Header declarations do not validate per-genotype DP/GQ or phase.',
                             'Recovery retains originally multiallelic records after splitting; downstream selects ORIG_NALLELES=2.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'vcf', 'index', 'raw-index', 'scratch'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--inventory-sha256', required=True)
    p.add_argument('--chromosome', required=True)
    p.add_argument('--required-free-bytes', required=True, type=int)
    p.add_argument('--minimum-free-gib', required=True, type=float)
    p.add_argument('--count-evidence', type=Path)
    p.add_argument('--count-evidence-sha256')
    p.add_argument('--expected-raw-count', type=int)
    a = p.parse_args()
    print(json.dumps(admit(a.inventory, a.inventory_sha256, a.chromosome, a.vcf, a.index,
                           a.raw_index, a.scratch, a.required_free_bytes, a.minimum_free_gib,
                           a.count_evidence, a.count_evidence_sha256, a.expected_raw_count), sort_keys=True))


if __name__ == '__main__':
    main()
