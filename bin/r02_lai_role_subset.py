#!/usr/bin/env python3
"""Mechanical SOURCE_VALID projection, preserving sites and opaque selected GT.

No frequencies, phase validation, simulation or SOURCE_TEST GT interpretation.
Inherited INFO/AC/AN describe the full source and MUST NOT be used as subset AC/AN.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def axis_hash(values):
    return hashlib.sha256(('\n'.join(values) + '\n').encode()).hexdigest()


def lines(path, limit):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        while True:
            line = handle.readline(limit + 1)
            if not line:
                return
            require(len(line) <= limit, 'VCF line exceeds bound')
            yield line.rstrip('\n').rstrip('\r')


def header(path, limit):
    for line in lines(path, limit):
        if line.startswith('#CHROM\t'):
            return line.split('\t')[9:]
    raise ValueError('VCF sample header missing')


def allowlist(roles, samples, expected, expected_ancestries):
    with Path(roles).open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require({'sample_id', 'role', 'ancestry'} <= set(reader.fieldnames or []), 'Roles columns missing')
        rows = list(reader)
    require(len(samples) == len(set(samples)), 'Duplicate panel sample')
    require([r['sample_id'] for r in rows] == samples, 'Roles/panel order or membership mismatch')
    selected = [r['sample_id'] for r in rows if r['role'] == 'SOURCE_VALID']
    require(len(selected) == expected and len(set(selected)) == expected, 'SOURCE_VALID count mismatch')
    require(dict(Counter(r['ancestry'] for r in rows if r['role'] == 'SOURCE_VALID')) == expected_ancestries,
            'SOURCE_VALID ancestry counts mismatch')
    return selected


def verify_projection(source, output, selected, expected_records, limit):
    samples = header(source, limit)
    require(header(output, limit) == selected, 'Output sample order differs')
    indices = [samples.index(s) + 9 for s in selected]
    source_rows = (x for x in lines(source, limit) if not x.startswith('#'))
    output_rows = (x for x in lines(output, limit) if not x.startswith('#'))
    count, digest = 0, hashlib.sha256()
    for left, right in itertools.zip_longest(source_rows, output_rows):
        require(left is not None and right is not None, 'Record count differs')
        a, b = left.split('\t'), right.split('\t')
        require(len(a) == 9 + len(samples) and len(b) == 9 + len(selected), 'VCF width mismatch')
        require(a[:9] == b[:9], 'Site/alleles/INFO/FORMAT changed')
        # Compare selected cells as opaque text, without decoding any unselected GT.
        require([a[i] for i in indices] == b[9:], 'Selected genotype text changed')
        digest.update(('\t'.join((a[0], a[1], a[3], a[4])) + '\n').encode())
        count += 1
    require(count == expected_records, 'Unexpected source record count')
    return count, digest.hexdigest()


def run(args):
    contract = json.loads(Path(args.contract).read_text())
    require(contract['schema'] == 'r02_lai_role_subset_contract_v1', 'Unknown subset contract')
    require(contract['role'] == 'SOURCE_VALID', 'Only SOURCE_VALID is authorized')
    inputs = {key: Path(getattr(args, key)) for key in ('source_vcf', 'roles', 'ref_receipt')}
    for key, path in inputs.items():
        require(sha(path) == contract['input_sha256'][key], 'Input hash mismatch: ' + key)
    ref = json.loads(inputs['ref_receipt'].read_text())
    require(ref['schema'] == 'r02_lai_genotype_support_v1' and
            ref['status'] == 'COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION' and
            ref['role_validation']['role'] == 'REF_TRAIN' and
            ref['role_validation']['n_role_members'] == contract['expected_ref_samples'] and
            ref['input_files']['roles']['sha256'] == contract['input_sha256']['roles'], 'REF receipt/roles mismatch')
    output = Path(args.outdir)
    require(not output.exists(), 'Output already exists')
    require(shutil.disk_usage(output.parent).free >= args.min_free_gib * 1024**3, 'Insufficient free disk reserve')
    samples = header(inputs['source_vcf'], args.max_line_bytes)
    require(len(samples) == contract['expected_source_samples'], 'Source sample count mismatch')
    selected = allowlist(inputs['roles'], samples, contract['expected_selected_samples'], contract['expected_ancestries'])
    version = subprocess.check_output([args.bcftools, '--version'], text=True).splitlines()[0]
    require(version == contract['bcftools_version'], 'Unexpected bcftools version')
    output.mkdir(mode=0o700)
    keep = output / 'samples.private.txt'
    keep.write_text('\n'.join(selected) + '\n'); keep.chmod(0o600)
    vcf = output / 'source_valid.chr22.vcf.gz'
    command = [args.bcftools, 'view', '-S', str(keep), '-I', '--no-version', '-Oz', '-o', str(vcf), str(inputs['source_vcf'])]
    subprocess.run(command, check=True, capture_output=True, timeout=args.timeout_seconds)
    count, digest = verify_projection(inputs['source_vcf'], vcf, selected, contract['expected_records'], args.max_line_bytes)
    subprocess.run([args.bcftools, 'index', '--csi', str(vcf)], check=True, capture_output=True, timeout=args.timeout_seconds)
    for key, path in inputs.items():
        require(sha(path) == contract['input_sha256'][key], 'Input changed during projection')
    receipt = dict(schema='r02_lai_role_subset_v1', status='COMPLETE_MECHANICAL_SUBSET_NOT_BUILD_QC_OR_PHASE_VALIDATION',
                   role='SOURCE_VALID', source_samples=len(samples), selected_samples=len(selected), records=count,
                   ancestry_counts=contract['expected_ancestries'], variant_key_order_sha256=digest,
                   sample_order_sha256=axis_hash(selected), sample_members_sha256=axis_hash(sorted(selected)),
                   input_sha256=contract['input_sha256'], contract_sha256=sha(args.contract), code_sha256=sha(__file__),
                   output_sha256={p.name: sha(p) for p in (vcf, Path(str(vcf)+'.csi'), keep)},
                   bcftools_version=version, variant_filters=[], genotype_filters=[], info_ac_an_updated=False,
                   inherited_info_scope='Full source cohort; prohibited for downstream subset frequencies',
                   source_test_projected=False, unselected_genotype_statistics_computed=False,
                   source_panel_parsed_by_bcftools=True, source_test_genotype_statistics_computed=False,
                   unselected_cells_in_python_audit='Opaque tab-delimited text only; no genotype interpretation',
                   selected_genotype_text_preserved=True, phase_reliability='UNKNOWN', allele_representation='UNKNOWN',
                   command_template='bcftools view -S <SOURCE_VALID_in_source_order> -I --no-version -Oz -o <output> <source>',
                   contains_sample_identifiers=False)
    (output/'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-vcf', 'roles', 'ref-receipt', 'contract', 'outdir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--bcftools', default='bcftools')
    p.add_argument('--min-free-gib', type=float, required=True)
    p.add_argument('--max-line-bytes', type=int, required=True)
    p.add_argument('--timeout-seconds', type=int, required=True)
    a = p.parse_args()
    require(a.min_free_gib >= 0 and a.max_line_bytes > 0 and a.timeout_seconds > 0, 'Invalid resource limits')
    run(a)


if __name__ == '__main__':
    main()
