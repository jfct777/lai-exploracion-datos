#!/usr/bin/env python3
"""Prepare one external-panel autosome, not simulations or independent donors.

CLI: --contract FILE --contract-sha256 SHA --outdir NEW_DIRECTORY.
Input schema r02_lai_panel_prepare_input_v1 (all paths are absolute):
  chromosome: "1".."22"; build: "GRCh38"; roles: {path,sha256};
  projections: {REF_TRAIN: {...}, SOURCE_VALID: {...}}, each with authenticated
    vcf, index (.csi), projection_manifest, build_audit, reference_identity;
    expected_samples and expected_ancestries explicitly fixed by the caller.
  genetic_map: {path,sha256,build};
  long_ld_bed: {path,sha256,build,coordinate_system:"0-based-half-open"};
  common: {maf: number in (0,.5), max_missing: number in [0,1)};
  tools: {bcftools:{path,sha256,version}, plink2:{path,sha256,version}};
  resources: {threads,plink_memory_mib,max_line_bytes,max_records,
    max_output_bytes,min_free_gib,timeout_seconds}.

Haplotype output REFERENCES existing immutable VCF/CSI and limits subsequent
copying to map endpoints; it does not duplicate, prune, impute or rephase them.
Kinship BED is a distinct UNPHASED derivative: PASS/unfiltered, biallelic ACGT
SNPs, outside the historical hg38 exclusion mask (long LD, gaps, blacklist),
frequency/missingness recomputed in these two
roles. No HWE filter or LD pruning. The later aggregate GDS stage owns pruning.
No SOURCE_TEST genotypes are opened. Metadata may contain other roles.

Reference identity and all-record REF/FASTA audits are reused only if their
exact VCF/reference digests agree; no fresh FASTA check is claimed. There is no
sample-independence or phase-accuracy certificate. Output BED/FAM and source
paths are private; callers must not publicly distribute the output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

from r02_lai_role_subset import require, sha, axis_hash, header, lines
from r02_lai_phased_audit import read_map
from r02_lai_build_verification import reference_contract

SCHEMA = 'r02_lai_panel_prepare_input_v1'
STATUS = 'COMPLETE_PANEL_PREPARATION_NOT_KINSHIP_PHASE_OR_BIOLOGICAL_VALIDATION'
ROLES = ('REF_TRAIN', 'SOURCE_VALID')
CODE_FILES = ('r02_lai_panel_prepare.py', 'r02_lai_role_subset.py',
              'r02_lai_phased_audit.py', 'r02_lai_build_verification.py',
              'r02_lai_allele_support.py', 'r02_genomic_pair_evidence.py')


def write_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


class Inputs:
    def __init__(self):
        self.files = {}

    def add(self, item, name):
        require(isinstance(item, dict) and {'path', 'sha256'} <= set(item), 'Missing input descriptor: ' + name)
        p = Path(item['path'])
        require(p.is_absolute() and p.is_file(), 'Missing absolute input: ' + name)
        require(re.fullmatch('[0-9a-f]{64}', item['sha256'] or ''), 'Invalid SHA256: ' + name)
        require(sha(p) == item['sha256'], 'Input SHA256 mismatch: ' + name)
        self.files[name] = dict(path=str(p.resolve()), sha256=item['sha256'], size_bytes=p.stat().st_size)
        return p.resolve()

    def unchanged(self):
        for name, item in self.files.items():
            require(Path(item['path']).stat().st_size == item['size_bytes'] and
                    sha(item['path']) == item['sha256'], 'Input changed during preparation: ' + name)


def validate_contract(data):
    require(data.get('schema') == SCHEMA and data.get('build') == 'GRCh38', 'Unknown schema/build')
    require(data.get('chromosome') in {str(i) for i in range(1, 23)}, 'Explicit autosome required')
    require(set(data.get('projections', {})) == set(ROLES), 'Only REF_TRAIN and SOURCE_VALID are authorized')
    require(set(data.get('common', {})) == {'maf', 'max_missing'}, 'Unknown common filters; LD pruning belongs to GDS')
    for field, low, high in (('maf', 0, .5), ('max_missing', 0, 1)):
        value = data['common'][field]
        require(type(value) in (int, float) and math.isfinite(value) and low <= value < high,
                'Invalid common parameter: ' + field)
    require(data['common']['maf'] > 0, 'Common MAF must be positive')
    resources = data['resources']
    required = {'threads', 'plink_memory_mib', 'max_line_bytes', 'max_records', 'max_output_bytes',
                'min_free_gib', 'timeout_seconds'}
    require(set(resources) == required, 'Explicit resource bounds required')
    for key in required - {'min_free_gib'}:
        require(type(resources[key]) is int and resources[key] > 0, 'Positive integer required: ' + key)
    require(resources['plink_memory_mib'] >= 640, 'PLINK requires at least 640 MiB')
    require(type(resources['min_free_gib']) in (int, float) and math.isfinite(resources['min_free_gib'])
            and resources['min_free_gib'] >= 0, 'Invalid disk reserve')
    require(set(data.get('tools', {})) == {'bcftools', 'plink2'}, 'Pinned tools required')
    for key in ('genetic_map', 'long_ld_bed'):
        require(data[key].get('build') == data['build'], key + ' build differs')
    require(data['long_ld_bed'].get('coordinate_system') == '0-based-half-open', 'Explicit BED coordinates required')


def load_roles(path):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require({'sample_id', 'role', 'ancestry'} <= set(reader.fieldnames or []), 'Missing roles columns')
        rows = list(reader)
    ids = [r['sample_id'] for r in rows]
    require(all(ids) and len(ids) == len(set(ids)), 'Empty or duplicated role sample')
    return {role: {r['sample_id']: r['ancestry'] for r in rows if r['role'] == role} for role in ROLES}


def authenticate_projection(item, role, data, roster, inputs):
    paths = {k: inputs.add(item[k], role + '.' + k) for k in
             ('vcf', 'index', 'projection_manifest', 'build_audit', 'reference_identity')}
    require(str(paths['index']) == str(paths['vcf']) + '.csi', 'CSI must be adjacent to its authenticated VCF')
    samples = header(paths['vcf'], data['resources']['max_line_bytes'])
    require(type(item['expected_samples']) is int and item['expected_samples'] > 0 and
            len(samples) == item['expected_samples'] == len(set(samples)), 'Role sample count differs')
    require(set(samples) == set(roster[role]), 'Projected samples differ from authorized role; TEST stays closed')
    require(dict(Counter(roster[role].values())) == item['expected_ancestries'], 'Ancestry metadata counts differ')
    receipt = json.loads(paths['projection_manifest'].read_text())
    require(receipt.get('schema') == 'r02_lai_role_subset_v1' and
            receipt.get('status') == 'COMPLETE_MECHANICAL_SUBSET_NOT_BUILD_QC_OR_PHASE_VALIDATION' and
            receipt.get('role') == role and receipt.get('chromosome') == data['chromosome'], 'Wrong projection receipt')
    require(receipt.get('selected_samples') == len(samples) and
            receipt.get('sample_order_sha256') == axis_hash(samples) and
            receipt.get('sample_members_sha256') == axis_hash(sorted(samples)), 'Projection sample axis differs')
    require(receipt.get('input_sha256', {}).get('roles') == data['roles']['sha256'] and
            receipt.get('selected_genotype_text_preserved') is True and
            receipt.get('source_test_projected') is False and not receipt.get('variant_filters') and
            not receipt.get('genotype_filters'), 'Projection provenance/filters differ')
    outputs = receipt.get('output_sha256', {})
    require(item['vcf']['sha256'] in outputs.values() and item['index']['sha256'] in outputs.values(),
            'Projection does not authenticate VCF and CSI')
    reference = json.loads(paths['reference_identity'].read_text())
    reference_contract(reference)
    audit = json.loads(paths['build_audit'].read_text())
    evidence = audit.get('reference_audit', {})
    require(audit.get('schema') == 'r02_lai_build_verification_v1' and audit.get('status') == 'VERIFIED' and
            audit.get('build') == data['build'] == reference['assembly'] and
            audit.get('vcf_sha256') == item['vcf']['sha256'] and
            audit.get('method') == 'bcftools_norm_check_ref_e_all_records_v1' and
            audit.get('fasta_sha256') == reference['fasta_sha256'] and
            audit.get('reference_provenance') == reference and
            audit.get('input_files', {}).get('reference_contract', {}).get('sha256') == item['reference_identity']['sha256'],
            'REF/FASTA identity audit differs from source')
    require(evidence.get('chromosome') == data['chromosome'] and evidence.get('mismatches') == 0 and
            evidence.get('bcftools_exit_code') == 0 and evidence.get('scope') == 'ALL_VCF_RECORDS' and
            evidence.get('records_checked') == receipt.get('records') and
            type(receipt.get('records')) is int and 0 < receipt['records'] <= data['resources']['max_records'],
            'Incomplete or incompatible all-record REF audit')
    length = reference['contigs'].get('chr' + data['chromosome'])
    require(type(length) is int and length > 0, 'Reference contig length missing')
    return dict(paths=paths, samples=samples, receipt=receipt, length_bp=length, reference=reference)


def ld_intervals(path, chromosome, length):
    intervals = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip() or line.startswith('#'):
                continue
            f = line.split()
            require(len(f) >= 3 and re.fullmatch(r'[A-Za-z0-9_.-]+', f[0]), 'Invalid exclusion BED row')
            start, end = int(f[1]), int(f[2])
            require(0 <= start < end, 'Invalid half-open LD BED interval')
            if f[0].removeprefix('chr') == chromosome:
                require(end <= length, 'LD BED exceeds reference chromosome')
                intervals.append((start, end))
    # Overlapping definitions are a union for exclusion, never duplicate sites.
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def genotype_category(gt, n_alt):
    parts = re.split(r'[/|]', gt)
    require(all(a == '.' or (a.isdigit() and int(a) <= n_alt) for a in parts), 'Invalid allele index in GT')
    if all(a == '.' for a in parts):
        return 'fully_missing'
    if len(parts) != 2:
        return 'non_diploid'
    if '.' in parts:
        return 'partially_missing'
    if '|' in gt:
        return 'complete_phased_diploid'
    return 'complete_unphased_homozygous' if parts[0] == parts[1] else 'complete_unphased_heterozygous'


def rows(path, limit):
    return (line.split('\t') for line in lines(path, limit) if not line.startswith('#'))


def scan_sources(prepared, data, map_range, intervals, exclude_path):
    """One bounded synchronized pass; GT histogram avoids Python work per cell."""
    limit = data['resources']['max_line_bytes']
    streams = [rows(prepared[r]['paths']['vcf'], limit) for r in ROLES]
    stats = {r: {scope: Counter() for scope in ('inside_map', 'outside_map')} for r in ROLES}
    counts = Counter(); digest = hashlib.sha256(); previous = 0; ld_i = 0
    with Path(exclude_path).open('x') as exclude:
        for pair in itertools.zip_longest(*streams):
            require(None not in pair, 'Role VCF record counts differ')
            left, right = pair
            require(left[:5] == right[:5], 'Role VCF site/allele/order differs')
            chromosome, position, ref, alt = left[0], int(left[1]), left[3], left[4]
            require(chromosome == 'chr' + data['chromosome'] and previous < position <= prepared[ROLES[0]]['length_bp'],
                    'Duplicate/nonmonotone position or wrong chromosome; reconcile sites before merging')
            previous = position; counts['source_records'] += 1
            require(counts['source_records'] <= data['resources']['max_records'], 'Record bound exceeded')
            digest.update(('\t'.join((chromosome, str(position), ref, alt)) + '\n').encode())
            scope = 'inside_map' if map_range[0] <= position <= map_range[1] else 'outside_map'
            counts['records_' + scope] += 1
            for role, row in zip(ROLES, pair):
                require(len(row) == 9 + len(prepared[role]['samples']), 'VCF sample width differs')
                fields = row[8].split(':')
                require(fields.count('GT') == 1, 'Exactly one GT field required')
                gt_i = fields.index('GT')
                histogram = Counter(row[9:])
                for cell, multiplicity in histogram.items():
                    parts = cell.split(':')
                    require(gt_i < len(parts), 'GT cell is absent')
                    stats[role][scope][genotype_category(parts[gt_i], len(alt.split(',')))] += multiplicity
                stats[role][scope]['person_site_denominator'] += len(prepared[role]['samples'])
                stats[role][scope]['sites'] += 1
                stats[role][scope]['sites_with_PS_field'] += int('PS' in fields)
            while ld_i < len(intervals) and intervals[ld_i][1] <= position - 1:
                ld_i += 1
            in_ld = ld_i < len(intervals) and intervals[ld_i][0] <= position - 1 < intervals[ld_i][1]
            snp = len(ref) == len(alt) == 1 and ref in 'ACGT' and alt in 'ACGT' and ref != alt
            pass_both = all(row[6] in ('.', 'PASS') for row in pair)
            counts['records_excluded_historical_mask'] += int(in_ld)
            counts['records_non_biallelic_acgt_snp'] += int(not snp)
            counts['records_nonpass_in_either_role'] += int(not pass_both)
            if in_ld or not snp or not pass_both:
                # PLINK normalizes chr22 to 22 in the @ variable-ID template.
                exclude.write(f'{data["chromosome"]}:{position}:{ref}:{alt}\n')
                counts['records_excluded_before_common_filter'] += 1
    require(counts['records_inside_map'] > 0, 'No source markers within genetic-map support')
    for role in ROLES:
        require(counts['source_records'] == prepared[role]['receipt']['records'] and
                digest.hexdigest() == prepared[role]['receipt']['variant_key_order_sha256'], 'Authenticated source site axis differs')
    return stats, dict(counts), digest.hexdigest()


def verify_merged(path, prepared, data):
    expected = sum((prepared[r]['samples'] for r in ROLES), [])
    limit = data['resources']['max_line_bytes']
    require(header(path, limit) == expected, 'Merge changed sample order/membership')
    count = 0; missing_phase_delimiters_changed = 0
    for triple in itertools.zip_longest(*(rows(prepared[r]['paths']['vcf'], limit) for r in ROLES), rows(path, limit)):
        require(None not in triple, 'Merge changed number of records')
        a, b, c = triple
        require(a[:2] == b[:2] == c[:2] and a[3:5] == b[3:5] == c[3:5], 'Merge changed position or alleles')
        # Compare GT independently of INFO recomputation and FORMAT field order.
        def gt_cells(row):
            if row[8] == 'GT':
                return row[9:]
            i = row[8].split(':').index('GT')
            return [x.split(':')[i] for x in row[9:]]
        original = gt_cells(a) + gt_cells(b); merged = gt_cells(c)
        require(len(original) == len(merged), 'Merge sample count differs')
        if original == merged:
            count += 1
            continue
        # bcftools 1.16 writes 0|. as 0/. and .|. as ./.; no phased
        # haplotype is established by either incomplete call. Preserve allele
        # order/missing positions, and document this syntactic change. Complete
        # GT must remain exact, including phase and haplotype allele order.
        for before, after in zip(original, merged):
            if before == after:
                continue
            require('.' in before and re.split(r'[/|]', before) == re.split(r'[/|]', after),
                    'Merge changed a GT or filled a missing genotype')
            missing_phase_delimiters_changed += 1
        count += 1
    require(count == prepared[ROLES[0]]['receipt']['records'], 'Merge record count differs')
    return dict(records=count, missing_phase_delimiters_changed=missing_phase_delimiters_changed,
                complete_gt_identical=True, missing_allele_positions_identical=True)


def command_plan(data, prepared, combined, exclusions, prefix):
    bcftools, plink = (data['tools'][k]['path'] for k in ('bcftools', 'plink2'))
    resources = data['resources']
    # bcftools reserves one main thread in addition to BGZF compression workers.
    merge = [bcftools, 'merge', '--no-version', '--merge', 'none', '--threads', str(max(0, resources['threads'] - 1)),
             '-Oz', '-o', str(combined)] + [str(prepared[r]['paths']['vcf']) for r in ROLES]
    make = [plink, '--vcf', str(combined), '--double-id', '--vcf-require-gt', '--vcf-half-call', 'missing',
            '--snps-only', 'just-acgt', '--min-alleles', '2', '--max-alleles', '2',
            '--set-all-var-ids', '@:#:$r:$a', '--maf', str(data['common']['maf']),
            '--geno', str(data['common']['max_missing']), '--make-bed', '--threads', str(resources['threads']),
            '--memory', str(resources['plink_memory_mib']), '--out', str(prefix)]
    if Path(exclusions).stat().st_size:
        make += ['--exclude', str(exclusions)]
    return merge, make


def verify_bed(prefix, samples, chromosome, intervals, exclusions):
    """Independently verify sample axis and that nominal exclusion reached BIM."""
    prefix = Path(prefix)
    with prefix.with_suffix('.fam').open() as handle:
        fam = [line.split() for line in handle]
    require(all(len(row) == 6 for row in fam) and [r[1] for r in fam] == samples and
            all(r[0] == r[1] for r in fam), 'PLINK sample axis differs')
    excluded = set(Path(exclusions).read_text().splitlines())
    n_common = 0; previous = 0; ld_i = 0
    with prefix.with_suffix('.bim').open() as handle:
        for line in handle:
            fields = line.split()
            require(len(fields) == 6, 'Wrong BIM columns')
            marker = fields[1].split(':'); position = int(fields[3])
            require(len(marker) == 4 and fields[0] == marker[0] == chromosome and
                    marker[1] == str(position) and position > previous, 'BIM chromosome/ID/order differs')
            require(len(marker[2]) == len(marker[3]) == 1 and set(marker[2:]) <= set('ACGT') and
                    marker[2] != marker[3] and set(fields[4:]) == set(marker[2:]), 'BIM source alleles differ')
            while ld_i < len(intervals) and intervals[ld_i][1] <= position - 1:
                ld_i += 1
            require(not (ld_i < len(intervals) and intervals[ld_i][0] <= position - 1 < intervals[ld_i][1]),
                    'Historical-mask region survived PLINK exclusion')
            require(fields[1] not in excluded, 'Excluded non-PASS/non-SNP site survived PLINK filtering')
            previous = position; n_common += 1
    require(n_common > 0, 'No common SNP retained; do not launch ancestry/kinship estimation')
    with prefix.with_suffix('.bed').open('rb') as handle:
        require(handle.read(3) == b'\x6c\x1b\x01', 'Wrong SNP-major BED encoding')
    require(prefix.with_suffix('.bed').stat().st_size == 3 + n_common * math.ceil(len(samples) / 4),
            'BED size/sample axis differs')
    return n_common


def check_storage(out, resources, private=None):
    roots = [out] + ([private] if private is not None else [])
    total = sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
    require(total <= resources['max_output_bytes'], 'Output/intermediate size bound exceeded')
    require(shutil.disk_usage(out).free >= resources['min_free_gib'] * 1024 ** 3, 'Disk reserve exhausted')


def run_command(command, log_path, timeout, env, out, private, resources):
    """Bound total live intermediate bytes and kill only this tool's process group."""
    started = time.monotonic()
    with Path(log_path).open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while True:
                check_storage(out, resources, private)
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = process.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    continue
                if code:
                    raise subprocess.CalledProcessError(code, command)
                check_storage(out, resources, private)
                return
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise


def run(contract_path, contract_sha256, outdir):
    inputs = Inputs()
    contract_path = inputs.add(dict(path=str(Path(contract_path).absolute()), sha256=contract_sha256), 'contract')
    data = json.loads(contract_path.read_text()); validate_contract(data)
    out = Path(outdir).absolute()
    require(not out.exists() and out.parent.is_dir(), 'Output must be a new directory in an existing parent')
    resources = data['resources']
    require(shutil.disk_usage(out.parent).free >= resources['min_free_gib'] * 1024 ** 3 + resources['max_output_bytes'],
            'Disk admission cannot preserve reserve at maximum output size')
    roles_path = inputs.add(data['roles'], 'roles'); roster = load_roles(roles_path)
    prepared = {r: authenticate_projection(data['projections'][r], r, data, roster, inputs) for r in ROLES}
    require(not set(prepared[ROLES[0]]['samples']) & set(prepared[ROLES[1]]['samples']), 'Roles overlap')
    require(prepared[ROLES[0]]['reference'] == prepared[ROLES[1]]['reference'], 'Roles use different reference identities')
    map_path = inputs.add(data['genetic_map'], 'genetic_map')
    low, high, map_points = read_map(map_path, data['chromosome'])
    require(1 <= low < high <= prepared[ROLES[0]]['length_bp'], 'Map outside reference chromosome')
    bed_path = inputs.add(data['long_ld_bed'], 'long_ld_bed')
    intervals = ld_intervals(bed_path, data['chromosome'], prepared[ROLES[0]]['length_bp'])
    versions = {}
    for name, tool in data['tools'].items():
        executable = inputs.add(tool, 'tool.' + name)
        version = subprocess.check_output([str(executable), '--version'], text=True, timeout=30).splitlines()[0]
        require(version == tool['version'], 'Tool version differs: ' + name)
        versions[name] = version
    private = out.with_name(out.name + '.private_work')
    require(not private.exists(), 'Private task workspace exists; use Nextflow resume, not in-place restart')
    out.mkdir(mode=0o700); private.mkdir(mode=0o700)
    exclusions = private / 'exclude.ids'
    stats, counts, site_hash = scan_sources(prepared, data, (low, high), intervals, exclusions)
    combined = private / 'combined.vcf.gz'; prefix = out / 'kinship'
    commands = command_plan(data, prepared, combined, exclusions, prefix)
    started = time.monotonic()
    env = {**os.environ, 'OMP_NUM_THREADS': str(resources['threads']), 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    for index, command in enumerate(commands):
        check_storage(out, resources, private)
        remaining = resources['timeout_seconds'] - (time.monotonic() - started)
        require(remaining > 0, 'Tool runtime budget exhausted')
        run_command(command, out / f'command{index + 1}.log', remaining, env, out, private, resources)
        if index == 0:
            merge_verification = verify_merged(combined, prepared, data)
    # BED is intentionally unphased; its full genotype preservation claim belongs
    # only to the source VCF and verified merge, not this common-SNP derivative.
    samples = sum((prepared[r]['samples'] for r in ROLES), [])
    n_common = verify_bed(prefix, samples, data['chromosome'], intervals, exclusions)
    gt_categories = ('complete_phased_diploid', 'complete_unphased_homozygous', 'complete_unphased_heterozygous',
                     'fully_missing', 'partially_missing', 'non_diploid')
    fields = ('role', 'scope', 'n_people', 'sites', 'person_site_denominator', *gt_categories, 'sites_with_PS_field')
    with (out / 'genotype_stats.tsv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=fields); writer.writeheader()
        for role in ROLES:
            for scope, counter in stats[role].items():
                require(sum(counter[k] for k in gt_categories) == counter['person_site_denominator'], 'GT denominator inconsistent')
                writer.writerow(dict(role=role, scope=scope, n_people=len(prepared[role]['samples']),
                                     **{k: counter[k] for k in fields[3:]}))
    haplotypes = dict(schema='r02_lai_haplotype_inputs_v1', status='REFERENCES_TO_OBSERVED_PHASED_INPUTS_NOT_SIMULATION_READY_CERTIFICATE',
        chromosome=data['chromosome'], build=data['build'], genetic_map=inputs.files['genetic_map'],
        usable_range=dict(first_bp=low, last_bp=high, coordinate_system='1-based-closed', extrapolation=False, map_points=map_points),
        projections={r: dict(vcf=inputs.files[r + '.vcf'], index=inputs.files[r + '.index'],
            sample_order_sha256=axis_hash(prepared[r]['samples']), n_people=len(prepared[r]['samples'])) for r in ROLES},
        copy_rule='Only complete diploid phased GT gives ordered observed haplotypes; missing/unphased/non-diploid cells remain unresolved, never filled with REF.',
        phase_accuracy='UNKNOWN; inferred phase in source VCF, not true or independently validated phase. Missing PS does not invalidate | GT.',
        source_filters=[], ld_pruned=False, genotype_stats_sha256=sha(out / 'genotype_stats.tsv'),
        source_test_opened=False, frequency_info='Inherited INFO AC/AN must not be used as subset frequencies.',
        independence='Not assessed; cross-role family audit is a later PC-AiR/PC-Relate stage.')
    write_json(out / 'haplotype_inputs.json', haplotypes)
    inputs.unchanged(); check_storage(out, resources, private)
    products = ('kinship.bed', 'kinship.bim', 'kinship.fam', 'haplotype_inputs.json', 'genotype_stats.tsv',
                'command1.log', 'command2.log', 'kinship.log')
    manifest = dict(schema='r02_lai_panel_prepare_v1', status=STATUS, chromosome=data['chromosome'], build=data['build'],
        inputs=inputs.files, code_sha256={p: sha(Path(__file__).parent / p) for p in CODE_FILES}, tools=versions,
        source_records=counts['source_records'], source_variant_key_order_sha256=site_hash, source_counts=counts,
        n_people=len(samples), role_counts={r: len(prepared[r]['samples']) for r in ROLES}, sample_order_sha256=axis_hash(samples),
        sample_members_sha256=axis_hash(sorted(samples)), common_snps=n_common, common=data['common'],
        common_frequency_population='REF_TRAIN plus SOURCE_VALID; joint common-SNP relationship audit, not held-out prediction performance.',
        common_filters=dict(biallelic_acgt_snps=True, site_filter='PASS_OR_UNFILTERED_IN_BOTH_ROLES',
            long_ld_bed_sha256=data['long_ld_bed']['sha256'], half_call='missing', no_hwe_filter=True, ld_pruning=False),
        exclusion_mask_kind='historical_hg38_exclusion_mask: long-LD regions, gaps and blacklist, not only LD',
        exclusion_intervals_zero_based_half_open=[list(x) for x in intervals], map_range=[low, high], no_map_extrapolation=True,
        reference_audit_reused=True, new_fasta_comparison=False, merge_verification=merge_verification,
        bed_allele_coding='PLINK BIM A1/A2; original REF/ALT retained in marker ID, not certified by BIM column order.',
        source_test_opened=False, simulation_executed=False, kinship_estimation_executed=False,
        phase_accuracy='UNKNOWN', public_distribution_allowed=False, contains_sample_identifiers=True,
        contains_individual_genotypes=True, resources=resources, commands=commands,
        bcftools_cpu_budget=dict(total_requested=resources['threads'], main_thread=1,
                                compression_threads=max(0, resources['threads'] - 1)),
        outputs_sha256={name: sha(out / name) for name in products},
        intermediate_combined=dict(path=str(combined), sha256=sha(combined),
            description='Verified temporary merge, not the immutable haplotype source; no pruning or missing-to-REF applied.'))
    write_json(out / 'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('contract', 'contract-sha256', 'outdir'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    run(args.contract, args.contract_sha256, args.outdir)


if __name__ == '__main__':
    main()
