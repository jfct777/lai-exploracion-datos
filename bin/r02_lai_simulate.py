#!/usr/bin/env python3
"""Run pinned haptools, retaining simulation truth outside predictor inputs.

This is an execution contract, not a donor-selection or demographic model.
Real data require a separate, hash-bound scientific admission. No phase is
created and no missing genotype is silently made homozygous-reference.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from r02_lai_support_review import require, sha, read, write

VERSION = '0.6.2'
SCHEMA = 'r02_lai_simulate_input_v1'
SCHEMA_V2 = 'r02_lai_simulate_input_v2'
CHECKS = ('phase_continuity', 'genotype_quality', 'build_and_alleles',
          'genetic_map', 'kinship_pcair_pcrelate', 'donor_reference_separation',
          'donor_split_separation', 'ancestry_representativeness', 'demographic_scenario')
CODE_FILES = ('r02_lai_simulate.py', 'r02_lai_support_review.py')


def input_policy(c):
    """Keep historical contracts strict; new contracts name both decisions."""
    strict = dict(site_filter='PASS_ONLY', unphased_homozygotes='REQUIRE_PHASED_GT')
    if c.get('schema') == SCHEMA:
        require('input_policy' not in c, 'Legacy contracts cannot silently change their input policy')
        return strict
    require(c.get('schema') == SCHEMA_V2, 'Invalid simulation contract')
    policy = c.get('input_policy')
    require(isinstance(policy, dict) and set(policy) == set(strict), 'Explicit input policy required')
    require(policy['site_filter'] in {'PASS_ONLY', 'PASS_OR_UNFILTERED'}, 'Invalid site filter policy')
    require(policy['unphased_homozygotes'] in {'REQUIRE_PHASED_GT', 'ALLOW_IDENTICAL_COPIES'},
            'Invalid unphased homozygote policy')
    return dict(policy)


def descriptor(entry):
    require(isinstance(entry, dict) and set(entry) == {'path', 'sha256'}, 'Invalid descriptor')
    p = Path(entry['path'])
    require(p.is_absolute() and p.is_file(), 'Input must be an existing absolute file')
    require(re.fullmatch('[0-9a-f]{64}', entry['sha256']) and sha(p) == entry['sha256'],
            'Input SHA256 mismatch: ' + p.name)
    return p


def text_open(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()


def validate_parameters(c):
    """Validate the supplied scenario without inventing an admission document."""
    policy = input_policy(c)
    require(c.get('scope') in {'SYNTHETIC_TEST', 'DNABR_AUTHORIZED_SIMULATION'}, 'Explicit scope required')
    for key in ('job_id', 'sample_namespace', 'donor_split'):
        require(re.fullmatch('[A-Za-z][A-Za-z0-9_.-]{0,79}', c.get(key, '')), 'Invalid ' + key)
    chroms = c.get('chromosomes')
    require(isinstance(chroms, list) and chroms and chroms == sorted(set(chroms), key=int)
            and all(isinstance(x,str) and re.fullmatch(r'([1-9]|1[0-9]|2[0-2])',x) for x in chroms), 'Sorted unique autosomes required')
    require(c.get('genome_build') in {'GRCh37', 'GRCh38', 'SYNTHETIC'}, 'Explicit genomic build required')
    require(type(c.get('seed')) is int and 1 <= c['seed'] < 2**32, 'Seed must be positive; haptools ignores zero')
    require(type(c.get('replacement')) is bool, 'Explicit replacement policy required')
    require(c.get('haptools_version') == VERSION, 'Pinned haptools version required')
    r = c.get('resources', {})
    for key in ('max_variants', 'max_genotype_cells', 'max_generation_marker_cells', 'timeout_seconds', 'popsize'):
        require(type(r.get(key)) is int and r[key] > 0, 'Positive resource limit required: '+key)
    for key in ('memory_gib', 'disk_reserve_gib', 'max_output_gib'):
        require(type(r.get(key)) in (int, float) and math.isfinite(r[key]) and r[key] > 0,
                'Positive resource limit required: '+key)
    require(r.get('cpus') == 1, 'haptools task uses one computation CPU; parallelize scenarios with Nextflow')
    return policy


def load_contract(path, digest):
    c = read(descriptor(dict(path=str(Path(path).resolve()), sha256=digest)))
    policy = validate_parameters(c)
    chroms = c['chromosomes']
    require(set(c.get('inputs', {})) == {'donor_vcf', 'donor_index', 'roles', 'model', 'admission', 'maps'}, 'Explicit inputs required')
    files = {k: descriptor(c['inputs'][k]) for k in ('donor_vcf', 'donor_index', 'roles', 'model', 'admission')}
    require(str(files['donor_vcf']).endswith('.vcf.gz') and str(files['donor_index']) in
            {str(files['donor_vcf'])+'.tbi', str(files['donor_vcf'])+'.csi'}, 'Indexed BGZF VCF required')
    require(set(c['inputs']['maps']) == set(chroms), 'One exact map per chromosome required')
    files['maps'] = {ch: descriptor(v) for ch, v in c['inputs']['maps'].items()}
    admission = read(files['admission'])
    require(admission.get('schema') == 'r02_lai_simulation_admission_v1', 'Scientific admission schema required')
    require(admission.get('scope') == c['scope'] and admission.get('genome_build') == c['genome_build'], 'Admission scope/build mismatch')
    expected = {k: c['inputs'][k]['sha256'] for k in ('donor_vcf', 'donor_index', 'roles', 'model')}
    expected['maps'] = {k: v['sha256'] for k, v in c['inputs']['maps'].items()}
    require(admission.get('source_sha256') == expected, 'Admission does not bind these exact inputs')
    if c['schema'] == SCHEMA_V2:
        require(admission.get('input_policy') == policy, 'Admission does not bind the input policy')
        provenance = c.get('preparation_provenance', {})
        require(isinstance(provenance, dict), 'Invalid preparation provenance')
        if c['scope'] == 'DNABR_AUTHORIZED_SIMULATION':
            require(set(provenance) == {'material_report', 'original_roles'}, 'L2 report and original roles required')
        for entry in provenance.values(): descriptor(entry)
        require(admission.get('preparation_sha256', {}) == {k: v['sha256'] for k, v in provenance.items()},
                'Admission does not bind the L2 preparation provenance')
    if c['scope'] == 'SYNTHETIC_TEST':
        require(c['genome_build'] == 'SYNTHETIC' and admission.get('status') == 'SYNTHETIC_FIXTURE_ONLY', 'Synthetic fixture scope mismatch')
    else:
        require(c['genome_build'] in {'GRCh37','GRCh38'}, 'Real simulation requires a real genomic assembly')
        require(admission.get('status') == 'REVIEWED_FOR_DECLARED_SCENARIO', 'Real simulation admission not reviewed')
        require(admission.get('reviewer') and admission.get('limitations'), 'Reviewer and scientific limitations required')
        require(admission.get('donor_split') == c['donor_split'], 'Donor split not admitted')
        require(admission.get('copy_provenance_policy') == 'DONOR_EXACT_HOMOLOG_COMPATIBILITY_SET', 'Acknowledge tool homolog provenance limitation')
        for key in CHECKS:
            item = admission.get('checks', {}).get(key, {})
            require(item.get('status') == 'ACCEPTED_FOR_SCENARIO' and item.get('interpretation'), 'Unresolved scientific admission: '+key)
            descriptor(item['evidence'])
    return c, files, admission


def read_model(path, popsize):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    require(len(rows) >= 2 and len(rows[0]) >= 3, 'Model needs sample count, Admixed, populations and generations')
    n = int(rows[0][0]); labels = rows[0][1:]
    require(n > 0 and labels[0] == 'Admixed' and len(set(labels)) == len(labels), 'Invalid model header')
    require(all(re.fullmatch('[A-Za-z][A-Za-z0-9_.-]*', x) for x in labels), 'Unsafe population label')
    require(popsize >= 10*n, 'Set popsize explicitly >=10 times samples; do not allow silent haptools clamping')
    last = 0
    for i, row in enumerate(rows[1:]):
        require(len(row) == len(labels)+1, 'Wrong number of model fractions')
        generation = int(row[0]); weights = list(map(float, row[1:]))
        require(generation > last and all(math.isfinite(x) and 0 <= x <= 1 for x in weights)
                and abs(sum(weights)-1) <= 1e-7, 'Invalid generation or admixture fractions')
        require(i != 0 or weights[0] == 0, 'First generation cannot sample a nonexistent admixed generation')
        last = generation
    return n, labels[1:]


def read_roles(path, split, populations, synthetic):
    donors = {}; units = defaultdict(set); samples = set()
    with Path(path).open() as h:
        reader = csv.DictReader(h, delimiter='\t')
        require(set(reader.fieldnames or []) == {'sample_id', 'role', 'ancestry', 'split', 'family_component'}, 'Role table schema mismatch')
        for row in reader:
            sid = row['sample_id']
            require(re.fullmatch('[A-Za-z0-9_.-]+', sid or '') and sid not in samples, 'Invalid or duplicate individual')
            samples.add(sid)
            require(row['family_component'] and row['split'], 'Missing kinship component or split')
            require(row['role'] in {'SOURCE_VALID', 'REF_TRAIN'}, 'SOURCE_TEST is not authorized')
            units[row['family_component']].add((row['role'], row['split']))
            if row['role'] == 'SOURCE_VALID' and row['split'] == split:
                require(row['ancestry'] in populations, 'Donor population absent from model')
                if synthetic: require(sid.startswith('SYNTH_'), 'Synthetic scope requires synthetic identifiers')
                donors[sid] = row['ancestry']
    require(donors and set(donors.values()) == set(populations), 'Missing donor population')
    require(all(len(x) == 1 for x in units.values()), 'A relatedness component crosses role/split boundaries')
    return donors


def read_map(path, chromosome):
    result = []; last_bp = 0; last_cm = -1.0
    with Path(path).open() as h:
        for line in h:
            row = line.split()
            require(len(row) == 4 and row[0] == chromosome, 'PLINK map requires chromosome marker cM bp')
            cm = float(row[2]); bp = int(row[3])
            require(math.isfinite(cm) and cm >= last_cm and bp > last_bp, 'Map must be monotone in bp and cM')
            result.append((bp, cm)); last_bp, last_cm = bp, cm
    require(len(result) >= 2 and result[-1][1] > result[0][1], 'Map needs nonzero genetic extent')
    return result


def vcf_rows(path):
    with text_open(path) as h:
        samples = None
        for line in h:
            if line.startswith('##'): continue
            if line.startswith('#CHROM'):
                samples = line.rstrip().split('\t')[9:]; yield samples, None; continue
            require(samples is not None, 'VCF header missing')
            row = line.rstrip().split('\t')
            require(len(row) == 9+len(samples), 'VCF column count mismatch')
            yield samples, row


def genotypes(row, allow_identical_copies=False):
    """Read copy alleles without inventing heterozygote phase or changing GT."""
    formats = row[8].split(':')
    require('GT' in formats, 'GT absent')
    gi = formats.index('GT'); result = []
    for field in row[9:]:
        parts = field.split(':'); gt = parts[gi] if len(parts) > gi else '.'
        phased = re.fullmatch(r'[01]\|[01]', gt) is not None
        identical = allow_identical_copies and gt in {'0/0', '1/1'}
        require(phased or identical, 'Only complete phased heterozygous or explicitly admitted homozygous diploid biallelic genotypes admitted')
        result.append((int(gt[0]), int(gt[2])))
    return result


def audit_donors(c, files, donors, maps, n):
    previous = None; counts = Counter(); positions = Counter()
    policy = input_policy(c); filters = Counter(); unphased_homozygotes = 0
    allowed = {'PASS'} if policy['site_filter'] == 'PASS_ONLY' else {'PASS', '.'}
    allow_identical = policy['unphased_homozygotes'] == 'ALLOW_IDENTICAL_COPIES'
    for samples, row in vcf_rows(files['donor_vcf']):
        if row is None:
            require(len(set(samples)) == len(samples) and set(samples) == set(donors), 'VCF must contain exactly admitted donors, not references or another split')
            continue
        chrom, pos = row[0], int(row[1]); key = (int(chrom), pos)
        require(chrom in maps and (previous is None or key > previous), 'VCF chromosome/order/duplicate error')
        require(maps[chrom][0][0] <= pos <= maps[chrom][-1][0], 'Variant outside map coverage')
        require(len(row[3]) == 1 and len(row[4]) == 1 and row[3] in 'ACGT' and row[4] in 'ACGT' and row[3] != row[4], 'Only biallelic SNVs supported by this simulation contract')
        require(row[6] in allowed, 'Donor site FILTER violates declared policy; unfiltered is not PASS')
        filters[row[6]] += 1
        genotypes(row, allow_identical)
        gi = row[8].split(':').index('GT')
        unphased_homozygotes += sum(field.split(':')[gi] in {'0/0','1/1'} for field in row[9:])
        counts[chrom] += 1; positions[chrom] = pos; previous = key
        total = sum(counts.values())
        require(total <= c['resources']['max_variants'], 'Variant resource bound exceeded')
        require(total*(len(donors)+n)*2 <= c['resources']['max_genotype_cells'], 'Genotype-cell resource bound exceeded')
    require(set(counts) == set(c['chromosomes']), 'At least one usable donor marker per requested chromosome required')
    return dict(variants_by_chromosome=dict(counts), distinct_donors=len(donors), simulated_individuals=n,
                copied_haplotypes=2*n, independent_donors_not_mosaics=True,
                input_policy=policy, original_site_filter_counts=dict(filters),
                unphased_homozygous_calls=unphased_homozygotes,
                original_genotypes_or_filters_changed=False, quality_or_phase_accuracy_validated=False)


def prepare(c, files, out):
    n, populations = read_model(files['model'], c['resources']['popsize'])
    donors = read_roles(files['roles'], c['donor_split'], populations, c['scope'] == 'SYNTHETIC_TEST')
    maps = {ch: read_map(p, ch) for ch, p in files['maps'].items()}
    generations = int([r for r in Path(files['model']).read_text().splitlines() if r.strip()][-1].split()[0])
    work_units = c['resources']['popsize'] * generations * sum(len(m) for m in maps.values())
    require(work_units <= c['resources']['max_generation_marker_cells'], 'Generation-map work bound exceeded')
    summary = audit_donors(c, files, donors, maps, n)
    summary['generation_marker_work_units'] = work_units
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out/'maps').mkdir(mode=0o700)
    for ch, p in files['maps'].items(): shutil.copyfile(p, out/'maps'/f'chr{ch}.map')
    with (out/'sample_info.tsv').open('x') as h:
        for sid, ancestry in donors.items(): h.write(sid+'\t'+ancestry+'\n')
    write(out/'admission.json', dict(status='INPUTS_CHECKED_NOT_SIMULATED', summary=summary,
          input_contract=c, provenance_policy='DONOR_EXACT_HOMOLOG_COMPATIBILITY_SET'))
    return summary, donors, maps


def command(c, files, prepared, output, executable='haptools'):
    args = [executable, 'simgenotype', '--model', str(files['model']), '--mapdir', str(prepared/'maps'),
            '--chroms', ','.join(c['chromosomes']), '--seed', str(c['seed']), '--popsize', str(c['resources']['popsize']),
            '--ref_vcf', str(files['donor_vcf']), '--sample_info', str(prepared/'sample_info.tsv'),
            '--sample_field', '--out', str(output)]
    # In 0.6.2 POP+SAMPLE together silently omits SAMPLE. Ancestry truth is in
    # the .bp file; do not rerun simulation or patch the scientific algorithm.
    if not c['replacement']: args.append('--no_replacement')
    return args


def breakpoints(path, maps, n, populations):
    result = {}; current = None; previous = {}
    with path.open() as h:
        for line in h:
            row = line.split()
            if len(row) == 1:
                require(re.fullmatch(r'Sample_[1-9][0-9]*_[12]', row[0]), 'Unexpected breakpoint sample')
                current = row[0]; require(current not in result, 'Duplicate breakpoint haplotype')
                result[current] = defaultdict(list); previous = {}; continue
            require(current and len(row) == 4, 'Malformed breakpoint row')
            population, ch, end, cm = row; end = int(end); cm = float(cm)
            require(ch in maps and population in populations and math.isfinite(cm), 'Breakpoint population/chromosome invalid')
            start = previous.get(ch, maps[ch][0][0]-1)+1
            require(end >= start and (end <= maps[ch][-1][0] or end == 2147483647), 'Unexpected terminal coordinate')
            clipped_end = min(end, maps[ch][-1][0])
            result[current][ch].append(dict(start=start, end=clipped_end, ancestry=population,
                                            donor=None, possible_copies={0, 1}, markers=0))
            previous[ch] = clipped_end
    expected = {f'Sample_{s}_{h}' for s in range(1,n+1) for h in (1,2)}
    require(set(result) == expected, 'Breakpoint sample count mismatch')
    for blocks in result.values():
        require(set(blocks) == set(maps), 'Breakpoint chromosome missing')
        require(all(blocks[ch][-1]['end'] == maps[ch][-1][0] for ch in maps), 'Incomplete breakpoint territory')
    return result


def separate_and_verify(c, files, raw, bp, out, donors, maps, summary, started=None):
    """Verify each copied allele; preserve unresolved homolog identity honestly."""
    n, populations = read_model(files['model'], c['resources']['popsize'])
    blocks = breakpoints(bp, maps, n, populations)
    endpoints = {s:{ch:[b['end'] for b in intervals] for ch,intervals in by_chrom.items()}
                 for s,by_chrom in blocks.items()}
    donor_iter = vcf_rows(files['donor_vcf']); donor_samples, sentinel = next(donor_iter)
    require(sentinel is None, 'Donor VCF header malformed')
    donor_index = {name: i for i, name in enumerate(donor_samples)}
    allow_identical = input_policy(c)['unphased_homozygotes'] == 'ALLOW_IDENTICAL_COPIES'
    rows = 0
    with text_open(raw) as h, (out/'genotypes.vcf').open('x') as predictor, gzip.open(out/'variant_annotations.private.tsv.gz','wt') as annotation:
        annotation.write('chromosome\tposition_bp\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
        for line in h:
            if line.startswith('##FORMAT=') and 'ID=GT,' not in line: continue
            if line.startswith('##INFO='): continue
            if line.startswith('##'): predictor.write(line); continue
            if line.startswith('#CHROM'):
                header = line.rstrip().split('\t'); samples = header[9:]
                require(samples == [f'Sample_{i}' for i in range(1,n+1)], 'Simulator sample order mismatch')
                predictor.write('\t'.join(header[:9]+[c['sample_namespace']+'__'+s for s in samples])+'\n'); continue
            row = line.rstrip().split('\t'); require(len(row) == n+9, 'Simulated VCF sample count mismatch')
            original = next(donor_iter, None); require(original is not None, 'Simulator added variants')
            donor_gts = genotypes(original[1], allow_identical)
            simulated_gts = genotypes(row)
            require(row[:5] == original[1][:5], 'Simulator altered variant keys/alleles/order')
            annotation.write('\t'.join(original[1][:8])+'\n')
            fields = row[8].split(':'); require('SAMPLE' in fields and 'POP' not in fields, 'Expected SAMPLE-only haptools trace')
            si = fields.index('SAMPLE'); ch, pos = row[0], int(row[1])
            for i, field in enumerate(row[9:]):
                names = field.split(':')[si].split(','); require(len(names) == 2, 'Missing donor-copy provenance')
                for hap, name in enumerate(names):
                    require(name in donors, 'Unadmitted donor in simulated output')
                    key = f'{samples[i]}_{hap+1}'; intervals = blocks[key][ch]
                    index = bisect_left(endpoints[key][ch], pos); block = intervals[index]
                    require(block['start'] <= pos <= block['end'], 'Variant outside ancestry truth')
                    require(donors[name] == block['ancestry'], 'Donor ancestry disagrees with simulated ancestry truth')
                    require(block['donor'] in (None, name), 'Donor changed within one simulated ancestry block')
                    block['donor'] = name; block['markers'] += 1
                    compatible = {k for k in (0,1) if donor_gts[donor_index[name]][k] == simulated_gts[i][hap]}
                    block['possible_copies'] &= compatible
                    require(block['possible_copies'], 'Copied alleles incompatible with a single donor homolog')
            predictor.write('\t'.join(row[:7]+['.','GT']+['|'.join(map(str,g)) for g in simulated_gts])+'\n')
            rows += 1
            if started is not None and rows % 4096 == 0: resource_guard(out,c['resources'],started)
    require(next(donor_iter, None) is None and rows == sum(summary['variants_by_chromosome'].values()), 'Simulator removed variants')
    import pysam
    pysam.tabix_compress(str(out/'genotypes.vcf'), str(out/'genotypes.vcf.gz'), force=False)
    pysam.tabix_index(str(out/'genotypes.vcf.gz'), preset='vcf', force=False)
    (out/'genotypes.vcf').unlink()  # Only the temporary file created above, after indexed output succeeds.
    states = Counter()
    with gzip.open(out/'truth.private.tsv.gz', 'wt') as h:
        writer = csv.writer(h, delimiter='\t'); writer.writerow(('sample_id','haplotype','chromosome','start_bp_inclusive',
          'end_bp_inclusive','ancestry','donor_id','compatible_donor_copies_zero_based','markers','copy_identity_status'))
        for sample_hap, by_chrom in blocks.items():
            sample, hap = sample_hap.rsplit('_',1)
            for ch, intervals in by_chrom.items():
                for block in intervals:
                    status = ('NO_OBSERVED_MARKER' if not block['markers'] else
                              'UNIQUE_BY_ALLELE_COMPATIBILITY' if len(block['possible_copies']) == 1 else 'AMBIGUOUS_IDENTICAL_ALLELES')
                    states[status] += 1
                    writer.writerow((c['sample_namespace']+'__'+sample,hap,ch,block['start'],block['end'],block['ancestry'],
                      block['donor'] or 'UNKNOWN',','.join(map(str,sorted(block['possible_copies']))) if block['markers'] else '.',block['markers'],status))
    return dict(copied_alleles_checked=rows*n*2, blocks_by_copy_status=dict(states),
                truth_is_simulated_not_observed_dnabr=True, donor_homolog_exported_by_haptools=False)


def resource_guard(out, resources, started):
    require(time.monotonic()-started <= resources['timeout_seconds'], 'Simulation task deadline exceeded')
    require(shutil.disk_usage(out).free >= resources['disk_reserve_gib']*1024**3, 'Disk reserve violated')
    require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) <= resources['max_output_gib']*1024**3,
            'Output budget exceeded')


def local_filesystem(path):
    """Inspect Linux mount metadata; a directory name is not a storage check."""
    resolved = Path(path).resolve(); matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        mount = Path(re.sub(r'\\([0-7]{3})',lambda m:chr(int(m.group(1),8)),left.split()[4]))
        if resolved == mount or mount in resolved.parents: matches.append((len(str(mount)),right.split()[0]))
    require(matches, 'Cannot identify work filesystem')
    kind = max(matches)[1]
    require(kind in {'ext2','ext3','ext4','xfs','btrfs','overlay','tmpfs'}, 'Work must use local/block disk, not '+kind)
    return kind


def execute_tool(args, out, resources, started, *, log_name='haptools.private.log', tool_name='haptools'):
    """Bound the actual child, preserving logs and incomplete files on failure."""
    require(Path(log_name).name == log_name and log_name.endswith('.log'), 'Plain log filename required')
    with (out/log_name).open('x') as h:
        child = subprocess.Popen(args, stdout=h, stderr=subprocess.STDOUT,
                                 env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        try:
            while child.poll() is None:
                resource_guard(out, resources, started)
                try: child.wait(timeout=1)
                except subprocess.TimeoutExpired: pass
            require(child.returncode == 0, tool_name+' failed; inspect retained private log')
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait()


def run(path, digest, out, executable='haptools'):
    started = time.monotonic(); c, files, admission = load_contract(path, digest)
    require(importlib.metadata.version('haptools') == VERSION, 'Installed haptools version mismatch')
    executable = shutil.which(executable)
    require(executable is not None, 'haptools executable unavailable')
    observed_version = subprocess.run([executable,'--version'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    require(observed_version == VERSION, 'Executable haptools version mismatch')
    require(not out.exists(), 'Output directory exists; use Nextflow resume, never overwrite outputs')
    filesystem = local_filesystem(out.parent)
    require(shutil.disk_usage(out.parent).free >= c['resources']['disk_reserve_gib']*1024**3, 'Insufficient local disk reserve')
    out.mkdir(mode=0o700)
    summary, donors, maps = prepare(c, files, out/'prepared')
    args = command(c, files, out/'prepared', out/'raw.private.vcf.gz', executable)
    execute_tool(args, out, c['resources'], started)
    verification = separate_and_verify(c, files, out/'raw.private.vcf.gz', out/'raw.private.bp', out, donors, maps, summary,started)
    load_contract(path, digest)
    resource_guard(out, c['resources'], started)
    outputs = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    manifest = dict(schema='r02_lai_simulate_output_v1', scope=c['scope'],
      status='COMPLETE_SYNTHETIC_SOFTWARE_TEST' if c['scope']=='SYNTHETIC_TEST' else 'COMPLETE_SIMULATED_SCENARIO_NOT_LAI_VALIDATION',
      input_contract_sha256=digest, job_id=c['job_id'], donor_split=c['donor_split'], genome_build=c['genome_build'],
      chromosomes=c['chromosomes'], sample_namespace=c['sample_namespace'], summary=summary, verification=verification,
      command=args, haptools_version=VERSION, environment={p:importlib.metadata.version(p) for p in ('haptools','numpy','pysam','cyvcf2','pgenlib')},
      executable_sha256=sha(executable),
      outputs_sha256=outputs, predictor_inputs=['genotypes.vcf.gz','genotypes.vcf.gz.tbi'],
      private_evaluation_only=['truth.private.tsv.gz','raw.private.bp','raw.private.vcf.gz'],
      contains_sample_identifiers=True, public_distribution_allowed=False,
      elapsed_seconds=time.monotonic()-started, distinct_donor_count=summary['distinct_donors'],
      work_filesystem=filesystem,
      independent_biological_replicates='NOT_ESTIMATED_BY_SIMULATOR',
      scientific_admission_sha256=c['inputs']['admission']['sha256'])
    if c['schema'] == SCHEMA_V2:
        manifest['preparation_provenance'] = c.get('preparation_provenance', {})
        manifest['input_policy'] = input_policy(c)
    write(out/'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True, type=Path); p.add_argument('--contract-sha256', required=True)
    p.add_argument('--output-dir', required=True, type=Path); p.add_argument('--haptools', default='haptools')
    args = p.parse_args(); os.umask(0o077)
    run(args.contract, args.contract_sha256, args.output_dir, args.haptools)


if __name__ == '__main__': main()
