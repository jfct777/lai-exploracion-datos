#!/usr/bin/env python3
"""Compose existing L1 projection, REF audit and genotype-count algorithms.

Each invocation is one cached Nextflow stage. The job manifest authenticates
inputs and fixes roles, chromosome and QC semantics. No TEST, simulation,
phasing, ancestry reassignment or new rarity threshold is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import r02_lai_allele_support as base
import r02_lai_build_verification as build
import r02_lai_role_subset as subset


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def authenticate_job(path, input_dir):
    data = base.read_json(path)
    base.require(data.get('schema') == 'r02_lai_phased_job_v1', 'Unknown phased L1 job')
    base.require(data.get('chromosome') in {str(i) for i in range(1, 23)}, 'Autosome required')
    base.require(data.get('panel_role') in subset.ALLOWED_ROLES, 'Forbidden panel role; TEST is closed')
    base.require(data.get('site_filter_policy') == 'PASS_OR_UNFILTERED', 'Frozen phased-panel site policy differs')
    base.require(data.get('quality') == {'mode': 'GT_ONLY', 'contract_id': 'observed_gt_no_dp_gq'},
                 'This audit requires explicit GT_ONLY; quality fields are not available')
    files = {}
    names = []
    for key, item in data['files'].items():
        name = item['name']
        base.require(Path(name).name == name and name not in ('.', '..'), 'Unsafe staged filename')
        names.append(name)
        target = Path(input_dir) / name
        base.require(target.is_file() and base.sha256(target) == item['sha256'], 'Input hash mismatch: ' + key)
        files[key] = target
    base.require(len(names) == len(set(names)), 'Staged input basenames must be unique')
    base.require(set(files) == {'rare_vcf', 'rare_contract', 'phased_vcf', 'roles', 'fasta', 'fai',
                               'reference_contract', 'contig_map', 'genetic_map'}, 'Incomplete or unexpected job files')
    base.require(str(files['fai']) == str(files['fasta']) + '.fai', 'FASTA index must be adjacent')
    for key in ('expected_source_samples', 'expected_rare_sites', 'expected_panel_samples',
                'expected_panel_records', 'expected_role_samples', 'max_line_bytes', 'timeout_seconds',
                'max_panel_bytes', 'max_panel_index_bytes', 'resource_check_rows'):
        base.require(type(data.get(key)) is int and data[key] > 0, 'Positive integer required: ' + key)
    base.require(type(data.get('min_free_gib')) in (int, float) and math.isfinite(data['min_free_gib']) and data['min_free_gib'] >= 0,
                 'Invalid disk reserve')
    return data, files


def audit_reference(data, files, vcf, out):
    args = SimpleNamespace(vcf=str(vcf), expected_vcf_sha256=base.sha256(vcf),
        fasta=str(files['fasta']), reference_contract=str(files['reference_contract']),
        expected_reference_contract_sha256=data['files']['reference_contract']['sha256'],
        contig_map=str(files['contig_map']), expected_contig_map_sha256=data['files']['contig_map']['sha256'],
        chromosome=data['chromosome'], reference_contig='chr' + data['chromosome'],
        max_records=max(data['expected_rare_sites'], data['expected_panel_records']),
        max_line_bytes=data['max_line_bytes'], timeout_seconds=data['timeout_seconds'],
        min_free_gib=data['min_free_gib'], bcftools='bcftools', out=str(out))
    return build.run(args)


def prepare_role(data, files, out):
    contract = dict(schema='r02_lai_role_subset_contract_v2', role=data['panel_role'],
        chromosome=data['chromosome'], expected_source_samples=data['expected_panel_samples'],
        expected_selected_samples=data['expected_role_samples'], expected_ancestries=data['expected_ancestries'],
        expected_records=data['expected_panel_records'], bcftools_version=data['bcftools_version'],
        input_sha256={key: base.sha256(files['phased_vcf'] if key == 'source_vcf' else files[key])
                      for key in ('source_vcf', 'roles')})
    contract_path = out / 'subset_contract.json'
    write_json(contract_path, contract)
    receipt = subset.run(SimpleNamespace(contract=str(contract_path), source_vcf=str(files['phased_vcf']),
        roles=str(files['roles']), ref_receipt=None, outdir=str(out / 'projection'),
        min_free_gib=data['min_free_gib'], max_line_bytes=data['max_line_bytes'],
        timeout_seconds=data['timeout_seconds'], bcftools='bcftools'))
    panel = out / 'projection' / f"{data['panel_role'].lower()}.chr{data['chromosome']}.vcf.gz"
    audit_reference(data, files, panel, out / 'panel_build_verification.json')
    write_json(out / 'complete.json', {'status': 'PREPARED_ROLE_AND_REFERENCE_CHECKED_NOT_PHASE_VALIDATED',
        'panel_role': data['panel_role'], 'chromosome': data['chromosome'],
        'projection_manifest_sha256': base.sha256(out / 'projection' / 'manifest.json'),
        'panel_vcf_sha256': base.sha256(panel), 'source_test_projected': False})
    return receipt


def read_map(path, chromosome):
    """Read the established SHAPEIT Position(bp)/Rate(cM/Mb)/Map(cM) map."""
    opener = gzip.open if str(path).endswith('.gz') else open
    first, last, prev_cm, records = None, 0, -math.inf, 0
    with opener(path, 'rt') as handle:
        header = handle.readline().strip().split()
        base.require(header in (['pos', 'chr', 'cM'], ['Position(bp)', 'Rate(cM/Mb)', 'Map(cM)'],
                                ['position', 'COMBINED_rate(cM/Mb)', 'Genetic_Map(cM)']),
                     'Unrecognized genetic-map columns')
        for line in handle:
            fields = line.split()
            base.require(len(fields) == 3, 'Invalid map row')
            bp, cm = int(fields[0]), float(fields[2])
            if header[1] == 'chr':
                base.require(fields[1] in (chromosome, 'chr' + chromosome), 'Map chromosome differs')
            base.require(bp > last and math.isfinite(cm) and cm >= prev_cm, 'Nonmonotone or invalid genetic map')
            if first is None:
                first = bp
            last, prev_cm, records = bp, cm, records + 1
    base.require(records >= 2, 'Map requires at least two points')
    return first, last, records


def summarize_map_support(table, genetic_map, chromosome, output):
    low, high, points = read_map(genetic_map, chromosome)
    counts = Counter()
    with gzip.open(table, 'rt') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['panel_ancestry'] != 'ALL':
                continue
            scope = 'inside' if low <= int(row['position_bp']) <= high else 'outside'
            counts['source_sites_' + scope] += 1
            carriers = row['n_phased_carriers_called']
            if carriers != 'NA' and int(carriers) > 0:
                counts['sites_with_observed_phased_carriers_' + scope] += 1
                counts['observed_phased_person_site_portations_' + scope] += int(carriers)
    result = dict(schema='r02_lai_phased_map_support_v1', status='DESCRIPTIVE_NOT_PHASE_OR_LAI_VALIDATION',
        chromosome=chromosome, map_sha256=base.sha256(genetic_map), map_points=points,
        first_bp=low, last_bp=high, counts=dict(counts), extrapolation=False,
        interpretation='ALL sites counted once; person-site portations are not independent people. Within endpoints is not phase accuracy or local map validation.')
    write_json(output, result)
    return result


def genotype_support(data, files, prepared, source_proof, out):
    prepared = Path(prepared)
    projection = base.read_json(prepared / 'projection' / 'manifest.json')
    base.require(projection['role'] == data['panel_role'] and projection['chromosome'] == data['chromosome'],
                 'Projection chromosome/role mismatch')
    panel = prepared / 'projection' / f"{data['panel_role'].lower()}.chr{data['chromosome']}.vcf.gz"
    base.require(base.sha256(panel) == projection['output_sha256'][panel.name], 'Projected VCF hash mismatch')
    panel_proof = prepared / 'panel_build_verification.json'
    inputs = {k: files[k] for k in ('rare_vcf', 'rare_contract', 'roles', 'contig_map')}
    inputs.update(panel_vcf=panel, source_build_verification=Path(source_proof), panel_build_verification=panel_proof)
    contract = dict(schema='r02_lai_genotype_contract_v1', panel_role=data['panel_role'],
        panel_sample_order_sha256=projection['sample_order_sha256'],
        panel_sample_members_sha256=projection['sample_members_sha256'], site_filter_policy=data['site_filter_policy'],
        quality=data['quality'], allele_representation=dict(status='UNKNOWN', panel_vcf_sha256=base.sha256(panel),
            evidence='Mechanical role projection verified against authenticated phased VCF. Original unsplit/decomposition history is not established; REF dosage may remain unresolved.'))
    inputs['genotype_contract'] = out / 'genotype_contract.json'
    write_json(inputs['genotype_contract'], contract)
    evidence = {'schema': 'r02_lai_build_evidence_v1'}
    for name, proof in (('source', Path(source_proof)), ('panel', panel_proof)):
        record = base.read_json(proof)
        evidence[name] = dict(build=record['build'], status='VERIFIED', fasta_sha256=record['fasta_sha256'],
            evidence='Authenticated all-record REF/FASTA audit, not QC or phase certification',
            verification_receipt={'path': str(proof), 'sha256': base.sha256(proof)})
    inputs['build_evidence'] = out / 'build_evidence.json'
    write_json(inputs['build_evidence'], evidence)
    hashes = out / 'input_hashes.json'
    write_json(hashes, {key: base.sha256(value) for key, value in inputs.items()})
    args = SimpleNamespace(**{k: str(v) for k, v in inputs.items()}, input_hashes=str(hashes),
        mode='genotype_support', panel_role=data['panel_role'], chromosome=data['chromosome'],
        expected_source_samples=data['expected_source_samples'], expected_rare_sites=data['expected_rare_sites'],
        max_panel_records=data['expected_panel_records'], max_panel_bytes=data['max_panel_bytes'],
        max_panel_index_bytes=data['max_panel_index_bytes'], max_line_bytes=data['max_line_bytes'],
        min_free_gib=data['min_free_gib'], resource_check_rows=data['resource_check_rows'], outdir=str(out / 'genotype_support'))
    result = base.run(args)
    summarize_map_support(out / 'genotype_support' / 'soporte_alelos_dnabr.tsv.gz', files['genetic_map'],
                          data['chromosome'], out / 'map_support.json')
    # Separate publication bundle leaves the counter's original receipt intact.
    # Hard links avoid a second chromosome-sized table copy on the local disk.
    bundle = out / 'publication_bundle'
    bundle.mkdir(mode=0o700)
    for source, name in ((out / 'genotype_support' / 'soporte_alelos_dnabr.tsv.gz', 'soporte_alelos_dnabr.tsv.gz'),
                         (out / 'map_support.json', 'map_support.json')):
        os.link(source, bundle / name)
    published = dict(result)
    published['genotype_counter_manifest_sha256'] = base.sha256(out / 'genotype_support' / 'manifest.json')
    published['map_coverage'] = 'ENDPOINT_RANGE_ONLY_NO_EXTRAPOLATION_NOT_PHASE_VALIDATION'
    published['outputs_sha256'] = {p.name: base.sha256(p) for p in bundle.iterdir()}
    published['composition_code_sha256'] = base.sha256(__file__)
    write_json(bundle / 'manifest.json', published)
    write_json(out / 'complete.json', {'status': 'COMPLETE_GENOTYPE_AND_MAP_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION',
        'chromosome': data['chromosome'], 'panel_role': data['panel_role'],
        'manifest_sha256': base.sha256(out / 'genotype_support' / 'manifest.json'),
        'publication_manifest_sha256': base.sha256(bundle / 'manifest.json'),
        'map_support_sha256': base.sha256(out / 'map_support.json'), 'source_test_projected': False})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('source', 'role', 'support'))
    for key in ('job', 'input-dir', 'outdir'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--prepared')
    parser.add_argument('--source-proof')
    args = parser.parse_args(argv)
    data, files = authenticate_job(args.job, args.input_dir)
    out = Path(args.outdir)
    base.require(not out.exists(), 'Output exists; Nextflow resume must select completed tasks')
    base.free_disk(out.parent, data['min_free_gib'])
    out.mkdir(mode=0o700)
    if args.stage == 'source':
        audit_reference(data, files, files['rare_vcf'], out / 'source_build_verification.json')
    elif args.stage == 'role':
        prepare_role(data, files, out)
    else:
        base.require(args.prepared and args.source_proof, 'Support requires both completed audit stages')
        genotype_support(data, files, args.prepared, args.source_proof, out)
    # Re-authenticate immutable sources after the stage, before its receipt.
    authenticate_job(args.job, args.input_dir)
    write_json(out / 'stage_receipt.json', {'schema': 'r02_lai_phased_stage_v1', 'stage': args.stage,
        'job_sha256': base.sha256(args.job), 'code_sha256': base.sha256(__file__),
        'chromosome': data['chromosome'], 'panel_role': data['panel_role'], 'source_test_projected': False})


if __name__ == '__main__':
    base.os.umask(0o077)
    main()
