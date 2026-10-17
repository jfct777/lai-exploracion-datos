#!/usr/bin/env python3
"""Extract frozen REF--VALID pairs from existing Refined-IBD; never call IBD.

M27E's sum of reported copy-segment lengths and denominator are reproduced
using its reader. Territorial unions are additional, not IBD1/IBD2 or
callability. All outputs contain private identifiers and must stay private.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import audit_m27e_ibd_rare_transfer as m27e
from r02_segment_evidence import merge_intervals, interval_bp, require

SCHEMA = 'r02_lai_external_ibd_pairs_private_v1'
STATUS = 'COMPLETE_EXISTING_EXTERNAL_IBD_EXTRACTION_NOT_PEDIGREE_OR_SIMULATION'


def sha(path, compressed=False):
    digest = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')


def table(path, rows, fields=None):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def checked(item):
    path = Path(item['path'])
    require(path.is_file() and sha(path) == item['sha256'], 'Input SHA256 mismatch')
    if item.get('compression') == 'gzip':
        require(sha(path, True) == item['original_sha256'], 'Original-byte SHA256 mismatch')
    return path


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def strict_map(path, chrom):
    positions, cms = m27e.read_map(path, chrom)
    require(all(b > a for a, b in zip(positions, positions[1:])), 'Non-increasing map')
    require(all(math.isfinite(x) for x in cms), 'Nonfinite map')
    require(all(b >= a for a, b in zip(cms, cms[1:])), 'Decreasing map')
    return positions, cms


def cm_at(bp, positions, cms):
    require(positions[0] <= bp <= positions[-1], 'Outside authenticated map; extrapolation forbidden')
    return m27e.interpolate_cm(bp, positions, cms)


def extract_geometry(path, chrom, target_pairs, identities):
    """Stream all records strictly; retain only selected pairs and copy IDs.

    Exact or mirrored duplicate geometry is rejected, not silently discarded.
    No callability mask is invented; the complete reported callset is scanned.
    """
    selected, seen = defaultdict(list), set()
    with m27e.open_text(path) as source:
        for line_number, line in enumerate(source, 1):
            fields = line.rstrip('\n').split('\t')
            require(len(fields) == 9, 'Expected nine Refined-IBD fields')
            a, ha, b, hb, cc, lo, hi, lod, length = fields
            require(a in identities and b in identities and a != b, 'Unknown/self identity')
            require(ha in ('1', '2') and hb in ('1', '2'), 'Unknown haplotype copy')
            require(int(cc.removeprefix('chr')) == chrom, 'Chromosome mismatch')
            lo, hi, lod, length = int(lo), int(hi), float(lod), float(length)
            require(1 <= lo <= hi and math.isfinite(lod) and math.isfinite(length), 'Invalid segment')
            require(lod >= 3.0 and length >= 2.0, 'Frozen reporting floors violated')
            if a > b:
                a, b, ha, hb = b, a, hb, ha
            key = (a, ha, b, hb, lo, hi)
            require(key not in seen, 'Duplicate/mirrored segment geometry')
            seen.add(key)
            if (a, b) in target_pairs:
                selected[(a, b)].append(dict(sample_a=a, copy_a=ha, sample_b=b, copy_b=hb,
                    chrom=chrom, start_bp=lo, end_bp=hi, lod=lod, reported_cm=length,
                    source_line=line_number))
    return selected, len(seen)


def geometry(records, positions, cms):
    union = merge_intervals([(r['start_bp'], r['end_bp']) for r in records])
    by_copy = defaultdict(list)
    for row in records:
        by_copy[(row['copy_a'], row['copy_b'])].append((row['start_bp'], row['end_bp']))
    measure = lambda intervals: sum(cm_at(hi, positions, cms) - cm_at(lo, positions, cms)
                                   for lo, hi in merge_intervals(intervals))
    supported = all(positions[0] <= lo <= hi <= positions[-1] for lo, hi in union)
    return dict(any_copy_union_bp=interval_bp(union), any_copy_union_cm=measure(union) if supported else None,
                copy_pair_union_bp_sum=sum(interval_bp(v) for v in by_copy.values()),
                copy_pair_union_cm_sum=sum(measure(v) for v in by_copy.values()) if supported else None,
                union_cm_status='WITHIN_AUTHENTICATED_MAP' if supported else 'NA_SELECTED_SEGMENT_OUTSIDE_MAP',
                copy_pairs=','.join(a + '-' + b for a, b in sorted(by_copy)),
                union_pieces=len(union))


def historical_span(start, end, positions, cms):
    """Reproduce M27E only; explicitly expose its endpoint extrapolation."""
    left, right = max(0, positions[0]-start), max(0, end-positions[-1])
    return dict(observed_span_cm=m27e.interpolate_cm(end, positions, cms)-m27e.interpolate_cm(start, positions, cms),
                historical_left_extrapolated_bp=left, historical_right_extrapolated_bp=right,
                historical_left_extrapolated_cm=(cms[0]-m27e.interpolate_cm(start,positions,cms)) if left else 0.,
                historical_right_extrapolated_cm=(m27e.interpolate_cm(end,positions,cms)-cms[-1]) if right else 0.,
                historical_map_status='HISTORICAL_ENDPOINT_EXTRAPOLATION' if left or right else 'WITHIN_AUTHENTICATED_MAP')


def sum_known(values):
    return None if any(value is None for value in values) else sum(values)


def policy_values(total_cm, maximum_cm, genome_cm, policy):
    require(genome_cm > 0, 'Nonpositive denominator')
    phi = total_cm / (4.0 * genome_cm)
    return dict(m27e_reported_phi=phi,
        m27f_primary_edge=maximum_cm >= policy['max_segment_floor_cm'] and phi >= policy['kinship_floor'],
        diagnostic_ibd_phi_00221=phi >= .0221,
        diagnostic_ibd_phi_00221_and_anchor10=phi >= .0221 and maximum_cm >= policy['max_segment_floor_cm'])


def run(contract_path, digest, outdir):
    require(sha(contract_path) == digest, 'Contract hash mismatch')
    contract = json.loads(Path(contract_path).read_text())
    require(contract['chromosomes'] == list(range(1, 23)), 'Complete 22-autosome scan required')
    require(contract['mask_status'] == 'CALLABILITY_UNKNOWN_NO_ADDITIONAL_MASK', 'Undeclared mask')
    require(contract['map_policy'] == 'M27E_HISTORICAL_EXPLICIT_EXTRAPOLATION__NEW_UNION_STRICT_NA',
            'Historical reproduction and strict diagnostic map policies must be explicit')
    files = {key: checked(item) for key, item in contract['inputs'].items()}
    manifest = json.loads(files['m27e_manifest'].read_text())
    historical = json.loads(files['m27e_summary'].read_text())
    prereg = json.loads(files['m27f_policy'].read_text())
    policy = prereg['block_policy']
    require(policy['max_segment_floor_cm'] == 10.0 and policy['kinship_floor'] == 0.04419417382415922,
            'Historical policy changed')
    role_rows = rows(files['roles'])
    roles = {r['sample_id']: r for r in role_rows}
    require(len(roles) == len(role_rows) == 3685, 'Frozen role identity contract')
    queue = [r for r in rows(files['pc_pairs']) if r['role_pair'] == 'REF_TRAIN--SOURCE_VALID'
             and r['status_phi_00221'] == 'PC_SENSITIVE']
    pairs = {tuple(sorted((r['sample_a'], r['sample_b']))): r for r in queue}
    require(len(pairs) == len(queue) == 20, 'Expected exactly 20 frozen pairs')
    for a, b in pairs:
        require({roles[a]['role'], roles[b]['role']} == {'REF_TRAIN', 'SOURCE_VALID'}, 'Forbidden role')
        require(roles[a]['atomic_unit_id'] != roles[b]['atomic_unit_id'], 'Original unit crosses roles')
    nam = [r for r in role_rows if r['role'] == 'SOURCE_VALID' and r['ancestry'] == 'Native_American']
    require(len(nam) == 14 and len({r['atomic_unit_id'] for r in nam}) == 2, 'NAM unit contract')
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=False)
    all_segments, chr_rows, source_rows = [], [], []
    endpoints, spans, totals = {}, {}, {p: m27e.PairIbd() for p in pairs}
    n_all = 0
    for chrom in contract['chromosomes']:
        ipath, mpath, log = (files[f'{kind}{chrom}'] for kind in ('ibd', 'map', 'log'))
        for kind, name in (('ibd', f'ibd_chr_{chrom}.ibd'), ('map', f'genetic.map.chr{chrom}'),
                           ('log', f'ibd_chr_{chrom}.log')):
            item = contract['inputs'][f'{kind}{chrom}']
            require(item.get('original_sha256', item['sha256']) == manifest['inputs'][name],
                    'Source is not authenticated by M27E')
        require(m27e.parse_refined_ibd_log(log) == dict(version='refined-ibd.17Jan20.102.jar',
            n_samples=3685, minimum_length_cm=2.0, minimum_lod=3.0), 'Refined-IBD log contract')
        # Original reader: preserve its exact reported-sum algorithm and global endpoints.
        original, ends, receipt = m27e.read_ibd({chrom: ipath}, {s:s for s in roles},
            dict(reported_segment_min_lod=3.0, reported_segment_min_cm=2.0))
        selected, scanned = extract_geometry(ipath, chrom, pairs, roles)
        require(scanned == receipt['n_segments'] and receipt['n_duplicate_segment_keys'] == 0,
                'Scan/duplicate disagreement')
        endpoints[chrom] = ends[chrom]
        with tempfile.TemporaryDirectory(prefix='map-', dir=out) as tmp:
            plain = Path(tmp) / f'genetic.map.chr{chrom}'
            with gzip.open(mpath, 'rb') as src, plain.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            positions, cms = strict_map(plain, chrom)
        start, end = ends[chrom]
        historical_map = historical_span(start, end, positions, cms)
        span = historical_map['observed_span_cm']
        require(span > 0, 'Nonpositive observed chromosome span')
        spans[str(chrom)] = span
        for pair in sorted(pairs):
            values = original.get(pair, m27e.PairIbd())
            records = selected.get(pair, [])
            require(len(records) == values.n_segments and math.isclose(sum(r['reported_cm'] for r in records),
                    values.total_cm, abs_tol=1e-10), 'Original reader versus selected records disagreement')
            totals[pair].total_cm += values.total_cm
            totals[pair].max_cm = max(totals[pair].max_cm, values.max_cm)
            totals[pair].n_segments += values.n_segments
            row = dict(sample_a=pair[0], sample_b=pair[1], chrom=chrom,
                source_status='AUTHENTICATED_COMPLETE_REPORTED_CALLSET',
                reported_status='REPORTED_SEGMENTS' if records else 'NO_REPORTED_SEGMENT_NOT_BIOLOGICAL_ABSENCE',
                records=values.n_segments, max_reported_cm=values.max_cm, sum_reported_cm=values.total_cm,
                callability_status='UNKNOWN', **geometry(records, positions, cms))
            chr_rows.append(row)
            all_segments.extend(records)
        source_rows.append(dict(chrom=chrom, scanned_segments=scanned, observed_start_bp=start,
            observed_end_bp=end, map_start_bp=positions[0], map_end_bp=positions[-1], **historical_map,
            selected_segments=sum(len(v) for v in selected.values()), duplicate_segments=0,
            callability_status='UNKNOWN', **{kind + '_original_sha256': contract['inputs'][f'{kind}{chrom}'].get(
                'original_sha256', contract['inputs'][f'{kind}{chrom}']['sha256']) for kind in ('ibd','map','log')}))
        n_all += scanned
        print(json.dumps(dict(chromosome_complete=chrom, scanned_segments=scanned)), flush=True)
    genome = sum(spans.values())
    require(n_all == historical['n_segments'] == 925163, 'Historical segment count mismatch')
    require(math.isclose(genome, historical['observed_autosomal_map_span_cm'], abs_tol=1e-9, rel_tol=0),
            'Historical denominator mismatch')
    for chrom, span in spans.items():
        require(math.isclose(span, historical['per_chromosome_observed_map_span_cm'][chrom], abs_tol=1e-9, rel_tol=0),
                'Historical chromosome denominator mismatch')
    aggregate = []
    for pair, old in sorted(pairs.items()):
        value = totals[pair]
        per = [r for r in chr_rows if (r['sample_a'], r['sample_b']) == pair]
        aggregate.append(dict(old, source_autosomes_authenticated=22, reported_records=value.n_segments,
            reported_status='REPORTED_SEGMENTS' if value.n_segments else 'NO_REPORTED_SEGMENT_NOT_BIOLOGICAL_ABSENCE',
            sample_a_original_unit=roles[pair[0]]['atomic_unit_id'], sample_b_original_unit=roles[pair[1]]['atomic_unit_id'],
            max_reported_cm=value.max_cm, sum_reported_cm=value.total_cm,
            chromosomes_with_reported_segments=','.join(str(r['chrom']) for r in per if r['records']),
            any_copy_union_bp=sum(r['any_copy_union_bp'] for r in per),
            any_copy_union_cm=sum_known([r['any_copy_union_cm'] for r in per]),
            copy_pair_union_bp_sum=sum(r['copy_pair_union_bp_sum'] for r in per),
            copy_pair_union_cm_sum=sum_known([r['copy_pair_union_cm_sum'] for r in per]),
            union_cm_status='NA_SELECTED_SEGMENT_OUTSIDE_MAP' if any(r['any_copy_union_cm'] is None for r in per) else 'WITHIN_AUTHENTICATED_MAP',
            callability_status='UNKNOWN', **policy_values(value.total_cm, value.max_cm, genome, policy)))
    table(out/'pairs.private.tsv', aggregate)
    table(out/'pair_chromosomes.private.tsv', chr_rows)
    table(out/'segments.private.tsv', all_segments, ['sample_a','copy_a','sample_b','copy_b','chrom',
          'start_bp','end_bp','lod','reported_cm','source_line'])
    table(out/'sources.tsv', source_rows)
    summary = dict(status=STATUS, pairs=20, authenticated_ibd=22, authenticated_maps=22, authenticated_logs=22,
        all_source_records=n_all, selected_records=len(all_segments), observed_autosomal_map_span_cm=genome,
        original_nam_source_valid_people=14, original_nam_source_valid_units=2,
        pairs_with_reported_segments=sum(r['reported_records'] > 0 for r in aggregate),
        pairs_with_anchor10=sum(r['max_reported_cm'] >= 10 for r in aggregate),
        pairs_m27f_primary=sum(r['m27f_primary_edge'] for r in aggregate),
        pairs_ibd_phi_00221=sum(r['diagnostic_ibd_phi_00221'] for r in aggregate),
        pairs_ibd_phi_00221_and_anchor10=sum(r['diagnostic_ibd_phi_00221_and_anchor10'] for r in aggregate),
        selected_people=len({s for p in pairs for s in p}),
        selected_source_valid_people=len({s for p in pairs for s in p if roles[s]['role']=='SOURCE_VALID'}),
        selected_ref_train_people=len({s for p in pairs for s in p if roles[s]['role']=='REF_TRAIN'}),
        m27e_reported_phi_min=min(r['m27e_reported_phi'] for r in aggregate),
        m27e_reported_phi_max=max(r['m27e_reported_phi'] for r in aggregate),
        max_segment_cm_min=min(r['max_reported_cm'] for r in aggregate),
        max_segment_cm_max=max(r['max_reported_cm'] for r in aggregate),
        historical_denominator_extrapolated_chromosomes=[r['chrom'] for r in source_rows if r['historical_map_status']=='HISTORICAL_ENDPOINT_EXTRAPOLATION'],
        selected_pair_chromosomes_with_na_union_cm=sum(r['any_copy_union_cm'] is None for r in chr_rows),
        selected_pairs_with_na_union_cm=sum(r['any_copy_union_cm'] is None for r in aggregate),
        historical_map_policy=contract['map_policy'],
        mask_status=contract['mask_status'], callability_status='UNKNOWN', ibd_states='NOT_INFERRED',
        new_ibd_called=False, source_test_genotypes_opened=False, roles_changed=False, simulation_ready=False,
        independence_certified=False, interpretation='Reported call geometry and historical split-policy compatibility only')
    save(out/'summary.json', summary)
    save(out/'provenance.json', contract)
    outputs = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    save(out/'manifest.json', dict(schema=SCHEMA,status=STATUS,contract_sha256=digest,outputs_sha256=outputs))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.contract, args.sha256, args.outdir), sort_keys=True))
