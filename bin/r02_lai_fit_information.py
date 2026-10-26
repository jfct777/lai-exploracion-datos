#!/usr/bin/env python3
"""Describe existing FIT EVENT information; never select filters or fit models.

Reuse authenticated unmodified FULL doses from the four-condition preparation.
Coordinates are not independent observations and cM spacing is not measured LD.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from r02_lai_control_views import array_sha256, sha256
from r02_lai_event_annotations import counted_is_alt


def require(ok, message):
    if not ok:
        raise ValueError(message)


def authenticated(path, expected):
    path = Path(path)
    require(sha256(path) == expected, f"Hash differs: {path}")
    return path


def describe(values):
    values = np.asarray(values)
    return (dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'),
                     map(float, np.quantile(values, [0, .25, .5, .75, .95, 1]))))
            if values.size else None)


def counts(dose, observed):
    require(dose.ndim == 2 and dose.shape == observed.shape
            and observed.dtype == np.bool_, 'Invalid dose/mask')
    require(np.isin(dose[observed], [0, 1, 2]).all(), 'Invalid diploid dose')
    n = observed.sum(axis=0)
    hist = np.stack([((dose == v) & observed).sum(axis=0) for v in (0, 1, 2)])
    ac = hist[1] + 2 * hist[2]
    mean = np.divide(ac, n, out=np.zeros(len(n), dtype=float), where=n > 0)
    second = np.divide(hist[1] + 4 * hist[2], n,
                       out=np.zeros(len(n), dtype=float), where=n > 0)
    variance = np.maximum(0, second - mean**2)
    variance[n == 0] = np.nan
    return dict(n=n, n0=hist[0], n1=hist[1], n2=hist[2], ac=ac,
                an=2*n, carriers=hist[1]+hist[2], variance=variance,
                variable=(hist > 0).sum(axis=0) > 1)


def window_counts(evidence_cm, query_cm, radius_cm):
    evidence_cm, query_cm = np.asarray(evidence_cm), np.asarray(query_cm)
    require(radius_cm > 0 and np.isfinite(radius_cm), 'Invalid radius')
    require(np.isfinite(evidence_cm).all() and np.isfinite(query_cm).all()
            and np.all(np.diff(evidence_cm) >= 0), 'Invalid genetic coordinates')
    return (np.searchsorted(evidence_cm, query_cm + radius_cm, side='right')
            - np.searchsorted(evidence_cm, query_cm - radius_cm, side='left'))


def table(desc):
    p = authenticated(desc['path'], desc['sha256'])
    with gzip.open(p, 'rt') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    require(len(rows) == desc['rows'], 'Catalogue row count differs')
    return rows


def identity(row):
    allele = row['alt'] if counted_is_alt(row['ref'], row['alt'], row['counted_allele']) else row['ref']
    return (row['chromosome'], row['position_bp'], row['ref'], row['alt'], allele)


def run(manifest_path, expected_manifest_sha256, task_id, output):
    started = time.monotonic()
    manifest = json.loads(authenticated(manifest_path, expected_manifest_sha256).read_text())
    require(len(manifest['tasks']) == 6 and len({t['task_id'] for t in manifest['tasks']}) == 6,
            'Expected six distinct FIT tasks')
    plan = json.loads(authenticated(manifest['source_plan'], manifest['source_plan_sha256']).read_text())
    admitted = {t['task_id']: t for t in plan['tasks'] if t['role'] == 'FIT'}
    require(set(admitted) == {t['task_id'] for t in manifest['tasks']}, 'FIT case universe differs')
    completion = json.loads(authenticated(manifest['control_completion']['path'],
        manifest['control_completion']['sha256']).read_text())
    completed = {t['task_id']: t for t in completion['products']}
    task = next(t for t in manifest['tasks'] if t['task_id'] == task_id)
    require(task['role'] == 'FIT', 'Only FIT is admitted')
    require(task['control_receipt_sha256'] == completed[task_id]['receipt_sha256']
            and task['control_batch'] == completed[task_id]['batch']
            and task['event_catalogue'] == admitted[task_id]['inputs']['common_event_core_catalogues']['event'],
            'Manifest differs from original admitted plan/completion')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = output / 'summary.json'
    signature = {'manifest_sha256': sha256(manifest_path), 'code_sha256': sha256(__file__),
                 'task_id': task_id}
    if final.exists():
        prior = json.loads(final.read_text())
        require(prior['signature'] == signature, 'Existing output belongs to different diagnostic')
        authenticated(output/'locus-counts.tsv.gz', prior['locus_table_sha256'])
        return {'status': 'FINISHED_SKIP', 'task_id': task_id}
    cr = json.loads(authenticated(task['control_receipt'], task['control_receipt_sha256']).read_text())
    ir = json.loads(authenticated(task['index_receipt'], task['index_receipt_sha256']).read_text())
    require(cr['role'] == ir['role'] == 'FIT' and cr['task_id'] == ir['task_id'] == task_id
            and cr['factorized_receipt_sha256'] == task['index_receipt_sha256']
            and ir['plan_sha256'] == manifest['source_plan_sha256']
            and cr['batch_sha256'] == completed[task_id]['batch_sha256']
            and not cr['truth_loaded'] and not cr['evaluate_opened'], 'Receipt identities differ')
    cp = authenticated(task['control_batch'], cr['batch_sha256'])
    ip = authenticated(task['index_arrays'], ir['outputs_sha256']['indices.npz'])
    with np.load(cp, allow_pickle=False) as src:
        control = {k: src[k] for k in ('NONE_event_dosage', 'NONE_event_valid',
            'person_id', 'event_source_index', 'event_chromosome', 'event_position_bp',
            'event_ref', 'event_alt', 'event_counted_allele', 'event_is_alt')}
    for k, v in control.items():
        require(array_sha256(v, np) == cr['arrays_sha256'][k], 'Control array differs: ' + k)
    with np.load(ip, allow_pickle=False) as src:
        indices = {k: src[k] for k in src.files if k.startswith('event_') or k == 'person_id'}
    for k in control:
        if not k.startswith('NONE_'):
            require(np.array_equal(control[k], indices[k]), 'Axis differs: ' + k)
    dose, obs = control['NONE_event_dosage'], control['NONE_event_valid']
    require(dose.shape == (ir['people'], ir['loci']['event']), 'Dose dimensions differ')
    rows, old = table(task['event_catalogue']), table(task['old_event_catalogue'])
    current_ids, old_ids = set(map(identity, rows)), set(map(identity, old))
    require(len(current_ids) == len(rows) and old_ids <= current_ids,
            'Historical EVENT removed or duplicated; investigate before comparing')
    cm = np.asarray(indices['event_position_cm'], dtype=float)
    bp = np.asarray(indices['event_position_bp'], dtype=np.int64)
    require(np.all(np.diff(bp) > 0) and np.all(np.diff(cm) >= 0), 'Unsorted EVENT axis')
    for k in ('chromosome', 'position_bp', 'ref', 'alt', 'counted_allele'):
        require(np.array_equal(np.array([r[k] for r in rows]), indices['event_' + k]),
                'Catalogue/index identity differs: ' + k)
    require(np.array_equal(np.array([counted_is_alt(r['ref'], r['alt'], r['counted_allele'])
                                    for r in rows]), indices['event_is_alt']), 'Counted allele differs')
    source_ac = np.array([int(r['source_minor_ac']) for r in rows])
    source_an = np.array([int(r['source_an']) for r in rows])
    require((source_an > 0).all() and (source_ac >= 0).all()
            and (2*source_ac <= source_an).all(), 'Invalid source minor count')
    qp = authenticated(ir['query_overlay']['path'], ir['query_overlay']['sha256'])
    query = {}
    query_members = ('query_cm', 'query_pos_bp', 'query_valid', 'query_score_weight_cm')
    with np.load(qp, allow_pickle=False) as src:
        for k in query_members:
            a = src[k]
            require(a.shape[0] == ir['people'] and np.array_equal(a, np.broadcast_to(a[0], a.shape)),
                    'Person-specific query geometry requires different summary')
            query[k] = a[0]
    require(query['query_valid'].dtype == np.bool_ and query['query_valid'].all(),
            'Query validity differs')
    q, w = query['query_cm'], query['query_score_weight_cm']
    require(np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0, 'Query weights differ')
    c = counts(dose, obs)
    carried = (dose > 0) & obs
    geometry = {}
    old_cm = np.array([float(r['position_cm']) for r in old])
    for radius in manifest['descriptive_radii_cm']:
        all_n = window_counts(cm, q, radius)
        variable_n = window_counts(cm[c['variable']], q, radius)
        old_n = window_counts(old_cm, q, radius)
        exposed = np.array([(window_counts(cm[person], q, radius) > 0).mean() for person in carried])
        geometry[str(radius)] = dict(queries=len(q), queries_without_catalogue_event=int((all_n == 0).sum()),
            queries_without_variable_event=int((variable_n == 0).sum()),
            queries_without_old_event=int((old_n == 0).sum()), events_per_query=describe(all_n),
            genetic_weight_fraction_without_catalogue_event=float(w[all_n == 0].sum()/w.sum()),
            person_fraction_queries_with_carried_event=describe(exposed))
    tmp = output/'locus-counts.tmp.gz'
    with gzip.open(tmp, 'wt', newline='') as stream:
        fields = ('chromosome', 'position_bp', 'position_cm', 'ref', 'alt', 'counted_allele',
                  'n', 'n0', 'n1', 'n2', 'ac', 'an', 'carriers', 'variance', 'variable')
        writer = csv.DictWriter(stream, fields, delimiter='\t'); writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({**{k: row[k] for k in fields[:6]},
                             **{k: (None if k == 'variance' and not c['n'][i] else c[k][i]) for k in fields[6:]}})
    os.replace(tmp, output/'locus-counts.tsv.gz')
    references = {}
    require(indices['event_reference_ancestry_order'].tolist() == ['African', 'European', 'Native_American'],
            'Reference ancestry order differs')
    for k in ('event_reference_ac', 'event_reference_an', 'event_reference_known'):
        require(indices[k].shape == (len(cm), 3), 'Reference shape differs: ' + k)
    known = indices['event_reference_known']
    rac, ran = indices['event_reference_ac'], indices['event_reference_an']
    require(known.dtype == np.bool_ and np.issubdtype(rac.dtype, np.integer)
            and np.issubdtype(ran.dtype, np.integer) and (rac[known] >= 0).all()
            and (ran[known] >= rac[known]).all(),
            'Reference AC/AN domain differs')
    for j, ancestry in enumerate(indices['event_reference_ancestry_order']):
        known = indices['event_reference_known'][:, j]
        ac, an = indices['event_reference_ac'][:, j], indices['event_reference_an'][:, j]
        references[str(ancestry)] = dict(known_loci=int(known.sum()),
            loci_with_counted_copies=int((known & (ac > 0)).sum()),
            ac=describe(ac[known]), an=describe(an[known]),
            effective_independent_donors=None)
    result = dict(signature=signature, status='FIT_INFORMATION_DESCRIBED_NOT_POWER_OR_TRAINING',
        chromosome=ir['chromosome'], role='FIT', people=ir['people'], event_loci=len(cm),
        old_event_loci=len(old), added_loci=len(current_ids-old_ids),
        added_ref_minor=sum(identity(r) not in old_ids and r['counted_allele'] == r['ref'] for r in rows),
        common_loci=ir['loci']['common'], query_positions=len(q),
        query_bp_extent=[int(query['query_pos_bp'][0]), int(query['query_pos_bp'][-1])],
        query_cm_extent=[float(q[0]), float(q[-1])],
        event_bp_extent=[int(bp[0]), int(bp[-1])], event_spacing_cm=describe(np.diff(cm)),
        event_spacing_bp=describe(np.diff(bp)), event_equal_cm_neighbors=int((np.diff(cm)==0).sum()),
        observed_genotypes=int(obs.sum()), possible_genotypes=int(obs.size),
        loci_without_observed_genotypes=int((c['n']==0).sum()),
        loci_without_carriers=int(((c['carriers']==0)&(c['n']>0)).sum()),
        variable_loci=int(c['variable'].sum()), carrier_count_per_locus=describe(c['carriers']),
        person_carried_loci=describe(carried.sum(axis=1)),
        person_counted_copies=describe(np.where(obs,dose,0).sum(axis=1)),
        person_observed_loci=describe(obs.sum(axis=1)), references=references,
        geometry=geometry, truth_members_opened=[], select_or_evaluate_arrays_opened=[],
        query_members_opened=list(query_members), radius_selected=False, filters_changed=False,
        training_started=False, locus_table_sha256=sha256(output/'locus-counts.tsv.gz'),
        elapsed_seconds=time.monotonic()-started,
        maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations=['Mosaics and loci are dependent; no power calculation or independent donor count.',
                    'Physical span is not sequenced/callable territory; cM spacing is not LD.',
                    'Reference AC/AN describe reference panel, not true local ancestry in mosaics.',
                    'Radii are descriptive previously proposed scales, not selected hyperparameters.'])
    (output/'summary.tmp').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    os.replace(output/'summary.tmp', final)
    return {k: result[k] for k in ('status', 'chromosome', 'variable_loci', 'elapsed_seconds')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--task-id', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(run(a.manifest, a.manifest_sha256, a.task_id, a.output)))
