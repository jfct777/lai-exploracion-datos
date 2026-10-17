#!/usr/bin/env python3
"""Contract-bound diploid LAI evaluation; no fitting or biological promotion.

Reuse M39 probability losses, M34 genetic-coordinate weights and M37 ordered
boundary matching. Arrays may be read-only NPY memmaps. Every model is scored
on the same truth-defined mask, never on its own available predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np

import m39_calibration as calibration
from m34_score_predictions import marker_weights
from m37_trace_score import boundaries, match_pairs

require = calibration.require
sha = calibration.sha256
STATES = calibration.STATE_NAMES
ANCESTRIES = ('AFR', 'EUR', 'NAM')
METRICS = ('clipped_log_loss', 'brier', 'accuracy', 'dosage_mae_AFR',
           'dosage_mae_EUR', 'dosage_mae_NAM', 'raw_log_loss')
TOLERANCES = (0.1, 0.2, 0.5)
WEIGHTINGS = ('observed_positions', 'genetic_cm_voronoi')
CODE_FILES = ('r02_lai_evaluate.py', 'm39_calibration.py', 'm34_score_predictions.py',
              'm37_trace_score.py', 'm37_trace_core.py')


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def grid_ranges(chromosome, positions, cm):
    """Exact ordered position grid; no interpolation, duplicates or reordering."""
    chrom, pos, cm = np.asarray(chromosome), np.asarray(positions), np.asarray(cm)
    require(chrom.ndim == pos.ndim == cm.ndim == 1 and len(chrom) == len(pos) == len(cm) > 0,
            'Grid dimensions differ')
    require(pos.dtype.kind in 'iu' and (pos > 0).all() and (pos <= np.iinfo(np.int64).max).all()
            and np.isfinite(cm).all(), 'Invalid grid coordinates')
    pos = pos.astype(np.int64, copy=False)
    values = [str(x) for x in chrom]
    require(all(x in {str(i) for i in range(1, 23)} for x in values), 'Autosomal chromosome labels required')
    groups = []
    start = 0
    for stop in range(1, len(values)+1):
        if stop == len(values) or values[stop] != values[start]:
            require(np.all(np.diff(pos[start:stop]) > 0) and np.all(np.diff(cm[start:stop]) >= 0),
                    'Duplicate, unordered or decreasing grid coordinates')
            groups.append((values[start], start, stop))
            start = stop
    require([int(x[0]) for x in groups] == sorted(set(int(x) for x in values)),
            'Chromosome grid must be contiguous and numerically ordered')
    return groups


def boundary_rates(counts):
    truth, predicted, matched = (int(counts[k]) for k in ('truth', 'predicted', 'matched'))
    return {**counts, 'precision': ratio(matched, predicted), 'recall': ratio(matched, truth),
            'f1': ratio(2*matched, truth+predicted),
            'mean_matched_error_cm': ratio(counts['absolute_error_cm'], matched),
            'false_transitions_per_cm': ratio(predicted-matched, counts['observed_span_cm'])}


def boundary_report(truth, hard, mask, cm, groups, memory_bytes):
    """Missing positions split runs; unordered states never invent haplotypes."""
    totals = {str(t): dict(truth=0, predicted=0, matched=0,
                          absolute_error_cm=0., observed_span_cm=0.) for t in TOLERANCES}
    for _, left, right in groups:
        observed = np.asarray(mask[left:right], dtype=bool)
        edges = np.flatnonzero(np.diff(np.r_[False, observed, False]))
        for begin, end in zip(edges[::2], edges[1::2]):
            begin, end = int(begin)+left, int(end)+left
            actual = boundaries(truth[begin:end], cm[begin:end])
            predicted = boundaries(hard[begin:end], cm[begin:end])
            # Existing M37 matcher stores a path in each dynamic-program cell.
            # Account for Python objects and copied paths, not just float cells.
            bound = (len(actual)+1)*(len(predicted)+1)*(256+32*min(len(actual), len(predicted)))
            require(bound <= memory_bytes, 'Boundary matcher exceeds configured memory bound')
            for tolerance in TOLERANCES:
                matches = match_pairs(actual, predicted, tolerance)
                row = totals[str(tolerance)]
                row['truth'] += len(actual)
                row['predicted'] += len(predicted)
                row['matched'] += len(matches)
                row['absolute_error_cm'] += sum(abs(actual[a][0]-predicted[b][0]) for a, b in matches)
                row['observed_span_cm'] += float(cm[end-1]-cm[begin])
    return {key: boundary_rates(value) for key, value in totals.items()}


def evaluate_arrays(probabilities, truth_state, observed_mask, *, chromosome, positions_bp,
                    genetic_cm, floor, chunk_positions, max_working_bytes,
                    expected_dosage_output=None, check_budget=None, max_query_gap_cm=None,
                    expected_dosage_input=None):
    """Score one arm, preserving masks and equal-person vs genomic weights.

    `floor` is an explicit numerical log-loss convention, not a harm tolerance.
    Missing truth has zero weight; invalid predictions at observed cells fail.
    An optional writable NPY array receives expected AFR/EUR/NAM copy counts.
    """
    p, y, mask = probabilities, truth_state, observed_mask
    require(p.ndim == 3 and p.shape[-1] == 6 and y.shape == mask.shape == p.shape[:2],
            'Expected predictions [person,position,6] and shared truth/mask axes')
    n, sites = y.shape
    if expected_dosage_input is not None:
        require(expected_dosage_input.shape == (n,sites,3),'Direct marginal dosage axes differ')
    require(n > 0 and sites > 0 and y.dtype.kind in 'iu' and mask.dtype.kind in 'bui',
            'Empty axes or invalid truth/mask dtype')
    require(type(chunk_positions) is int and chunk_positions > 0 and
            type(max_working_bytes) is int and max_working_bytes > 0, 'Invalid working budget')
    require(math.isfinite(floor) and 0 < floor < 1, 'Explicit probability floor required')
    base_bound = sites*96 + n*32768 + min(chunk_positions, sites)*2048
    require(base_bound < max_working_bytes, 'Array/block working-memory bound exceeded')
    groups = grid_ranges(chromosome, positions_bp, genetic_cm)
    require(sites == len(genetic_cm), 'Prediction and query grid differ')
    cm = np.asarray(genetic_cm, dtype=np.float64)
    gaps = 0
    if max_query_gap_cm is not None:
        require(math.isfinite(max_query_gap_cm) and max_query_gap_cm > 0, 'Explicit maximum query gap must be positive')
        supported = []
        for chrom, left, right in groups:
            cuts = [left, *(np.flatnonzero(np.diff(cm[left:right]) > max_query_gap_cm)+left+1).tolist(), right]
            gaps += len(cuts)-2
            supported.extend((chrom,a,b) for a,b in zip(cuts[:-1],cuts[1:]))
        groups = supported
    weights = np.zeros(sites, dtype=np.float64)
    for _, left, right in groups:
        if right-left >= 2 and cm[right-1] > cm[left]:
            weights[left:right] = marker_weights(cm[left:right])
    if expected_dosage_output is not None:
        require(expected_dosage_output.shape == (n, sites, 3), 'Dosage output shape differs')
    people = []
    for person in range(n):
        numerator = np.zeros((2, len(METRICS)), dtype=np.float64)
        denominator = np.zeros(2, dtype=np.float64)
        stratum_num, stratum_den = np.zeros((2, 3, 2)), np.zeros((2, 3, 2))
        stratum_counts = np.zeros((3, 2), dtype=np.int64)
        raw_infinite = np.zeros(2, dtype=bool)
        n_observed = true_zeros = below_floor = 0
        hard = np.full(sites, -1, dtype=np.int8)
        for left in range(0, sites, chunk_positions):
            if check_budget is not None:
                check_budget()
            right = min(sites, left+chunk_positions)
            valid = np.asarray(mask[person, left:right])
            require(np.isin(valid, (0, 1)).all(), 'Mask must be boolean or exact 0/1')
            valid = valid.astype(bool)
            if expected_dosage_output is not None:
                expected_dosage_output[person, left:right] = np.nan
            count = int(valid.sum())
            if not count:
                continue
            labels = np.asarray(y[person, left:right])[valid]
            predicted, true = calibration.true_probabilities(
                np.asarray(p[person, left:right])[valid][None, :, :], labels[None, :])
            predicted = predicted[0]
            true = true[0]
            # M39's per-person output becomes a per-cell kernel by giving each
            # cell a length-one position axis. This is only computational reuse;
            # these cells are never claimed to be independent people.
            cell = calibration.evaluate(predicted[:, None, :], labels[:, None], floor)
            losses = np.column_stack((cell['log_loss_per_person'], cell['brier_per_person'],
                cell['accuracy_per_person'], np.asarray(cell['dosage_mae_per_person']),
                -np.log(np.where(true > 0, true, 1.))))
            dosage = predicted @ calibration.DOSAGES
            if expected_dosage_input is not None:
                direct_dosage = np.asarray(expected_dosage_input[person,left:right],dtype=np.float64)[valid]
                require(np.isfinite(direct_dosage).all() and np.allclose(direct_dosage,dosage,atol=5e-6,rtol=0),
                        'Direct marginal dosage differs beyond bounded float arithmetic')
                dosage = direct_dosage
            require(np.all((dosage >= 0) & (dosage <= 2)) and np.allclose(dosage.sum(-1), 2),
                    'Expected ancestry copy counts differ')
            hard[left:right][valid] = predicted.argmax(-1)
            if expected_dosage_output is not None:
                expected_dosage_output[person, left:right][valid] = dosage
            actual_dose = calibration.DOSAGES[labels]
            losses[:,3:6] = np.abs(dosage-actual_dose)
            n_observed += count
            true_zeros += cell['true_zero_count']
            below_floor += cell['true_below_floor_count']
            for ancestry in range(3):
                present = actual_dose[:, ancestry] > 0
                stratum_counts[ancestry] += (int((~present).sum()), int(present.sum()))
            for weighting, w in enumerate((np.ones(count), weights[left:right][valid])):
                numerator[weighting] += (losses*w[:, None]).sum(0)
                denominator[weighting] += w.sum()
                raw_infinite[weighting] |= bool(((true == 0) & (w > 0)).any())
                for ancestry in range(3):
                    for presence in (0, 1):
                        selected = (actual_dose[:, ancestry] > 0) == bool(presence)
                        stratum_num[weighting, ancestry, presence] += float((losses[selected, 3+ancestry]*w[selected]).sum())
                        stratum_den[weighting, ancestry, presence] += float(w[selected].sum())
        metrics = {}
        for j, name in enumerate(WEIGHTINGS):
            value = {metric: ratio(numerator[j, k], denominator[j]) for k, metric in enumerate(METRICS)}
            if raw_infinite[j]:
                value['raw_log_loss'] = None
            metrics[name] = dict(metrics=value, denominator=float(denominator[j]),
                numerator={metric: None if metric == 'raw_log_loss' and raw_infinite[j] else float(numerator[j, k])
                           for k, metric in enumerate(METRICS)}, raw_log_loss_infinite=bool(raw_infinite[j]),
                dosage_by_truth_presence={ancestry:{label:dict(
                    mae=ratio(stratum_num[j, a, b], stratum_den[j, a, b]),
                    numerator=float(stratum_num[j, a, b]), denominator=float(stratum_den[j, a, b]),
                    cells=int(stratum_counts[a, b])) for b, label in enumerate(('absent', 'present'))}
                    for a, ancestry in enumerate(ANCESTRIES)})
        people.append(dict(person_index=person, observed_cells=n_observed, missing_cells=sites-n_observed,
            true_probability_zero_cells=true_zeros, true_probability_below_floor_cells=below_floor,
            weightings=metrics, boundaries=boundary_report(y[person], hard, mask[person], cm, groups,
                                                         max_working_bytes-base_bound)))
    result = summarize(people, sites, floor)
    result.update(max_query_gap_cm=max_query_gap_cm, genomic_gap_splits=gaps,
                  dense_grid_admission='NOT_CLAIMED' if max_query_gap_cm is None else 'SUPPLIED_MAX_GAP_SPLITS_ONLY',
                  dosage_semantics='HAPLOTYPE_MARGINAL_SUM_NO_INDEPENDENCE' if expected_dosage_input is not None else 'EXPECTATION_UNDER_SUPPLIED_SIX_STATE_PREDICTOR')
    return result


def summarize(people, sites, floor):
    summary = {}
    for weighting in WEIGHTINGS:
        rows = [p['weightings'][weighting] for p in people]
        total = sum(r['denominator'] for r in rows)
        report = {}
        for metric in METRICS:
            finite = [r['metrics'][metric] for r in rows if r['metrics'][metric] is not None]
            infinite = metric == 'raw_log_loss' and any(r['raw_log_loss_infinite'] for r in rows)
            report[metric] = dict(equal_person_mean=None if infinite or not finite else float(np.mean(finite)),
                people_with_denominator=sum(r['denominator'] > 0 for r in rows),
                pooled_numerator=None if infinite else sum(r['numerator'][metric] for r in rows),
                pooled_denominator=total,
                pooled_mean=None if infinite else ratio(sum(r['numerator'][metric] for r in rows), total),
                infinite=infinite)
        strata = {}
        for ancestry in ANCESTRIES:
            strata[ancestry] = {}
            for presence in ('absent', 'present'):
                selected = [r['dosage_by_truth_presence'][ancestry][presence] for r in rows]
                finite = [r['mae'] for r in selected if r['mae'] is not None]
                strata[ancestry][presence] = dict(equal_person_mae=float(np.mean(finite)) if finite else None,
                    people_with_denominator=len(finite), cells=sum(r['cells'] for r in selected),
                    numerator=sum(r['numerator'] for r in selected), denominator=sum(r['denominator'] for r in selected))
        summary[weighting] = dict(metrics=report, dosage_by_truth_presence=strata)
    boundary = {}
    for tolerance in TOLERANCES:
        rows = [p['boundaries'][str(tolerance)] for p in people]
        totals = {k:sum(r[k] for r in rows) for k in ('truth','predicted','matched','absolute_error_cm','observed_span_cm')}
        finite = [r['f1'] for r in rows if r['f1'] is not None]
        boundary[str(tolerance)] = dict(**boundary_rates(totals),
            equal_person_f1=float(np.mean(finite)) if finite else None,
            people_with_f1_denominator=len(finite))
    primary = boundary['0.2']
    return dict(people=len(people), positions=sites, observed_cells=sum(p['observed_cells'] for p in people),
        missing_cells=sum(p['missing_cells'] for p in people), weightings=summary, boundaries=boundary,
        per_person=people, floor=floor, probability_state_order=list(STATES),
        dosage_unit='ancestral_copies_0_to_2_not_divided_by_ploidy',
        boundary_definition='unordered_diploid_grid_state_transitions_midpoint_cm_within_observed_runs',
        missing_policy='fixed_truth_mask_shared_by_all_arms_invalid_observed_prediction_fails',
        zero_denominator='null_not_perfect_score', primary_boundary_tolerance_cm=0.2,
        primary_boundary_loss_pooled=None if primary['f1'] is None else 1-primary['f1'],
        primary_boundary_loss_equal_person=None if primary['equal_person_f1'] is None else 1-primary['equal_person_f1'],
        selection_aggregation='NOT_ADJUDICATED_BY_EVALUATOR',
        clipped_log_loss_is_strictly_proper=False,
        biological_selection_authorized=False)


def donor_bootstrap(delta, people, *, replicates, seed, max_working_bytes):
    """Conditional paired cluster bootstrap, never independent-window resampling."""
    require(type(replicates) is int and replicates > 0 and type(seed) is int and seed >= 0,
            'Explicit bootstrap count and seed required')
    require(len(delta) == len(people) and np.isfinite(delta).all(), 'Invalid paired unit losses')
    clusters, donor_owner = {}, {}
    for value, person in zip(delta, people):
        unit = person['evaluation_unit']
        donors = person['donor_units'].split(',')
        require(unit and donors and all(donors) and len(donors) == len(set(donors)), 'Donor unit metadata missing')
        clusters.setdefault(unit, []).append(float(value))
        for donor in donors:
            require(donor not in donor_owner or donor_owner[donor] == unit,
                    'Donors shared across bootstrap clusters; mosaics are not independent units')
            donor_owner[donor] = unit
    require(len(clusters) >= 2, 'At least two declared donor units needed for requested bootstrap')
    require(replicates*8 + len(clusters)*64 < max_working_bytes, 'Bootstrap exceeds memory budget')
    values = np.asarray([np.mean(clusters[key]) for key in sorted(clusters)])
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        draws[i] = values[rng.integers(len(values), size=len(values))].mean()
    return dict(delta_equal_donor_unit=float(values.mean()), conditional_cluster_bootstrap_95=np.quantile(draws, [.025,.975]).tolist(),
        units=len(values), donor_units=len(donor_owner), people=len(people), replicates=replicates, seed=seed,
        estimand='equal_declared_donor_unit_mean_of_equal_person_differences',
        scope='conditional_on_declared_disjoint_donor_units_not_population_independence_certification')


def read_table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def authenticate(descriptor, inputs):
    require(set(descriptor) == {'path', 'sha256'} and calibration.valid_hash(descriptor['sha256']),
            'Authenticated path/SHA256 descriptor required')
    path = Path(descriptor['path'])
    require(path.is_absolute() and '..' not in path.parts and path.is_file() and not path.is_symlink(),
            'Input must be a regular absolute file')
    calibration.authenticated(path, descriptor['sha256'])
    inputs[str(path)] = descriptor['sha256']
    return path


def load_contract(path, digest):
    """Verify all local artifacts before opening arrays; used by preflight too."""
    inputs = {}
    # Nextflow may stage the contract itself as a symlink; authenticate its
    # resolved regular target. Explicit scientific input descriptors stay strict.
    path = authenticate(dict(path=str(Path(path).resolve()), sha256=digest), inputs)
    require(path.stat().st_size <= 2**20, 'Evaluation contract exceeds one MiB')
    c = json.loads(path.read_text())
    require(c['schema'] == 'r02_lai_evaluate_input_v1' and c['usage_scope'] == 'DEVELOPMENT_ONLY',
            'Development evaluation contract required; no confirmatory TEST')
    require(c['evidence_kind'] in {'FIXTURE', 'SIMULATED_RECORDED_TRUTH'}, 'Unrecognized truth evidence class')
    require(c['ancestry_names'] == list(ANCESTRIES) and c['state_names'] == list(STATES), 'Ancestry/state semantics differ')
    require(c['boundary_tolerances_cm'] == list(TOLERANCES) and c['primary_boundary_tolerance_cm'] == 0.2,
            'Registered L4 boundary tolerance curve must be preserved')
    require(c['source_test_opened'] is False and c['grid_selected_from_truth'] is False,
            'TEST or truth-selected evaluation grid forbidden')
    if c['evidence_kind'] != 'FIXTURE':
        require(isinstance(c.get('max_query_gap_cm'), (float,int)) and c['max_query_gap_cm'] > 0 and
                math.isfinite(c['max_query_gap_cm']), 'Real evaluation requires an explicit admitted maximum query gap')
    expected = {'grid', 'persons', 'roles', 'split', 'truth_provenance', 'truth_state', 'observed_mask'}
    require(set(c['inputs']) == expected, 'Evaluation input inventory differs')
    paths = {key:authenticate(record, inputs) for key,record in c['inputs'].items()}
    require(type(c['resources']['max_working_bytes']) is int and c['resources']['max_working_bytes'] > 0 and
            32*sum(paths[key].stat().st_size for key in ('grid','persons','truth_provenance','roles','split')) < c['resources']['max_working_bytes'],
            'Metadata table memory admission exceeded before parsing')
    for name, digest in c['code_sha256'].items():
        require(name in CODE_FILES and sha(Path(__file__).parent/name) == digest, 'Evaluation code hash differs')
    require(set(c['code_sha256']) == set(CODE_FILES), 'Complete metric dependency hashes required')
    binding = {name+'_sha256':c['inputs'][name]['sha256'] for name in ('grid','persons','roles','split','observed_mask')}
    provenance = json.loads(paths['truth_provenance'].read_text())
    require(provenance['schema'] == 'r02_lai_evaluation_truth_v1' and
            provenance['evidence_kind'] == c['evidence_kind'] and
            provenance['truth_origin'] == ('SYNTHETIC_FIXTURE' if c['evidence_kind'] == 'FIXTURE' else 'SIMULATOR_RECORDED_ANCESTRY'),
            'Truth must be registered simulator ancestry, never FLARE or another estimator')
    require(provenance['truth_state_sha256'] == c['inputs']['truth_state']['sha256'] and
            all(provenance[k] == v for k,v in binding.items()), 'Truth provenance axes/roles differ')
    if c['evidence_kind'] != 'FIXTURE':
        original = authenticate(provenance['recorded_truth'], inputs)
        simulation = authenticate(provenance['simulation_manifest'], inputs)
        document = json.loads(simulation.read_text())
        require(document['schema'] == 'r02_lai_simulate_output_v1' and
                document['scope'] == 'DNABR_AUTHORIZED_SIMULATION' and
                document['status'] == 'COMPLETE_SIMULATED_SCENARIO_NOT_LAI_VALIDATION' and
                document['outputs_sha256'].get(original.name) == provenance['recorded_truth']['sha256'],
                'Simulator manifest does not authenticate real recorded truth; fixtures cannot be promoted')
        require(calibration.valid_hash(document['scientific_admission_sha256']) and
                calibration.valid_hash(document['input_contract_sha256']), 'Simulation provenance hashes absent')
    grid, people = read_table(paths['grid']), read_table(paths['persons'])
    require(grid and set(grid[0]) == {'chromosome','position_bp','genetic_cm'}, 'Grid columns differ')
    require(people and set(people[0]) == {'sample_key_sha256','role','evaluation_unit','donor_units'}, 'Person/unit columns differ')
    require(len({p['sample_key_sha256'] for p in people}) == len(people) and
            all(calibration.valid_hash(p['sample_key_sha256']) for p in people), 'Duplicate/invalid person hashes')
    require(c['evaluation_roles'] and set(c['evaluation_roles']) <= {'SELECT','VALID','SCORE','EVALUATE','DEVELOPMENT'} and
            all(p['role'] in c['evaluation_roles'] and p['evaluation_unit'] and p['donor_units'] for p in people),
            'Evaluation roles/units absent or forbidden')
    roles = json.loads(paths['roles'].read_text())
    split = json.loads(paths['split'].read_text())
    require(roles['schema'] == 'r02_lai_evaluation_roles_v1' and
            roles['persons'] == {p['sample_key_sha256']:p['role'] for p in people},
            'Person roles differ from actual role artifact contents')
    require(split['schema'] == 'r02_lai_evaluation_split_v1' and
            split['persons'] == {p['sample_key_sha256']:dict(evaluation_unit=p['evaluation_unit'],
                donor_units=p['donor_units'].split(',')) for p in people},
            'Person donor units differ from actual split artifact contents')
    coordinates = dict(chromosome=np.asarray([r['chromosome'] for r in grid]),
        positions_bp=np.asarray([int(r['position_bp']) for r in grid], dtype=np.int64),
        genetic_cm=np.asarray([float(r['genetic_cm']) for r in grid]))
    grid_ranges(**dict(chromosome=coordinates['chromosome'],positions=coordinates['positions_bp'],cm=coordinates['genetic_cm']))
    prediction_rows = []
    require(c['predictions'] and len({p['name'] for p in c['predictions']}) == len(c['predictions']), 'Unique prediction arms required')
    for row in c['predictions']:
        require(row['name'] and all(x.isalnum() or x in '_-' for x in row['name']), 'Unsafe prediction name')
        prediction = authenticate(row['probabilities'], inputs)
        receipt_path = authenticate(row['receipt'], inputs)
        require(receipt_path.stat().st_size <= 2**20, 'Prediction receipt exceeds one MiB')
        receipt = json.loads(receipt_path.read_text())
        require(receipt['schema'] == 'r02_lai_evaluation_prediction_v1' and
                receipt['probabilities_sha256'] == row['probabilities']['sha256'] and
                receipt['state_names'] == list(STATES) and receipt['evidence_kind'] == c['evidence_kind'] and
                all(receipt[k] == v for k,v in binding.items()), 'Prediction identity/grid/mask/roles differ')
        require(receipt['probability_semantics'] in {'JOINT_SIX_STATE', 'PRODUCT_OF_HAPLOTYPE_MARGINALS_APPROXIMATION'} and
                calibration.valid_hash(receipt['budget_sha256']) and
                calibration.valid_hash(receipt['reference_panel_sha256']), 'Prediction semantics/budget/reference absent')
        direct = authenticate(row['expected_dosage'],inputs) if 'expected_dosage' in row else None
        if direct is not None:
            require(direct.suffix=='.npy' and receipt.get('expected_dosage_sha256')==row['expected_dosage']['sha256'] and
                    receipt.get('dosage_semantics')=='HAPLOTYPE_MARGINAL_SUM_NO_INDEPENDENCE', 'Direct dosage provenance differs')
        prediction_rows.append((row['name'], prediction, receipt, direct))
    names = {r[0]:r[2] for r in prediction_rows}
    for comparison in c['comparisons']:
        a, b = comparison['candidate'], comparison['comparator']
        require(a in names and b in names and a != b, 'Invalid paired comparison')
        require(type(comparison['require_matched_budget']) is bool and
                type(comparison['require_matched_reference']) is bool, 'Explicit budget/reference matching rules required')
        if comparison['require_matched_budget']:
            require(names[a]['budget_sha256'] == names[b]['budget_sha256'], 'Control budgets differ')
        if comparison['require_matched_reference']:
            require(names[a]['reference_panel_sha256'] == names[b]['reference_panel_sha256'], 'Control reference panels differ')
    if c.get('harm_tolerances'):
        receipt_path = authenticate(c['harm_tolerance_provenance'], inputs)
        receipt = json.loads(receipt_path.read_text())
        require(receipt['prespecified'] is True and receipt['harm_tolerances'] == c['harm_tolerances'],
                'Harm tolerances lack an exact predefined receipt')
        for rule in c['harm_tolerances']:
            require(rule['weighting'] in WEIGHTINGS and rule['metric'] in METRICS and rule['metric'] != 'accuracy' and
                    math.isfinite(rule['max_delta']) and rule['max_delta'] >= 0, 'Invalid loss harm tolerance')
    return c, inputs, paths, coordinates, people, prediction_rows


def comparisons(reports, specifications, people, config):
    rows = []
    for spec in specifications:
        candidate, baseline = reports[spec['candidate']], reports[spec['comparator']]
        row = dict(**spec, deltas={}, harm_checks=[], biological_selection_authorized=False)
        for weighting in WEIGHTINGS:
            for metric in METRICS:
                deltas = []
                for a, b in zip(candidate['per_person'], baseline['per_person']):
                    x, y = a['weightings'][weighting], b['weightings'][weighting]
                    require(x['denominator'] == y['denominator'], 'Paired metric denominators differ')
                    av, bv = x['metrics'][metric], y['metrics'][metric]
                    deltas.append(None if av is None or bv is None else av-bv)
                finite = [x for x in deltas if x is not None]
                infinite_raw = metric == 'raw_log_loss' and any(
                    a['weightings'][weighting]['raw_log_loss_infinite'] or
                    b['weightings'][weighting]['raw_log_loss_infinite']
                    for a,b in zip(candidate['per_person'], baseline['per_person']))
                key = weighting+'/'+metric
                row['deltas'][key] = dict(candidate_minus_comparator=float(np.mean(finite)) if finite and not infinite_raw else None,
                    people_paired=len(finite), people_total=len(people), direction='higher_better' if metric == 'accuracy' else 'lower_better',
                    per_person=deltas, undefined_due_to_infinite_raw_loss=infinite_raw)
                if config['bootstrap']['enabled'] and len(finite) == len(people):
                    row['deltas'][key]['bootstrap'] = donor_bootstrap(np.asarray(finite), people,
                        replicates=config['bootstrap']['replicates'], seed=config['bootstrap']['seed'],
                        max_working_bytes=config['resources']['max_working_bytes'])
        for rule in config.get('harm_tolerances', []):
            result = row['deltas'][rule['weighting']+'/'+rule['metric']]['candidate_minus_comparator']
            row['harm_checks'].append(dict(**rule, delta=result,
                status='NOT_MEASURED' if result is None else 'EXCEEDS_PREDEFINED_TOLERANCE' if result > rule['max_delta'] else 'WITHIN_PREDEFINED_TOLERANCE',
                no_automatic_biological_adjudication=True))
        row['boundary_deltas'] = {str(t):dict(candidate_minus_comparator_f1=(
            candidate['boundaries'][str(t)]['f1']-baseline['boundaries'][str(t)]['f1'])
            if candidate['boundaries'][str(t)]['f1'] is not None and baseline['boundaries'][str(t)]['f1'] is not None else None)
            for t in TOLERANCES}
        rows.append(row)
    return rows


def run(contract_path, digest, output):
    started = time.monotonic()
    c, inputs, paths, coordinates, people, predictions = load_contract(contract_path, digest)
    limits = c['resources']
    # Tiny software fixtures do not need the 12GiB production disk reserve.
    # Real simulator evidence keeps the same admission bound.
    minimum_reserve = 0 if c['evidence_kind'] == 'FIXTURE' else 12
    require(limits['cpus'] == 1 and limits['min_free_gib'] >= minimum_reserve and
            limits['timeout_seconds'] > 0 and limits['max_output_bytes'] > 0, 'Invalid bounded evaluation resources')
    output = Path(output)
    require(not output.exists(), 'Output exists; use Nextflow resume, never overwrite')
    require(output.parent.is_dir(), 'Output parent missing')
    def guard():
        require(time.monotonic()-started <= limits['timeout_seconds'], 'Evaluation time budget exceeded')
        require(shutil.disk_usage(output.parent).free >= limits['min_free_gib']*2**30, 'Free disk reserve reached')
    guard()
    # Reject compressed archives: mmap gives bounded resident blocks, unlike NPZ.
    require(paths['truth_state'].suffix == paths['observed_mask'].suffix == '.npy' and
            all(path.suffix == '.npy' for _,path,_,_ in predictions), 'Evaluation arrays must be uncompressed NPY for mmap')
    y, mask = (np.load(paths[key], mmap_mode='r', allow_pickle=False) for key in ('truth_state','observed_mask'))
    require(y.shape == mask.shape == (len(people), len(coordinates['genetic_cm'])), 'Truth axes differ from authenticated tables')
    require(len(people)*len(predictions)*65536 + len(coordinates['genetic_cm'])*96 +
            min(limits['chunk_positions'],y.shape[1])*2048 < limits['max_working_bytes'],
            'All-arm report and block working-memory admission exceeded')
    output_bound = len(predictions)*(y.size*12 if c['emit_expected_dosages'] else 0) + len(people)*len(predictions)*32768 + 2**20
    require(output_bound <= limits['max_output_bytes'] and
            shutil.disk_usage(output.parent).free-output_bound >= limits['min_free_gib']*2**30,
            'Evaluation output storage budget exceeded')
    output.mkdir(mode=0o700)
    reports = {}
    for name, path, receipt, direct_path in predictions:
        guard()
        p = np.load(path, mmap_mode='r', allow_pickle=False)
        direct = np.load(direct_path,mmap_mode='r',allow_pickle=False) if direct_path is not None else None
        dose = np.lib.format.open_memmap(output/(name+'.expected_dosage.npy'), mode='w+', dtype=np.float32,
                                       shape=(*y.shape,3)) if c['emit_expected_dosages'] else None
        reports[name] = evaluate_arrays(p, y, mask, **coordinates, floor=c['probability_floor'],
            chunk_positions=limits['chunk_positions'], max_working_bytes=limits['max_working_bytes'],
            expected_dosage_output=dose, check_budget=guard, max_query_gap_cm=c.get('max_query_gap_cm'),expected_dosage_input=direct)
        reports[name]['prediction_provenance'] = receipt
        if dose is not None:
            dose.flush()
            del dose
    contrasts = comparisons(reports, c['comparisons'], people, c)
    for name, report in reports.items():
        write_json(output/(name+'.metrics.private.json'), report)
    write_json(output/'comparisons.private.json', contrasts)
    for path, expected in inputs.items():
        require(sha(Path(path)) == expected, 'Evaluation source changed while running')
    guard()
    require(sum(p.stat().st_size for p in output.iterdir()) <= limits['max_output_bytes'], 'Output size cap exceeded')
    manifest = dict(schema='r02_lai_evaluate_output_v1', status='COMPLETE_TECHNICAL_EVALUATION_NOT_BIOLOGICAL_GO',
        evidence_kind=c['evidence_kind'], claim_level='OPERATIONAL_FIXTURE' if c['evidence_kind']=='FIXTURE' else 'CONDITIONAL_SIMULATION_EVALUATION',
        input_contract_sha256=digest, input_sha256=inputs, code_sha256=c['code_sha256'],
        people=len(people), query_positions=len(coordinates['genetic_cm']),
        prediction_arms=[name for name,_,_,_ in predictions], simulations_executed=False, training_executed=False,
        source_test_opened=False, biological_selection_authorized=False,
        boundary_truth_resolution='query_grid_midpoints_not_exact_parental_breakpoints',
        outputs_sha256={p.name:sha(p) for p in sorted(output.iterdir())},
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(output/'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.verify_only:
        c, _, _, _, people, predictions = load_contract(args.contract, args.contract_sha256)
        result = dict(status='EVALUATION_CONTRACT_VERIFIED_NOT_RUN', evidence_kind=c['evidence_kind'],
                      people=len(people), arms=len(predictions))
    else:
        result = run(args.contract, args.contract_sha256, args.output_dir)
    print(json.dumps({key:value for key,value in result.items() if key in {'status','evidence_kind','people','arms','elapsed_seconds'}}, allow_nan=False))


if __name__ == '__main__':
    main()
