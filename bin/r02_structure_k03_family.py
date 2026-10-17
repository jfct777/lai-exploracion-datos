#!/usr/bin/env python3
"""Authenticate a declared K03 FIT/SELECT family before any selection.

This barrier never ranks candidates, aligns community IDs across blocks, or
opens EVALUATE. Completeness means completeness of the supplied, externally
frozen inventory; it does not establish adequacy of the scientific search.
"""
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(
        c in '0123456789abcdef' for c in value)


IDENTITY = ('arm', 'candidate_id', 'block_id', 'exclusion_scope', 'alpha',
            'baseline_identity', 'select_target_identity')


def validate_complete_family(family, records):
    """Validate already authenticated manifests/metrics, without choosing a model."""
    require(family.get('schema') == 'r02_k03_family_inventory_v1' and
            family.get('evaluate_allowed') is False and
            valid_digest(family.get('scientific_spec_sha256')),
            'Explicit frozen FIT/SELECT family required')
    rows = family.get('models')
    require(isinstance(rows, list) and rows, 'Nonempty model inventory required')
    expected = {}
    for row in rows:
        name = row.get('model_id')
        require(isinstance(name, str) and name and name not in expected,
                'Duplicate or absent model identity')
        require(all(k in row for k in IDENTITY) and
                row['arm'] in {'CONTINUOUS', 'COMMUNITY_A', 'COMMUNITY_B'} and
                row['exclusion_scope'] in {'BLOCK_AND_LD_MARGIN', 'WHOLE_CHROMOSOME'} and
                type(row['alpha']) in (int, float) and math.isfinite(row['alpha']) and
                row['alpha'] > 0, 'Invalid declared model identity')
        require(all(valid_digest(row.get(k)) for k in (
            'request_sha256', 'inputs_manifest_sha256', 'baseline_identity',
            'select_target_identity')), 'Declared model provenance required')
        expected[name] = row
    observed = {}
    for record in records:
        name = record.get('model_id')
        require(name in expected and name not in observed,
                'Duplicate or undeclared result; no silent overwrite')
        observed[name] = record
    require(set(observed) == set(expected),
            'Incomplete family: selecting only successful models is forbidden')
    views = family.get('required_views')
    require(isinstance(views, list) and views, 'Explicit target-view inventory required')
    view_set = {(x['block_id'], x['exclusion_scope']) for x in views}
    require(len(view_set) == len(views) and view_set == {
        (r['block_id'], r['exclusion_scope']) for r in rows},
        'Target-view inventory differs')
    scientific_spec = family['scientific_spec_sha256']
    complete = []
    for name in sorted(expected):
        row, record = expected[name], observed[name]
        m, metrics = record['manifest'], record['metrics']
        require(m.get('schema') == 'r02_k03_fit_select_result_v1' and
                m.get('stage') == 'ONE_MODEL_COMPLETE_SELECTION_PENDING',
                'Result is not one complete FIT/SELECT model')
        require(m.get('request_sha256') == row['request_sha256'] and
                m.get('inputs_manifest_sha256') == row['inputs_manifest_sha256'] and
                m.get('scientific_spec_sha256') == scientific_spec,
                'Result request, prepared input or scientific specification differs')
        require(m.get('evaluate_targets_computed') is False and
                m.get('alpha_selection_performed') is False and
                m.get('biological_validation') is False and
                metrics.get('evaluate_targets_computed') is False and
                metrics.get('selection_performed') is False and
                metrics.get('biological_validation') is False,
                'Result exceeds FIT/SELECT scope')
        require(all(metrics.get(k) == row[k] for k in IDENTITY),
                'Model metrics identity differs; never align candidates by name alone')
        require(m.get('baseline_identity') == row['baseline_identity'] and
                m.get('select_target_identity') == row['select_target_identity'],
                'Manifest baseline/target identity differs')
        require(type(metrics.get('fit_pairs')) is int and metrics['fit_pairs'] > 0 and
                type(metrics.get('select_pairs')) is int and metrics['select_pairs'] > 0,
                'Explicit nonempty pair denominators required')
        diagnostics = metrics.get('numerical_diagnostics', {})
        residual = diagnostics.get('relative_penalized_gradient')
        tolerance = diagnostics.get('gradient_tolerance')
        require(diagnostics.get('lsqr_status') in (0, 1, 2, 4, 5) and
                diagnostics.get('alpha') == row['alpha'] and
                all(type(x) in (int, float) and math.isfinite(x) for x in (residual, tolerance)) and
                0 <= residual <= tolerance < 1 and tolerance > 0,
                'Unconverged or inconsistent numerical model')
        loss = metrics.get('weighted_select_mse')
        require(type(loss) in (int, float) and math.isfinite(loss) and loss >= 0,
                'Finite nonnegative SELECT loss required')
        numerical = metrics.get('numerical_loss', {})
        lower, upper = numerical.get('exact_ridge_mse_lower'), numerical.get('exact_ridge_mse_upper')
        require(all(type(x) in (int, float) and math.isfinite(x) for x in (lower, upper)) and
                0 <= lower <= upper and numerical.get('statistical_confidence_interval') is False and
                numerical.get('bound_scope') == 'CONDITIONAL_ON_RESIDUAL_ACCURACY_AND_FIXED_PREPROCESSING',
                'Explicit conditional numerical bounds required; these are not confidence intervals')
        baseline_id = row.get('baseline_model_id')
        if row['arm'] == 'CONTINUOUS':
            require(baseline_id is None, 'Continuous model cannot reference another baseline')
        else:
            require(baseline_id in expected and expected[baseline_id]['arm'] == 'CONTINUOUS',
                    'Community model needs an explicitly declared continuous baseline')
            baseline = expected[baseline_id]
            require(all(row[k] == baseline[k] for k in (
                'block_id', 'exclusion_scope', 'baseline_identity', 'select_target_identity')),
                'Compared baseline has different view, availability, controls or targets')
            require(all(metrics[k] == observed[baseline_id]['metrics'][k] for k in (
                'select_pairs', 'fit_pairs')), 'Compared pair denominators differ')
        complete.append(dict(model_id=name, **{k: row[k] for k in IDENTITY},
                             baseline_model_id=baseline_id,
                             weighted_select_mse=loss, numerical_loss=numerical,
                             fit_pairs=metrics['fit_pairs'], select_pairs=metrics['select_pairs']))
    return dict(schema='r02_k03_complete_family_v1',
                stage='DECLARED_FAMILY_AUTHENTICATED_SELECTION_NOT_PERFORMED',
                models=complete, model_count=len(complete), view_count=len(view_set),
                scientific_spec_sha256=scientific_spec, selection_performed=False,
                evaluate_targets_computed=False, biological_validation=False,
                global_partition_alignment_assumed=False,
                scientific_search_adequacy_certified=False)


def collect(family_path, family_sha256, descriptors, *, max_metadata_bytes,
            max_result_bytes, max_models):
    """Read pinned results with explicit bounds; descriptors come after execution.

    `family_sha256` must come from an independently frozen admission. Result
    hashes cannot exist before fitting, so each post-execution descriptor pins
    its manifest and that manifest pins all three result files.
    """
    require(all(type(x) is int and x > 0 for x in (
        max_metadata_bytes, max_result_bytes, max_models)), 'Positive resource bounds required')
    pins = {}

    def pin(path, digest, limit):
        path = Path(path).resolve()
        require(path not in pins or pins[path] == digest, 'Conflicting hashes for the same path')
        require(valid_digest(digest) and path.is_file() and path.stat().st_size <= limit and
                sha256(path) == digest, 'Changed, absent or oversized result')
        pins[path] = digest
        return path

    f = pin(family_path, family_sha256, max_metadata_bytes)
    family = json.loads(f.read_text())
    require(isinstance(descriptors, list) and 0 < len(descriptors) <= max_models and
            len(family['models']) <= max_models, 'Family exceeds explicit model limit')
    records = []
    for descriptor in descriptors:
        root = Path(descriptor['directory']).resolve()
        manifest_path = pin(root/'manifest.json', descriptor['manifest_sha256'], max_metadata_bytes)
        manifest = json.loads(manifest_path.read_text())
        outputs = manifest['outputs_sha256']
        require(set(outputs) == {'metrics.json', 'model.private.npz', 'select_predictions.private.npz'},
                'Exact model result inventory required')
        require(sum((root/name).stat().st_size for name in outputs) <= max_result_bytes,
                'Result exceeds total byte limit')
        for name, digest in outputs.items():
            require((root/name).resolve().parent == root and not (root/name).is_symlink(),
                    'Result outputs must be regular files inside their result directory')
            pin(root/name, digest, max_metadata_bytes if name == 'metrics.json' else max_result_bytes)
        records.append(dict(model_id=descriptor['model_id'], manifest=manifest,
                            metrics=json.loads((root/'metrics.json').read_text())))
    result = validate_complete_family(family, records)
    require(all(sha256(p) == digest for p, digest in pins.items()),
            'Source changed during family authentication')
    result['family_sha256'] = family_sha256
    result['results'] = descriptors
    return result
