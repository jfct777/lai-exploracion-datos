#!/usr/bin/env python3
"""Materialize family-explicit C and R+C affinities from frozen FIT products.

C uses the complete (not rank-selected) common representation prepared by
``r02_structure_common_inputs.py``.  Its source quantity is the FIT-centred,
FIT-weighted, mean-imputed common kernel reconstructed as ``scores @ scores.T``
and divided by the selected common-site count.  It is not relabelled as the
historical observed-pair PLINK GRM.  The graph transform is its positive part.

R+C uses the historical algebra with an explicit request weight, but its common
component retains the current kernel definition above.  Positive scales are
estimated only over eligible FIT-to-FIT edges.  Query-to-FIT rows never change
the scales or FIT graph, and EVALUATE outcomes are never read.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import zipfile

import numpy as np

import r02_structure_affinity_io as affinity_io
import r02_structure_continuous as continuous

discrete = continuous.discrete
require, sha, load_json = discrete.require, discrete.sha, discrete.load_json
SCHEMA = 'r02_structure_common_family_request_v1'
FAMILIES = {'C', 'R_PLUS_C'}
CODE_FILES = tuple(dict.fromkeys(('r02_structure_common_affinities.py',
    'r02_structure_affinity_io.py', 'r02_structure_continuous.py', *discrete.CODE_FILES)))
REQUEST_FIELDS = {'schema', 'evidence_scope', 'scientific_spec_sha256',
                  'implementation_sha256', 'family', 'mix_weight',
                  'normalization_policy', 'source_contract', 'common', 'rare', 'resources'}
LIMIT_FIELDS = {'max_input_bytes', 'max_metadata_bytes', 'max_memory_bytes',
                'max_output_bytes', 'min_free_bytes'}


def implementation_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def pin(path, digest, maximum):
    path = Path(path).resolve()
    require(path.is_absolute() and path.is_file() and discrete.valid_hash(digest)
            and type(maximum) is int and maximum > 0 and path.stat().st_size <= maximum
            and sha(path) == digest, 'Pinned common-family input changed or exceeds bound')
    return path


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write('\n')
    Path(path).chmod(0o600)


def _npz_uncompressed_bytes(path, expected, maximum):
    total = 0
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {name+'.npy' for name in expected}
                and len(archive.namelist()) == len(expected),
                'Common representation array inventory differs')
        for info in archive.infolist():
            total += info.file_size
            require(total <= maximum, 'Common representation decode exceeds input bound')
    return total


def load_roster(source, common_manifest, evidence_scope, maximum):
    require(set(source) == {'manifest', 'persons'}, 'Exact source contract/person pins required')
    contract_path = pin(source['manifest']['path'], source['manifest']['sha256'], maximum)
    persons_path = pin(source['persons']['path'], source['persons']['sha256'], maximum)
    contract = load_json(contract_path)
    require(sha(contract_path) == common_manifest['source_contract_sha256']
            and contract['roles_state'] == 'EXPLICITLY_FROZEN'
            and contract['metadata_sha256']['persons.tsv'] == sha(persons_path),
            'Common representation and frozen roster contract differ')
    persons = discrete.shared.table(persons_path, ['sample_id', 'component', 'role'])
    roles = discrete.shared.validate_roles(persons)
    if evidence_scope == 'DNABR_R02':
        require(len(persons) == 2619 and {k: len(v) for k, v in roles.items()} ==
                {'FIT': 1460, 'SELECT': 476, 'EVALUATE': 683},
                'Common-family producer must retain frozen R02 roles')
    return contract_path, persons_path, contract, persons, roles


def load_common(desc, *, maximum_input, maximum_metadata, maximum_memory):
    require(set(desc) == {'directory', 'manifest_sha256'}
            and Path(desc['directory']).is_absolute(),
            'Exact absolute common representation descriptor required')
    root = Path(desc['directory']).resolve()
    manifest_path = pin(root/'manifest.json', desc['manifest_sha256'], maximum_metadata)
    manifest = load_json(manifest_path)
    require(manifest['schema'] == 'r02_structure_materialized_common_merge_v2'
            and manifest['stage'] == 'FULL_NUMERICAL_COMMON_SPECTRUM_NOT_PREDICTIVE_SELECTION'
            and manifest['targets_read'] is False and manifest['biological_validation'] is False
            and set(manifest['outputs_sha256']) == {'continuous.private.npz'}
            and type(manifest['n_sites']) is int and manifest['n_sites'] > 0
            and type(manifest['numerical_rank']) is int and manifest['numerical_rank'] > 0,
            'Full frozen common representation required')
    arrays_path = pin(root/'continuous.private.npz',
                      manifest['outputs_sha256']['continuous.private.npz'], maximum_input)
    fields = {'scores', 'dual_loadings', 'eigenvalues', 'fit_weights',
              'fit_indices', 'observed_counts'}
    decoded = _npz_uncompressed_bytes(arrays_path, fields, maximum_input)
    require(decoded*3 <= maximum_memory, 'Common representation workspace exceeds memory bound')
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in fields}
    scores, eigenvalues = arrays['scores'], arrays['eigenvalues']
    require(scores.dtype == np.float64 and scores.ndim == 2
            and eigenvalues.dtype == np.float64 and eigenvalues.ndim == 1
            and scores.shape[1] == len(eigenvalues) == manifest['numerical_rank']
            and np.all(np.isfinite(scores)) and np.all(np.isfinite(eigenvalues))
            and np.all(eigenvalues > 0),
            'Common scores must retain the complete finite numerical spectrum')
    return root, manifest_path, manifest, arrays_path, arrays


def _eligible_pairs(persons, fit):
    for offset, i in enumerate(fit):
        for j in fit[offset+1:]:
            if persons[i]['component'] != persons[j]['component']:
                yield int(i), int(j)


def _common_value(scores, n_sites, i, j):
    raw = float(np.dot(scores[i], scores[j]))
    require(math.isfinite(raw), 'Nonfinite reconstructed common kernel')
    return raw, max(raw/n_sites, 0.0)


def _common_fit_edges(persons, fit, scores, n_sites):
    """BLAS one FIT row at a time; avoid millions of tiny dot-product calls."""
    fit_array=np.asarray(fit,dtype=np.int64)
    for offset,i in enumerate(fit):
        later=fit_array[offset+1:]
        raw_values=scores[i]@scores[later].T
        for j,raw in zip(later,raw_values):
            j=int(j)
            if persons[i]['component']!=persons[j]['component']:
                raw=float(raw);require(math.isfinite(raw),'Nonfinite reconstructed common kernel')
                yield i,j,raw,max(raw/n_sites,0.0)


def _common_query_row(query, fit, scores, n_sites):
    raw=np.asarray(scores[query]@scores[np.asarray(fit,dtype=np.int64)].T,dtype=np.float64)
    require(np.all(np.isfinite(raw)),'Nonfinite reconstructed common query kernel')
    return raw,np.maximum(raw/n_sites,0.0)


def _component_pair_positive_mean(edges, persons, name):
    """Equal total opportunity per frozen FIT component pair, positives only."""
    component_sizes = {}
    for person in persons:
        if person['role'] == 'FIT':
            component_sizes[person['component']] = component_sizes.get(person['component'], 0)+1
    total = mass = 0.0; count = 0
    for i, j, value in edges:
        if value > 0:
            pair_weight = 1.0/(component_sizes[persons[i]['component']]
                               * component_sizes[persons[j]['component']])
            total += pair_weight*value; mass += pair_weight; count += 1
    require(count > 0 and math.isfinite(total) and total > 0
            and math.isfinite(mass) and mass > 0,
            name+' FIT-positive scale is undefined')
    result = total/mass
    require(math.isfinite(result) and result > 0, name+' FIT-positive scale is undefined')
    return result, count, mass


def _rare_inputs(desc):
    require(isinstance(desc, dict) and set(desc) ==
            {'persons.tsv', 'fit_affinities.private.tsv', 'query_affinities.private.tsv',
             'exclusion.json', 'affinity_inputs.json'},
            'R+C requires exact authenticated rare FIT/query inputs')
    return desc


def _write_row(writer, i, j, value, numerator=None, denominator=None):
    if value is None:
        writer.writerow([i, j, 0, 'NA', 'NA', 'NA'])
    else:
        require(math.isfinite(value) and value >= 0, 'Graph affinity must be finite and nonnegative')
        if numerator is None:
            writer.writerow([i, j, 1, format(value, '.17g'), 'NA', 'NA'])
        else:
            require(math.isfinite(numerator) and denominator > 0,
                    'Common support numerator/denominator invalid')
            writer.writerow([i, j, 1, format(value, '.17g'),
                             format(numerator, '.17g'), str(denominator)])


def materialize(request_path, request_sha256, output):
    request_path = pin(request_path, request_sha256, 2**20)
    request = load_json(request_path)
    require(set(request) == REQUEST_FIELDS and request['schema'] == SCHEMA
            and request['implementation_sha256'] == implementation_hashes()
            and request['evidence_scope'] in {'DNABR_R02', 'SYNTHETIC_INSTRUMENT_ONLY'}
            and request['family'] in FAMILIES
            and discrete.valid_hash(request['scientific_spec_sha256']),
            'Exact common-family request/current implementation required')
    limits = request['resources']
    require(set(limits) == LIMIT_FIELDS and all(type(v) is int and v > 0 for v in limits.values()),
            'Explicit positive common-family resource limits required')
    family = request['family']; weight = request['mix_weight']
    if family == 'C':
        require(weight is None and request['rare'] is None
                and request['normalization_policy'] is None,
                'C does not accept a hidden mixture weight or rare source')
    else:
        require(type(weight) in (int, float) and math.isfinite(weight) and 0 <= weight <= 1,
                'R+C requires an explicit finite convex mixture weight')
        require(request['normalization_policy'] == 'COMPONENT_PAIR_EQUAL_POSITIVE_MEAN',
                'R+C requires the admitted explicit FIT normalization policy')
    common_root, common_manifest_path, common_manifest, arrays_path, arrays = load_common(
        request['common'], maximum_input=limits['max_input_bytes'],
        maximum_metadata=limits['max_metadata_bytes'], maximum_memory=limits['max_memory_bytes'])
    contract_path, persons_path, contract, persons, roles = load_roster(
        request['source_contract'], common_manifest, request['evidence_scope'],
        limits['max_metadata_bytes'])
    scores, fit_indices = arrays['scores'], arrays['fit_indices']
    require(scores.shape[0] == len(persons)
            and fit_indices.dtype.kind in 'iu'
            and np.array_equal(fit_indices, roles['FIT'])
            and arrays['fit_weights'].shape == (len(fit_indices),)
            and arrays['dual_loadings'].shape == (len(fit_indices), scores.shape[1])
            and arrays['observed_counts'].dtype.kind in 'iu'
            and arrays['observed_counts'].shape == (len(persons),)
            and np.all((arrays['observed_counts'] >= 0)
                       & (arrays['observed_counts'] <= common_manifest['n_sites'])),
            'Common representation roster, dimensions or call masks differ')
    fit = list(map(int, roles['FIT']))
    query = sorted(int(i) for role in ('SELECT', 'EVALUATE') for i in roles[role])
    common_scale = rare_scale = None; common_positive = rare_positive = None
    common_mass = rare_mass = None
    rare = None
    if family == 'R_PLUS_C':
        rare = affinity_io.AuthenticatedRareView(
            _rare_inputs(request['rare']), evidence_scope=request['evidence_scope'],
            max_input_bytes=limits['max_input_bytes'],
            max_metadata_bytes=limits['max_metadata_bytes'])
        require(rare.persons == persons
                and rare.upstream['source_contract_sha256'] == common_manifest['source_contract_sha256']
                and rare.upstream['block_id'] == common_manifest['block_id']
                and rare.upstream['exclusion_scope'] == common_manifest['exclusion_scope'],
                'R and common roster/source/territory differ')
        rare_scale, rare_positive, rare_mass = _component_pair_positive_mean(
            rare.fit_edges(), persons, 'Rare')
        common_scale, common_positive, common_mass = _component_pair_positive_mean(
            ((i,j,value)for i,j,_,value in _common_fit_edges(
                persons,fit,scores,common_manifest['n_sites'])), persons, 'Common')
    out = Path(output)
    require(not out.exists() and out.parent.is_dir()
            and shutil.disk_usage(out.parent).free >= limits['min_free_bytes'] + limits['max_output_bytes'],
            'New output and declared common-family disk reserve required')
    out.mkdir(mode=0o700)
    shutil.copyfile(persons_path, out/'persons.tsv'); (out/'persons.tsv').chmod(0o600)
    fit_path, query_path = out/'fit_affinities.private.tsv', out/'query_affinities.private.tsv'
    clipped_fit = clipped_query = missing_query = 0
    with fit_path.open('x', newline='') as fit_stream:
        writer = csv.writer(fit_stream, delimiter='\t', lineterminator='\n')
        writer.writerow(affinity_io.FAMILY_EDGE_COLUMNS)
        rare_edges = iter(rare.fit_edges()) if rare else None
        for row_number, (i,j,raw,common_value) in enumerate(
                _common_fit_edges(persons,fit,scores,common_manifest['n_sites']),1):
            clipped_fit += raw < 0
            if rare:
                ri, rj, rare_value = next(rare_edges)
                require((ri, rj) == (i, j), 'R and common FIT edge order differs')
                value = (1-weight)*rare_value/rare_scale + weight*common_value/common_scale
                _write_row(writer, i, j, value)
            else:
                _write_row(writer, i, j, common_value, raw, common_manifest['n_sites'])
            if row_number % 4096 == 0:
                require(fit_stream.tell() <= limits['max_output_bytes']
                        and shutil.disk_usage(out).free >= limits['min_free_bytes'],
                        'FIT family table exceeds output/disk bound')
        if rare:
            require(next(rare_edges, None) is None, 'Unexpected extra rare FIT edge')
    with query_path.open('x', newline='') as query_stream:
        writer = csv.writer(query_stream, delimiter='\t', lineterminator='\n')
        writer.writerow(affinity_io.FAMILY_EDGE_COLUMNS)
        rare_rows = iter(rare.query_rows()) if rare else None
        row_number = 0
        for q in query:
            rare_observations = None
            if rare:
                rq, rare_observations, _ = next(rare_rows)
                require(rq == q, 'R and common query order differs')
            raw_values,common_values=_common_query_row(q,fit,scores,common_manifest['n_sites'])
            for reference,raw,common_value in zip(fit,raw_values,common_values):
                raw=float(raw);common_value=float(common_value)
                clipped_query += raw < 0; row_number += 1
                if rare:
                    rare_value = rare_observations[reference]
                    if rare_value is None:
                        missing_query += 1; _write_row(writer, q, reference, None)
                    else:
                        value = ((1-weight)*rare_value/rare_scale
                                 + weight*common_value/common_scale)
                        _write_row(writer, q, reference, value)
                else:
                    _write_row(writer, q, reference, common_value, raw, common_manifest['n_sites'])
                if row_number % 4096 == 0:
                    require(fit_path.stat().st_size + query_stream.tell() <= limits['max_output_bytes']
                            and shutil.disk_usage(out).free >= limits['min_free_bytes'],
                            'Query family table exceeds output/disk bound')
        if rare:
            require(next(rare_rows, None) is None, 'Unexpected extra rare query row')
    for path in (fit_path, query_path): path.chmod(0o600)
    require(sum(p.stat().st_size for p in out.iterdir()) <= limits['max_output_bytes'],
            'Common-family output exceeds byte bound')
    territory = dict(block_id=common_manifest['block_id'],
        exclusion_scope=common_manifest['exclusion_scope'],
        allowed_input_intervals=common_manifest['allowed_input_intervals'],
        source_contract_sha256=common_manifest['source_contract_sha256'])
    sources = dict(common=dict(directory=str(common_root),
        manifest_sha256=sha(common_manifest_path), arrays_sha256=sha(arrays_path),
        representation='FIT_WEIGHTED_CENTERED_MEAN_IMPUTED_COMPLETE_NUMERICAL_KERNEL',
        n_sites=common_manifest['n_sites'], weighting=common_manifest['weighting'],
        observed_counts_preserved_in_source=True),
        source_contract=dict(manifest=str(contract_path), manifest_sha256=sha(contract_path),
                             persons_sha256=sha(persons_path)),
        rare=None if rare is None else dict(affinity_inputs_sha256=rare.source_sha256,
                                            measure='SOURCE_MINOR_JACCARD'))
    if family == 'C':
        affinity = dict(measure='POSITIVE_PART_FIT_CENTERED_COMMON_KERNEL',
            units='STANDARDIZED_DOSAGE_CROSS_PRODUCT_PER_SELECTED_COMMON_SITE',
            transformation='MAX(FIT_WEIGHTED_CENTERED_MEAN_IMPUTED_KERNEL,0)',
            minimum=0.0, maximum=None)
        normalization = dict(scope='NONE', mix_weight=None,
            fit_positive_common_scale=None, fit_positive_rare_scale=None,
            queries_used_for_scale=False)
        support = dict(row_encoding='NUMERIC_SOURCE_COMPONENT',
            numerator='SIGNED_COMMON_KERNEL_CROSS_PRODUCT_BEFORE_POSITIVE_PART',
            denominator='SELECTED_COMMON_SITE_COUNT',
            person_call_counts_source='continuous.private.npz:observed_counts')
    else:
        affinity = dict(measure='FIT_NORMALIZED_RARE_PLUS_COMMON_AFFINITY',
            units='WEIGHTED_SUM_OF_COMPONENTS_IN_FIT_POSITIVE_MEAN_UNITS',
            transformation='(1-w)*R/sR+w*C/sC_WITH_C_POSITIVE_COMMON_KERNEL',
            minimum=0.0, maximum=None)
        normalization = dict(scope='ELIGIBLE_CROSS_COMPONENT_FIT_EDGES_ONLY',
            policy='COMPONENT_PAIR_EQUAL_POSITIVE_MEAN',
            mix_weight=weight, fit_positive_common_scale=common_scale,
            fit_positive_rare_scale=rare_scale,
            n_positive_common_fit_edges=common_positive,
            n_positive_rare_fit_edges=rare_positive,
            positive_common_fit_component_pair_mass=common_mass,
            positive_rare_fit_component_pair_mass=rare_mass,
            queries_used_for_scale=False)
        support = dict(row_encoding='COMPOSITE_SOURCES_PINNED',
            numerator='NOT_SINGLE_RATIO', denominator='COMPONENT_DENOMINATORS_IN_PINNED_SOURCES',
            common_person_call_counts_source='continuous.private.npz:observed_counts',
            rare_pair_denominators_source='pinned rare affinity tables')
    sizes = {}
    for i in fit: sizes[persons[i]['component']] = sizes.get(persons[i]['component'], 0)+1
    fit_edges = len(fit)*(len(fit)-1)//2-sum(n*(n-1)//2 for n in sizes.values())
    metadata = dict(schema=affinity_io.FAMILY_SCHEMA,
        stage='FAMILY_AFFINITIES_READY_NOT_SCIENTIFICALLY_ADMITTED', family=family,
        affinity=affinity, missing_edge_policy='UNKNOWN_NOT_ZERO',
        same_component_fit_edges='EXCLUDE', territory=territory,
        role_counts={k: len(v) for k, v in roles.items()},
        persons_content_sha256=discrete.materialize.digest(persons),
        normalization=normalization, support=support, sources=sources,
        capacity=dict(fit_edges=fit_edges, query_fit_edges=len(query)*len(fit),
            fit_support='COMPLETE_OBSERVED_REQUIRED',
            query_support='EXPLICIT_OBSERVED_OR_UNKNOWN',
            dense_spectrum_memory_order='O_N_FIT_SQUARED'),
        sha256={p.name: sha(p) for p in (out/'persons.tsv', fit_path, query_path)},
        target_outcomes_read=False, evaluate_scored=False,
        scientific_admission=False, biological_validation=False)
    write_json(out/'affinity_inputs.json', metadata)
    require(sha(request_path) == request_sha256
            and request['implementation_sha256'] == implementation_hashes()
            and sha(common_manifest_path) == request['common']['manifest_sha256'],
            'Request/code/common source changed during family materialization')
    return dict(family=family, fit_edges=fit_edges,
                query_fit_edges=len(query)*len(fit), missing_query_edges=missing_query,
                common_negative_fit_pairs_clipped=clipped_fit,
                common_negative_query_pairs_clipped=clipped_query,
                manifest_sha256=sha(out/'affinity_inputs.json'),
                scientific_admission=False, biological_validation=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.request, args.request_sha256, args.output),
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
