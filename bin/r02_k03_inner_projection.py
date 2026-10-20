#!/usr/bin/env python3
"""Connect one frozen INNER_FIT split to existing graph, Leiden and projection APIs.

This module supplies no resolution, seed, stopping rule, or support threshold.
The caller must provide an immutable episode made by
``r02_structure_search_policy.freeze_episode``.  Inputs are already decoded by
the authenticated affinity producer; genomic parsing remains its responsibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

import r02_structure_continuous as continuous
import r02_structure_discrete as discrete
import r02_structure_fit_partitions as partitions
import r02_structure_search_policy as policy
import r02_resolution_profile as profile

GRAPH_ARRAYS = ('fit_indices','adjacency','degree','active_fit','inverse_sqrt_degree',
    'normalized','connected_component','eigenvalues','eigenvectors','trivial_modes',
    'zero_modes','projectable_modes','fit_scores','shared_component_ids',
    'shared_component_volumes','shared_component_basis','shared_component_fit_scores',
    'shared_fit_states')
QUERY_ARRAYS = ('query_indices','query_scores','query_shared_scores','query_status',
                'query_observed_fit_count')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _require_hash(value, name):
    discrete.require(discrete.valid_hash(value), name + ' SHA256 required')


def load_spectrum(*, artifact_directory, manifest_sha256, persons,
                  fold_plan_bytes, fold_plan_sha256, source_sha256,
                  max_input_bytes, max_memory_bytes):
    """Restore the producer's full spectrum without eigendecomposition."""
    for value,name in ((manifest_sha256,'Spectrum manifest'),(fold_plan_sha256,'Fold plan'),
                       (source_sha256,'Affinity source')): _require_hash(value,name)
    discrete.require(hashlib.sha256(fold_plan_bytes).hexdigest()==fold_plan_sha256,
                     'Fold plan file identity differs')
    plan=json.loads(fold_plan_bytes); root=Path(artifact_directory); mp=root/'manifest.json'
    discrete.require(mp.is_file() and mp.stat().st_size<=1<<20
                     and hashlib.sha256(mp.read_bytes()).hexdigest()==manifest_sha256,
                     'Spectrum manifest identity differs')
    manifest=json.loads(mp.read_text()); payload=root/manifest.get('file','__invalid__')
    discrete.require(manifest.get('schema')=='r02_k03_full_inner_spectrum_artifacts_v1'
        and manifest.get('source_sha256')==source_sha256
        and manifest.get('fold_plan_sha256')==fold_plan_sha256
        and manifest.get('full_spectrum_preserved') is True
        and manifest.get('predictive_rank_selected') is False
        and manifest.get('communities_learned') is False
        and manifest.get('genetic_response_read') is False,
        'Spectrum source/fold/semantics differ')
    discrete.require(payload.is_file() and payload.stat().st_size==manifest['bytes']
        and payload.stat().st_size<=max_input_bytes
        and hashlib.sha256(payload.read_bytes()).hexdigest()==manifest['sha256'],
        'Spectrum payload identity differs')
    expected=set(GRAPH_ARRAYS+QUERY_ARRAYS)
    with zipfile.ZipFile(payload) as z:
        discrete.require(set(z.namelist())=={n+'.npy' for n in expected},
                         'Spectrum array inventory differs')
        discrete.require(sum(i.file_size for i in z.infolist())<=max_memory_bytes,
                         'Spectrum decoded size exceeds memory budget')
    train=tuple(plan['inner_fit_indices']); project=tuple(plan['inner_project_indices'])
    discrete.require(train == tuple(sorted(set(train)))
        and project == tuple(sorted(set(project))), 'Fold indices must be unique and ordered')
    with zipfile.ZipFile(payload) as z:
        expanded = sum(i.file_size for i in z.infolist())
    # Include array copies, Python graph-identity construction and JSON hashing.
    workspace_bound = 8*expanded + 80*len(train)**2 + 256*(len(train)+len(project))
    discrete.require(workspace_bound <= max_memory_bytes,
                     'Spectrum restoration workspace exceeds memory budget')
    with np.load(payload,allow_pickle=False) as saved:
        arrays={n:saved[n] for n in expected}
    decoded=sum(a.nbytes for a in arrays.values())
    discrete.require(decoded<=max_memory_bytes,'Decoded spectrum exceeds memory budget')
    roles=discrete.shared.validate_roles(persons)
    discrete.require(plan.get('schema')=='r02_k03_inner_fit_pair_complete_plan_v1'
        and np.array_equal(arrays['fit_indices'],train)
        and np.array_equal(arrays['query_indices'],project)
        and set(train)|set(project)==set(map(int,roles['FIT']))
        and set(train).isdisjoint(project)
        and {persons[i]['component'] for i in train}.isdisjoint(
            {persons[i]['component'] for i in project}),
        'Spectrum indices differ from INNER_FIT plan')
    n=len(train); q=len(project)
    discrete.require(arrays['adjacency'].shape==arrays['normalized'].shape==
        arrays['eigenvectors'].shape==(n,n)
        and all(arrays[k].shape==(n,) for k in ('degree','active_fit','inverse_sqrt_degree',
            'connected_component','eigenvalues','trivial_modes','zero_modes','projectable_modes',
            'shared_fit_states'))
        and arrays['fit_scores'].shape[0]==n
        and arrays['shared_component_basis'].shape[0]==n
        and arrays['shared_component_fit_scores'].shape[0]==n
        and all(arrays[k].shape[0]==q for k in QUERY_ARRAYS),
        'Spectrum array dimensions differ from fold')
    graph=continuous.FitRareGraphSpectrum.__new__(continuous.FitRareGraphSpectrum)
    for name in GRAPH_ARRAYS:
        arrays[name].setflags(write=False); setattr(graph,name,arrays[name])
    graph._arrays=GRAPH_ARRAYS
    graph._pins={n:hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in GRAPH_ARRAYS}
    graph.memory_bound=(20*len(train)*len(train)+128*len(train))*8
    graph.persons_sha256=discrete.materialize.digest(persons)
    graph.source_sha256=source_sha256; graph.query_indices=project
    graph.affinity_semantics=manifest['affinity_semantics']
    graph.numerical_tolerance=manifest['numerical_tolerance']
    identity=dict(source_sha256=source_sha256,persons_sha256=graph.persons_sha256,
        fit_indices=list(train),adjacency=graph.adjacency.tolist(),
        same_component_fit_edges='EXCLUDE',missing_edge_policy='UNKNOWN_NOT_ZERO',
        affinity_semantics=graph.affinity_semantics)
    graph.graph_sha256=discrete.materialize.digest(identity)
    graph.restored_identity_scope='EXPLICIT_AFFINITY_SEMANTICS'
    default=dict(family='R',measure='SOURCE_MINOR_JACCARD',
        units='DIMENSIONLESS_INTERSECTION_OVER_UNION',minimum=0.0,maximum=1.0)
    if graph.graph_sha256!=manifest['graph_sha256'] and graph.affinity_semantics==default:
        # The existing constructor deliberately preserves this exact historical digest.
        legacy=dict(identity);legacy.pop('affinity_semantics')
        graph.graph_sha256=discrete.materialize.digest(legacy)
        graph.restored_identity_scope='LEGACY_DEFAULT_SOURCE_MINOR_JACCARD'
    discrete.require(graph.graph_sha256==manifest['graph_sha256'],
                     'Restored graph identity differs')
    graph.verify(source_sha256)
    continuous_rows={str(int(q)):{'scores':arrays['query_scores'][i].copy(),
        'shared_component_scores':arrays['query_shared_scores'][i].copy(),
        'status':str(arrays['query_status'][i]),
        'observed_fit_count':int(arrays['query_observed_fit_count'][i]),
        'query_index':int(q),'graph_sha256':graph.graph_sha256,
        'source_sha256':source_sha256} for i,q in enumerate(arrays['query_indices'])}
    return graph,plan,continuous_rows


def run_fold(*, persons, artifact_directory, spectrum_manifest_sha256,
             fold_plan_bytes, fold_plan_sha256, source_sha256, view_id,
             query_observations, episode, max_input_bytes, max_memory_bytes,
             max_working_bytes, max_iterations):
    """Run an already frozen episode and project its candidates.

    Saved spectrum and ``query_observations`` must come from the same authenticated
    INNER_FIT affinity product. The observations map each INNER_PROJECT index
    to its observed INNER_FIT neighbors. Missing affinities stay absent and are never converted to zero.
    """
    _require_hash(fold_plan_sha256, 'Fold plan')
    _require_hash(source_sha256, 'Affinity source')
    discrete.require(isinstance(fold_plan_bytes, bytes)
                     and hashlib.sha256(fold_plan_bytes).hexdigest() == fold_plan_sha256,
                     'Fold plan file identity differs')
    graph,fold_plan,continuous_rows=load_spectrum(artifact_directory=artifact_directory,
        manifest_sha256=spectrum_manifest_sha256,persons=persons,
        fold_plan_bytes=fold_plan_bytes,fold_plan_sha256=fold_plan_sha256,
        source_sha256=source_sha256,max_input_bytes=max_input_bytes,
        max_memory_bytes=max_memory_bytes)
    discrete.require(fold_plan.get('schema') == 'r02_k03_inner_fit_pair_complete_plan_v1',
                     'Exact pair-complete INNER_FIT plan required')
    train = tuple(fold_plan['inner_fit_indices'])
    project = tuple(fold_plan['inner_project_indices'])
    discrete.require(train == tuple(sorted(set(train))) and project == tuple(sorted(set(project)))
                     and set(train).isdisjoint(project), 'Invalid INNER_FIT/INNER_PROJECT indices')
    discrete.require(set(query_observations) == set(project),
                     'Projection rows must cover exactly INNER_PROJECT')
    policy.verify_episode(episode)
    discrete.require(isinstance(view_id, str) and view_id
                     and episode['source_sha256'] == source_sha256
                     and episode['view_id'] == view_id,
                     'Episode source or view differs')
    gamma = float.fromhex(episode['gamma_hex'])
    candidates, projections, episode_results = [], {}, []
    for seed in episode['seeds']:
        candidate = partitions.learn_candidate(
            graph, source_sha256=source_sha256, resolution=gamma, seed=seed,
            max_iterations=max_iterations, max_working_bytes=max_working_bytes)
        discrete.require(candidate['status'] == 'FIT_ZERO_IMPROVEMENT_ITERATION_REACHED',
                         'Unfinished Leiden seed keeps episode incomplete')
        fit_labels = {int(i): label for i, label in candidate['labels'].items()}
        rows = {}
        for query in project:
            observations = query_observations[query]
            # Invoke the existing continuous projection and community assignment;
            # neither operation is reimplemented here.
            community_row = discrete.observed_community_means(fit_labels, observations)
            rows[str(query)] = {'continuous': continuous_rows[str(query)],
                                'community': community_row}
        candidates.append(candidate)
        projections[str(seed)] = rows
        episode_results.append({
            'seed': seed, 'source_sha256': source_sha256,
            'gamma_hex': episode['gamma_hex'], 'status': candidate['status'],
            'membership_key': profile.make_line(graph, candidate['labels']).key})
    coverage = policy.assess_episode(episode, episode_results)
    graph.verify(source_sha256)
    return {
        'schema': 'r02_k03_inner_fit_leiden_projection_v1',
        'status': 'EPISODE_COMPLETE_PREDICTIONS_UNSCORED' if coverage['status'] == 'DECLARED_SEED_MASS_COVERED'
                  else 'NEW_MEMBERSHIPS_REQUIRE_FROZEN_FOLLOWUP_EPISODE',
        'fold_id': fold_plan['fold_id'], 'view_id': view_id,
        'fold_plan_sha256': fold_plan_sha256,
        'source_sha256': source_sha256, 'graph_sha256': graph.graph_sha256,
        'episode_sha256': episode['episode_sha256'], 'candidates': candidates,
        'projections': projections, 'coverage': coverage,
        'inner_fit_people': len(train), 'inner_project_people': len(project),
        'select_evaluate_read': False, 'responses_read': False,
        'ridge_interface': {
            'module': 'r02_ridge_finite_family',
            'required_calls': ['fit_gram_statistics', 'finite_family'],
            'required_identity': ['fold_plan_sha256', 'source_sha256', 'graph_sha256',
                                  'response_sha256', 'preprocessing_sha256'],
            'released': False},
        'scientific_calibration_released': False, 'biological_validation': False}


def implementation_hashes():
    here = Path(__file__)
    paths = [here, Path(continuous.__file__), Path(discrete.__file__),
             Path(partitions.__file__), Path(policy.__file__), Path(profile.__file__)]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
