#!/usr/bin/env python3
"""FIT-only community candidates on the exact validated continuous rare graph.

This component does not choose a human resolution grid or a winning partition.
The caller supplies each resolution/seed and a computational iteration limit.
No query edges, outcomes, global M16.5 labels, or cohort filters are accepted.
Convergence means that a complete Leiden iteration found no improvement; it
does not prove a global optimum or biological support. Isolates stay UNASSIGNED.
"""
from __future__ import annotations

import hashlib
import argparse
from importlib.metadata import version
import json
import shutil
from pathlib import Path

import igraph
import leidenalg
import numpy as np

from r02_structure_continuous import FitRareGraphSpectrum
import r02_structure_continuous as continuous
import r02_structure_discrete as discrete
from r02_structure_affinity_io import AuthenticatedFitAffinities

require = discrete.require
CODE_FILES = ('r02_structure_fit_partitions.py', 'r02_structure_affinity_io.py', *continuous.CODE_FILES)


def implementation_hashes():
    return {name: discrete.sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def canonical_labels(membership, global_indices):
    """Stable names depend on FIT member indices, never library label numbers."""
    groups = {}
    for label, index in zip(membership, global_indices):
        groups.setdefault(int(label), []).append(int(index))
    ordered = sorted(tuple(sorted(v)) for v in groups.values())
    return {i: 'FIT_C%04d' % (k + 1) for k, members in enumerate(ordered) for i in members}


def learn_candidate(spectrum, *, source_sha256, resolution, seed,
                    max_iterations, max_working_bytes):
    """Return one explicit RB-configuration Leiden candidate and diagnostics.

    Reuses the authenticated FIT graph rather than implementing another affinity
    builder. The graph excludes within-component FIT edges by its existing API.
    A complete observed graph is required by that API; that restriction is not
    authorization to drop people whose affinities are missing.

    The workspace check is a conservative estimate for this extra graph, not an
    OS memory cap. Deployment must also cap the process and measure its peak.
    """
    require(isinstance(spectrum, FitRareGraphSpectrum), 'Validated FIT rare graph required')
    spectrum.verify(source_sha256)
    require(type(resolution) in (int, float) and np.isfinite(resolution) and resolution > 0,
            'Explicit positive finite resolution required; no default grid')
    require(type(seed) is int and 0 <= seed < 2**31, 'Explicit nonnegative 31-bit seed required')
    require(type(max_iterations) is int and max_iterations > 0, 'Positive iteration bound required')
    require(type(max_working_bytes) is int and max_working_bytes > 0, 'Explicit workspace bound required')
    active = np.flatnonzero(spectrum.active_fit)
    n_edges = int(np.count_nonzero(spectrum.adjacency)) // 2
    bound = 512*n_edges + 1024*len(spectrum.fit_indices)
    require(bound <= max_working_bytes, 'Community graph workspace estimate exceeds budget')
    adjacency = spectrum.adjacency[np.ix_(active, active)]
    left, right = np.nonzero(np.triu(adjacency, 1))
    weights = adjacency[left, right]
    graph = igraph.Graph(n=len(active), edges=zip(left.tolist(), right.tolist()), directed=False)
    partition = leidenalg.RBConfigurationVertexPartition(
        graph, weights=weights.tolist(), resolution_parameter=float(resolution))
    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(seed)
    history, converged = [], False
    previous = float(partition.quality())
    for iteration in range(1, max_iterations + 1):
        improvement = float(optimiser.optimise_partition(partition, n_iterations=1))
        quality = float(partition.quality())
        # Floating arithmetic tolerance scales with the unnormalised objective.
        tolerance = 64*np.finfo(float).eps*max(1., abs(previous), abs(quality), float(weights.sum()))
        require(np.isfinite(quality) and np.isfinite(improvement)
                and quality >= previous-tolerance and improvement >= -tolerance,
                'Nonfinite or decreasing Leiden objective')
        history.append(dict(iteration=iteration, improvement=improvement, objective=quality))
        previous = quality
        if improvement == 0.0:
            converged = True
            break
    membership = np.asarray(partition.membership, dtype=np.int64)
    degree = adjacency.sum(axis=1)
    volume = float(degree.sum())
    within = float(2*weights[membership[left] == membership[right]].sum())
    degree_per_community = np.bincount(membership, weights=degree)
    independent_quality = within - float(resolution)*float(degree_per_community @ degree_per_community)/volume
    require(np.isclose(independent_quality, previous, rtol=1e-11, atol=1e-11*volume),
            'Leiden objective differs from explicit RB-configuration calculation')
    active_labels = canonical_labels(membership, spectrum.fit_indices[active])
    # String keys are canonical before hashing and survive JSON roundtrips.
    labels = {str(int(i)): active_labels.get(int(i), 'UNASSIGNED') for i in spectrum.fit_indices}
    spectrum.verify(source_sha256)
    result = dict(
        schema='r02_fit_leiden_candidate_v1',
        status='FIT_ZERO_IMPROVEMENT_ITERATION_REACHED' if converged else 'ITERATION_LIMIT_NOT_ADMITTED',
        labels=labels, source_sha256=source_sha256, graph_sha256=spectrum.graph_sha256,
        persons_sha256=spectrum.persons_sha256, fit_indices=spectrum.fit_indices.tolist(),
        resolution=float(resolution), seed=seed, max_iterations=max_iterations,
        iterations=history, objective=previous, normalized_objective=previous/volume,
        objective_definition='sum_ij(A_ij-gamma*k_i*k_j/sum_ij(A_ij))*1(c_i=c_j)',
        libraries=dict(igraph=version('igraph'), leidenalg=version('leidenalg')),
        graph_edges=n_edges, graph_volume=volume, workspace_estimate_bytes=bound,
        assigned_fit=len(active), unassigned_fit=len(labels)-len(active),
        same_component_fit_edges='EXCLUDE', missing_edge_policy='UNKNOWN_NOT_ZERO',
        learning_roles=['FIT'], target_used=False, heldout_edges_used=False,
        global_optimum_proven=False, resolution_selected=False, biological_validation=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result['candidate_sha256'] = discrete.materialize.digest(result)
    return result


def write_frozen_candidate(output, candidate, *, persons, persons_sha256,
                           fit_affinities_sha256, exclusion_sha256):
    """Write an admitted candidate in the existing ProjectionBundle schema.

    This is upstream provenance, not a substitute for the downstream reader's
    authentication of the actual persons/affinity/exclusion files. No files are
    overwritten. Iteration-capped candidates cannot be exposed for projection.
    """
    payload = {k: v for k, v in candidate.items() if k != 'candidate_sha256'}
    require(candidate.get('candidate_sha256') == discrete.materialize.digest(payload),
            'Candidate changed after FIT learning')
    require(candidate['status'] == 'FIT_ZERO_IMPROVEMENT_ITERATION_REACHED', 'Unconverged candidate cannot be frozen')
    roles = discrete.shared.validate_roles(persons)
    require(discrete.materialize.digest(persons) == candidate['persons_sha256']
            and np.array_equal(roles['FIT'], candidate['fit_indices']), 'Candidate FIT roster/person metadata changed')
    for value in (persons_sha256, fit_affinities_sha256, exclusion_sha256):
        require(discrete.valid_hash(value), 'Explicit artifact hashes required')
    out = Path(output)
    require(not out.exists(), 'Candidate output must be new; never overwrite')
    out.mkdir(parents=True, mode=0o700)
    table = 'sample_id\tcommunity\n' + ''.join(
        persons[i]['sample_id']+'\t'+candidate['labels'][str(i)]+'\n' for i in roles['FIT'])
    (out/'communities.private.tsv').write_text(table)
    method = dict(schema='r02_fit_community_method_v1',
                  algorithm='Leiden RBConfigurationVertexPartition',
                  parameters={k: v for k, v in candidate.items() if k not in ('labels', 'code_sha256')},
                  code_sha256=candidate['code_sha256'], fit_only=True, exclusion_sha256=exclusion_sha256)
    (out/'community_method.json').write_text(json.dumps(method, indent=2, allow_nan=False)+'\n')
    receipt = dict(schema='r02_fit_communities_frozen_v1', status='FIT_ONLY_FROZEN_BEFORE_PROJECTION',
                   fit_ids_sha256=discrete.ordered_ids_hash(persons[i]['sample_id'] for i in roles['FIT']),
                   fit_affinities_sha256=fit_affinities_sha256,
                   communities_sha256=discrete.sha(out/'communities.private.tsv'),
                   persons_sha256=persons_sha256, exclusion_sha256=exclusion_sha256,
                   learning_roles=['FIT'], target_used=False, heldout_edges_used=False,
                   method_manifest_sha256=discrete.sha(out/'community_method.json'))
    (out/'frozen_communities.json').write_text(json.dumps(receipt, indent=2)+'\n')
    for path in out.iterdir():
        path.chmod(0o600)
    return receipt


def run_request(request_path, request_sha256, output):
    """One explicit candidate from authenticated upstream affinity products.

    This adapter consumes no query edges. Shared authentication accepts the
    original whole-chromosome producer or the exact-interval producer for both
    reviewed geometries. No implicit resolution/seed or global labels are used.
    """
    path = Path(request_path)
    require(discrete.valid_hash(request_sha256) and path.is_file()
            and path.stat().st_size <= 2**20 and discrete.sha(path) == request_sha256,
            'FIT learning request identity changed')
    r = discrete.load_json(path)
    require(set(r) == {'schema', 'evidence_scope', 'inputs', 'implementation_sha256', 'parameters', 'resources'}
            and r['schema'] == 'r02_fit_learning_request_v1'
            and r['evidence_scope'] in {'SYNTHETIC_INSTRUMENT_ONLY', 'DNABR_R02'}
            and r['implementation_sha256'] == implementation_hashes(), 'Exact FIT request/code required')
    names = {'persons.tsv', 'fit_affinities.private.tsv', 'exclusion.json', 'affinity_inputs.json'}
    require(set(r['inputs']) == names, 'Exact FIT-only input inventory required')
    limits = r['resources']
    require(set(limits) == {'max_input_bytes', 'max_memory_bytes', 'min_free_bytes'}
            and all(type(v) is int and v > 0 for v in limits.values()), 'Explicit FIT resource limits required')
    params = r['parameters']
    require(set(params) == {'resolution', 'seed', 'max_iterations'}, 'Explicit candidate parameters required')
    admitted = AuthenticatedFitAffinities(r['inputs'], evidence_scope=r['evidence_scope'],
        max_input_bytes=limits['max_input_bytes'], max_metadata_bytes=limits['max_input_bytes'])
    persons = admitted.persons
    graph_pin = admitted.source_sha256
    graph = FitRareGraphSpectrum(persons, admitted.fit_edges(), source_sha256=graph_pin,
                                max_memory_bytes=limits['max_memory_bytes'])
    available = limits['max_memory_bytes']-graph.memory_bound
    require(available > 0, 'No memory budget left for community graph')
    candidate = learn_candidate(graph, source_sha256=graph_pin, max_working_bytes=available, **params)
    out = Path(output)
    require(out.parent.exists() and shutil.disk_usage(out.parent).free >= limits['min_free_bytes'],
            'Insufficient FIT output disk reserve')
    # All sources remain immutable through the computation.
    admitted.verify()
    require(discrete.sha(path) == request_sha256 and implementation_hashes() == r['implementation_sha256'],
            'FIT source or code changed during learning')
    receipt = write_frozen_candidate(out, candidate, persons=persons,
        persons_sha256=r['inputs']['persons.tsv']['sha256'],
        fit_affinities_sha256=r['inputs']['fit_affinities.private.tsv']['sha256'],
        exclusion_sha256=r['inputs']['exclusion.json']['sha256'])
    (out/'candidate.json').write_text(json.dumps(candidate, indent=2)+'\n')
    (out/'candidate.json').chmod(0o600)
    manifest = dict(schema='r02_fit_learning_result_v1', request_sha256=request_sha256,
        inputs=r['inputs'], implementation_sha256=r['implementation_sha256'],
        evidence_scope=r['evidence_scope'], graph_sha256=graph.graph_sha256,
        outputs_sha256={p.name: discrete.sha(p) for p in out.iterdir()},
        biological_validation=False, evaluate_scored=False, query_edges_used=False,
        frozen_communities_sha256=discrete.sha(out/'frozen_communities.json'))
    (out/'learning_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'learning_manifest.json').chmod(0o600)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run_request(args.request, args.request_sha256, args.output)))
