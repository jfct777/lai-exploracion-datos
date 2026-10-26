"""Parameter-free crossing locations for a bounded, FIT-only RB search.

Leiden still performs community optimization. This controller chooses queries
from intersections of known partition objectives a-gamma*b. A closed search
means no undiscovered envelope improvement under the supplied oracle/seed
policy. An approximate oracle such as Leiden never certifies the global upper
envelope, all community partitions, or biological structure.
"""
from dataclasses import dataclass
import math
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PartitionLine:
    key: str
    intercept: float
    penalty: float

    def __post_init__(self):
        require(isinstance(self.key, str) and bool(self.key), 'Canonical membership key required')
        require(math.isfinite(self.intercept) and self.intercept >= 0 and
                math.isfinite(self.penalty) and self.penalty > 0,
                'Finite nonnegative internal affinity and positive null penalty required')

    def quality(self, gamma):
        return self.intercept-gamma*self.penalty


def envelope(lines):
    """Upper hull, preserving every membership sharing an identical line.

    Lines are sorted by increasing slope (-penalty). Start locations are exact
    intersections in floating arithmetic, not a fixed biological resolution
    grid. Equal slope with smaller intercept is dominated for every gamma.
    """
    groups = {}
    for line in lines:
        groups.setdefault((line.penalty, line.intercept), []).append(line.key)
    coefficients = []
    for (penalty, intercept), keys in sorted(groups.items(), key=lambda x: (-x[0][0], -x[0][1])):
        if coefficients and penalty == coefficients[-1]['penalty']:
            continue
        coefficients.append(dict(penalty=penalty, intercept=intercept, keys=sorted(keys)))
    hull = []
    for line in coefficients:
        start = -math.inf
        while hull:
            previous = hull[-1]
            start = (previous['intercept']-line['intercept'])/(previous['penalty']-line['penalty'])
            require(math.isfinite(start), 'Nonfinite objective crossing')
            if start > previous['start']:
                break
            hull.pop()
        hull.append(line | dict(start=start if hull else -math.inf))
    return hull


def bounded_profile(initial, oracle, *, gamma_low, gamma_high, max_queries):
    """Call the supplied optimizer only at endpoints, gamma1 and hull crossings.

    Initial lines must contain the analytically optimal endpoint partitions.
    The caller authenticates graph, line coefficients, membership and seeds.
    No SELECT responses or outcomes enter this function. Query caps produce an
    incomplete result, never a pruned successful search. Every returned line is
    retained, including distinct equal-objective memberships. Floating-point
    tolerance is reported, not used to discard a scientific candidate.
    """
    require(math.isfinite(gamma_low) and math.isfinite(gamma_high) and
            0 < gamma_low <= gamma_high and type(max_queries) is int and max_queries > 0,
            'Explicit finite positive domain and query bound required')
    known = {}

    def insert(line):
        require(isinstance(line, PartitionLine), 'Authenticated partition line required')
        require(line.key not in known or known[line.key] == line,
                'Same membership changed objective coefficients')
        known[line.key] = line

    for line in initial:
        insert(line)
    require(known, 'Analytic endpoint partitions required')
    queried, records = set(), []
    anchors = sorted({gamma_low, gamma_high} | ({1.} if gamma_low <= 1 <= gamma_high else set()))
    while True:
        hull = envelope(known.values())
        desired = anchors + [h['start'] for h in hull if gamma_low < h['start'] < gamma_high]
        pending = [g for g in desired if g not in queried]
        if not pending:
            status = 'NO_NEW_ENVELOPE_IMPROVEMENT_UNDER_DECLARED_ORACLE'
            break
        if len(queried) >= max_queries:
            status = 'QUERY_LIMIT_INCOMPLETE_NOT_ADMITTED'
            break
        gamma = pending[0]
        prior = max(line.quality(gamma) for line in known.values())
        new = list(oracle(gamma))
        require(new, 'Optimizer returned no candidates')
        for line in new:
            insert(line)
        best = max(line.quality(gamma) for line in known.values())
        tolerance = 64*sys.float_info.epsilon*max(
            1., *(abs(l.intercept)+gamma*abs(l.penalty) for l in known.values()))
        records.append(dict(gamma=gamma, candidates=[l.key for l in new],
                            previous_best=prior, new_best=best,
                            improvement=best-prior, arithmetic_tolerance=tolerance))
        queried.add(gamma)
    active = set()
    # Retain even partitions optimal only at a crossing (a zero-width regime).
    # The hull may discard their line, but this scan cannot discard memberships
    # with tied/near-tied quality at queried points due to numerical arithmetic.
    positions = anchors + [h['start'] for h in hull if gamma_low <= h['start'] <= gamma_high]
    for gamma in positions:
        best = max(line.quality(gamma) for line in known.values())
        tolerance = 64*sys.float_info.epsilon*max(
            1., *(abs(l.intercept)+gamma*abs(l.penalty) for l in known.values()))
        active.update(l.key for l in known.values() if best-l.quality(gamma) <= tolerance)
    return dict(schema='r02_rb_crossing_profile_v1', status=status,
                gamma_low=gamma_low, gamma_high=gamma_high, max_queries=max_queries,
                queries=records, discovered_memberships=len(known),
                envelope_candidate_keys=sorted(active),
                all_lines=[vars(l) for l in sorted(known.values(), key=lambda x: x.key)],
                unresolved_queries=pending if status.startswith('QUERY_LIMIT') else [],
                global_optimum_proven=False, all_partitions_found=False,
                heldout_used=False, biological_validation=False)


def make_line(spectrum, labels, prepared_workspace=None):
    """Canonical membership and RB coefficients shared by both profile drivers.

    This is the identical formula formerly nested in fit_leiden_profile. It
    does not select or optimize a partition, or use held-out observations.
    """
    import hashlib
    import json
    import numpy as np
    if prepared_workspace is None:
        a, d = spectrum.adjacency, spectrum.degree
        left, right = np.nonzero(np.triu(a, 1))
        weights = a[left, right]
        volume = float(d.sum())
        active = np.flatnonzero(spectrum.active_fit)
    else:
        prepared_workspace.verify(
            spectrum, source_sha256=spectrum.source_sha256,
            max_working_bytes=prepared_workspace.workspace_estimate_bytes)
        d = spectrum.degree
        left, right, weights = (prepared_workspace.line_left,
                                prepared_workspace.line_right,
                                prepared_workspace.line_weights)
        volume = prepared_workspace.volume
        active = prepared_workspace.active
    require(set(labels) == {str(int(i)) for i in spectrum.fit_indices}, 'Candidate FIT membership differs')
    ordered = [labels[str(int(i))] for i in spectrum.fit_indices]
    names = {}
    canonical = []
    for index, label in enumerate(ordered):
        if not spectrum.active_fit[index]:
            require(label == 'UNASSIGNED', 'Isolated FIT received an identity')
            canonical.append(-1)
        else:
            require(label != 'UNASSIGNED', 'Positive-degree FIT is unassigned')
            if label not in names:
                names[label] = len(names)
            canonical.append(names[label])
    membership = np.asarray(canonical)
    degrees = np.bincount(membership[active], weights=d[active])
    key = hashlib.sha256(json.dumps(canonical, separators=(',', ':')).encode()).hexdigest()
    return PartitionLine(key, float(2*weights[membership[left] == membership[right]].sum()/volume),
                         float((degrees/volume) @ (degrees/volume)))


def fit_leiden_profile(spectrum, *, source_sha256, seeds, max_iterations,
                       max_queries, max_working_bytes):
    """Bind the controller to the same authenticated FIT graph/Leiden learner.

    Seeds are explicit, reproducible search replications. This API chooses no
    human seed count or computational budget. The analytic endpoint lines are
    known independently of Leiden; failure to reproduce their memberships is
    reported and prevents exposing an unlearned label file as a learned result.
    Query outcomes, held-out edges and M16.5 community labels are never read.
    """
    import hashlib
    import json
    import numpy as np
    import r02_structure_fit_partitions as learner

    require(isinstance(spectrum, learner.FitRareGraphSpectrum), 'Authenticated FIT spectrum required')
    spectrum.verify(source_sha256)
    require(isinstance(seeds, (list, tuple)) and seeds and len(set(seeds)) == len(seeds)
            and all(type(s) is int and 0 <= s < 2**31 for s in seeds), 'Explicit distinct 31-bit seeds required')
    require(type(max_queries) is int and max_queries > 0 and type(max_working_bytes) is int
            and max_working_bytes > 0 and type(max_iterations) is int and max_iterations > 0,
            'Explicit positive profile resource bounds required')
    a, d = spectrum.adjacency, spectrum.degree
    left, right = np.nonzero(np.triu(a, 1))
    volume = float(d.sum())
    require(len(left) > 0 and volume > 0 and math.isfinite(volume), 'Nonempty positive affinity required')
    low = float(a[left, right].min()/volume)
    high = float(np.max(a[left, right]*volume/(d[left]*d[right])))
    require(math.isfinite(low) and math.isfinite(high) and 0 < low <= high,
            'Analytic gamma domain underflowed or overflowed')
    record_bound = max_queries*len(seeds)*(len(d)*256+max_iterations*256+16384)
    graph_bound = 512*len(left)+1024*len(d)
    require(record_bound+graph_bound <= max_working_bytes, 'Profile candidate/graph memory bound exceeded')
    workspace = learner.prepare_leiden_workspace(
        spectrum, source_sha256=source_sha256,
        max_working_bytes=max_working_bytes-record_bound)
    active = np.flatnonzero(spectrum.active_fit)
    candidates, attempts = {}, []

    endpoint_labels = []
    for membership in (spectrum.connected_component, np.arange(len(d))):
        labels = learner.canonical_labels(membership[active], spectrum.fit_indices[active])
        endpoint_labels.append({str(int(i)):labels.get(int(i), 'UNASSIGNED') for i in spectrum.fit_indices})
    initial = [make_line(spectrum, labels, prepared_workspace=workspace)
               for labels in endpoint_labels]

    def oracle(gamma):
        lines = []
        for seed in seeds:
            candidate = learner.learn_candidate(spectrum, source_sha256=source_sha256,
                resolution=gamma, seed=seed, max_iterations=max_iterations,
                max_working_bytes=max_working_bytes-record_bound,
                prepared_workspace=workspace)
            require(candidate['status'] == 'FIT_ZERO_IMPROVEMENT_ITERATION_REACHED',
                    'Iteration-limited query cannot close a profile')
            line = make_line(spectrum, candidate['labels'], prepared_workspace=workspace)
            require(np.isclose(line.quality(gamma), candidate['normalized_objective'],
                               rtol=1e-11, atol=1e-11), 'Profile line differs from Leiden objective')
            attempts.append(dict(gamma=gamma, seed=seed, membership_key=line.key,
                                 normalized_objective=candidate['normalized_objective'],
                                 iterations=len(candidate['iterations'])))
            candidates.setdefault(line.key, candidate)
            lines.append(line)
        return lines

    result = bounded_profile(initial, oracle, gamma_low=low, gamma_high=high, max_queries=max_queries)
    missing = sorted(set(result['envelope_candidate_keys'])-set(candidates))
    if missing and not result['status'].startswith('QUERY_LIMIT'):
        result['status'] = 'ANALYTIC_ENDPOINT_NOT_REPRODUCED_BY_OPTIMIZER_NOT_ADMITTED'
    spectrum.verify(source_sha256)
    result.update(source_sha256=source_sha256, graph_sha256=spectrum.graph_sha256,
        seeds=list(seeds), max_iterations=max_iterations, graph_volume=volume,
        profile_workspace_estimate_bytes=record_bound+graph_bound,
        objective_normalization='Q_gamma / sum_ij A_ij', attempts=attempts,
        analytic_memberships_without_learned_candidate=missing,
        candidates=candidates,
        predictive_candidate_keys=sorted(candidates),
        candidate_policy='ALL_DISCOVERED_CONVERGED_FIT_MEMBERSHIPS',
        rb_dominated_candidate_keys=sorted(set(candidates)-set(result['envelope_candidate_keys'])),
        envelope_role='DIRECT_RESOLUTION_QUERIES_NOT_FILTER_PREDICTIVE_CANDIDATES',
        alpha_or_outcome_selection=False,
        software_versions=next(iter(candidates.values()))['libraries'])
    return result


def implementation_hashes():
    from pathlib import Path
    import r02_structure_fit_partitions as learner
    names = ('r02_resolution_profile.py', *learner.CODE_FILES)
    return {name:learner.discrete.sha(Path(__file__).with_name(name)) for name in names}


def run_request(request_path, request_sha256, output):
    """Authenticated profile task; a bounded incomplete search publishes no labels."""
    from pathlib import Path
    import json
    import shutil
    import r02_structure_fit_partitions as learner
    from r02_structure_affinity_io import AuthenticatedFitAffinities
    io = learner.discrete
    path = Path(request_path)
    require(io.valid_hash(request_sha256) and path.is_file() and path.stat().st_size <= 2**20
            and io.sha(path) == request_sha256, 'Profile request identity changed')
    request = io.load_json(path)
    require(set(request) == {'schema','evidence_scope','inputs','implementation_sha256','parameters','resources'}
            and request['schema'] == 'r02_fit_profile_request_v2'
            and request['implementation_sha256'] == implementation_hashes(), 'Exact profile request/code required')
    limits = request['resources']
    require(set(limits) == {'max_input_bytes','max_metadata_bytes','max_memory_bytes','min_free_bytes','max_output_bytes'}
            and all(type(v) is int and v > 0 for v in limits.values()), 'Explicit profile limits required')
    require(set(request['parameters']) == {'seeds','max_iterations','max_queries'}, 'Explicit profile search required')
    source = AuthenticatedFitAffinities(request['inputs'], evidence_scope=request['evidence_scope'],
        max_input_bytes=limits['max_input_bytes'], max_metadata_bytes=limits['max_metadata_bytes'])
    graph = learner.FitRareGraphSpectrum(source.persons, source.fit_edges(),
        source_sha256=source.source_sha256, max_memory_bytes=limits['max_memory_bytes'])
    result = fit_leiden_profile(graph, source_sha256=source.source_sha256,
        max_working_bytes=limits['max_memory_bytes']-graph.memory_bound, **request['parameters'])
    source.verify()
    require(io.sha(path) == request_sha256 and implementation_hashes() == request['implementation_sha256'],
            'Profile source or code changed during search')
    out = Path(output)
    require(not out.exists() and out.parent.is_dir() and shutil.disk_usage(out.parent).free >= limits['min_free_bytes'],
            'New profile output and disk reserve required')
    out.mkdir(mode=0o700)
    encoded = json.dumps(result, indent=2, allow_nan=False)+'\n'
    require(len(encoded.encode()) <= limits['max_output_bytes'], 'Profile output exceeds byte bound')
    (out/'profile.json').write_text(encoded)
    admitted = result['status'] == 'NO_NEW_ENVELOPE_IMPROVEMENT_UNDER_DECLARED_ORACLE'
    if admitted:
        for key in result['predictive_candidate_keys']:
            learner.write_frozen_candidate(out/'candidates'/key, result['candidates'][key],
                persons=source.persons, persons_sha256=source.pins['persons.tsv'],
                fit_affinities_sha256=source.pins['fit_affinities.private.tsv'],
                exclusion_sha256=source.pins['exclusion.json'])
    written = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    require(written <= limits['max_output_bytes'], 'Profile plus candidate outputs exceed byte bound')
    manifest = dict(schema='r02_fit_profile_result_v2', status=result['status'],
        profile_complete=admitted, request_sha256=request_sha256, inputs=request['inputs'],
        implementation_sha256=request['implementation_sha256'], evidence_scope=request['evidence_scope'],
        graph_sha256=graph.graph_sha256, parameters=request['parameters'],
        candidate_policy=result['candidate_policy'],
        predictive_candidate_keys=result['predictive_candidate_keys'] if admitted else [],
        rb_envelope_candidate_keys=result['envelope_candidate_keys'],
        rb_dominated_candidate_keys=result['rb_dominated_candidate_keys'],
        outputs_sha256={str(p.relative_to(out)):io.sha(p) for p in out.rglob('*') if p.is_file()},
        biological_validation=False, global_optimum_proven=False, query_edges_used=False, evaluate_scored=False)
    (out/'profile_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for p in out.rglob('*'):
        p.chmod(0o700 if p.is_dir() else 0o600)
    return manifest


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run_request(args.request, args.request_sha256, args.output)
    print(json.dumps(result))
    if not result['profile_complete']:
        raise SystemExit(2)
