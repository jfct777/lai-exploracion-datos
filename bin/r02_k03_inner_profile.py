#!/usr/bin/env python3
"""Checkpointed inner-FIT Leiden search using existing resolution/seed policies.

No responses, SELECT scores or EVALUATE scores are read. Finishing this search
does not finish predictive calibration. Every completed seed is reusable.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tarfile
import tempfile
import time

import numpy as np
import r02_k03_inner_projection as projection
import r02_resolution_profile as profile
import r02_structure_fit_partitions as partitions
import r02_structure_search_policy as policy
from r02_structure_profile_queue import write_once


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(path, root):
    path = Path(path)
    if str(path).startswith('/repo/'):
        return root / path.relative_to('/repo')
    return path if path.is_absolute() else root / path


def spectrum_directory(task, temporary, root):
    if task['spectrum_directory']:
        return resolve(task['spectrum_directory'], root)
    # The legacy first view is stored in one authenticated archive. Extract only
    # the exact manifest and its named payload; no traversal or links accepted.
    with tarfile.open(resolve(task['spectrum_archive'], root)) as archive:
        matches = []
        for member in archive:
            if member.isfile() and member.name.endswith('/manifest.json') and member.size <= 2**20:
                raw = archive.extractfile(member).read()
                if hashlib.sha256(raw).hexdigest() == task['spectrum_manifest_sha256']:
                    matches.append((member.name, raw))
        require(len(matches) == 1, 'Exactly one pinned spectrum manifest required')
        name, raw = matches[0]
        manifest = json.loads(raw)
        require(Path(manifest['file']).name == manifest['file'], 'Unsafe payload name')
        member = archive.getmember(str(Path(name).parent / manifest['file']))
        require(member.isfile() and member.size == manifest['bytes'], 'Spectrum member differs')
        target = Path(temporary)
        (target / 'manifest.json').write_bytes(raw)
        with archive.extractfile(member) as source, (target / manifest['file']).open('wb') as out:
            shutil.copyfileobj(source, out)
        return target


class BudgetReached(RuntimeError):
    pass


def seed_time_limit(max_seed_seconds, profile_remaining, campaign_remaining):
    """Return the earliest deadline and its cause; do not reset any budget."""
    values = (max_seed_seconds, profile_remaining, campaign_remaining)
    require(all(math.isfinite(value) for value in values) and max_seed_seconds > 0,
            'Finite seed/profile/campaign time budgets required')
    # On ties retain the enclosing limit, which cannot be retried as a seed.
    deadlines = ((campaign_remaining, 'CAMPAIGN_TIME_LIMIT'),
                 (profile_remaining, 'CUMULATIVE_PROFILE_TIME_LIMIT'),
                 (max_seed_seconds, 'SINGLE_SEED_TIME_LIMIT'))
    seconds, reason = min(deadlines, key=lambda item: item[0])
    if seconds <= 0:
        raise BudgetReached(reason)
    return seconds, reason


def execute(request, output, root):
    task = request['task']; limits = request['limits']; search = request['search']
    finite = search.get('policy') == 'HISTORICAL_SCALES_FINITE_RESTARTS_RETAIN_PRIOR_V1'
    require(request['schema'] == 'r02_inner_fit_profile_request_v1'
            and request['scientific_scope'] == 'INNER_FIT_COMMUNITY_DISCOVERY_ONLY',
            'Reviewed inner-FIT scope required')
    if finite:
        from r02_k03_finite_search import validate_request
        validate_request(request)
    else:
        require(search['views'] == 52140 and search['family_error'] == .05
                and search['missing_mass'] == .1, 'Reviewed legacy policy required')
    code_dir = Path(__file__).parent
    require(all(Path(name).name == name and name.endswith('.py') for name in request['code_sha256']),
            'Unsafe code file identity')
    require({p.name for p in code_dir.glob('*.py')} == set(request['code_sha256'])
        and sha(code_dir/'CODE_MANIFEST.json') == request['code_manifest_sha256']
        and json.loads((code_dir/'CODE_MANIFEST.json').read_text()) == request['code_sha256'],
        'Exact deployed code inventory required')
    for name, digest in request['code_sha256'].items():
        require(sha(Path(__file__).parent / name) == digest, 'Frozen code differs: ' + name)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable = {k:v for k,v in request.items() if k not in ('limits','campaign_deadline_epoch')}
        binding = dict(problem_sha256=projection.digest(immutable), scope=request['scientific_scope'])
        immutable_write = write_once
        if finite:
            from r02_k03_finite_search import write_once as immutable_write
        immutable_write(output / 'binding.json', binding, 2**20)
        immutable_write(output / 'attempt_requests' / (projection.digest(request) + '.json'), request, 2**20)
        persons_path = resolve(task['persons'], root)
        plan_path = resolve(task['fold_plan'], root)
        require(sha(persons_path) == task['persons_sha256']
                and sha(plan_path) == task['fold_plan_sha256'], 'Roster/fold identity differs')
        with persons_path.open() as stream:
            persons = list(csv.DictReader(stream, delimiter='\t'))
        with tempfile.TemporaryDirectory(prefix='spectrum-', dir=output) as temporary:
            directory = spectrum_directory(task, temporary, root)
            graph, plan, _ = projection.load_spectrum(artifact_directory=directory,
                manifest_sha256=task['spectrum_manifest_sha256'], persons=persons,
                fold_plan_bytes=plan_path.read_bytes(), fold_plan_sha256=task['fold_plan_sha256'],
                source_sha256=task['source_sha256'], max_input_bytes=limits['max_input_bytes'],
                max_memory_bytes=limits['max_memory_bytes'])
            require(task['fold_id'] == plan['fold_id'], 'Published fold identity differs from authenticated plan')
        if finite:
            from r02_k03_finite_search import search_graph as finite_search
            result = finite_search(graph, request, output)
        else:
            result = search_graph(graph, request, output)
        result.update(fold_id=task['fold_id'], physical_view=task['physical_view'],
            logical_target_ids=request['logical_target_ids'], inner_fit_people=len(plan['inner_fit_indices']),
            inner_project_people=len(plan['inner_project_indices']),
            scientific_calibration_complete=False, select_evaluate_scored=False,
            biological_validation=False)
        temporary = output / 'STATUS.json.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, output / 'STATUS.json')
        return result


def search_graph(graph, request, output):
    search, limits = request['search'], request['limits']
    source = graph.source_sha256; started = time.monotonic()
    budget_path = output / 'BUDGET.json'
    previous = json.loads(budget_path.read_text()) if budget_path.exists() else {'spent_seconds': 0.}
    # An interrupted seed has no measured end time. Charge its reserved ceiling
    # conservatively, rather than silently resetting its unobserved cost.
    interrupted_charge = previous.get('active_seed_reserved_seconds', 0.)
    spent_before = previous['spent_seconds'] + interrupted_charge
    counters = dict(queries=0, seeds=0, new_seeds=0, reused_seeds=0)
    a, d = graph.adjacency, graph.degree
    left, right = np.nonzero(np.triu(a, 1)); volume = float(d.sum())
    require(len(left) > 0 and volume > 0, 'Nonempty positive inner-FIT graph required')
    low = float(a[left, right].min() / volume)
    high = float(np.max(a[left, right] * volume / (d[left] * d[right])))
    active = np.flatnonzero(graph.active_fit)
    initial = []
    for membership in (graph.connected_component, np.arange(len(d))):
        labels = partitions.canonical_labels(membership[active], graph.fit_indices[active])
        labels = {str(int(i)): labels.get(int(i), 'UNASSIGNED') for i in graph.fit_indices}
        initial.append(profile.make_line(graph, labels))
    known = {line.key: line for line in initial}
    discovered = {}; assessments = []

    def save_budget(active_seed_seconds=0.):
        record = dict(spent_seconds=spent_before + time.monotonic() - started,
            max_seconds=limits['max_seconds'], counters=counters,
            active_seed_reserved_seconds=active_seed_seconds,
            interrupted_seed_charge_this_attempt=interrupted_charge,
            interruption_charge_is_measurement=False)
        temporary = output / 'BUDGET.json.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(temporary, budget_path)

    def check_budget():
        save_budget()
        if spent_before + time.monotonic() - started >= limits['max_seconds']:
            raise BudgetReached('CUMULATIVE_PROFILE_TIME_LIMIT')
        if time.time() >= request['campaign_deadline_epoch']:
            raise BudgetReached('CAMPAIGN_TIME_LIMIT')
        if shutil.disk_usage(output).free < limits['min_free_bytes']:
            raise BudgetReached('DISK_RESERVE')

    def oracle(gamma):
        counters['queries'] += 1; j = counters['queries']; prior = []; query_lines = {}
        for k in range(1, search['max_episodes_per_query'] + 1):
            ep = output / 'episodes' / f'q{j:04d}_e{k:04d}.json'
            if ep.exists():
                episode = json.loads(ep.read_text()); policy.verify_episode(episode)
                require(episode['source_sha256'] == source
                    and episode['view_id'] == request['id']
                    and episode['gamma_hex'] == gamma.hex()
                    and episode['known_memberships'] == sorted(known)
                    and episode['query_index'] == j and episode['episode_index'] == k
                    and episode['prior_seeds'] == sorted(r['seed'] for r in prior)
                    and all(episode[name] == search[name] for name in ('family_error','views','missing_mass')),
                    'Frozen episode history changed')
            else:
                check_budget()
                episode = policy.freeze_episode(source_sha256=source, view_id=request['id'], gamma=gamma,
                    known_memberships=list(known), prior_results=prior, query_index=j, episode_index=k,
                    family_error=search['family_error'], views=search['views'], missing_mass=search['missing_mass'])
                write_once(ep, episode, 2**20)
            if counters['seeds'] + len(episode['seeds']) > search['max_total_seeds']:
                raise BudgetReached('SEED_LIMIT_INCOMPLETE_EPISODE')
            results = []
            for seed in episode['seeds']:
                check_budget(); counters['seeds'] += 1
                record_path = output / 'seeds' / f'q{j:04d}_e{k:04d}_s{seed}.json'
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    require(record['episode_sha256'] == episode['episode_sha256'], 'Seed belongs to other episode')
                    candidate_path = output / 'partitions' / (record['membership_key'] + '.json')
                    require(sha(candidate_path) == record['partition_file_sha256'], 'Saved membership differs')
                    labels = json.loads(candidate_path.read_text())['labels']
                    line = profile.make_line(graph, labels)
                    require(line.key == record['membership_key'] and record['seed'] == seed
                        and record['source_sha256'] == source and record['gamma_hex'] == gamma.hex()
                        and record['status'] == 'FIT_ZERO_IMPROVEMENT_ITERATION_REACHED', 'Seed identity/status differs')
                    counters['reused_seeds'] += 1
                else:
                    seconds, timeout_reason = seed_time_limit(
                        limits['max_seed_seconds'],
                        limits['max_seconds'] - spent_before - time.monotonic() + started,
                        request['campaign_deadline_epoch'] - time.time())
                    save_budget(active_seed_seconds=seconds)
                    began = time.monotonic()
                    def expired(signum, frame):
                        raise BudgetReached(timeout_reason)
                    previous_handler = signal.signal(signal.SIGALRM, expired)
                    try:
                        # Checkpoint I/O consumes the enclosing budgets. Recheck
                        # immediately before arming, so it cannot change a
                        # profile timeout into a misleading seed timeout.
                        seconds, timeout_reason = seed_time_limit(
                            limits['max_seed_seconds'],
                            limits['max_seconds'] - spent_before - time.monotonic() + started,
                            request['campaign_deadline_epoch'] - time.time())
                        signal.setitimer(signal.ITIMER_REAL, seconds)
                        candidate = partitions.learn_candidate(graph, source_sha256=source, resolution=gamma,
                            seed=seed, max_iterations=search['max_iterations'],
                            max_working_bytes=limits['max_working_bytes'])
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(signal.SIGALRM, previous_handler)
                    require(candidate['status'] == 'FIT_ZERO_IMPROVEMENT_ITERATION_REACHED',
                            'Unfinished seed cannot close the episode')
                    line = profile.make_line(graph, candidate['labels'])
                    require(np.isclose(line.quality(gamma), candidate['normalized_objective'], rtol=1e-11, atol=1e-11),
                            'Leiden objective differs from membership')
                    candidate_path = output / 'partitions' / (line.key + '.json')
                    if candidate_path.exists():
                        saved = json.loads(candidate_path.read_text())
                        require(profile.make_line(graph, saved['labels']).key == line.key, 'Membership cache conflict')
                    else:
                        write_once(candidate_path, dict(labels=candidate['labels'], rb_line=vars(line),
                            graph_sha256=graph.graph_sha256), 2**20)
                    record = {k:v for k,v in candidate.items() if k not in ('labels','fit_indices','candidate_sha256')}
                    record.update(seed=seed, source_sha256=source, gamma_hex=gamma.hex(), membership_key=line.key,
                        episode_sha256=episode['episode_sha256'], partition_file_sha256=sha(candidate_path),
                        elapsed_seconds=time.monotonic()-began)
                    write_once(record_path, record, 2**20)
                    counters['new_seeds'] += 1
                results.append({name: record[name] for name in ('seed','source_sha256','gamma_hex','membership_key','status')})
                query_lines[line.key] = line; discovered[line.key] = line
                save_budget()
            assessment = policy.assess_episode(episode, results)
            assessments.append(dict(query=j, episode=k, **assessment))
            known.update(query_lines); prior.extend(results)
            if assessment['status'] == 'DECLARED_SEED_MASS_COVERED':
                return list(query_lines.values())
        raise BudgetReached('EPISODE_LIMIT')

    try:
        result = profile.bounded_profile(initial, oracle, gamma_low=low, gamma_high=high,
            max_queries=search['max_queries'])
        missing = sorted(set(result['envelope_candidate_keys']) - set(discovered))
        complete = result['status'] == 'NO_NEW_ENVELOPE_IMPROVEMENT_UNDER_DECLARED_ORACLE' and not missing
        result.update(status='INNER_FIT_PROFILE_COMPLETE' if complete else 'INCOMPLETE_PROFILE',
            missing_analytic_memberships=missing, assessments=assessments)
    except BudgetReached as error:
        result = dict(status='INCOMPLETE_RECOVERABLE', reason=str(error))
    finally:
        save_budget()
    graph.verify(source)
    result.update(counters=counters, discovered_partitions=len(discovered),
        graph_sha256=graph.graph_sha256, source_sha256=source,
        spent_seconds=spent_before+time.monotonic()-started,
        complete_calibration=False, scientific_selection=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.request) == args.request_sha256, 'Request identity differs')
    print(json.dumps(execute(json.loads(args.request.read_text()), args.output, args.root), indent=2))
