"""Finite, prospective Leiden candidate search; no seed-coverage claim.

Reuse authenticated earlier memberships without promoting an incomplete old
search. All declared seed calls remain required, including failures.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

import numpy as np
import r02_resolution_profile as profile
import r02_structure_fit_partitions as partitions

POLICY = 'HISTORICAL_SCALES_FINITE_RESTARTS_RETAIN_PRIOR_V1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()

def write_once(path, value, maximum):
    """Install complete bytes atomically, never replacing an existing result."""
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw)<=maximum, 'Immutable result exceeds bound')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:
            os.link(name,path)
        except FileExistsError:
            require(path.read_bytes()==raw, 'Immutable result differs: '+str(path))
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(name)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def atomic(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)

def validate_search(search):
    require(search['policy'] == POLICY, 'Unreviewed finite search policy')
    expected = {'R':[.5,1.,2.], 'C':[.5,1.,2.], 'R_PLUS_C':[.5,1.,2.],
                'H':[.5,.8,1.,1.2,1.5,2.,3.]}
    require(search['family'] in expected, 'Unknown family')
    queries = search['queries']
    require([float.fromhex(q['gamma_hex']) for q in queries] == expected[search['family']],
            'Resolution list differs from prospective finite specification')
    for q in queries:
        require(len(q['seeds']) == 25 and len(set(q['seeds'])) == 25
                and all(type(s) is int and 0 <= s < 2**31 for s in q['seeds']),
                'Exactly 25 frozen distinct integer seeds required')
        require(q['seeds']==queries[0]['seeds'], 'Paired seeds required across resolutions')
        reused=[r['seed'] for r in q['reuse']]
        require(len(reused)==len(set(reused)) and set(reused)<=set(q['seeds']), 'Invalid seed reuse list')
    require(type(search['max_iterations']) is int and search['max_iterations'] > 0, 'Iteration limit required')

def validate_request(request):
    validate_search(request['search'])
    snapshot=request['search']['import_snapshot']
    require(snapshot['snapshot_sha256']==digest({k:v for k,v in snapshot.items() if k!='snapshot_sha256'}),
            'Import snapshot identity differs')
    context=snapshot['context']
    require(context['task']==request['task'] and context['logical_target_ids']==request['logical_target_ids']
            and context['objective']=='RBConfigurationVertexPartition', 'Import context differs')
    admission=request['admission']
    require(sha(admission['prior_request'])==admission['prior_request_sha256'], 'Prior request changed')
    prior=json.loads(Path(admission['prior_request']).read_text())
    require(prior['task']==context['task'] and prior['code_sha256']==context['source_code_sha256']
            and prior['logical_target_ids']==context['logical_target_ids'], 'Prior procedure differs')
    require(sha(admission['scientific_rule'])==admission['scientific_rule_sha256'], 'Scientific amendment changed')
    for name in ('legacy_binding','legacy_budget'):
        item=snapshot.get(name)
        if item:
            require(sha(item['path'])==item['sha256'], 'Legacy checkpoint changed')

def search_graph(graph, request, output):
    # Import after module initialization; share the tested timeout calculation.
    from r02_k03_inner_profile import BudgetReached, seed_time_limit
    search, limits = request['search'], request['limits']
    validate_request(request)
    source = graph.source_sha256
    started = time.monotonic()
    budget_path = output/'BUDGET.json'
    previous = json.loads(budget_path.read_text()) if budget_path.exists() else {}
    require(not previous or (previous.get('schema')=='r02_finite_search_budget_v1'
            and previous.get('snapshot_sha256')==search['import_snapshot']['snapshot_sha256']),
            'Old-policy or incompatible budget cannot initialize finite search')
    interrupted_charge = previous.get('active_seed_reserved_seconds', 0.)
    spent_before = previous.get('new_policy_spent_seconds', 0.)+interrupted_charge
    counts = dict(required_seeds=sum(len(q['seeds']) for q in search['queries']),
                  completed_seeds=0, new_seeds=0, reused_seeds=0, prior_memberships=0)
    workspace = partitions.prepare_leiden_workspace(graph, source_sha256=source,
                         max_working_bytes=limits['max_working_bytes'])
    lines, origins = {}, {}
    imported = search['import_snapshot']
    historical_seconds = imported['historical_spent_seconds']

    def save_budget(reserve=0.):
        new_spent = spent_before+time.monotonic()-started
        atomic(budget_path, dict(schema='r02_finite_search_budget_v1',
            snapshot_sha256=imported['snapshot_sha256'], new_policy_spent_seconds=new_spent,
            historical_spent_seconds=historical_seconds,
            spent_seconds=historical_seconds+new_spent,
            active_seed_reserved_seconds=reserve, counters=counts,
            additional_policy_limit_seconds=limits['max_seconds'],
            interrupted_seed_charge_this_attempt=interrupted_charge,
            interruption_charge_is_measurement=False))

    def check_budget():
        save_budget()
        if time.monotonic()-started+spent_before >= limits['max_seconds']:
            raise BudgetReached('CUMULATIVE_NEW_POLICY_PROFILE_TIME_LIMIT')
        if time.time() >= request['campaign_deadline_epoch']:
            raise BudgetReached('CAMPAIGN_TIME_LIMIT')
        if shutil.disk_usage(output).free < limits['min_free_bytes']:
            raise BudgetReached('DISK_RESERVE')

    def keep(labels, origin):
        line = profile.make_line(graph, labels, prepared_workspace=workspace)
        target = output/'partitions'/(line.key+'.json')
        value = dict(labels=labels, rb_line=vars(line), graph_sha256=graph.graph_sha256)
        write_once(target, value, 2**20)
        lines[line.key] = line
        origins.setdefault(line.key, []).append(origin)
        return line, target

    def load_prior(item):
        path = Path(item['path'])
        require(sha(path) == item['sha256'], 'Prior membership bytes changed')
        record = json.loads(path.read_text())
        require(record['graph_sha256'] == graph.graph_sha256, 'Prior graph identity differs')
        witness=item['witness']
        require(sha(witness['seed_path'])==witness['seed_sha256']
                and sha(witness['episode_path'])==witness['episode_sha256'], 'Prior evidence changed')
        seed_record=json.loads(Path(witness['seed_path']).read_text())
        episode=json.loads(Path(witness['episode_path']).read_text())
        require(seed_record['status']=='FIT_ZERO_IMPROVEMENT_ITERATION_REACHED'
                and seed_record['source_sha256']==source
                and seed_record['partition_file_sha256']==item['sha256']
                and seed_record['membership_key']==item['membership_key']
                and seed_record['episode_sha256']==episode['episode_sha256']
                and seed_record['seed'] in episode['seeds'], 'Prior convergence provenance differs')
        line, target = keep(record['labels'], dict(kind='prior_snapshot', **item))
        require(line.key == item['membership_key'], 'Prior membership identity differs')
        return line, target

    try:
        for item in imported['partitions']:
            check_budget(); load_prior(item); counts['prior_memberships'] += 1
        # Analytic endpoints are retained as candidates, not claimed as a
        # biologically meaningful population partition.
        active = np.flatnonzero(graph.active_fit)
        for name, membership in (('connected_components',graph.connected_component),
                                 ('singletons',np.arange(len(graph.degree)))):
            labels = partitions.canonical_labels(membership[active], graph.fit_indices[active])
            labels = {str(int(i)): labels.get(int(i),'UNASSIGNED') for i in graph.fit_indices}
            keep(labels, dict(kind='analytic', name=name))
        for qi, query in enumerate(search['queries']):
            gamma = float.fromhex(query['gamma_hex'])
            reuse = {int(x['seed']): x for x in query['reuse']}
            for seed in query['seeds']:
                check_budget()
                record_path = output/'seeds'/f'q{qi:02d}_s{seed}.json'
                alias = dict(kind='finite_seed', gamma_hex=gamma.hex(), seed=seed)
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    candidate_path = output/'partitions'/(record['membership_key']+'.json')
                    require(record['source_sha256']==source and record['gamma_hex']==gamma.hex()
                            and record['seed']==seed and record['status']=='FIT_ZERO_IMPROVEMENT_ITERATION_REACHED'
                            and record['partition_file_sha256']==sha(candidate_path),
                            'Finished finite seed changed or incompatible')
                    saved = json.loads(candidate_path.read_text())
                    require(saved['graph_sha256']==graph.graph_sha256, 'Saved graph differs')
                    line, _ = keep(saved['labels'], alias)
                    require(line.key==record['membership_key'], 'Saved labels differ')
                    counts['reused_seeds'] += 1
                elif seed in reuse:
                    item = reuse[seed]
                    require(sha(item['path'])==item['sha256'], 'Prior seed changed')
                    record = json.loads(Path(item['path']).read_text())
                    require(record['source_sha256']==source and record['gamma_hex']==gamma.hex()
                            and record['seed']==seed and record['status']=='FIT_ZERO_IMPROVEMENT_ITERATION_REACHED',
                            'Prior seed incompatible')
                    line, target = load_prior(item['partition'])
                    require(line.key==record['membership_key'] and
                            record['partition_file_sha256']==item['partition']['sha256'],
                            'Prior seed and membership differ')
                    origins[line.key].append(alias)
                    record = dict(record, partition_file_sha256=sha(target), reused_from=item['path'])
                    write_once(record_path, record, 2**20)
                    counts['reused_seeds'] += 1
                else:
                    seconds, reason = seed_time_limit(limits['max_seed_seconds'],
                        limits['max_seconds']-spent_before-time.monotonic()+started,
                        request['campaign_deadline_epoch']-time.time())
                    save_budget(seconds)
                    began = time.monotonic()
                    def expired(signum, frame):
                        raise BudgetReached(reason)
                    previous_handler = signal.signal(signal.SIGALRM, expired)
                    try:
                        seconds, reason = seed_time_limit(limits['max_seed_seconds'],
                            limits['max_seconds']-spent_before-time.monotonic()+started,
                            request['campaign_deadline_epoch']-time.time())
                        signal.setitimer(signal.ITIMER_REAL,seconds)
                        candidate = partitions.learn_candidate(graph, source_sha256=source,
                            resolution=gamma, seed=seed, max_iterations=search['max_iterations'],
                            max_working_bytes=limits['max_working_bytes'], prepared_workspace=workspace)
                    finally:
                        signal.setitimer(signal.ITIMER_REAL,0)
                        signal.signal(signal.SIGALRM,previous_handler)
                    require(candidate['status']=='FIT_ZERO_IMPROVEMENT_ITERATION_REACHED',
                            'Incomplete optimization cannot count as finished')
                    line, target = keep(candidate['labels'],alias)
                    require(np.isclose(line.quality(gamma),candidate['normalized_objective'],
                                       rtol=1e-11,atol=1e-11), 'Objective differs from labels')
                    record = {k:v for k,v in candidate.items() if k not in ('labels','fit_indices','candidate_sha256')}
                    record.update(seed=seed,source_sha256=source,gamma_hex=gamma.hex(),
                        membership_key=line.key,partition_file_sha256=sha(target),
                        elapsed_seconds=time.monotonic()-began)
                    write_once(record_path,record,2**20)
                    counts['new_seeds'] += 1
                counts['completed_seeds'] += 1
                save_budget()
        require(counts['completed_seeds']==counts['required_seeds'], 'Finite list is incomplete')
        status = 'FINITE_DECLARED_FAMILY_COMPLETE_CONDITIONAL'
        reason = None
    except BudgetReached as error:
        status, reason = 'INCOMPLETE_RECOVERABLE', str(error)
    finally:
        save_budget()
    graph.verify(source)
    atomic(output/'CANDIDATES.json',dict(schema='r02_finite_fit_candidates_v1',
        status=status,complete=(status=='FINITE_DECLARED_FAMILY_COMPLETE_CONDITIONAL'),
        required_seeds=counts['required_seeds'],completed_seeds=counts['completed_seeds'],
        scientific_selection_enabled=False,
        graph_sha256=graph.graph_sha256, candidates=[dict(membership_key=k,
        partition_sha256=sha(output/'partitions'/(k+'.json')),origins=origins[k],
        candidate_class=('PRIMARY_FINITE' if any(o['kind']=='finite_seed' for o in origins[k])
        else 'ANALYTIC_CONTROL' if any(o['kind']=='analytic' for o in origins[k])
        else 'RETAINED_SENSITIVITY')) for k in sorted(lines)],
        historical_search_effort_equal=False, biological_selection=False))
    return dict(status=status,reason=reason,counters=counts,discovered_partitions=len(lines),
        graph_sha256=graph.graph_sha256,source_sha256=source,
        spent_seconds=historical_seconds+spent_before+time.monotonic()-started,
        seed_mass_coverage_certified=False,complete_calibration=False,scientific_selection=False,
        historical_receipts_superseded=False, boundary_evaluation_pending=True)
