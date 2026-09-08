#!/usr/bin/env python3
"""Timing-only, stdlib forecast for the fixed-geometry 12+20 multichannel pilot.

This reads authenticated timing/resource/provenance fields, never accesses scores,
labels, predictions or checkpoints, and cannot launch jobs. Anchor choice is made
before scientific outcomes exist. The estimate is not a bill or a speed guarantee.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import struct

from m39_gpu_serial_profile import sha256
from m39_ordered_gpu_manifest import load_plan, require, STAGE_ARMS


FAMILIES = ('cnn', 'attention')
ARMS = STAGE_ARMS['multichannel_followup']


def number(value, name: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f'invalid {name}')
    return float(value)


def read_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f'not a regular input: {path}')
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f'expected JSON object: {path}')
    return value


def file_receipt(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def scalar_radius(store: Path, expected_manifest: str) -> dict:
    """Authenticate the scalar radius only; no genomic or outcome arrays are opened."""
    path = store / 'manifest.json'
    require(sha256(path) == expected_manifest, 'radius store manifest hash differs')
    manifest = read_json(path)
    spec = manifest['arrays']['radius_cm']
    require(spec['file'] == 'radius_cm.npy' and spec['dtype'] == '<f8'
            and spec['shape'] == [1] and spec['nbytes'] == 8, 'radius array contract differs')
    member = store / spec['file']
    require(member.is_file() and not member.is_symlink()
            and sha256(member) == spec['sha256'], 'radius scalar hash differs')
    with member.open('rb') as stream:
        require(stream.read(6) == b'\x93NUMPY', 'radius scalar is not NPY')
        version = stream.read(2)
        require(version in (b'\x01\x00', b'\x02\x00'), 'unsupported radius NPY version')
        length_bytes = 2 if version == b'\x01\x00' else 4
        length = int.from_bytes(stream.read(length_bytes), 'little')
        require(0 < length <= 4096, 'invalid radius NPY header length')
        header = ast.literal_eval(stream.read(length).decode('latin1'))
        require(header == {'descr':'<f8', 'fortran_order':False, 'shape':(1,)},
                'radius NPY header differs')
        raw = stream.read(9)
        require(len(raw) == 8, 'radius NPY payload differs')
    value = number(struct.unpack('<d', raw)[0], 'radius', positive=True)
    return {'radius_cm':value, 'manifest':file_receipt(path), 'scalar':file_receipt(member)}


def load_timing_profile(plan_path: Path, outputs: Path, launch_path: Path,
                        completion_path: Path, seal_path: Path) -> dict:
    """Authenticate the full family×arm inventory but project only timing fields."""
    plan = load_plan(plan_path)
    require(plan['stage'] == 'multichannel_technical', 'timing requires multichannel_technical')
    launch, completion, seal = map(read_json, (launch_path, completion_path, seal_path))
    plan_sha, seal_sha = sha256(plan_path), sha256(seal_path)
    require(launch['schema_version'] == 'm39-gpu-launch-v1'
            and launch['stage'] == plan['stage'] and launch['plan_sha256'] == plan_sha
            and launch['source_seal_sha256'] == seal_sha
            and launch['input_sha256'] == plan['inputs'], 'profile launch binding differs')
    require(completion['schema_version'] == 'm39-gpu-controller-completion-v1'
            and completion['status'] == 'NEXTFLOW_COMPLETED_NEEDS_PRIMARY_POST'
            and completion['exit_code'] == 0 and completion['run_id'] == launch['run_id']
            and completion['launch_sha256'] == sha256(launch_path), 'profile controller not complete/bound')
    require(seal['profile_sha256'] == plan_sha and seal['source_commit'] == launch['source_commit'],
            'profile source seal differs')
    require(launch['group_ids'] == [group['id'] for group in plan['groups']], 'profile group IDs differ')
    controller_seconds = number(completion['elapsed_seconds'], 'profile controller time', positive=True)
    workers = launch['resources']['max_concurrent_workers']
    require(type(workers) is int and workers == plan['resources']['max_workers'], 'profile worker count differs')
    radii = {role:scalar_radius(Path(launch['inputs'][f'{role}_store']),
                               plan['inputs'][f'{role}_manifest_sha256']) for role in ('train','select')}
    require(all(record['radius_cm'] == .2 for record in radii.values()),
            'forecast requires the same authenticated 0.2 cM radius in both roles')
    cases, groups, geometry = [], [], None
    for group in plan['groups']:
        group_path = outputs / f"training-{group['id']}"
        done_path = group_path / 'group.completion.json'
        done = read_json(done_path)
        require(done['status'] == 'COMPLETED_DECLARED_PAIRED_ARMS_NEEDS_SCIENTIFIC_POST'
                and done['exit_code'] == 0 and done['stage'] == plan['stage']
                and done['group_id'] == group['id'] and done['plan_sha256'] == plan_sha
                and done['source_seal_sha256'] == seal_sha
                and done['source_commit'] == seal['source_commit']
                and done['SCORE_opened'] is False and done['completed_arms'] == list(ARMS)
                and set(done['case_receipt_sha256']) == set(ARMS), 'profile group incomplete or unbound')
        group_cases = []
        for spec in group['configs']:
            config_path = plan_path.parent / spec['file']
            cfg = read_json(config_path)
            receipt_path = group_path / cfg['arm'] / 'training.receipt.json'
            require(sha256(receipt_path) == done['case_receipt_sha256'][cfg['arm']],
                    'profile case receipt hash differs')
            receipt = read_json(receipt_path)
            require(receipt['schema_version'] == 'm39-ordered-multichannel-training-v1'
                    and receipt['decision'] == 'COMPLETED_EXPLORATORY_DEVELOPMENT_CASE'
                    and receipt['config'] == cfg and receipt['sources']['config_sha256'] == spec['sha256']
                    and receipt['scope']['SCORE_opened'] is False, 'profile case binding differs')
            require(all(receipt['sources'][name] == value for name, value in plan['inputs'].items())
                    and bool(receipt['sources']['code_sha256'])
                    and all(seal['source_sha256'].get(name) == value
                            for name, value in receipt['sources']['code_sha256'].items()),
                    'profile case source/input seal differs')
            # No selected_step, SELECT/metrics, probabilities, truth or checkpoints
            # are accessed. Curve steps establish how many timing calls occurred.
            steps = cfg['steps']
            curve_steps = [row['step'] for row in receipt['curve']]
            expected_steps = ([0] if cfg.get('evaluate_initial', False) else []) + list(
                range(cfg['evaluate_every_steps'], steps + 1, cfg['evaluate_every_steps']))
            if expected_steps[-1] != steps:
                expected_steps.append(steps)
            require(curve_steps == expected_steps, 'profile evaluation timing count differs')
            training = number(receipt['training_seconds'], 'profile training time', positive=True)
            evaluation = number(receipt['evaluation_seconds'], 'profile evaluation time', positive=True)
            selected = number(receipt['SELECT_seconds'], 'profile SELECT time')
            probe = number(receipt['TRAIN_probe_seconds'], 'profile TRAIN probe time')
            elapsed = number(receipt['elapsed_seconds'], 'profile case elapsed', positive=True)
            require(math.isclose(selected + probe, evaluation, rel_tol=0, abs_tol=1e-6),
                    'profile evaluation components differ')
            require(elapsed + 1e-6 >= training + evaluation, 'profile timing components exceed elapsed')
            available_pairs = receipt['TRAIN_exposure']['available_pairs']
            anchors, available_anchors = cfg['anchor_count'], receipt['available_anchors']
            require(type(available_pairs) is int and available_pairs > 0 and available_pairs % anchors == 0,
                    'profile training population cannot be derived')
            require(type(available_anchors) is int and available_anchors >= anchors,
                    'invalid available anchor count')
            current_geometry = {'train_people':available_pairs // anchors, 'available_anchors':available_anchors,
                'profile_anchors':anchors, 'profile_steps':steps, 'batch_size':cfg['batch_size'],
                'train_probe_people':cfg.get('train_probe_people',0),
                'evaluate_initial':cfg.get('evaluate_initial',False),
                'model':{key:value for key,value in cfg['model'].items() if key != 'family'},
                'multichannel':cfg['multichannel'], 'cpu_threads':cfg['cpu_threads'], 'radius_cm':.2}
            require(geometry is None or geometry == current_geometry, 'profile geometry differs between cases')
            geometry = current_geometry
            case = {'group_id':group['id'], 'case_id':cfg['case_id'], 'family':cfg['model']['family'],
                'arm':cfg['arm'], 'receipt':file_receipt(receipt_path),
                'elapsed_seconds':elapsed, 'training_seconds':training, 'evaluation_seconds':evaluation,
                'SELECT_seconds':selected, 'TRAIN_probe_seconds':probe, 'evaluation_count':len(curve_steps),
                'setup_seconds':max(0., elapsed-training-evaluation), 'training_seconds_per_step':training/steps,
                'evaluation_seconds_per_call_per_anchor':evaluation/len(curve_steps)/anchors,
                'rss_peak_bytes':receipt['rss_peak_bytes'], 'gpu_memory':receipt['gpu_memory']}
            require(case['family'] in FAMILIES, 'profile has unknown family')
            group_cases.append(case)
        require(len({case['family'] for case in group_cases}) == 1, 'profile group mixes families')
        group_elapsed = number(done['elapsed_seconds'], 'profile group elapsed', positive=True)
        case_elapsed = sum(case['elapsed_seconds'] for case in group_cases)
        require(group_elapsed + 1e-6 >= case_elapsed, 'profile group elapsed is shorter than case timers')
        require(controller_seconds + 1e-6 >= group_elapsed,
                'profile controller elapsed is shorter than a worker group')
        groups.append({'group_id':group['id'], 'family':group_cases[0]['family'],
            'receipt':file_receipt(done_path), 'elapsed_seconds':group_elapsed,
            'case_seconds_sum':case_elapsed, 'group_overhead_seconds':max(0.,group_elapsed-case_elapsed)})
        cases.extend(group_cases)
    require({(case['family'],case['arm']) for case in cases} ==
            {(family,arm) for family in FAMILIES for arm in ARMS}, 'profile must cover both families and all five arms')
    return {'run_id':launch['run_id'], 'created_utc':launch['created_utc'],
            'source_commit':seal['source_commit'], 'controller_elapsed_seconds':controller_seconds,
            'max_workers':workers, 'geometry':geometry, 'radius_receipts':radii,
            'plan':file_receipt(plan_path), 'launch':file_receipt(launch_path),
            'completion':file_receipt(completion_path), 'source_seal':file_receipt(seal_path),
            'groups':groups, 'cases':cases}


@dataclass(frozen=True)
class BudgetPolicy:
    budget_usd: float
    wall_seconds: float
    worker_hourly_usd: float
    disk_hourly_usd: float
    safety_factor: float
    provision_seconds_per_worker: float
    reserve_usd: float
    profile_reserve_usd: float
    fixed_overhead_seconds: float

    def validate(self):
        for name, value in asdict(self).items():
            number(value, name, positive=name in ('budget_usd','wall_seconds','worker_hourly_usd','safety_factor'))
        require(self.safety_factor >= 1, 'safety factor must be at least one')


def timing_statistics(profile: dict) -> dict:
    result = {}
    for family in FAMILIES:
        result[family] = {}
        for arm in ARMS:
            cases = [case for case in profile['cases'] if (case['family'],case['arm']) == (family,arm)]
            require(bool(cases), f'missing profile timing for {family}/{arm}')
            result[family][arm] = {'measurements':len(cases), **{name:{
                'mean':statistics.mean(case[name] for case in cases), 'max':max(case[name] for case in cases)}
                for name in ('setup_seconds','training_seconds_per_step','evaluation_seconds_per_call_per_anchor')}}
    return result


def forecast(profile: dict, policy: BudgetPolicy, anchor_counts=(660,512,256,128), *,
             scientific_evaluate_initial: bool | None = None) -> dict:
    policy.validate()
    require(len(anchor_counts) > 0 and len(set(anchor_counts)) == len(anchor_counts)
            and all(type(count) is int and 0 < count <= profile['geometry']['available_anchors']
                    for count in anchor_counts), 'candidate anchors must be distinct and inside available inventory')
    stats = timing_statistics(profile)
    workers = profile['max_workers']
    hourly = policy.worker_hourly_usd + policy.disk_hourly_usd
    profile_upper = profile['controller_elapsed_seconds'] * workers / 3600 * hourly
    charged_profile = max(profile_upper, policy.profile_reserve_usd)
    geometry = profile['geometry']
    require(scientific_evaluate_initial is None or type(scientific_evaluate_initial) is bool,
            'invalid scientific initial-evaluation declaration')
    # Keep the profile's initial evaluation as a conservative allowance when
    # science omits it. An explicit extra scientific initial call is never omitted.
    charged_initial = geometry['evaluate_initial'] or scientific_evaluate_initial is True
    candidates = []
    for anchors in sorted(anchor_counts, reverse=True):
        stages = []
        for name, arms, replicas, passes in (('A',('none','both'),3,2), ('B',ARMS,2,4)):
            steps = math.ceil(geometry['train_people']/geometry['batch_size']) * anchors * passes
            evaluations = passes + int(charged_initial)
            fits, group_seconds, task_seconds = [], [], []
            for family in FAMILIES:
                family_fit_seconds = 0.
                for arm in arms:
                    timing = stats[family][arm]
                    estimates = {}
                    for statistic in ('mean','max'):
                        components = {'setup_seconds':timing['setup_seconds'][statistic],
                            'training_seconds':timing['training_seconds_per_step'][statistic]*steps,
                            'evaluation_seconds':timing['evaluation_seconds_per_call_per_anchor'][statistic]*evaluations*anchors}
                        estimates[statistic] = {**components, 'raw_fit_seconds':sum(components.values()),
                            'guarded_fit_seconds':sum(components.values())*policy.safety_factor}
                    family_fit_seconds += estimates['max']['guarded_fit_seconds']
                    fits.append({'family':family,'arm':arm,'replicas':replicas,'steps':steps,
                                 'evaluation_count':evaluations,'estimate':estimates})
                overhead = max(group['group_overhead_seconds'] for group in profile['groups']
                               if group['family'] == family)
                task_duration = family_fit_seconds + overhead*policy.safety_factor
                group_duration = task_duration + policy.provision_seconds_per_worker
                group_seconds.extend([group_duration]*replicas)
                task_seconds.extend([task_duration]*replicas)
            total = sum(group_seconds)
            # List-scheduling upper bound if ready jobs keep <=workers busy.
            # Provisioning per group and a separate fixed audit reserve are added.
            wall = total/workers + (1-1/workers)*max(group_seconds)
            stages.append({'stage':name,'groups':len(group_seconds),'fits':sum(fit['replicas'] for fit in fits),
                'passes':passes,'fit_estimates':fits,'group_worker_seconds':group_seconds,
                'group_task_seconds':task_seconds,
                'worker_seconds':total,'wall_seconds_upper_estimate':wall,
                'scheduling_bound':{'formula':'sum_group_seconds/workers + (1-1/workers)*max_group_seconds',
                    'workers':workers,'sum_group_seconds':total,'maximum_group_seconds':max(group_seconds),
                    'assumption':'work_conserving_async_mixed_groups_with_each_provision_inside_supplied_estimate'},
                'fits_native_task_ceiling':max(task_seconds) < 8*3600})
        worker_seconds = sum(stage['worker_seconds'] for stage in stages)
        science_wall = sum(stage['wall_seconds_upper_estimate'] for stage in stages) + policy.fixed_overhead_seconds
        total_wall = profile['controller_elapsed_seconds'] + science_wall
        science_compute = worker_seconds/3600*policy.worker_hourly_usd
        science_disks = worker_seconds/3600*policy.disk_hourly_usd
        total_cost = charged_profile + science_compute + science_disks + policy.reserve_usd
        native_feasible = all(stage['fits_native_task_ceiling'] for stage in stages)
        candidates.append({'anchor_count_A':anchors,'anchor_count_B':anchors,'radius_cm_A':.2,'radius_cm_B':.2,
            'neural_fits':32,'stages':stages,'science_worker_hours':worker_seconds/3600,
            'science_compute_usd':science_compute,'science_disks_usd':science_disks,
            'profile_charged_usd':charged_profile,'extra_reserve_usd':policy.reserve_usd,
            'total_estimated_usd':total_cost,'science_wall_seconds':science_wall,
            'profile_plus_science_wall_seconds':total_wall,
            'fits_cost_ceiling':total_cost <= policy.budget_usd,'fits_time_ceiling':total_wall <= policy.wall_seconds,
            'fits_native_task_ceiling':native_feasible,
            'feasible':total_cost <= policy.budget_usd and total_wall <= policy.wall_seconds and native_feasible})
    eligible = [candidate for candidate in candidates if candidate['feasible']]
    return {'schema_version':'m39-multichannel-profile-budget-v1',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'policy':asdict(policy),
        'profile':profile,'timing_statistics':stats,'candidates':candidates,
        'evaluation_schedule':{'profile_evaluate_initial':geometry['evaluate_initial'],
            'declared_scientific_evaluate_initial':scientific_evaluate_initial,
            'initial_evaluations_charged_per_fit':int(charged_initial),
            'conservative_extra_evaluations_per_fit':
                int(charged_initial and scientific_evaluate_initial is False),
            'policy':'retain_profile_initial_call_as_conservative_allowance_when_science_omits_it'},
        'selected_anchor_count':eligible[0]['anchor_count_A'] if eligible else None,
        'selection_rule':'largest_same_A_B_anchor_count_fitting_cost_total_wall_and_native_task_ceiling_before_efficacy',
        'status':'BOUNDED_CANDIDATE_FOR_RESPONSIBLE_FREEZE' if eligible else 'NO_CANDIDATE_FITS',
        'profile_compute_plus_disk_upper_estimate_usd':profile_upper,
        'actual_invoice':False,'efficacy_fields_used':False,'training_launched':False,
        'limitations':['Short-profile first CUDA setup/compilation can inflate per-step extrapolation.',
            'Mean equals max with one timing measurement per family/arm; no repeated-run precision is claimed.',
            'Evaluation scales with anchor count, with identical SELECT and TRAIN-probe sizes and checkpoint policy.',
            'Setup is held fixed per fit; short profile does not measure convergence or scientific effect.',
            'Feasibility uses componentwise maximum rates times supplied safety factor; no LR speedup is assumed.',
            'Wall bound assumes work-conserving scheduling; supplied provisioning and fixed overhead are estimates.',
            'The current plan validator permits at most eight hours per native worker task; provisioning is outside that timer.',
            'Rates are caller-supplied estimates, not current billing; VMdev/network/storage/tax must fit the extra reserve or remain excluded.',
            'Dense inference, FLARE2 and any extension beyond these 32 fits are excluded.']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','outputs','launch','completion','source-seal','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in BudgetPolicy.__dataclass_fields__:
        parser.add_argument('--'+name.replace('_','-'),type=float,required=True)
    parser.add_argument('--anchor-counts',type=int,nargs='+',default=[660,512,256,128])
    parser.add_argument('--scientific-evaluate-initial',type=int,choices=(0,1),
                        help='Record scientific policy; profile initial-call cost is retained conservatively')
    args = parser.parse_args()
    require(not args.out.exists() and not args.out.is_symlink(), 'budget output already exists')
    profile = load_timing_profile(args.plan,args.outputs,args.launch,args.completion,args.source_seal)
    policy = BudgetPolicy(**{name:getattr(args,name) for name in BudgetPolicy.__dataclass_fields__})
    result = forecast(profile,policy,args.anchor_counts,
                      scientific_evaluate_initial=None if args.scientific_evaluate_initial is None
                      else bool(args.scientific_evaluate_initial))
    result['generator'] = file_receipt(Path(__file__))
    with args.out.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':result['status'],'selected_anchor_count':result['selected_anchor_count'],
                      'out':str(args.out),'sha256':sha256(args.out)},sort_keys=True))


if __name__ == '__main__':
    main()
