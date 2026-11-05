"""Bounded FIT effective-rate calibration, with saved per-start results.

No biological generalization, IID uncertainty or neural training is licensed.
Interrupted starts restart from the best saved point, not L-BFGS history;
evaluation budgets are cumulative across invocations and invalid probes count.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
import torch
from r02_lai_d1_lumped import (OFFDIAGONALS, CLASSES, composite_nll,
    transition_probabilities, exact_delta_tensor, combine_counts)
from r02_lai_duration_fit_summary import _canonical, _write_fresh_atomic


def fit(count_data, initial, *, max_evaluations, seconds, output, identity,
        gradient_tolerance=1e-7):
    """Optimize NLL/changed-endpoint count; preserve best evaluated finite state."""
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    files = sorted(output.glob('evaluation-*.json'))
    saved = [json.loads(p.read_text()) for p in files]
    if any(s['identity'] != identity for s in saved):
        raise ValueError('incompatible calibration checkpoint')
    valid = [s for s in saved if s['finite']]
    best = min(valid, key=lambda s:s['objective']) if valid else None
    deltas = exact_delta_tensor(count_data)
    counts = torch.tensor(count_data['counts'],dtype=torch.int64)
    scale = int(count_data['changes'])
    if scale <= 0: raise ValueError('no changes: no finite positive-rate calibration')
    remaining = max(0.,seconds-sum(s['elapsed_seconds'] for s in saved))
    started, last, calls = time.monotonic(), time.monotonic(), len(saved)
    cache={np.asarray(s['log_rates'],dtype=np.float64).tobytes():s for s in saved}
    def fun(x):
        nonlocal last, calls, best
        key=np.asarray(x,dtype=np.float64).tobytes()
        if key in cache:
            previous=cache[key]
            if not previous['finite']: raise ValueError('repeat invalid probe; stop without recharging budget')
            return previous['objective'],np.asarray(previous['gradient'])
        if calls >= max_evaluations or time.monotonic()-started >= remaining:
            raise TimeoutError('cumulative calibration budget exhausted')
        record = dict(identity=identity, log_rates=x.tolist(), finite=False)
        try:
            rates = torch.tensor(x,dtype=torch.float64,requires_grad=True)
            value = composite_nll(rates,deltas,counts)/scale
            grad = torch.autograd.grad(value,rates)[0].detach().numpy()
            if not np.isfinite(float(value.detach())) or not np.isfinite(grad).all():
                raise ValueError('nonfinite objective/gradient')
            record.update(finite=True,objective=float(value.detach()),gradient=grad.tolist())
        except ValueError as error:
            record['error'] = str(error)
        record['elapsed_seconds'] = time.monotonic()-last
        _write_fresh_atomic(output/f'evaluation-{calls:04d}.json',_canonical(record))
        cache[key]=record
        calls += 1; last = time.monotonic()
        if not record['finite']: raise ValueError(record['error'])
        if best is None or record['objective'] < best['objective']: best = record
        return record['objective'],np.asarray(record['gradient'])
    initial = np.asarray(initial,dtype=np.float64)
    x0 = np.asarray(best['log_rates']) if best else initial
    solver = None
    try:
        solver = minimize(fun,x0,jac=True,method='L-BFGS-B',
            options=dict(maxiter=max_evaluations,maxfun=max_evaluations,
                         gtol=gradient_tolerance,ftol=1e-13,maxls=20))
        stop = str(solver.message)
    except (ValueError,TimeoutError) as error:
        stop = str(error)
    result = dict(identity=identity,best=best,evaluations=calls,stop=stop,
        elapsed_seconds=sum(s['elapsed_seconds'] for s in saved)+time.monotonic()-started,
        elapsed_seconds_this_invocation=time.monotonic()-started,
        invalid_evaluations=sum(not s['finite'] for s in cache.values()),
        budget_exhausted=bool(calls>=max_evaluations or time.monotonic()-started>=remaining),
        solver_success=bool(solver is not None and solver.success),
        converged=bool(best is not None and np.max(np.abs(best['gradient']))<=gradient_tolerance),
        gradient_tolerance=gradient_tolerance,objective_scale_changed_endpoints=scale,
        resumed_from_best_point=bool(saved),lbfgs_history_restored=False)
    return result


def run(request_path, counts_path, output_dir):
    request=json.loads(Path(request_path).read_text())
    if request['scope']!='FIT_EFFECTIVE_D1_RATE_CALIBRATION_NOT_NEURAL_TRAINING':
        raise ValueError('calibration scope differs')
    data=json.loads(Path(counts_path).read_text())
    if data['status']!='VERIFIED_COUNTS_NOT_BIOLOGICAL_IMPROVEMENT':
        raise ValueError('counts not admitted')
    if len(data['data']['delta_morgan_bits'])>request['max_distance_groups']:
        raise ValueError('distance workspace limit exceeded; preserve counts, do not round/drop')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    identity=dict(request_sha256=sha(request_path),counts_sha256=sha(counts_path),
                  code_sha256=sha(__file__),math_sha256=sha(Path(__file__).with_name('r02_lai_d1_lumped.py')))
    results=[]
    for i,start in enumerate(request['initial_log_rates']):
        path=out/f'start-{i}.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['identity'] != identity: raise ValueError('saved start differs')
        else:
            result=fit(data['data'],start,max_evaluations=request['max_evaluations_per_start'],
                       seconds=request['seconds_per_start'],output=out/f'start-{i}',identity=identity,
                       gradient_tolerance=request['gradient_tolerance'])
            _write_fresh_atomic(path,_canonical(result))
        results.append(result)
    valid=[r for r in results if r['converged']]
    best=min(valid,key=lambda r:r['best']['objective']) if valid else None
    summary=dict(status='INCOMPLETE',identity=identity,starts=results,
        neural_training_performed=False,demographic_rate_claim=False,
        uncertainty_estimated=False,independent_units_claimed=False)
    if best:
        x=torch.tensor(best['best']['log_rates'],dtype=torch.float64,requires_grad=True)
        counts=torch.tensor(data['data']['counts'],dtype=torch.int64)
        ds=exact_delta_tensor(data['data'])
        eig=None;positive=False
        try:
            h=torch.autograd.functional.hessian(lambda v:composite_nll(v,ds,counts),x).detach().numpy()
            eig=np.linalg.eigvalsh(h)
            positive=bool(eig[0]>0 and eig[-1]/eig[0]<request['max_hessian_condition'])
        except (RuntimeError,MemoryError,np.linalg.LinAlgError) as error:
            summary['hessian_unavailable']=str(error)
        objectives=[r['best']['objective']*r['objective_scale_changed_endpoints'] for r in valid]
        agrees=len(valid)==len(results) and max(objectives)-min(objectives)<=request['max_nll_difference']
        with torch.no_grad(): p=transition_probabilities(x,ds).numpy()
        observed=counts.numpy();expected=observed.sum(2)[:,:,None]*p
        summary.update(status='CONVERGED_EFFECTIVE_FIT_RATES' if agrees and positive else 'INCOMPLETE',
            rates_per_morgan=x.exp().detach().tolist(),offdiagonal_order=OFFDIAGONALS,
            nll=float(composite_nll(x,ds,counts).detach()),
            hessian_eigenvalues=None if eig is None else eig.tolist(),hessian_is_not_biological_uncertainty=True,
            multistart_max_nll_difference=max(objectives)-min(objectives),
            observed_endpoint_counts=observed.sum(0).tolist(),
            expected_endpoint_counts=expected.sum(0).tolist(),
            residuals_by_delta=(observed-expected).tolist(),
            delta_morgan_bits=data['data']['delta_morgan_bits'],class_order=CLASSES,
            scope='effective homogeneous D1 rates for these shared-donor FIT trajectories only')
        case_data=[]
        for case in data['cases']:
            if sha(case['path'])!=case['sha256']:raise ValueError('case count product changed')
            case_data.append(json.loads(Path(case['path']).read_text())['counts'])
        sensitivity=[]
        for i,held in enumerate(case_data):
            retained=combine_counts([c for j,c in enumerate(case_data) if j!=i])
            result=fit(retained,x.detach().numpy(),max_evaluations=request['max_evaluations_per_start'],
                seconds=request['seconds_per_start'],output=out/f'omitted-case-{i}',
                identity=dict(identity,omitted_fit_case=i),gradient_tolerance=request['gradient_tolerance'])
            if result['best']:
                result['rates_per_morgan']=np.exp(result['best']['log_rates']).tolist()
            sensitivity.append(result)
        summary['leave_one_fit_case_out_sensitivity']=sensitivity
        if not all(s['converged'] for s in sensitivity):summary['status']='INCOMPLETE'
    _write_fresh_atomic(out/'CALIBRATION.json',_canonical(summary))
    print(json.dumps({k:summary[k] for k in ('status','neural_training_performed')}))


if __name__=='__main__':
    torch.set_num_threads(1)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request',required=True);ap.add_argument('--counts',required=True)
    ap.add_argument('--output-dir',required=True);args=ap.parse_args()
    run(args.request,args.counts,args.output_dir)
