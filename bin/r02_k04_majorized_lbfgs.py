#!/usr/bin/env python3
"""Checkpointed L-BFGS iterations with the reviewed K04 quadratic majorizer."""
from __future__ import annotations
import contextlib,hashlib,json,math,signal,time
from pathlib import Path
import numpy as np

SCHEMA='r02_k04_majorized_lbfgs_checkpoint_v1'
def boot_id(): return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def require(x,m):
    if not x: raise ValueError(m)
def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def envelope(s):
    gap=s['best_primal_objective']-s['best_dual_objective']
    return {'primal_objective':s['best_primal_objective'],'feasible_dual_objective':s['best_dual_objective'],
        'primal_dual_gap':gap,'scope':'FP64_CONDITIONAL_PROPAGATED_NOT_EXACT'}
def bindings(state):
    return {'optimizer_point_sha256':array_sha(state['x']),'gradient_sha256':array_sha(state['gradient']),
        'gradient_point_sha256':array_sha(state['x']),'majorizer_rhs_sha256':array_sha(-state['gradient']),
        'current_repaired_dual_sha256':array_sha(state['current_repaired_dual'])}
def valid_diagnostic(d):
    keys=('primal_objective','feasible_dual_objective','primal_dual_gap')
    return (d.get('dual_certificate_eligible') is True and d.get('feasible_dual_sum_residual')==0.0 and
        all(math.isfinite(float(d[k])) for k in keys) and float(d['primal_dual_gap'])>=0 and
        float(d['primal_objective'])>=float(d['feasible_dual_objective']))
@contextlib.contextmanager
def deadline(seconds):
    require(math.isfinite(seconds) and seconds>0,'positive finite operation deadline required')
    old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    def expired(_signum,_frame):raise TimeoutError('bounded operator stage exceeded reserved seconds')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        if old_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,*old_timer)
def save(dual,path,state):
    names={'x','gradient','history_s','history_y','best_beta','best_dual','current_repaired_dual'}
    state['bindings']=bindings(state)
    dual._atomic_checkpoint(path,{k:state[k] for k in names},{k:v for k,v in state.items() if k not in names})
def load(dual,path,manifest_sha,selection_sha,dimension,rows,columns,history_size,max_products,max_seconds,max_iterations):
    with np.load(path,allow_pickle=False) as z:
        names={'metadata_utf8','x','gradient','history_s','history_y','best_beta','best_dual','current_repaired_dual'}
        require(set(z.files)==names,'checkpoint arrays differ');s=json.loads(z['metadata_utf8'].tobytes())
        for k in names-{'metadata_utf8'}: s[k]=np.asarray(z[k],dtype=np.float64).copy()
    require(s.get('schema')==SCHEMA and s.get('manifest_sha256')==manifest_sha and
        s.get('selection_sha256')==selection_sha,'checkpoint identity differs')
    require(s['x'].shape==s['gradient'].shape==(dimension,) and s['history_s'].shape==s['history_y'].shape==(history_size,dimension)
        and s['best_beta'].shape==(columns,) and s['best_dual'].shape==s['current_repaired_dual'].shape==(rows,),'checkpoint shapes differ')
    require(all(np.isfinite(s[k]).all() for k in ('x','gradient','history_s','history_y','best_beta','best_dual','current_repaired_dual')),'checkpoint nonfinite')
    integer_keys=('iteration','history_count','history_next','products_budget_charged','products_completed','products_lost_reserve','skipped_curvature_updates','history_resets')
    require(all(type(s.get(k)) is int and s[k]>=0 for k in integer_keys),'checkpoint counters invalid')
    require(s['iteration']<=max_iterations and s['history_count']<=history_size and s['history_next']<history_size and
        s['products_completed']+s['products_lost_reserve']<=s['products_budget_charged']<=max_products,
        'checkpoint counters exceed bounds')
    require(all(math.isfinite(float(s.get(k,float('nan')))) for k in ('value','best_intercept','best_primal_objective','best_dual_objective','elapsed_compute_seconds','lost_compute_reserve_seconds','interruption_observed_seconds_total')) and
        0<=s['lost_compute_reserve_seconds']<=s['elapsed_compute_seconds']<=max_seconds and
        s['interruption_observed_seconds_total']>=0 and
        s['best_primal_objective']>=s['best_dual_objective'],'checkpoint scalar/envelope invalid')
    require(s.get('pending_step') is None or s.get('pending_step')==s['iteration']+1,'checkpoint pending step invalid')
    if s.get('pending_step') is None:
        require(s.get('pending_started_monotonic') is None and s.get('pending_boot_id') is None,'completed checkpoint has pending clock')
    else:
        require(math.isfinite(float(s.get('pending_started_monotonic',float('nan')))) and s['pending_started_monotonic']>=0 and
            isinstance(s.get('pending_boot_id'),str) and s['pending_boot_id'],'pending clock invalid')
    require(s.get('bindings')==bindings(s),'checkpoint point/gradient binding differs')
    return s

def run(problem,dual,exact,parameters_fn,bt,product_counts,checkpoint,*,manifest_sha,selection_sha,source_checkpoint_sha,
        starting_dual,best_beta,best_intercept,curvature,history_size,max_iterations,max_products,max_seconds,
        step_seconds_bound,closing_reserve_seconds,curvature_epsilon,resume=False,invocation_iteration_limit=None,
        initial_products_budget_charged=4,initial_products_lost_reserve=0,
        initial_elapsed_compute_seconds=0.0,initial_lost_compute_reserve_seconds=0.0,initial_seconds_bound=150,
        source_primal_objective=None,source_dual_objective=None):
    started=time.monotonic(); checkpoint=Path(checkpoint)
    if resume:
        state=load(dual,checkpoint,manifest_sha,selection_sha,problem.subspace.dimension,problem.n_rows,problem.n_columns,history_size,max_products,max_seconds,max_iterations)
        if state.get('pending_step') is not None:
            require(state['pending_boot_id']==boot_id(),'cannot account interrupted step across system boot')
            observed=max(0.0,time.monotonic()-state['pending_started_monotonic'])
            charged_seconds=min(max_seconds-state['elapsed_compute_seconds'],max(step_seconds_bound,observed))
            state['products_lost_reserve']+=5
            state['lost_compute_reserve_seconds']+=charged_seconds
            state['elapsed_compute_seconds']+=charged_seconds
            state['interruption_observed_seconds_total']+=observed
            state['pending_step']=None;state['pending_started_monotonic']=None;state['pending_boot_id']=None;save(dual,checkpoint,state)
    else:
        require(not checkpoint.exists(),'fresh run refuses checkpoint overwrite')
        require(initial_products_budget_charged<=max_products-2 and
            initial_elapsed_compute_seconds+initial_seconds_bound+closing_reserve_seconds<max_seconds,
            'insufficient cumulative budget before initial four products')
        coordinates,conversion_l2,_=problem.subspace.coordinates_from_full(np.asarray(starting_dual)/problem.sqrt_a)
        before=sum(product_counts.values())
        with deadline(initial_seconds_bound):
            value,gradient,physical=problem.value_gradient(coordinates)
            # Preserve and re-certify the selected feasible incumbent itself.
            diagnostic,repaired=exact.certify_repaired(problem,float(best_intercept),np.asarray(best_beta),np.asarray(starting_dual),curvature)
        require(sum(product_counts.values())-before==4 and diagnostic['dual_certificate_eligible'] and diagnostic['feasible_dual_sum_residual']==0.0,'initial four products/certificate differ')
        require(valid_diagnostic(diagnostic),'initial certificate diagnostic invalid')
        require(source_primal_objective is not None and source_dual_objective is not None and
            float(diagnostic['primal_objective'])==float(source_primal_objective) and
            float(diagnostic['feasible_dual_objective'])==float(source_dual_objective),
            'recomputed BH certificate does not exactly reconcile with selected source')
        state={'schema':SCHEMA,'manifest_sha256':manifest_sha,'selection_sha256':selection_sha,
          'source_checkpoint_sha256':source_checkpoint_sha,'iteration':0,'x':coordinates,'value':float(value),'gradient':gradient,
          'history_s':np.zeros((history_size,len(coordinates))),'history_y':np.zeros((history_size,len(coordinates))),
          'history_count':0,'history_next':0,'skipped_curvature_updates':0,'history_resets':0,
          'best_beta':np.asarray(best_beta).copy(),'best_intercept':float(best_intercept),
          'best_primal_objective':float(diagnostic['primal_objective']),'best_dual':repaired.copy(),
          'best_dual_objective':float(diagnostic['feasible_dual_objective']),'current_repaired_dual':repaired.copy(),
          'coordinate_conversion_l2':conversion_l2,
          'coordinate_raw_dual_difference_l2':float(np.linalg.norm(physical['raw_dual']-np.asarray(starting_dual))),
          'source_repaired_dual_sha256':array_sha(np.asarray(starting_dual)),
          'products_budget_charged':initial_products_budget_charged,'products_completed':4,
          'products_lost_reserve':initial_products_lost_reserve,'pending_step':None,
          'pending_started_monotonic':None,'pending_boot_id':None,'interruption_observed_seconds_total':0.0,
          'elapsed_compute_seconds':initial_elapsed_compute_seconds+(time.monotonic()-started),
          'lost_compute_reserve_seconds':initial_lost_compute_reserve_seconds,'history':[]}
        started=time.monotonic();save(dual,checkpoint,state)
    status='INCOMPLETE_ITERATION_LIMIT'; accepted_this=0
    while state['iteration']<max_iterations:
        gap=envelope(state)['primal_dual_gap']
        if 0<=gap<=dual.GAP_TARGET: status='CONVERGED_ORIGINAL_GAP';break
        if state['products_budget_charged']+5+2>max_products: status='INCOMPLETE_PRODUCT_BUDGET';break
        if state['elapsed_compute_seconds']+(time.monotonic()-started)+step_seconds_bound+closing_reserve_seconds>=max_seconds:
            status='INCOMPLETE_WALL_BUDGET_WITH_CLOSING_RESERVE';break
        direction=dual._lbfgs_direction(state['gradient'],state['history_s'],state['history_y'],state['history_count'],state['history_next'])
        slope=float(state['gradient']@direction); source='LBFGS'
        if not math.isfinite(slope) or slope>=0:
            require(np.isfinite(state['gradient']).all() and float(state['gradient']@state['gradient'])>0,'zero/nonfinite gradient is not convergence')
            direction=-state['gradient'].copy();state['history_count']=0;state['history_next']=0;state['history_resets']+=1
            slope=float(state['gradient']@direction);source='NEGATIVE_GRADIENT_HISTORY_RESET'
        state['elapsed_compute_seconds']+=time.monotonic()-started;started=time.monotonic()
        state['pending_step']=state['iteration']+1;state['pending_started_monotonic']=time.monotonic();state['pending_boot_id']=boot_id()
        state['products_budget_charged']+=5;save(dual,checkpoint,state)
        before=sum(product_counts.values())
        with deadline(step_seconds_bound):
            transformed=bt(direction)
            parameters=parameters_fn(state['gradient'],direction,transformed,problem.penalty)
            require(parameters['status']=='DIRECTION_READY_NOT_CERTIFIED','majorizer direction rejected')
            trial_x=state['x']+parameters['step']*direction
            trial_value,trial_gradient,physical=problem.value_gradient(trial_x)
            actual=float(state['value']-trial_value)
            if actual>0 and actual>=parameters['predicted_decrease']:
                diagnostic,repaired=exact.certify_repaired(problem,physical['intercept'],physical['beta'],physical['raw_dual'],curvature)
        if actual<=0 or actual<parameters['predicted_decrease']:
            status='MAJORIZER_SAFEGUARD_UNRESOLVED';state['pending_step']=None;state['pending_started_monotonic']=None;state['pending_boot_id']=None;state['products_lost_reserve']+=5
            state['elapsed_compute_seconds']+=time.monotonic()-started;started=time.monotonic();save(dual,checkpoint,state);break
        require(sum(product_counts.values())-before==5,'five-product step accounting differs')
        proposed_primal=min(state['best_primal_objective'],float(diagnostic['primal_objective'])) if valid_diagnostic(diagnostic) else float('nan')
        proposed_dual=max(state['best_dual_objective'],float(diagnostic['feasible_dual_objective'])) if valid_diagnostic(diagnostic) else float('nan')
        if not valid_diagnostic(diagnostic) or not math.isfinite(proposed_primal-proposed_dual) or proposed_primal<proposed_dual:
            status='CERTIFICATE_OR_ENVELOPE_INVALID';state['pending_step']=None;state['pending_started_monotonic']=None;state['pending_boot_id']=None;state['products_lost_reserve']+=5
            state['elapsed_compute_seconds']+=time.monotonic()-started;started=time.monotonic();save(dual,checkpoint,state);break
        displacement=trial_x-state['x'];change=trial_gradient-state['gradient'];sy=float(displacement@change)
        scale=float(np.linalg.norm(displacement))*float(np.linalg.norm(change))
        if sy>0 and sy>curvature_epsilon*scale:
            i=state['history_next'];state['history_s'][i]=displacement;state['history_y'][i]=change
            state['history_next']=(i+1)%history_size;state['history_count']=min(history_size,state['history_count']+1)
        else: state['skipped_curvature_updates']+=1
        state.update(x=trial_x,value=float(trial_value),gradient=trial_gradient,current_repaired_dual=repaired,
            iteration=state['iteration']+1,pending_step=None,pending_started_monotonic=None,pending_boot_id=None,
            products_completed=state['products_completed']+5)
        if diagnostic['primal_objective']<state['best_primal_objective']:
            state.update(best_beta=physical['beta'].copy(),best_intercept=float(physical['intercept']),best_primal_objective=float(diagnostic['primal_objective']))
        if diagnostic['dual_certificate_eligible'] and diagnostic['feasible_dual_sum_residual']==0.0 and diagnostic['feasible_dual_objective']>state['best_dual_objective']:
            state.update(best_dual=repaired.copy(),best_dual_objective=float(diagnostic['feasible_dual_objective']))
        state['history'].append({'iteration':state['iteration'],'direction_source':source,'majorizer':parameters,'actual_decrease':actual,
            'diagnostic':diagnostic,'best':envelope(state)})
        state['elapsed_compute_seconds']+=time.monotonic()-started;started=time.monotonic();save(dual,checkpoint,state);accepted_this+=1
        if 0<=envelope(state)['primal_dual_gap']<=dual.GAP_TARGET:status='CONVERGED_ORIGINAL_GAP';break
        if invocation_iteration_limit is not None and accepted_this>=invocation_iteration_limit: status='INCOMPLETE_INVOCATION_TEST_LIMIT';break
    state['elapsed_compute_seconds']+=time.monotonic()-started;save(dual,checkpoint,state)
    return {'status':status,'iterations':state['iteration'],'best':envelope(state),'state':state,
      'scientific_model_admitted':False,'select_opened':False,'evaluate_opened':False}
