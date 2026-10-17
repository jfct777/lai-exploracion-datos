#!/usr/bin/env python3
"""One resumable planning step for the FIT-only adaptive Leiden profile.

Does not run Nextflow or cloud commands. A reviewed operator executes the emitted
batch and calls this planner again. Completed seed products are authenticated
and reused even when original Nextflow work directories are unavailable.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import r02_structure_fit_graph as graph_io

io=graph_io.io
require=io.require
profile=graph_io.profile
policy=graph_io.policy
FAMILY_QUEUE='r02_fit_family_profile_queue_request_v2'


class Pending(Exception):
    def __init__(self,batch):self.batch=batch


class Incomplete(Exception):
    pass


def implementation_hashes():
    return graph_io.implementation_hashes()|{'r02_structure_profile_queue.py':io.sha(Path(__file__))}


def write_bytes_once(path,raw,maximum):
    """Immutable artifact, durable before any child task is released."""
    require(len(raw)<=maximum,'Profile metadata byte limit exceeded')
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if path.exists():
        require(path.read_bytes()==raw,'Attempt to change an immutable profile record')
    else:
        with path.open('xb')as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        path.chmod(0o600)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return io.sha(path)


def write_once(path,value,maximum):
    return write_bytes_once(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode(),maximum)


def plan(request_path,request_sha256,admission_path,admission_sha256):
    request_path=graph_io.pin(request_path,request_sha256,2**20)
    r=io.load_json(request_path)
    family_queue=r.get('schema')==FAMILY_QUEUE
    fields={'schema','evidence_scope','scientific_spec_sha256','implementation_sha256',
        'graph','work_root','code_dir','search','candidate_resources','resources'}
    require(set(r)==fields|({'coverage'} if family_queue else set()) and
        r['schema'] in {'r02_fit_profile_queue_request_v1',FAMILY_QUEUE} and r['implementation_sha256']==implementation_hashes(),
        'Exact profile queue request/implementation required')
    require(r['evidence_scope']in {'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'} and io.valid_hash(r['scientific_spec_sha256']),
        'Explicit profile evidence and scientific specification required')
    limits=r['resources'];search=r['search']
    require(set(limits)=={'max_input_bytes','max_metadata_bytes','max_memory_bytes','max_output_bytes','min_free_bytes'}
        and all(type(v)is int and v>0 for v in limits.values()),'Explicit profile resource bounds required')
    require(set(search)=={'family_error','views','missing_mass','max_queries','max_episodes_per_query','max_total_seeds','max_iterations'}
        and all(type(search[k])is int and search[k]>0 for k in ('views','max_queries','max_episodes_per_query','max_total_seeds','max_iterations')),
        'Explicit optimizer coverage and resource policy required')
    policy.probability(search['family_error'],'family_error');policy.probability(search['missing_mass'],'missing_mass')
    require(family_queue or r['evidence_scope']!='DNABR_R02' or search['views']==88,
        'R02 legacy coverage family must retain all 88 views')
    a=io.load_json(graph_io.pin(admission_path,admission_sha256,limits['max_metadata_bytes']))
    require(a['schema']==('r02_fit_family_profile_queue_admission_v2' if family_queue else 'r02_fit_profile_queue_admission_v1') and a['status']=='FIT_PROFILE_POLICY_ADMITTED'
        and a['request_sha256']==request_sha256 and a['scientific_spec_sha256']==r['scientific_spec_sha256']
        and a['evidence_scope']==r['evidence_scope'] and a['evaluate_allowed']is False,
        'Parent scientific admission for adaptive FIT policy required')
    if family_queue:
        require(a.get('coverage')==r['coverage'], 'Family coverage must be explicitly admitted')
    for name,pin in r['implementation_sha256'].items():
        graph_io.pin(Path(r['code_dir'])/name,pin,limits['max_metadata_bytes'])
    root=Path(r['work_root'])
    require(root.is_absolute() and '..'not in root.parts,'Explicit safe absolute profile workspace required')
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with(root/'planner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding=dict(request_sha256=request_sha256,admission_sha256=admission_sha256,
                     implementation_sha256=r['implementation_sha256'])
        write_once(root/'binding.json',binding,limits['max_metadata_bytes'])
        return _plan_locked(r,root,request_sha256,admission_sha256)


def _plan_locked(r,root,request_sha256,admission_sha256):
    limits=r['resources'];search=r['search']
    graph,persons,gm=graph_io.load_graph(r['graph']['directory'],r['graph']['manifest_sha256'],
        max_input_bytes=limits['max_input_bytes'],max_metadata_bytes=limits['max_metadata_bytes'],max_memory_bytes=limits['max_memory_bytes'])
    require(gm['evidence_scope']==r['evidence_scope'],'Profile graph evidence differs')
    family_queue=r['schema']==FAMILY_QUEUE
    require(gm['schema']==(graph_io.FAMILY_GRAPH_SCHEMA if family_queue else 'r02_fit_graph_frozen_v1'),
            'Profile queue and graph family schema differ')
    identity=None
    if family_queue:
        coverage=r['coverage']
        require(set(coverage)=={'path','sha256'}, 'Pinned family coverage required')
        universe=io.load_json(graph_io.pin(coverage['path'],coverage['sha256'],limits['max_metadata_bytes']))
        require(set(universe)=={'schema','family','views'} and universe['schema']=='r02_fit_family_profile_coverage_v1'
            and universe['family']==gm['family'] and isinstance(universe['views'],list)
            and len(universe['views'])==search['views'], 'Declared family coverage differs from error denominator')
        entries=universe['views']
        require(all(set(x)=={'view_id','source_sha256'} and isinstance(x['view_id'],str)
            and x['view_id'] and io.valid_hash(x['source_sha256']) for x in entries)
            and len({x['view_id'] for x in entries})==len(entries)
            and {'view_id':gm['view_id'],'source_sha256':graph.source_sha256} in entries,
            'Family coverage must uniquely include the authenticated graph')
        identity={k:gm[k] for k in ('family','affinity_semantics','territory','source_sha256','graph_sha256')}
    import shutil
    require(shutil.disk_usage(root).free>=limits['min_free_bytes'],'Profile disk reserve not available')
    d=graph.degree;a=graph.adjacency;left,right=np.nonzero(np.triu(a,1));volume=float(d.sum())
    require(len(left)>0 and volume>0,'Nonempty FIT graph required')
    low=float(a[left,right].min()/volume);high=float(np.max(a[left,right]*volume/(d[left]*d[right])))
    active=np.flatnonzero(graph.active_fit)
    initial=[]
    for membership in (graph.connected_component,np.arange(len(d))):
        labels=graph_io.learner.canonical_labels(membership[active],graph.fit_indices[active])
        labels={str(int(i)):labels.get(int(i),'UNASSIGNED')for i in graph.fit_indices}
        initial.append(profile.make_line(graph,labels))
    known={line.key:line for line in initial};candidates={};candidate_episodes={};attempts=[];episodes=[]
    counters=dict(queries=0,seeds=0,candidate_bytes=0)
    code=graph_io.implementation_hashes()

    def oracle(gamma):
        counters['queries']+=1;j=counters['queries'];prior_results=[];query_lines={}
        for k in range(1,search['max_episodes_per_query']+1):
            ep=root/'episodes'/('query_%06d'%j)/('episode_%06d.json'%k)
            required=policy.confirmation_size(missing_mass=search['missing_mass'],error=policy.episode_error(
                family_error=search['family_error'],views=search['views'],query_index=j,episode_index=k))
            if counters['seeds']+required>search['max_total_seeds']:
                raise Incomplete('SEED_BUDGET_BEFORE_COMPLETE_EPISODE')
            if ep.exists():
                require(ep.stat().st_size<=limits['max_metadata_bytes'],'Episode metadata exceeds budget')
                e=io.load_json(ep);policy.verify_episode(e)
                require(e['source_sha256']==graph.source_sha256 and e['view_id']==gm['view_id']
                    and e['gamma_hex']==float(gamma).hex() and e['known_memberships']==sorted(known)
                    and e['query_index']==j and e['episode_index']==k and e['missing_mass']==search['missing_mass']
                    and e['family_error']==search['family_error'] and e['views']==search['views']
                    and e['prior_seeds']==sorted(x['seed']for x in prior_results),
                    'Replay history differs from precommitted episode')
            else:
                e=policy.freeze_episode(source_sha256=graph.source_sha256,view_id=gm['view_id'],gamma=gamma,
                    known_memberships=list(known),prior_results=prior_results,family_error=search['family_error'],
                    views=search['views'],query_index=j,episode_index=k,missing_mass=search['missing_mass'])
                write_once(ep,e,limits['max_metadata_bytes'])
            counters['seeds']+=len(e['seeds']);missing=[];results=[]
            for seed in e['seeds']:
                ident='%s_q%d_e%d_s%d'%(request_sha256[:12],j,k,seed)
                task=root/'requests'/(ident+'.json');adm=root/'admissions'/(ident+'.json')
                cr=dict(schema=(graph_io.FAMILY_CANDIDATE_REQUEST if family_queue else 'r02_fit_cached_candidate_request_v1'),evidence_scope=r['evidence_scope'],
                    scientific_spec_sha256=r['scientific_spec_sha256'],implementation_sha256=code,graph=r['graph'],
                    episode=dict(path=str(ep),sha256=io.sha(ep)),parameters=dict(resolution=gamma,seed=seed,
                    max_iterations=search['max_iterations']),resources=r['candidate_resources'])
                cp=write_once(task,cr,limits['max_metadata_bytes'])
                ar=dict(schema='r02_fit_graph_task_admission_v1',request_sha256=cp,
                    scientific_spec_sha256=r['scientific_spec_sha256'],mode='CANDIDATE',evidence_scope=r['evidence_scope'],
                    status='FIT_ONLY_ADMITTED',evaluate_allowed=False,parent_profile_request_sha256=request_sha256,
                    parent_profile_admission_sha256=admission_sha256)
                ap=write_once(adm,ar,limits['max_metadata_bytes'])
                product=root/'published'/('CANDIDATE_'+ident);manifest=product/'candidate_manifest.json'
                if not manifest.exists():
                    missing.append(dict(id=ident,request=str(task),sha256=cp,admission=str(adm),admission_sha256=ap));continue
                require(manifest.stat().st_size<=limits['max_metadata_bytes'],'Candidate manifest too large')
                m=io.load_json(manifest)
                require(m['schema']==('r02_fit_family_cached_candidate_result_v2' if family_queue else 'r02_fit_cached_candidate_result_v1') and m['request_sha256']==cp
                    and m['implementation_sha256']==code and m['graph']==r['graph']
                    and m['episode_sha256']==e['episode_sha256'] and m['episode_file_sha256']==io.sha(ep)
                    and m['source_sha256']==graph.source_sha256 and m['seed']==seed
                    and m['gamma_hex']==float(gamma).hex() and m['status']=='FIT_ZERO_IMPROVEMENT_ITERATION_REACHED',
                    'Cached candidate identity or convergence differs')
                if family_queue:
                    require(m.get('identity')==identity and m.get('profile_identity_sha256')==io.materialize.digest(
                        dict(identity=identity,membership_key=m['membership_key'],rb_line=m['rb_line'])),
                        'Cached family candidate semantics differ')
                require(set(m['outputs_sha256'])=={'candidate.json','communities.private.tsv','community_method.json','frozen_communities.json'},
                    'Exact per-seed candidate artifact inventory required')
                for name,pin in m['outputs_sha256'].items():graph_io.pin(product/name,pin,limits['max_metadata_bytes'])
                current_bytes=(product/'candidate.json').stat().st_size
                require(6*(counters['candidate_bytes']+current_bytes)+1024*len(attempts)+3*gm['arrays_bytes']<=limits['max_memory_bytes'],
                        'Accumulated candidate metadata exceeds memory budget')
                c=io.load_json(product/'candidate.json')
                require(c['candidate_sha256']==io.materialize.digest({n:v for n,v in c.items()if n!='candidate_sha256'})
                    and c['graph_sha256']==graph.graph_sha256 and c['source_sha256']==graph.source_sha256,
                    'Candidate content/provenance differs')
                line=profile.make_line(graph,c['labels'])
                require(vars(line)==m['rb_line'] and line.key==m['membership_key'],'Cached RB line differs from actual membership')
                record={name:m[name]for name in ('seed','source_sha256','gamma_hex','membership_key','status')}
                if line.key not in candidates:counters['candidate_bytes']+=current_bytes
                results.append(record);candidates.setdefault(line.key,c);query_lines[line.key]=line
                candidate_episodes.setdefault(line.key,e['episode_sha256'])
                attempts.append(record|dict(query_index=j,episode_index=k,request_sha256=cp,
                    candidate_manifest_sha256=io.sha(manifest),normalized_objective=c['normalized_objective'],
                    iterations=len(c['iterations'])))
            if missing:
                batch=dict(schema='r02_fit_graph_task_batch_v1',implementation_sha256=code,tasks=missing)
                bp=root/'batches'/('%s.json'%io.materialize.digest(batch));write_once(bp,batch,limits['max_output_bytes'])
                require(sum(p.stat().st_size for p in root.rglob('*')if p.is_file())<=limits['max_output_bytes'],
                        'Profile workspace publication budget exceeded')
                raise Pending(dict(path=str(bp),sha256=io.sha(bp),tasks=len(missing),query_index=j,episode_index=k))
            assessment=policy.assess_episode(e,results)
            episodes.append(dict(path=str(ep),sha256=io.sha(ep),assessment=assessment))
            known.update(query_lines);prior_results.extend(results)
            if assessment['status']=='DECLARED_SEED_MASS_COVERED':return list(query_lines.values())
        raise Incomplete('EPISODE_LIMIT_WITH_UNCOVERED_SEED_MASS')

    try:
        result=profile.bounded_profile(initial,oracle,gamma_low=low,gamma_high=high,max_queries=search['max_queries'])
    except Pending as p:
        return dict(status='PENDING_AUTHENTICATED_SEED_TASKS',next_batch=p.batch,completed_attempts=len(attempts),
                    distinct_candidates=len(candidates),source_sha256=graph.source_sha256,biological_validation=False)
    except Incomplete as error:
        return dict(status='PROFILE_LIMIT_INCOMPLETE_NOT_ADMITTED',reason=str(error),completed_attempts=len(attempts),
                    distinct_candidates=len(candidates),source_sha256=graph.source_sha256,biological_validation=False)
    missing=sorted(set(result['envelope_candidate_keys'])-set(candidates))
    complete=result['status']=='NO_NEW_ENVELOPE_IMPROVEMENT_UNDER_DECLARED_ORACLE'and not missing
    result.update(source_sha256=graph.source_sha256,graph_sha256=graph.graph_sha256,attempts=attempts,episodes=episodes,
        candidates=candidates,predictive_candidate_keys=sorted(candidates),candidate_policy='ALL_DISCOVERED_CONVERGED_FIT_MEMBERSHIPS',
        seed_policy=search,seed_coverage_complete=complete,analytic_memberships_without_learned_candidate=missing,
        rb_dominated_candidate_keys=sorted(set(candidates)-set(result['envelope_candidate_keys'])),
        envelope_role='DIRECT_RESOLUTION_QUERIES_NOT_FILTER_PREDICTIVE_CANDIDATES',alpha_or_outcome_selection=False)
    if not complete:
        return dict(status='PROFILE_LIMIT_INCOMPLETE_NOT_ADMITTED',reason=result['status'],
            analytic_memberships_without_learned_candidate=missing,completed_attempts=len(attempts),distinct_candidates=len(candidates))
    out=root/'complete'
    # An existing final result must be identical; scientific products are not overwritten.
    out.mkdir(exist_ok=True,mode=0o700)
    for key,c in candidates.items():
        candidate_out=out/'candidates'/key
        with tempfile.TemporaryDirectory(prefix='export-',dir=root)as tmp:
            temporary=Path(tmp)/'candidate'
            if family_queue:
                graph_io.write_family_candidate(temporary,c,persons,gm,candidate_episodes[key])
            else:
                graph_io.learner.write_frozen_candidate(temporary,c,persons=persons,
                    persons_sha256=gm['inputs']['persons.tsv']['sha256'],fit_affinities_sha256=gm['inputs']['fit_affinities.private.tsv']['sha256'],
                    exclusion_sha256=gm['inputs']['exclusion.json']['sha256'])
            for product in temporary.iterdir():
                write_bytes_once(candidate_out/product.name,product.read_bytes(),limits['max_metadata_bytes'])
    write_once(out/'profile.json',result,limits['max_output_bytes'])
    manifest=dict(schema=('r02_fit_family_profile_result_v4' if family_queue else 'r02_fit_profile_result_v3'),status=result['status'],profile_complete=True,
        request_sha256=request_sha256,admission_sha256=admission_sha256,graph=r['graph'],
        implementation_sha256=r['implementation_sha256'],evidence_scope=r['evidence_scope'],graph_sha256=graph.graph_sha256,
        candidate_policy=result['candidate_policy'],predictive_candidate_keys=sorted(candidates),
        rb_envelope_candidate_keys=result['envelope_candidate_keys'],rb_dominated_candidate_keys=result['rb_dominated_candidate_keys'],
        seed_policy=search,seed_coverage_complete=True,query_edges_used=False,evaluate_scored=False,biological_validation=False,
        outputs_sha256={str(p.relative_to(out)):io.sha(p)for p in out.rglob('*')if p.is_file()and p.name!='profile_manifest.json'})
    if family_queue:
        manifest.update(identity=identity,coverage=r['coverage'],source_files=gm['source_files'])
    else:
        manifest['inputs']=gm['inputs']
    require(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())<=limits['max_output_bytes'],
        'Complete profile exceeds total output budget')
    write_once(out/'profile_manifest.json',manifest,limits['max_metadata_bytes'])
    return dict(status='FIT_PROFILE_COMPLETE',directory=str(out),manifest_sha256=io.sha(out/'profile_manifest.json'),
                completed_attempts=len(attempts),distinct_candidates=len(candidates),biological_validation=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('request','request-sha256','admission','admission-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(plan(a.request,a.request_sha256,a.admission,a.admission_sha256),indent=2))
