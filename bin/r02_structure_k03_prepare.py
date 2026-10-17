#!/usr/bin/env python3
"""Authenticated K03 FIT/SELECT preparation with shared bases and compact labels.

No model fitting, alpha selection or EVALUATE target construction occurs here.
Common scores and rare graphs retain their complete upstream numerical ranks.
"""
from pathlib import Path
import collections
import hashlib
import json
import os
import shutil
import zipfile
import numpy as np
from sklearn.impute import SimpleImputer
import r02_structure_increment as core
import r02_structure_numerics as numeric
import r02_structure_affinity_io as affinity
import r02_structure_k03_features as features
import r02_structure_k03_design as design
import r02_structure_profile_provenance as profile_provenance

require=core.require
CODE_FILES=tuple(dict.fromkeys(('r02_structure_k03_prepare.py','r02_structure_k03_features.py',
    'r02_structure_k03_design.py','r02_structure_profile_provenance.py','r02_structure_numerics.py',*affinity.CODE_FILES,*features.continuous.CODE_FILES)))
def implementation():return {n:core.sha(Path(__file__).with_name(n))for n in CODE_FILES}


def target_arrays(persons,block,markers,genotypes,paired,*,min_fit_called,min_pair_markers,max_pairs,max_workspace_bytes):
    """Original masked target and B1 controls, FIT-only mean imputation.

    Unweighted mean imputation matches the established B1 design. Subsequent
    Ridge scaling remains weighted by component pairs. SELECT never fits either.
    """
    roles=core.validate_roles(persons);components=[p['component']for p in persons]
    gt=core._block_genotypes(block,markers,genotypes);records={};audit={}
    for role in ('FIT','SELECT'):
        i,j,y,n,info=numeric.target_pairs(gt,roles['FIT'],roles[role],components,min_fit_called,
            min_pair_markers,max_pairs,max_workspace_bytes=max_workspace_bytes)
        w,_=core.component_pair_weights(i,j,components)
        values=core.paired_rows(paired,'B1',i,j)
        missing=np.isnan(core.paired_rows(paired,'B2',i,j,column_indices=paired['indicator_indices']['B1'])).astype(float)
        require(values.ndim==2 and not np.isinf(values).any(),'B1 control values invalid')
        records[role]=dict(i=i,j=j,y=y,n=n,w=w,extra=np.concatenate((values,missing),axis=1))
        audit[role]=dict(info,pairs=len(i),min_joint_markers=int(n.min()),max_joint_markers=int(n.max()),
            weight_sum=float(w.sum()),paired_control_missing=np.isnan(values).sum(axis=0).tolist())
    require(np.isfinite(records['FIT']['extra']).any(axis=0).all(),'Paired column has no observed support in FIT')
    imputer=SimpleImputer(strategy='mean',keep_empty_features=True)
    records['FIT']['extra']=imputer.fit_transform(records['FIT']['extra'])
    records['SELECT']['extra']=imputer.transform(records['SELECT']['extra'])
    return records,dict(roles=audit,min_fit_called=min_fit_called,min_pair_markers=min_pair_markers,
        imputation_rule='UNWEIGHTED_FIT_COLUMN_MEAN_EXISTING_B1',imputation_statistics=imputer.statistics_.tolist(),
        indicator_indices=paired['indicator_indices']['B1'],evaluate_targets_computed=False)


class Authenticator:
    def __init__(self,limits):self.limits=limits;self.pins={};self.total=0;self.array_sizes={}
    def pin(self,path,digest,*,metadata=False):
        path=Path(path).resolve();limit=self.limits['max_metadata_bytes']if metadata else self.limits['max_input_bytes']
        require(path.is_file()and path.stat().st_size<=limit and core.sha(path)==digest,'K03 source changed or byte bound exceeded')
        if str(path)not in self.pins:self.total+=path.stat().st_size
        require(self.total<=self.limits['max_input_bytes'],'K03 aggregate input byte bound exceeded')
        self.pins[str(path)]=digest;return path
    def array(self,path,digest,fields=None):
        p=self.pin(path,digest)
        with zipfile.ZipFile(p)as z:
            size=sum(v.file_size for v in z.infolist())
            require(size<=self.limits['max_array_bytes'],'K03 decompressed array bound exceeded')
            if fields is not None:require({v.filename for v in z.infolist()}=={k+'.npy'for k in fields}and len(z.infolist())==len(fields),'K03 exact array inventory differs')
            self.array_sizes[str(p)]=size
        return p
    def verify(self):require(all(core.sha(p)==h for p,h in self.pins.items()),'K03 preparation source changed')


def common_scores(desc,auth,persons,*,source_sha256,block_id,scope,allowed,weighting):
    require(set(desc)=={'directory','manifest_sha256'},'Exact common view descriptor required')
    root=Path(desc['directory']);m=core.load_json(auth.pin(root/'manifest.json',desc['manifest_sha256'],metadata=True))
    require(m['schema']=='r02_structure_materialized_common_merge_v2'and
        m['stage']=='FULL_NUMERICAL_COMMON_SPECTRUM_NOT_PREDICTIVE_SELECTION'and m['source_contract_sha256']==source_sha256
        and m['block_id']==block_id and m['exclusion_scope']==scope and m['allowed_input_intervals']==allowed
        and m['weighting']==weighting and m['targets_read']is False and m['n_sites']>0
        and set(m['outputs_sha256'])=={'continuous.private.npz'},'Common selection or exclusion geometry differs')
    fields={'scores','dual_loadings','eigenvalues','fit_weights','fit_indices','observed_counts'}
    p=auth.array(root/'continuous.private.npz',m['outputs_sha256']['continuous.private.npz'],fields)
    with np.load(p,allow_pickle=False)as z:
        scores=z['scores'];indices=z['fit_indices'];weights=z['fit_weights'];eigenvalues=z['eigenvalues']
    roles=core.validate_roles(persons)
    require(scores.shape==(len(persons),m['numerical_rank'])and np.isfinite(scores).all()
        and np.array_equal(indices,roles['FIT'])and len(eigenvalues)==m['numerical_rank']
        and np.allclose(weights,features.continuous.fit_weights(persons,indices,weighting),rtol=1e-14,atol=0),
        'Full numerical common rank/FIT roster/weights differ')
    return scores,m


def profile_labels(desc,auth,rare):
    require(set(desc) in ({'directory','manifest_sha256'}, {'directory','manifest_sha256','provenance'}),
        'Exact FIT profile descriptor required')
    root=Path(desc['directory']);m=core.load_json(auth.pin(root/'profile_manifest.json',desc['manifest_sha256'],metadata=True))
    require(m['schema']in {'r02_fit_profile_result_v2','r02_fit_profile_result_v3'}and m['profile_complete']is True
        and m['status']=='NO_NEW_ENVELOPE_IMPROVEMENT_UNDER_DECLARED_ORACLE'
        and m['graph_sha256']==rare.graph_sha256 and m['query_edges_used']is False and m['evaluate_scored']is False,
        'Complete outcome-free FIT profile from identical graph required')
    require(m['inputs']['affinity_inputs.json']['sha256']==rare.source_sha256,'FIT profile source view differs')
    for name,digest in m['outputs_sha256'].items():
        require(not Path(name).is_absolute()and '..'not in Path(name).parts,'Unsafe FIT profile product')
        auth.pin(root/name,digest,metadata=True)
    profile=core.load_json(root/'profile.json');keys=m['predictive_candidate_keys']
    if m['schema']=='r02_fit_profile_result_v3':
        profile_provenance.authenticate(m,desc['manifest_sha256'],desc.get('provenance'),auth)
        require(m['seed_coverage_complete']is True and profile['seed_coverage_complete']is True
            and m['seed_policy']==profile['seed_policy'],'Adaptive profile lacks its declared FIT seed coverage')
    require(keys==profile['predictive_candidate_keys']and keys and len(set(keys))==len(keys),'Complete predictive profile family required')
    result={}
    for key in keys:
        c=profile['candidates'][key]
        require(c['status']=='FIT_ZERO_IMPROVEMENT_ITERATION_REACHED'and c['graph_sha256']==rare.graph_sha256
            and c['source_sha256']==rare.source_sha256 and c['persons_sha256']==rare.graph.persons_sha256
            and c['target_used']is False and c['heldout_edges_used']is False and c['learning_roles']==['FIT'],
            'Converged outcome-free FIT membership required')
        labels={int(k):v for k,v in c['labels'].items()};require(set(labels)==set(map(int,rare.graph.fit_indices)),'FIT labels omit people')
        result[key]=labels
    return result,m


def write_products(output,persons,base_points,records,candidates,*,metadata,rare_audit,target_audit,max_output_bytes,min_free_bytes):
    """Shared numerical arrays once; projected categorical codes per candidate."""
    out=Path(output);require(not out.exists()and out.parent.is_dir(),'New K03 prepared output required')
    require(shutil.disk_usage(out.parent).free>=min_free_bytes+max_output_bytes,'K03 preparation disk reserve insufficient')
    out.mkdir(mode=0o700);base=out/'base';base.mkdir();(out/'candidates').mkdir()
    features.discrete.materialize.input_plan.write_table(base/'persons.tsv',persons)
    np.savez(base/'arrays.private.npz',base_points=base_points,**{role+'_'+k:v for role,r in records.items()for k,v in r.items()})
    core.write_json(base/'support.json',dict(rare=rare_audit,target=target_audit))
    def identity(names,roles=('FIT','SELECT')):
        h=hashlib.sha256()
        for role in roles:
            for name in names:
                v=np.asarray(records[role][name]);h.update((role+':'+name+':'+str(v.dtype)+':'+str(v.shape)).encode());h.update(v.tobytes())
        return h.hexdigest()
    m=dict(schema='r02_k03_shared_base_v1',stage='FROZEN_FIT_SELECT_BASE_NO_EVALUATE_TARGETS',**metadata,
        target_identity=identity(('i','j','y','n','w')),mask_control_identity=identity(('i','j','n','w','extra')),
        select_target_identity=identity(('i','j','y','n','w'),('SELECT',)),
        evaluate_targets_computed=False,alpha_selection_performed=False,
        outputs_sha256={p.name:core.sha(p)for p in base.iterdir()if p.is_file()})
    core.write_json(base/'manifest.json',m);base_sha=core.sha(base/'manifest.json');inventory=[]
    def guard():require(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())<=max_output_bytes
        and shutil.disk_usage(out).free>=min_free_bytes,'K03 prepared output byte/disk bound exceeded')
    guard()
    for key,candidate in candidates:
        require(len(key)==64 and all(v in '0123456789abcdef'for v in key),'Safe unique candidate hash required')
        d=out/'candidates'/key;d.mkdir()
        labels=candidate['labels'];cats=candidate['categories'];index={v:k for k,v in enumerate(cats)}
        codes=np.asarray([index.get(v,-1)for v in labels],dtype=np.int32)
        require(len(codes)==len(persons)and np.array_equal(codes>=0,candidate['assigned']),'Projected membership availability differs')
        np.savez(d/'membership.private.npz',codes=codes)
        cm=dict(schema='r02_k03_projected_membership_v1',stage='FROZEN_PROJECTED_MEMBERSHIP_NO_SELECTION',
            candidate_id=key,parent_candidate_id=candidate['parent_candidate_id'],control=candidate['control'],sham_seed=candidate['sham_seed'],
            categories=list(cats),reasons=list(candidate['reasons']),n_categories=len(cats),
            base_manifest_sha256=base_sha,baseline_identity=design.baseline_identity(base_sha,codes),
            evidence_scope=metadata['evidence_scope'],scientific_spec_sha256=metadata['scientific_spec_sha256'],
            block_id=metadata['block_id'],exclusion_scope=metadata['exclusion_scope'],
            outputs_sha256={'membership.private.npz':core.sha(d/'membership.private.npz')},
            target_outcomes_used_for_membership=False,alpha_selection_performed=False)
        core.write_json(d/'manifest.json',cm);inventory.append(dict(candidate_id=key,directory=str(d.resolve()),
            manifest_sha256=core.sha(d/'manifest.json'),baseline_identity=cm['baseline_identity'],control=cm['control'],
            parent_candidate_id=cm['parent_candidate_id'],sham_seed=cm['sham_seed']))
        guard()
    require(inventory,'Nonempty complete candidate preparation required')
    return dict(base=dict(directory=str(base.resolve()),manifest_sha256=base_sha),candidates=inventory,
        baseline_count=len({v['baseline_identity']for v in inventory}),bytes=sum(p.stat().st_size for p in out.rglob('*')if p.is_file()))


def run_request(request_path,request_sha256,admission_path,admission_sha256,output):
    os.umask(0o077);path=Path(request_path);require(path.stat().st_size<=2**20 and core.sha(path)==request_sha256,'K03 prepare request changed')
    r=core.load_json(path)
    require(set(r)=={'schema','evidence_scope','scientific_spec_sha256','source_contract_sha256','implementation_sha256',
        'block_id','sources','weighting','sham_seeds','resources'}and r['schema']=='r02_k03_prepare_request_v1'
        and r['implementation_sha256']==implementation(),'Exact K03 prepare request/code required')
    require(r['evidence_scope']in{'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'}and r['weighting']=='COMPONENT_EQUAL','Explicit K03 population/FIT weighting required')
    require(isinstance(r['sham_seeds'],list)and len(set(r['sham_seeds']))==len(r['sham_seeds'])
        and all(type(s)is int and 0<=s<2**64 for s in r['sham_seeds']),'Explicit unique SHAM seed list required')
    require(core.sha(admission_path)==admission_sha256,'K03 preparation admission changed')
    a=core.load_json(admission_path)
    require(a==dict(schema='r02_k03_prepare_admission_v1',status='FIT_SELECT_INPUTS_ADMITTED',evidence_scope=r['evidence_scope'],
        scientific_spec_sha256=r['scientific_spec_sha256'],request_sha256=request_sha256,evaluate_allowed=False),
        'External admission of exact preparation required')
    limits=r['resources'];require(set(limits)=={'max_metadata_bytes','max_input_bytes','max_array_bytes','max_memory_bytes',
        'max_target_workspace_bytes','max_output_bytes','min_free_bytes'}and all(type(v)is int and v>0 for v in limits.values()),
        'Explicit positive preparation resource limits required')
    auth=Authenticator(limits);sources=r['sources'];require(set(sources)=={'common','rare','profile','prepared','evaluation'},'Exact independent K03 sources required')
    rd=affinity.AuthenticatedRareView(sources['rare'],evidence_scope=r['evidence_scope'],max_input_bytes=limits['max_input_bytes'],max_metadata_bytes=limits['max_metadata_bytes'])
    for name,desc in sources['rare'].items():auth.pin(desc['path'],desc['sha256'],metadata=name not in {'fit_affinities.private.tsv','query_affinities.private.tsv'})
    persons=rd.persons;roles=rd.roles
    require(rd.upstream['source_contract_sha256']==r['source_contract_sha256']and rd.upstream['block_id']==r['block_id'],'Rare source/block differs')
    scope=rd.upstream['exclusion_scope'];exclusion=rd.upstream['exclusion']
    common,cm=common_scores(sources['common'],auth,persons,source_sha256=r['source_contract_sha256'],block_id=r['block_id'],
        scope=scope,allowed=exclusion['allowed_input_intervals'],weighting=r['weighting'])
    graph=features.continuous.FitRareGraphSpectrum(persons,rd.fit_edges(),source_sha256=rd.source_sha256,max_memory_bytes=limits['max_memory_bytes'])
    rare=features.RareProjectionView(persons,graph,rd.query_rows(),source_sha256=rd.source_sha256,
        weighting=r['weighting'],max_memory_bytes=limits['max_memory_bytes'])
    labels,pm=profile_labels(sources['profile'],auth,rare)
    pd=sources['prepared'];require(set(pd)=={'directory','contract_sha256'},'Exact prepared catalogue required')
    root=Path(pd['directory']);contract=core.load_json(auth.pin(root/'contract.json',pd['contract_sha256'],metadata=True))
    ed=sources['evaluation'];require(set(ed)=={'path','sha256'},'Pinned existing target configuration required')
    evaluation=core.load_json(auth.pin(ed['path'],ed['sha256'],metadata=True))
    require(evaluation['prepared_contract_sha256']==pd['contract_sha256']and contract['source_contract_sha256']==r['source_contract_sha256']
        and contract['exclusion_scope']==scope,'Prepared target geometry/source differs')
    store=core.BlockBundles(root,contract,evaluation);require(store.persons==persons,'Common/rare/target persons differ')
    ks=[k for k,b in enumerate(store.blocks)if b['block_id']==r['block_id']];require(len(ks)==1,'Exact target block required')
    shard=store.path(store.entries[ks[0]]);receipt=core.load_json(shard/'manifest.json')
    for n,h in receipt['outputs_sha256'].items():
        if n.endswith('.npz'):auth.array(shard/n,h)
        else:auth.pin(shard/n,h,metadata=True)
    memory_bound=graph.memory_bound+rare.memory_bound+6*sum(auth.array_sizes.values())+limits['max_target_workspace_bytes']
    require(memory_bound<=limits['max_memory_bytes'],'Combined rare/common/target preparation workspace exceeds memory bound')
    values,c=store.load(ks[0]);people,_,blocks,markers,gt,_,pairs=values
    records,target_audit=target_arrays(people,blocks[0],markers,gt,pairs[r['block_id']],min_fit_called=c['min_fit_called'],
        min_pair_markers=c['min_pair_markers'],max_pairs=c['max_pairs'],max_workspace_bytes=limits['max_target_workspace_bytes'])
    base=np.concatenate((common,rare.coordinates,rare.state_points),axis=1)
    del values,gt,pairs
    def candidates():
        for key,real_labels in labels.items():
            projected=rare.project_communities(real_labels)
            yield key,dict(projected,parent_candidate_id=key,control='REAL',sham_seed=None)
            for seed in r['sham_seeds']:
                shuffled=features.conditional_sham_labels(graph,real_labels,source_sha256=rd.source_sha256,seed=seed)
                projected=rare.project_communities(shuffled)
                identifier=hashlib.sha256((key+':SHAM:'+str(seed)).encode()).hexdigest()
                yield identifier,dict(projected,parent_candidate_id=key,control='SHAM_WITHIN_FIT_COMPONENT',sham_seed=seed)
    metadata=dict(evidence_scope=r['evidence_scope'],scientific_spec_sha256=r['scientific_spec_sha256'],block_id=r['block_id'],
        exclusion_scope=scope,source_contract_sha256=r['source_contract_sha256'],implementation_sha256=implementation(),
        input_request_sha256=request_sha256,common_manifest_sha256=sources['common']['manifest_sha256'],
        rare_manifest_sha256=rd.source_sha256,profile_manifest_sha256=sources['profile']['manifest_sha256'],
        common_axes=common.shape[1],rare_axes=rare.coordinates.shape[1],rare_state_axes=rare.state_points.shape[1],
        rare_graph_sha256=graph.graph_sha256,rare_coordinate_sha256=hashlib.sha256(rare.coordinates.tobytes()).hexdigest(),
        exclusion_territory=dict(block_id=exclusion['block_id'],target=exclusion['target'],
            excluded=exclusion['excluded'],allowed_input_intervals=exclusion['allowed_input_intervals'],
            allowed_bp=exclusion['allowed_bp'],excluded_bp=exclusion['excluded_bp']),
        memory_bound_bytes=memory_bound,
        numerical_basis_claim='FROZEN_REALIZED_AXES_NOT_INVARIANT_TO_TIED_EIGENSPACE_ROTATION')
    rare_audit=dict(states=dict(collections.Counter(map(str,rare.states))),query=rare.diagnostics,
        fit_imputation_mean=rare.fit_imputation_mean.tolist(),imputation_rule='WEIGHTED_SUPPORTED_FIT_COORDINATE_MEAN')
    result=write_products(output,persons,base,records,candidates(),metadata=metadata,rare_audit=rare_audit,
        target_audit=target_audit,max_output_bytes=limits['max_output_bytes'],min_free_bytes=limits['min_free_bytes'])
    require(len(result['candidates'])==len(labels)*(1+len(r['sham_seeds'])),'Incomplete real/SHAM family cannot be frozen')
    auth.verify();rd.verify();store.verify();require(core.sha(path)==request_sha256 and implementation()==r['implementation_sha256'],'Preparation changed during run')
    result=dict(schema='r02_k03_prepared_family_v1',stage='COMPLETE_INPUT_FAMILY_NO_FIT_OR_SELECTION',**result,
        request_sha256=request_sha256,admission_sha256=admission_sha256,implementation_sha256=implementation(),
        evidence_scope=r['evidence_scope'],scientific_spec_sha256=r['scientific_spec_sha256'],
        source_contract_sha256=r['source_contract_sha256'],profile_candidate_keys=list(labels),sham_seeds=r['sham_seeds'],
        evaluate_targets_computed=False,alpha_selection_performed=False,biological_validation=False)
    core.write_json(Path(output)/'manifest.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['request','request-sha256','admission','admission-sha256','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();print(json.dumps(run_request(a.request,a.request_sha256,a.admission,a.admission_sha256,a.output)))
