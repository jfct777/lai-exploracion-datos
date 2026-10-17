#!/usr/bin/env python3
"""Bind paired FLARE outputs to recorded simulator truth on a fixed query grid.

No posterior interpolation, truth-selected positions or donor bootstrap. Stored
haplotype marginals produce explicitly approximate unordered joint states.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
import r02_lai_evaluate as score
import r02_lai_flare_plan as plan
import r02_lai_simulate as sim
from r02_lai_support_review import require,read,write,sha

SCHEMA='r02_lai_flare_evaluation_material_v1'
BASE='BASELINE_EXCLUDING_EVALUATED_DNABR_RARE'
FULL='BASELINE_PLUS_EVALUATED_DNABR_RARE'
ANCESTRY={'African':0,'European':1,'Native_American':2}
STATE=np.asarray([[0,1,2],[1,3,4],[2,4,5]],dtype=np.int8)
INPUTS={'material_manifest','simulation_manifest','recorded_truth','genetic_map',
        'common_query_inventory','common_query_material_manifest',
        'baseline_config','baseline_manifest','baseline_predictions',
        'full_config','full_manifest','full_predictions'}
CODE_FILES=('r02_lai_flare_evaluation_material.py',*score.CODE_FILES,*plan.CODE_FILES)


def descriptor(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def stable_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def joint_states(first,second):
    """Product approximation, preserving marginal copy expectation up to float error."""
    a,b=np.asarray(first,dtype=np.float32).astype(np.float64),np.asarray(second,dtype=np.float32).astype(np.float64)
    require(a.shape==b.shape and a.shape[-1]==3 and np.isfinite(a).all() and np.isfinite(b).all() and
            (a>=0).all() and (a<=1).all() and (b>=0).all() and (b<=1).all(), 'Invalid haplotype probability')
    residual=max(float(np.max(np.abs(a.sum(-1)-1))),float(np.max(np.abs(b.sum(-1)-1))))
    require(residual<=5e-7,'Stored float posterior mass invalid; rounded export is not silently repaired')
    result=np.stack((a[...,0]*b[...,0],a[...,0]*b[...,1]+a[...,1]*b[...,0],
        a[...,0]*b[...,2]+a[...,2]*b[...,0],a[...,1]*b[...,1],
        a[...,1]*b[...,2]+a[...,2]*b[...,1],a[...,2]*b[...,2]),axis=-1)
    return result.astype(np.float32),residual


def query_grid(path,genetic_map,chromosome):
    axis=[]
    with Path(path).open()as f:
        for r in csv.DictReader(f,delimiter='\t'):
            require(r['chromosome']==chromosome and r['channel']=='BASELINE_OTHER','Query must be baseline positions fixed before truth')
            axis.append((int(r['position']),r['ref'],r['alt']))
    pos=np.asarray([r[0]for r in axis],dtype=np.int64)
    require(len(pos)>=2 and (np.diff(pos)>0).all(),'Invalid query position axis')
    bp=[];cm=[]
    with Path(genetic_map).open()as f:
        for line in f:
            fields=line.split()
            require(len(fields)==4 and fields[0]==chromosome,'Explicit chromosome PLINK map required')
            bp.append(int(fields[3]));cm.append(float(fields[2]))
    bp=np.asarray(bp,dtype=np.int64);cm=np.asarray(cm,dtype=np.float64)
    require(len(bp)>=2 and (np.diff(bp)>0).all() and np.isfinite(cm).all() and (np.diff(cm)>=0).all() and
            bp[0]<=pos[0]<=pos[-1]<=bp[-1],'Map does not support the query axis')
    return axis,pos,np.interp(pos,bp,cm)


def truth_arrays(path,samples,positions,chromosome,out,guard):
    """Recorded inclusive ancestry intervals only; uncovered queries stay missing."""
    people={s:i for i,s in enumerate(samples)};shape=(len(samples),len(positions))
    hap=np.lib.format.open_memmap(out/'truth-haplotypes.partial.npy',mode='w+',dtype=np.int8,shape=(*shape,2));hap[:]=-1
    previous={};rows=0
    with gzip.open(path,'rt')as f:
        for r in csv.DictReader(f,delimiter='\t'):
            require(r['sample_id']in people and r['chromosome']==chromosome and r['haplotype']in {'1','2'} and r['ancestry']in ANCESTRY,'Truth person/homolog/chromosome/ancestry differs')
            i,h=people[r['sample_id']],int(r['haplotype'])-1;lo,hi=int(r['start_bp_inclusive']),int(r['end_bp_inclusive'])
            require(0<lo<=hi and lo>previous.get((i,h),0),'Overlapping or unordered recorded truth intervals')
            left,right=np.searchsorted(positions,[lo,hi],side='left');right=int(np.searchsorted(positions,hi,side='right'))
            hap[i,left:right,h]=ANCESTRY[r['ancestry']];previous[i,h]=hi;rows+=1
            if rows%1024==0:guard()
    require(set(previous)=={(i,h)for i in range(len(samples))for h in(0,1)},'Truth lacks a supplied target homolog')
    truth=np.lib.format.open_memmap(out/'truth_state.npy',mode='w+',dtype=np.int8,shape=shape)
    mask=np.lib.format.open_memmap(out/'observed_mask.npy',mode='w+',dtype=np.uint8,shape=shape)
    for i in range(len(samples)):
        mask[i]=np.all(hap[i]>=0,axis=-1);truth[i]=-1;ok=mask[i].astype(bool)
        truth[i,ok]=STATE[hap[i,ok,0],hap[i,ok,1]];guard()
    observed=int(mask.sum(dtype=np.int64));truth.flush();mask.flush();del truth,mask,hap
    (out/'truth-haplotypes.partial.npy').unlink()
    return dict(recorded_intervals=rows,observed_cells=observed,all_query_cells=shape[0]*shape[1],missing_cells=shape[0]*shape[1]-observed)


def probability_array(path,samples,axis,chromosome,out,chunk_positions,guard,dosage_output=None):
    n,m=len(samples),len(axis);arr=np.lib.format.open_memmap(out,mode='w+',dtype=np.float32,shape=(n,m,6))
    block=np.empty((n,min(chunk_positions,m),6),dtype=np.float32);q=0;start=0;last=0;mass=0.;skipped=0;header=False;ancestry=False
    dose=np.lib.format.open_memmap(dosage_output,mode='w+',dtype=np.float32,shape=(n,m,3))if dosage_output is not None else None
    doseblock=np.empty((n,min(chunk_positions,m),3),dtype=np.float32)if dose is not None else None
    with gzip.open(path,'rt')as f:
        for line in f:
            if line.startswith('##'):
                if line.strip()=='##ANCESTRY=<AFR=0,EUR=1,NAM=2>':ancestry=True
                continue
            if line.startswith('#CHROM\t'):
                require(not header and line.rstrip('\n').split('\t')[9:]==samples and ancestry,'Prediction sample/ancestry axes differ');header=True;continue
            r=line.rstrip('\n').split('\t');require(header and len(r)==9+n and r[0]==chromosome,'Malformed prediction axis')
            position=int(r[1]);require(position>last,'Duplicate or unordered prediction position');last=position
            if q==m:skipped+=1;continue
            if position<axis[q][0]:skipped+=1;continue
            require((position,r[3],r[4])==axis[q],'Missing or mismatched common query; posterior interpolation forbidden')
            fmt=r[8].split(':');require(len(set(fmt))==len(fmt) and {'ANP1','ANP2'}<=set(fmt),'Missing posterior format')
            x,y=fmt.index('ANP1'),fmt.index('ANP2');first=[];second=[]
            for cell in r[9:]:
                fields=cell.split(':');require(len(fields)==len(fmt),'Invalid posterior cell')
                first.append([float(v)for v in fields[x].split(',')]);second.append([float(v)for v in fields[y].split(',')])
            joint,residual=joint_states(first,second);mass=max(mass,residual);block[:,q-start]=joint;q+=1
            if dose is not None:doseblock[:,q-1-start]=np.asarray(first,dtype=np.float32)+np.asarray(second,dtype=np.float32)
            if q-start==block.shape[1] or q==m:
                arr[:,start:q]=block[:,:q-start]
                if dose is not None:dose[:,start:q]=doseblock[:,:q-start]
                start=q;guard()
    require(q==m,'Prediction lacks fixed query positions');arr.flush();del arr
    if dose is not None:dose.flush();del dose
    return dict(query_positions=m,extra_prediction_positions_not_scored=skipped,max_haplotype_mass_residual=mass,
        numerical_normalization_in_adapter=False,evaluator_normalizes_six_state_mass_within_5e_minus5=True,
        semantics='PRODUCT_OF_HAPLOTYPE_MARGINALS_APPROXIMATION',phase_accuracy_validated=False)


def prepare(config_path,digest,output):
    started=time.monotonic();c=read(sim.descriptor(dict(path=str(Path(config_path).resolve()),sha256=digest)))
    require(c.get('schema')==SCHEMA and c.get('scope')in {'SYNTHETIC_TEST','CONDITIONAL_DEVELOPMENT'} and
            set(c['inputs'])==INPUTS and c['query_policy']=='COMMON_BASELINE_AFTER_REFERENCE_FILTER_FIXED_BEFORE_TRUTH', 'Explicit paired evaluation preparation required')
    require(c['bootstrap']=={'enabled':False} and c['probability_floor']==1e-12 and c['max_query_gap_cm']in(.01,.005),
            'Preserve registered numerical/support sensitivities; no mosaic bootstrap')
    files={k:sim.descriptor(v)for k,v in c['inputs'].items()};bounds=c['resources'];out=Path(output).resolve()
    require(not out.exists() and out.parent.is_dir(),'Fresh evaluation material directory required');sim.local_filesystem(out.parent)
    def guard():sim.resource_guard(out,dict(timeout_seconds=bounds['timeout_seconds'],disk_reserve_gib=bounds['disk_reserve_gib'],max_output_gib=bounds['max_output_bytes']/2**30),started)
    require(sum(p.stat().st_size for p in files.values())<=bounds['max_input_bytes'],'Input byte admission exceeded')
    material=read(files['material_manifest']);source=read(files['simulation_manifest']);chromosome=str(c['chromosome'])
    require(material['schema']=='r02_lai_flare_material_pair_v1' and material['status']=='PAIRED_PREDICTOR_INPUTS_PREPARED_NO_INFERENCE' and
            material['chromosome']==chromosome and material['inputs_sha256']['simulation_manifest']==sha(files['simulation_manifest']) and
            material['inputs_sha256']['genetic_map']==sha(files['genetic_map']),'Original paired material binding differs')
    require(source['outputs_sha256'].get(files['recorded_truth'].name)==sha(files['recorded_truth']) and source['chromosomes']==[chromosome], 'Recorded truth is not the original simulation product')
    predictions={};configs={};inputs={}
    for label,arm in [('baseline',BASE),('full',FULL)]:
        config=read(files[label+'_config']);manifest=read(files[label+'_manifest']);plan.validate(config)
        alias=label=='full' and material.get('plan_aliases')=={FULL:BASE}
        expected_config=files['baseline_config']if alias else files[label+'_config']
        require(material['outputs_sha256'][arm+'/'+files[label+'_config'].name]==sha(files[label+'_config']) and config['marker_condition']==arm and
                manifest['input_contract_sha256']==sha(expected_config) and manifest['status']=='COMPLETE_TRUTH_BLIND_FLARE_NOT_LAI_VALIDATION' and
                manifest['outputs_sha256']['inference.anc.vcf.gz']==sha(files[label+'_predictions']), 'Model output differs from the original paired input')
        inp={k:sim.descriptor(v)for k,v in config['inputs'].items()};export=plan.validate_export(config,inp)
        require(export['stored_float_roundtrip'] and manifest['probability_export']==export,'Scoring requires explicitly admitted stored-float export')
        require(inp['simulation_manifest']==files['simulation_manifest'] and sha(inp['genetic_map'])==sha(files['genetic_map']), 'Paired model source/map differs')
        configs[label]=config;inputs[label]=inp;predictions[label]=manifest
        if alias:
            require(sha(files['baseline_manifest'])==sha(files['full_manifest']) and sha(files['baseline_predictions'])==sha(files['full_predictions']) and
                    all(sha(inp[k])==sha(inputs['baseline'][k])for k in ('reference_vcf','target_vcf','marker_inventory','roles','sample_map')),
                    'Reused model requires exact byte-identical predictor evidence')
    require(configs['baseline']['parameters']==configs['full']['parameters'] and configs['baseline']['resources']==configs['full']['resources'], 'Paired model parameters or budget differ')
    samples,sentinel=next(sim.vcf_rows(inputs['baseline']['target_vcf']));other,sentinel2=next(sim.vcf_rows(inputs['full']['target_vcf']))
    require(sentinel is sentinel2 is None and samples==other,'Paired target roster/order differs')
    query_source=read(files['common_query_material_manifest'])
    require(query_source['schema']=='r02_lai_flare_material_pair_v1' and query_source['chromosome']==chromosome and
            query_source['reference_frequency_policy']=='FLARE_DOCUMENTED_SEQUENCE_DEFAULTS' and
            query_source['inputs_sha256']['reference_vcf']==material['inputs_sha256']['reference_vcf'] and
            query_source['inputs_sha256']['simulation_manifest']==material['inputs_sha256']['simulation_manifest'] and
            query_source['outputs_sha256'][BASE+'/marker_inventory.tsv']==sha(files['common_query_inventory']),
            'Common query axis must precede truth and come from the same filtered parental material')
    axis,positions,cm=query_grid(files['common_query_inventory'],files['genetic_map'],chromosome)
    complete_count=query_source['source_loci']-query_source['candidate_loci']
    query_coverage=dict(common_queries=len(axis),complete_baseline_queries=complete_count,
        excluded_queries_without_all_four_predictions=complete_count-len(axis),
        original_truth_or_error_used_for_axis=False,posterior_interpolation=False)
    require(len(axis)<=bounds['max_query_positions'] and len(samples)<=bounds['max_people'],'Shape bounds exceeded')
    projected=len(samples)*len(axis)*76+16*2**20
    require(projected<=bounds['max_output_bytes'] and sim.shutil.disk_usage(out.parent).free-projected>=bounds['disk_reserve_gib']*2**30,'Evaluation array disk admission exceeded')
    out.mkdir(mode=0o700);guard()
    with(out/'grid.tsv').open('x')as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(('chromosome','position_bp','genetic_cm'));w.writerows((chromosome,p,g)for p,g in zip(positions,cm))
    units={s:set()for s in samples}
    with inputs['baseline']['roles'].open()as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['role']=='SIMULATED_TARGET':units[r['sample_id']].add(r['atomic_unit_id'])
    require(all(units.values()),'Original donor units absent')
    people=[]
    for sample in samples:
        key=hashlib.sha256(sample.encode()).hexdigest();people.append(dict(sample_key_sha256=key,role='DEVELOPMENT',evaluation_unit='SHARED_DONOR_FOLD_'+source['donor_split'],donor_units=','.join(sorted(units[sample]))))
    with(out/'persons.tsv').open('x')as f:
        w=csv.DictWriter(f,fieldnames=['sample_key_sha256','role','evaluation_unit','donor_units'],delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(people)
    write(out/'roles.json',dict(schema='r02_lai_evaluation_roles_v1',persons={p['sample_key_sha256']:p['role']for p in people}))
    write(out/'split.json',dict(schema='r02_lai_evaluation_split_v1',persons={p['sample_key_sha256']:dict(evaluation_unit=p['evaluation_unit'],donor_units=p['donor_units'].split(','))for p in people}))
    truth_audit=truth_arrays(files['recorded_truth'],samples,positions,chromosome,out,guard)
    paths={k:out/n for k,n in [('grid','grid.tsv'),('persons','persons.tsv'),('roles','roles.json'),('split','split.json'),('truth_state','truth_state.npy'),('observed_mask','observed_mask.npy')]}
    descriptors={k:descriptor(v)for k,v in paths.items()};binding={k+'_sha256':descriptors[k]['sha256']for k in ('grid','persons','roles','split','observed_mask')}
    evidence='FIXTURE'if c['scope']=='SYNTHETIC_TEST'else'SIMULATED_RECORDED_TRUTH'
    truth=dict(schema='r02_lai_evaluation_truth_v1',evidence_kind=evidence,truth_origin='SYNTHETIC_FIXTURE'if evidence=='FIXTURE'else'SIMULATOR_RECORDED_ANCESTRY',truth_state_sha256=descriptors['truth_state']['sha256'],recorded_truth=descriptor(files['recorded_truth']),simulation_manifest=descriptor(files['simulation_manifest']),**binding)
    write(out/'truth-provenance.json',truth);descriptors['truth_provenance']=descriptor(out/'truth-provenance.json')
    rows=[];audits={}
    for label in ('baseline','full'):
        probability=out/(label+'.npy');dosage=out/(label+'.marginal-dosage.npy');audits[label]=probability_array(files[label+'_predictions'],samples,axis,chromosome,probability,bounds['chunk_positions'],guard,dosage_output=dosage)
        receipt=dict(schema='r02_lai_evaluation_prediction_v1',evidence_kind=evidence,state_names=list(score.STATES),probabilities_sha256=sha(probability),probability_semantics='PRODUCT_OF_HAPLOTYPE_MARGINALS_APPROXIMATION',budget_sha256=stable_hash(dict(parameters=configs[label]['parameters'],resources=configs[label]['resources'])),reference_panel_sha256=material['inputs_sha256']['reference_vcf'],source_prediction_sha256=sha(files[label+'_predictions']),source_model_manifest_sha256=sha(files[label+'_manifest']),audit=audits[label],**binding)
        receipt.update(expected_dosage_sha256=sha(dosage),dosage_semantics='HAPLOTYPE_MARGINAL_SUM_NO_INDEPENDENCE')
        write(out/(label+'.receipt.json'),receipt);rows.append(dict(name=label,probabilities=descriptor(probability),expected_dosage=descriptor(dosage),receipt=descriptor(out/(label+'.receipt.json'))))
    evaluation=dict(schema='r02_lai_evaluate_input_v1',usage_scope='DEVELOPMENT_ONLY',evidence_kind=evidence,ancestry_names=list(score.ANCESTRIES),state_names=list(score.STATES),boundary_tolerances_cm=list(score.TOLERANCES),primary_boundary_tolerance_cm=.2,source_test_opened=False,grid_selected_from_truth=False,max_query_gap_cm=c['max_query_gap_cm'],evaluation_roles=['DEVELOPMENT'],inputs=descriptors,predictions=rows,code_sha256={name:sha(Path(__file__).with_name(name))for name in score.CODE_FILES},probability_floor=c['probability_floor'],emit_expected_dosages=False,comparisons=[dict(candidate='full',comparator='baseline',require_matched_budget=True,require_matched_reference=True)],bootstrap=c['bootstrap'],harm_tolerances=[],resources=c['evaluation_resources'])
    write(out/'evaluation-input.json',evaluation);score.load_contract(out/'evaluation-input.json',sha(out/'evaluation-input.json'))
    for d in c['inputs'].values():sim.descriptor(d)
    guard();result=dict(schema=SCHEMA,status='PAIRED_EVALUATION_INPUTS_PREPARED_NO_SCIENTIFIC_SELECTION',evidence_kind=evidence,people=len(samples),query_positions=len(axis),query_coverage=query_coverage,truth_audit=truth_audit,probability_audits=audits,source_test_opened=False,bootstrap_enabled=False,posterior_interpolation=False,population_generalization_authorized=False,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,inputs_sha256={k:v['sha256']for k,v in c['inputs'].items()},code_sha256={n:sha(Path(__file__).with_name(n))for n in set(CODE_FILES)},outputs_sha256={p.name:sha(p)for p in out.iterdir()if p.is_file()})
    write(out/'manifest.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--contract-sha256',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();os.umask(0o077);prepare(a.contract,a.contract_sha256,a.output_dir)
if __name__=='__main__':main()
