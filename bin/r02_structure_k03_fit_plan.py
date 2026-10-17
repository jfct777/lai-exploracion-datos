#!/usr/bin/env python3
"""Freeze the complete K03 model inventory before fitting; never select a subset.

No genotype arrays, model fits or human admissions are produced. An explicit
resource cap fails the entire plan, preserving a count preview for review.
"""
from pathlib import Path
import csv
import hashlib
import json
import math


def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def validate_alphas(alphas):
    require(isinstance(alphas,list)and alphas and len(set(alphas))==len(alphas)
        and all(type(a)in(int,float)and math.isfinite(a)and a>0 for a in alphas),'Explicit distinct positive alpha family required')


def person_axis(path,evidence_scope):
    with Path(path).open(encoding='utf-8',newline='')as handle:
        reader=csv.DictReader(handle,delimiter='\t');rows=list(reader)
    require(reader.fieldnames==['sample_id','component','role']and rows,
        'Exact nonempty person role table required')
    require(len({r['sample_id']for r in rows})==len(rows)and
        all(r['sample_id']and r['component']and r['role']in{'FIT','SELECT','EVALUATE'}for r in rows),
        'Person identities/components/roles are invalid')
    counts={role:sum(r['role']==role for r in rows)for role in('FIT','SELECT','EVALUATE')}
    if evidence_scope=='DNABR_R02':
        require(len(rows)==2619 and counts=={'FIT':1460,'SELECT':476,'EVALUATE':683},
            'Human K03 roster or fixed roles differ')
    return dict(persons_sha256=sha(path),person_count=len(rows),role_counts=counts)


def validate_territory(territory,block_id,scope):
    require(set(territory)=={'block_id','target','excluded','allowed_input_intervals','allowed_bp','excluded_bp'}
        and territory['block_id']==block_id and scope in{'BLOCK_AND_LD_MARGIN','WHOLE_CHROMOSOME'},
        'Exact block/exclusion territory required')
    target=territory['target'];excluded=territory['excluded'];allowed=territory['allowed_input_intervals']
    def valid_interval(value):
        return (isinstance(value,list)and len(value)==3 and isinstance(value[0],str)and value[0]
            and type(value[1])is int and type(value[2])is int and 0<value[1]<=value[2])
    require(valid_interval(target)and valid_interval(excluded)
        and isinstance(allowed,list)and allowed and all(valid_interval(v)for v in allowed)
        and all(type(v)is int and v>=0 for v in territory.values()if type(v)is int),
        'Malformed block/exclusion territory')
    require(target[0]==excluded[0]and target[1]>=excluded[1]and target[2]<=excluded[2]
        and territory['allowed_bp']>0 and territory['excluded_bp']>0,
        'Excluded territory does not contain the target block')
    require(all(v[0]!=excluded[0]or v[2]<excluded[1]or v[1]>excluded[2]for v in allowed),
        'Allowed and excluded territories overlap')
    if scope=='WHOLE_CHROMOSOME':
        require(all(v[0]!=target[0]for v in allowed),'Whole-chromosome exclusion leaks its target chromosome')
    else:
        require(any(v[0]==target[0]for v in allowed),'Block-and-margin exclusion removed the whole target chromosome')
    return digest(territory)


def validate_human_views(views):
    """Require the complete paired human universe before emitting any fit request."""
    require(len(views)==88,'Human K03 fit planning requires all 88 predeclared views')
    blocks={v['block_id']for v in views}
    require(len(blocks)==44,'Human K03 fit planning requires all 44 reserved blocks')
    for block in blocks:
        pair=[v for v in views if v['block_id']==block]
        require({v['exclusion_scope']for v in pair}=={'BLOCK_AND_LD_MARGIN','WHOLE_CHROMOSOME'}
            and len(pair)==2,'Each human block requires both exclusion geometries exactly once')
        require(len({v['territory_identity']for v in pair})==2,
            'Paired human exclusion geometries cannot share a territory')
    require(len({v['persons_sha256']for v in views})==1 and len({v['source_contract_sha256']for v in views})==1,
        'Human K03 views use incompatible person axes or source contracts')


def model_inventory(views,alphas,scientific_spec_sha256):
    """Canonical exact reuse; statistical group membership is separate from IO."""
    validate_alphas(alphas)
    models={};groups=[];required=[];seen=set()
    for view in views:
        base=view['base'];candidates=view['candidates'];key=(view['block_id'],view['exclusion_scope'])
        require(key not in seen and candidates,'Unique nonempty input view required');seen.add(key)
        required.append(dict(block_id=key[0],exclusion_scope=key[1]));grouped={}
        for candidate in candidates:
            group=(candidate['control'],candidate['sham_seed'])
            entries=grouped.setdefault(group,{arm:set()for arm in ('CONTINUOUS','COMMUNITY_A','COMMUNITY_B')})
            for alpha in alphas:
                bid='m_'+digest(['CONTINUOUS',candidate['baseline_identity'],float(alpha)])
                if bid not in models:
                    models[bid]=dict(model_id=bid,arm='CONTINUOUS',candidate_id=candidate['candidate_id'],alpha=float(alpha),
                        block_id=key[0],exclusion_scope=key[1],baseline_identity=candidate['baseline_identity'],
                        select_target_identity=view['select_target_identity'],inputs_manifest_sha256=base['manifest_sha256'],
                        inputs=dict(base=base,candidate=candidate['descriptor'],arm='CONTINUOUS'),baseline_model_id=None)
                entries['CONTINUOUS'].add(bid)
                for arm in ('COMMUNITY_A','COMMUNITY_B'):
                    mid='m_'+digest([arm,base['manifest_sha256'],candidate['candidate_id'],float(alpha)])
                    require(mid not in models,'Duplicate candidate in the declared family')
                    models[mid]=dict(model_id=mid,arm=arm,candidate_id=candidate['candidate_id'],alpha=float(alpha),
                        block_id=key[0],exclusion_scope=key[1],baseline_identity=candidate['baseline_identity'],
                        select_target_identity=view['select_target_identity'],inputs_manifest_sha256=base['manifest_sha256'],
                        inputs=dict(base=base,candidate=candidate['descriptor'],arm=arm),baseline_model_id=bid)
                    entries[arm].add(mid)
        for (control,seed),entries in grouped.items():
            groups.append(dict(block_id=key[0],exclusion_scope=key[1],control=control,sham_seed=seed,
                models_by_arm={a:sorted(ids)for a,ids in entries.items()}))
    return dict(schema='r02_k03_family_inventory_v1',scientific_spec_sha256=scientific_spec_sha256,evaluate_allowed=False,
        models=list(models.values()),required_views=required,selection_groups=groups,alphas=alphas,
        selection_performed=False,global_partition_alignment_assumed=False,
        baseline_group_rule='EXACT_IO_REUSE_GROUP_ELIGIBILITY_FROM_SELECTION_GROUPS_NOT_REPRESENTATIVE_LABEL')


def prepare(config_path,config_sha256,output):
    require(sha(config_path)==config_sha256,'Fit plan configuration changed')
    config=json.loads(Path(config_path).read_text())
    require(set(config)=={'schema','evidence_scope','scientific_spec_sha256','families','fit_code','alphas','parameters',
        'fit_resources','resources'}and config['schema']=='r02_k03_fit_plan_request_v1','Exact explicit fit plan required')
    require(config['evidence_scope']in{'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'},'Explicit evidence scope required')
    limits=config['resources'];require(set(limits)=={'max_metadata_bytes','max_input_bytes','max_models','max_output_bytes'}
        and all(type(x)is int and x>0 for x in limits.values()),'Explicit fit planner limits required')
    pins={};total=0
    def pin(path,h,metadata=False):
        nonlocal total
        p=Path(path).resolve();limit=limits['max_metadata_bytes']if metadata else limits['max_input_bytes']
        require(p.is_file()and p.stat().st_size<=limit and sha(p)==h,'Prepared family changed or byte bound exceeded')
        if str(p)not in pins:total+=p.stat().st_size
        require(total<=limits['max_input_bytes'],'Fit planner aggregate byte bound exceeded');pins[str(p)]=h;return p
    code=config['fit_code'];require(set(code)=={'directory','implementation_sha256'},'Exact fitted implementation required')
    for name,h in code['implementation_sha256'].items():
        require(Path(name).name==name and name.startswith('r02_')and name.endswith('.py'),'Unsafe fit source name');pin(Path(code['directory'])/name,h)
    views=[]
    for descriptor in config['families']:
        require(set(descriptor)=={'directory','manifest_sha256'},'Exact prepared family descriptor required')
        root=Path(descriptor['directory']).resolve();fm=json.loads(pin(root/'manifest.json',descriptor['manifest_sha256'],True).read_text())
        require(fm['schema']=='r02_k03_prepared_family_v1'and fm['stage']=='COMPLETE_INPUT_FAMILY_NO_FIT_OR_SELECTION'
            and fm['scientific_spec_sha256']==config['scientific_spec_sha256']and fm['evaluate_targets_computed']is False
            and fm['alpha_selection_performed']is False,'Complete unselected FIT/SELECT input family required')
        bp=pin(root/'base/manifest.json',fm['base']['manifest_sha256'],True);bm=json.loads(bp.read_text())
        require(bm['evidence_scope']==config['evidence_scope']and bm['scientific_spec_sha256']==config['scientific_spec_sha256'],'Prepared base population differs')
        require(fm['base']['directory']==str((root/'base').resolve()),'Prepared family/base directory differs')
        if config['evidence_scope']=='DNABR_R02':
            require(all(fm.get(k)==bm.get(k)for k in('evidence_scope','scientific_spec_sha256','source_contract_sha256')),
                'Prepared family/base source identity differs')
            source_identity=bm.get('source_contract_sha256')
            require(isinstance(source_identity,str)and len(source_identity)==64
                and all(c in '0123456789abcdef'for c in source_identity),
                'Human prepared source identity is missing or malformed')
        for name,h in bm['outputs_sha256'].items():
            require(Path(name).name==name,'Unsafe base file');pin(root/'base'/name,h,not name.endswith('.npz'))
        axis=person_axis(root/'base/persons.tsv',config['evidence_scope'])
        territory_identity=(validate_territory(bm['exclusion_territory'],bm['block_id'],bm['exclusion_scope'])
            if config['evidence_scope']=='DNABR_R02'or'exclusion_territory'in bm else None)
        candidates=[]
        for row in fm['candidates']:
            key=row['candidate_id'];require(len(key)==64 and all(c in '0123456789abcdef'for c in key),'Unsafe candidate key')
            directory=root/'candidates'/key;mp=pin(directory/'manifest.json',row['manifest_sha256'],True);cm=json.loads(mp.read_text())
            require(cm['candidate_id']==key and cm['base_manifest_sha256']==fm['base']['manifest_sha256']
                and cm['baseline_identity']==row['baseline_identity']and cm['control']==row['control']and cm['sham_seed']==row['sham_seed']
                and all(cm.get(k)==bm.get(k)for k in('evidence_scope','scientific_spec_sha256','block_id','exclusion_scope')),
                'Prepared candidate identity differs')
            for name,h in cm['outputs_sha256'].items():
                require(Path(name).name==name,'Unsafe membership file');pin(directory/name,h)
            candidates.append(dict(candidate_id=key,baseline_identity=cm['baseline_identity'],control=cm['control'],sham_seed=cm['sham_seed'],
                descriptor=dict(directory=str(directory),manifest_sha256=row['manifest_sha256'])))
        expected={(key,'REAL',None)for key in fm['profile_candidate_keys']}|{
            (key,'SHAM_WITHIN_FIT_COMPONENT',seed)for key in fm['profile_candidate_keys']for seed in fm['sham_seeds']}
        actual={(row['parent_candidate_id'],row['control'],row['sham_seed'])for row in fm['candidates']}
        require(actual==expected and len(actual)==len(fm['candidates']),'Complete original/SHAM family required')
        views.append(dict(base=dict(directory=str(root/'base'),manifest_sha256=fm['base']['manifest_sha256']),candidates=candidates,
            block_id=bm['block_id'],exclusion_scope=bm['exclusion_scope'],select_target_identity=bm['select_target_identity'],
            territory_identity=territory_identity,source_contract_sha256=bm.get('source_contract_sha256'),**axis))
    if config['evidence_scope']=='DNABR_R02':validate_human_views(views)
    validate_alphas(config['alphas'])
    count_baseline=sum(len({v['baseline_identity']for v in view['candidates']})for view in views)*len(config['alphas'])
    count_extension=sum(len(view['candidates'])for view in views)*len(config['alphas'])
    expected_count=count_baseline+2*count_extension
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    preview=dict(status='COMPLETE_FAMILY_SIZE_PREVIEW_NOT_ADMISSION',models=expected_count,views=len(views),
        model_counts_by_arm=dict(CONTINUOUS=count_baseline,COMMUNITY_A=count_extension,COMMUNITY_B=count_extension),
        max_models=limits['max_models'],config_sha256=config_sha256,genotype_arrays_loaded=False,models_fitted=False)
    write(out/'preview.json',preview)
    require(expected_count<=limits['max_models'],'Complete family exceeds resource cap; no truncation or model requests generated')
    family=model_inventory(views,config['alphas'],config['scientific_spec_sha256'])
    require(len(family['models'])==expected_count,'Canonical model count differs from preview')
    (out/'requests').mkdir();model_rows=[]
    for row in family['models']:
        request=dict(schema='r02_k03_fit_select_request_v2',evidence_scope=config['evidence_scope'],scientific_spec_sha256=config['scientific_spec_sha256'],
            inputs=row['inputs'],implementation_sha256=code['implementation_sha256'],alpha=row['alpha'],parameters=config['parameters'],resources=config['fit_resources'])
        rp=out/'requests'/(row['model_id']+'.json');write(rp,request);row['request_sha256']=sha(rp)
        model_rows.append(dict(id=row['model_id'],request=str(rp.resolve()),request_sha256=row['request_sha256']))
    write(out/'family.json',family)
    write(out/'unadmitted-models.json',dict(schema='r02_k03_unadmitted_model_inventory_v1',models=model_rows,
        family_sha256=sha(out/'family.json'),implementation_sha256=code['implementation_sha256'],scientific_spec_sha256=config['scientific_spec_sha256'],
        status='EXTERNAL_ADMISSION_REQUIRED_NO_ADMISSIONS_GENERATED'))
    require(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())<=limits['max_output_bytes'],'Fit plan output limit exceeded')
    require(sha(config_path)==config_sha256 and all(sha(p)==h for p,h in pins.items()),'Fit plan source changed')
    result=dict(status='COMPLETE_REQUEST_FAMILY_PREPARED_NOT_ADMITTED_OR_FITTED',**{k:preview[k]for k in ('models','views','model_counts_by_arm')},
        config_sha256=config_sha256,family_sha256=sha(out/'family.json'),source_pins=pins,
        outputs_sha256={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()},
        genotype_arrays_loaded=False,models_fitted=False,admissions_generated=False)
    write(out/'manifest.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.config,a.config_sha256,a.output)))
