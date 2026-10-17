#!/usr/bin/env python3
"""Plan and run the local common-combined C/R+C branch after source staging.

The controller performs no cloud action.  It freezes 88 C requests, runs their
Nextflow workflow, freezes 3 lazy R+C weights per verified C result, runs that
workflow, and writes create-only task/phase receipts for the cloud wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

COMMON_SCHEMA='r02_structure_common_family_request_v1'
COMPOSITE_SCHEMA='r02_structure_lazy_composite_request_v1'
BASE_CODE_FILES=('r02_structure_affinity_io.py','r02_structure_discrete.py',
    'r02_structure_materialize.py','r02_structure_increment.py',
    'r02_structure_input_plan.py','r02_genomic_pair_evidence.py')
COMMON_CODE_FILES=('r02_structure_common_affinities.py','r02_structure_continuous.py',*BASE_CODE_FILES)
COMPOSITE_CODE_FILES=('r02_structure_composite_affinities.py',*BASE_CODE_FILES)
WEIGHTS=(.25,.5,.75)
BRANCH='common-combined';PHASE='materialize-c-rc';EXPECTED_VIEWS=88


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open()as stream:return json.load(stream)


def implementation_hashes(code_dir,names):
    code_dir=Path(code_dir).resolve()
    result={name:sha(code_dir/name)for name in names}
    require(all((code_dir/name).is_file()for name in names),'Frozen family implementation is incomplete')
    return result


def safe_id(value):
    require(isinstance(value,str)and re.fullmatch(r'[A-Za-z0-9_-]+',value),'Unsafe view/task identifier')
    return value


def write_exact(path,value):
    path=Path(path);text=json.dumps(value,indent=2,allow_nan=False,sort_keys=True)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(path.read_text()==text,'Existing frozen branch artifact differs: '+str(path))
    else:
        with path.open('x')as stream:stream.write(text)
        path.chmod(0o600)
    return sha(path)


def descriptor(path):
    path=Path(path).resolve();require(path.is_file(),'Missing staged source: '+str(path))
    return dict(path=str(path),sha256=sha(path))


def view_key(manifest):
    block=safe_id(manifest['block_id']);scope=safe_id(manifest['exclusion_scope'])
    return block+'__'+scope


def common_sources(root):
    found={}
    for path in sorted(Path(root).resolve().rglob('manifest.json')):
        m=load_json(path)
        if m.get('schema')!='r02_structure_materialized_common_merge_v2':continue
        require(m['stage']=='FULL_NUMERICAL_COMMON_SPECTRUM_NOT_PREDICTIVE_SELECTION'
                and set(m['outputs_sha256'])=={'continuous.private.npz'},'Invalid staged common merge')
        arrays=path.parent/'continuous.private.npz'
        require(arrays.is_file()and sha(arrays)==m['outputs_sha256'][arrays.name],
                'Staged common arrays differ from manifest')
        key=view_key(m);require(key not in found,'Repeated staged common view: '+key)
        found[key]=dict(directory=str(path.parent),manifest_sha256=sha(path),manifest=m)
    require(len(found)==EXPECTED_VIEWS,f'Expected {EXPECTED_VIEWS} staged common views, found {len(found)}')
    return found


def rare_sources(root):
    found={}
    for path in sorted(Path(root).resolve().rglob('affinity_inputs.json')):
        m=load_json(path)
        if m.get('schema')not in {'r02_structure_rare_affinity_inputs_v1','r02_structure_discrete_reuse_v1'}:continue
        key=safe_id(m['block_id'])+'__'+safe_id(m['exclusion_scope'])
        require(key not in found,'Repeated staged rare view: '+key)
        names={'persons.tsv','fit_affinities.private.tsv','query_affinities.private.tsv',
               'exclusion.json','affinity_inputs.json'}
        inputs={name:descriptor(path.parent/name)for name in names}
        for name in names-{'affinity_inputs.json'}:
            require(m['sha256'][name]==inputs[name]['sha256'],'Staged rare source differs: '+key+'/'+name)
        found[key]=dict(inputs=inputs,manifest=m)
    require(len(found)==EXPECTED_VIEWS,f'Expected {EXPECTED_VIEWS} staged rare views, found {len(found)}')
    return found


def roster(bundle,common_catalog):
    bundle=Path(bundle).resolve();manifest=descriptor(bundle/'materialize.json');persons=descriptor(bundle/'persons.tsv')
    contract=load_json(manifest['path'])
    require(contract['roles_state']=='EXPLICITLY_FROZEN'
            and contract['metadata_sha256']['persons.tsv']==persons['sha256'],
            'Frozen bundle roster binding differs')
    require(all(entry['manifest']['source_contract_sha256']==manifest['sha256']
                for entry in common_catalog.values()),'Common views and shared original bundle differ')
    return dict(manifest=manifest,persons=persons)


def plan_c(common_root,bundle,spec_sha,code_dir,requests_dir,batch_path):
    catalog=common_sources(common_root);source_contract=roster(bundle,catalog)
    implementation=implementation_hashes(code_dir,COMMON_CODE_FILES);tasks=[];requests_dir=Path(requests_dir)
    for key,entry in sorted(catalog.items()):
        request=dict(schema=COMMON_SCHEMA,evidence_scope='DNABR_R02',scientific_spec_sha256=spec_sha,
            implementation_sha256=implementation,family='C',mix_weight=None,normalization_policy=None,
            source_contract=source_contract,
            common=dict(directory=entry['directory'],manifest_sha256=entry['manifest_sha256']),rare=None,
            resources=dict(max_input_bytes=1073741824,max_metadata_bytes=16777216,
                max_memory_bytes=1879048192,max_output_bytes=1073741824,min_free_bytes=1073741824))
        path=requests_dir/(key+'.json');pin=write_exact(path,request)
        tasks.append(dict(id=safe_id(key),request=str(path.resolve()),sha256=pin))
    batch=dict(schema='r02_structure_common_family_batch_v1',implementation_sha256=implementation,tasks=tasks)
    return write_exact(batch_path,batch),catalog


def family_inputs(directory):
    directory=Path(directory).resolve();names={'persons.tsv','fit_affinities.private.tsv',
        'query_affinities.private.tsv','affinity_inputs.json'}
    require(all((directory/name).is_file()for name in names),'Incomplete C family output: '+str(directory))
    inputs={name:descriptor(directory/name)for name in names};m=load_json(directory/'affinity_inputs.json')
    require(m['family']=='C'and m['scientific_admission']is False
            and m['sha256']=={name:inputs[name]['sha256']for name in names-{'affinity_inputs.json'}},
            'C family output pins/status differ')
    return inputs,m


def plan_rc(c_results,rare_root,spec_sha,code_dir,requests_dir,batch_path):
    rare=rare_sources(rare_root);c_found={}
    for path in sorted(Path(c_results).resolve().rglob('affinity_inputs.json')):
        m=load_json(path)
        if m.get('family')!='C':continue
        key=safe_id(m['territory']['block_id'])+'__'+safe_id(m['territory']['exclusion_scope'])
        require(key not in c_found,'Repeated C result: '+key);c_found[key]=family_inputs(path.parent)
    require(set(c_found)==set(rare)and len(c_found)==EXPECTED_VIEWS,'C results and staged R views differ')
    implementation=implementation_hashes(code_dir,COMPOSITE_CODE_FILES);tasks=[];requests_dir=Path(requests_dir)
    for key in sorted(c_found):
        c_inputs,c_manifest=c_found[key];r=rare[key]
        require(c_manifest['territory']['source_contract_sha256']==r['manifest']['source_contract_sha256'],
                'C/R source contract differs: '+key)
        grouped=[]
        for weight in WEIGHTS:
            label=f'w{int(weight*100):03d}';task_id=safe_id(key+'__'+label)
            request=dict(schema=COMPOSITE_SCHEMA,evidence_scope='DNABR_R02',scientific_spec_sha256=spec_sha,
                implementation_sha256=implementation,mix_weight=weight,
                normalization_policy='COMPONENT_PAIR_EQUAL_POSITIVE_MEAN',
                common_inputs=c_inputs,rare_inputs=r['inputs'],
                resources=dict(max_input_bytes=1073741824,max_metadata_bytes=16777216))
            path=requests_dir/(task_id+'.json');pin=write_exact(path,request)
            grouped.append(dict(id=task_id,path=str(path.resolve()),sha256=pin))
        set_path=requests_dir/'_sets'/(key+'.json')
        set_pin=write_exact(set_path,dict(schema='r02_structure_lazy_composite_request_set_v1',
            territory_id=key,requests=grouped))
        tasks.append(dict(id=key,request_set=str(set_path.resolve()),sha256=set_pin))
    require(len(tasks)==EXPECTED_VIEWS and sum(len(load_json(t['request_set'])['requests']) for t in tasks)
            ==EXPECTED_VIEWS*len(WEIGHTS),'Lazy R+C grouped request count differs')
    batch=dict(schema='r02_structure_lazy_composite_batch_v1',implementation_sha256=implementation,tasks=tasks)
    return write_exact(batch_path,batch)


def run_nextflow(executable,workflow,config,work,batch,batch_pin,code,output,forks,family_stage):
    command=[executable,'-C',str(Path(config).resolve()),'run',str(Path(workflow).resolve()),'-resume',
        '-work-dir',str(Path(work).resolve()),'--batch',str(Path(batch).resolve()),'--batch_sha256',batch_pin,
        '--code_dir',str(Path(code).resolve()),'--output_dir',str(Path(output).resolve()),
        '--memory_gb','2','--max_forks',str(forks),'--task_time','30m','--family_stage',family_stage]
    command.extend(['--expected_tasks',str(EXPECTED_VIEWS)])
    completed=subprocess.run(command,check=False)
    require(completed.returncode==0,'Nextflow branch phase failed: '+' '.join(command))
    return command


def emit_task_receipts(root,family,requests,results,pattern,*,strict=True):
    root=Path(root).resolve();markers=root/'branches'/BRANCH/'task-completions'/PHASE
    requests=Path(requests);results=Path(results)
    request_by_id={p.stem:p for p in requests.glob('*.json')};records={};request_records=[];errors=[]
    for directory in sorted(results.glob(pattern)):
        try:
            task_id=directory.name.split('_',1)[1]
            require(task_id in request_by_id,'Unexpected family result: '+task_id)
            request=request_by_id[task_id];request_body=load_json(request)
            if family=='C':
                family_inputs(directory)
            else:
                files_here=list(directory.iterdir())
                require(len(files_here)==1 and files_here[0].name=='composite_affinity.json',
                        'Incomplete lazy R+C output: '+task_id)
                value=load_json(files_here[0])
                require(value.get('schema')=='r02_structure_lazy_composite_affinity_v1'
                        and value.get('stage')=='LAZY_COMPOSITE_READY_NOT_SCIENTIFICALLY_ADMITTED'
                        and value.get('mix_weight')==request_body['mix_weight']
                        and value.get('normalization_policy')==request_body['normalization_policy']
                        and value.get('common_inputs')==request_body['common_inputs']
                        and value.get('rare_inputs')==request_body['rare_inputs']
                        and value.get('capacity',{}).get('materialized_pair_rows')==0
                        and value.get('scientific_admission') is False
                        and value.get('biological_validation') is False,
                        'Lazy R+C output/request pins or status differ: '+task_id)
            files=sorted(p for p in directory.iterdir()if p.is_file())
            outputs=[]
            for path in files:
                relative=path.relative_to(root/'branches'/BRANCH/'results')
                outputs.append(dict(path=str(Path('branches')/BRANCH/'results'/relative),
                                    sha256=sha(path),bytes=path.stat().st_size))
            marker=dict(schema='r02_family_task_complete_v1',status='COMPLETE',branch=BRANCH,
                        phase=PHASE,task_id=task_id,outputs=outputs)
            marker_path=markers/(task_id+'.json');marker_sha=write_exact(marker_path,marker)
            records[task_id]=marker_sha
            request_records.append(dict(task_id=task_id,family=family,path=str(request),
                                        sha256=sha(request),status='COMPLETE'))
        except Exception as error:
            errors.append(f'{directory.name}: {error}')
    require(not strict or not errors,'Invalid/incomplete family outputs: '+'; '.join(errors))
    return records,request_records


def finish_phase(root,records,request_records,commands):
    root=Path(root).resolve()
    require(len(records)==EXPECTED_VIEWS*(1+len(WEIGHTS))==352,'Final branch task count differs')
    phase=dict(schema='r02_family_phase_complete_v1',status='COMPLETE',branch=BRANCH,phase=PHASE,tasks=records)
    phase_path=root/'branches'/BRANCH/'phase-completions'/(PHASE+'.json');phase_sha=write_exact(phase_path,phase)
    receipt=dict(schema='r02_common_combined_final_receipt_v1',status='COMPLETE',branch=BRANCH,phase=PHASE,
        phase_manifest=str(phase_path),phase_manifest_sha256=phase_sha,requests=sorted(request_records,key=lambda v:v['task_id']),
        task_manifests=records,commands=commands,scientific_admission=False,biological_validation=False)
    write_exact(root/'branches'/BRANCH/'FINAL_RECEIPT.json',receipt)


def final_receipts(root,c_requests,rc_requests,c_results,rc_results,commands):
    c_records,c_requests_done=emit_task_receipts(root,'C',c_requests,c_results,'affinity_*')
    rc_records,rc_requests_done=emit_task_receipts(root,'RC',rc_requests,rc_results,'composite_*')
    require(len(c_records)==EXPECTED_VIEWS and len(rc_records)==EXPECTED_VIEWS*len(WEIGHTS),
            'Missing family outputs; phase completion forbidden')
    finish_phase(root,{**c_records,**rc_records},c_requests_done+rc_requests_done,commands)


def first_existing(candidates,label):
    for candidate in candidates:
        if Path(candidate).exists():return str(Path(candidate).resolve())
    raise ValueError('Missing staged '+label+': '+', '.join(map(str,candidates)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime-root',required=True);p.add_argument('--phase',required=True)
    for name in ('common-root','rare-root','bundle','scientific-spec','code-dir','workflow','nextflow-config'):
        p.add_argument('--'+name)
    p.add_argument('--nextflow');a=p.parse_args();root=Path(a.runtime_root).resolve()
    require(a.phase==PHASE,'Controller supports exactly phase '+PHASE)
    a.nextflow=a.nextflow or str(root/'nextflow')
    require(Path(a.nextflow).is_file(),'Pinned runtime Nextflow wrapper required')
    version=subprocess.run([a.nextflow,'-version'],check=False,text=True,capture_output=True)
    require(version.returncode==0 and 'version 26.04.6' in version.stdout+version.stderr,
            'Pinned Nextflow 26.04.6 required')
    frozen=root/'frozen'/BRANCH
    a.common_root=a.common_root or first_existing(
        [root/'sources/common/published/targets',root/'sources/common'],'common root')
    a.rare_root=a.rare_root or first_existing(
        [root/'sources/rare/published/affinities/views',root/'sources/rare'],'rare root')
    a.bundle=a.bundle or first_existing([root/'sources/common/bundle',root/'bundle',Path('/bundle')],'original bundle')
    a.scientific_spec=a.scientific_spec or str(frozen/'COMMON_FAMILY_SPEC.md')
    a.code_dir=a.code_dir or str(frozen/'code');a.workflow=a.workflow or str(frozen/'workflow.nf')
    a.nextflow_config=a.nextflow_config or str(frozen/'runtime.config')
    require(all(Path(value).exists()for value in (a.scientific_spec,a.code_dir,a.workflow,a.nextflow_config)),
            'Frozen common-combined payload is incomplete')
    spec=Path(a.scientific_spec).resolve();require(spec.is_file(),'Missing family specification');spec_sha=sha(spec)
    base=root/'branches'/BRANCH;requests=base/'requests';results=base/'results';work=base/'work'
    for directory in (requests/'C',requests/'RC',results/'C',results/'RC',work/'C',work/'RC'):directory.mkdir(parents=True,exist_ok=True)
    c_batch=requests/'c-batch.json';c_pin,_=plan_c(a.common_root,a.bundle,spec_sha,a.code_dir,requests/'C',c_batch)
    commands=[]
    try:
        commands.append(run_nextflow(a.nextflow,a.workflow,a.nextflow_config,work/'C',c_batch,c_pin,a.code_dir,results/'C',4,'C'))
    finally:
        c_records,c_done=emit_task_receipts(root,'C',requests/'C',results/'C','affinity_*',strict=False)
    require(len(c_records)==EXPECTED_VIEWS,'Incomplete C phase; R+C must not start')
    rc_batch=requests/'rc-batch.json';rc_pin=plan_rc(results/'C',a.rare_root,spec_sha,a.code_dir,requests/'RC',rc_batch)
    try:
        commands.append(run_nextflow(a.nextflow,a.workflow,a.nextflow_config,work/'RC',rc_batch,rc_pin,a.code_dir,results/'RC',3,'RC'))
    finally:
        rc_records,rc_done=emit_task_receipts(root,'RC',requests/'RC',results/'RC','composite_*',strict=False)
    require(len(rc_records)==EXPECTED_VIEWS*len(WEIGHTS),'Incomplete R+C phase; completion forbidden')
    finish_phase(root,{**c_records,**rc_records},c_done+rc_done,commands)


if __name__=='__main__':main()
