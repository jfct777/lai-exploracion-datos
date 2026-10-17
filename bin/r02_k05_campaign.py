#!/usr/bin/env python3
"""Bounded K05 partition queue, authenticated publication and family aggregation.

Scientific calculations remain in the frozen Nextflow workflow. Three serial
slots never dispatch the reused pilot, alter family size, or retry failed human
fits automatically. Cloud object mutation uses the shared create-only helper;
retirement uses the shared original-ID gate after a stopped worker is archived.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import threading
import time

import r02_autosome_pipeline as pipeline
import r02_publish_evidence as publication
import r02_retirement_gate as retirement

SCHEMA = 'r02_k05_campaign_v1'
FAMILY = [f'partition_{i:03d}' for i in range(57)]
REUSE = 'partition_009'
CONSTANTS = {'partition_'+s for s in ('007','008','014','015','028','042','048','054')}
require = publication.require
sha = pipeline.sha
read = publication.read_unique_json
GIB = 2**30


def save(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Existing immutable record differs: '+str(path))
    else:
        publication.write_receipt_new(path, value)
    return value


def validate_launch(job):
    require(job['schema'] == SCHEMA and job['family_partition_ids'] == FAMILY,
            'Exact prespecified family of 57 required')
    require(job['reuse_partition_id'] == REUSE, 'Completed pilot must be reused')
    tasks = [row for slot in job['slots'] for row in slot['tasks']]
    ids = [row['partition_id'] for row in tasks]
    require(len(ids) == 56 and len(set(ids)) == 56 and set(ids) == set(FAMILY)-{REUSE},
            'Exactly 56 unique remaining partitions; pilot cannot be dispatched')
    require(sorted((s['vm'],s['slot']) for s in job['slots']) ==
            [(vm,n) for vm in ('A','B') for n in (1,2,3)], 'Exactly two workers, three slots each')
    require({r['partition_id'] for r in tasks if r['geometrically_constant']} == CONSTANTS,
            'Constant partitions must remain explicit')
    for row in tasks:
        n = row['assigned_people']
        require(type(n) is int and n > 1 and row['candidate_pairs'] == n*(n-1)//2,
                'Support/candidate pair count differs')
    require(job['task_cpus'] == 1 and job['task_memory_gib'] == 16 and
            job['rss_guard_mib'] == 14336 and job['max_concurrent'] == 3,
            'Reviewed resource envelope changed')
    require(job['max_vm_seconds'] == 18000 and 0 < job['max_task_seconds'] <= 7200
            and job['min_free_bytes'] >= 8*GIB, 'Reviewed deadlines or disk reserve changed')
    require(job['output_prefix'].startswith(publication.PREFIX+'r02-k05-full-'), 'Wrong campaign destination')
    require(len(job['dependencies']) == 50, 'Exactly fifty authenticated dependencies required')
    for path, entry in job['dependencies'].items():
        require(path.startswith('/home/jose.tantalean/projects/lai-exploracion-datos/.claude/runs/')
                and '..' not in Path(path).parts, 'Input escaped frozen minimum hierarchy')
        publication.validate_sha(entry['sha256'])
    publication.validate_sha(job['contract_sha256'])
    require('@sha256:' in job['container'] and job['container'].startswith('us-central1-docker.pkg.dev/uspbr-242713/'),
            'Portable pinned image required')
    require(job['container_id'].startswith('sha256:'), 'Frozen image configuration ID required')
    return tasks


def verify_inputs(root, job):
    for relative, digest in job['files_sha256'].items():
        path = root/relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts and
                path.resolve()==path and path.is_relative_to(root), 'Unsafe frozen path')
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                'Frozen source or runtime changed: '+relative)
    for path, entry in job['dependencies'].items():
        p = Path(path)
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == entry['bytes']
                and sha(p) == entry['sha256'], 'Frozen dependency changed: '+path)
    require(sha(root/'frozen/contract.json') == job['contract_sha256'], 'Scientific contract changed')
    settings = read(root/'frozen/contract.json')['settings']
    require(settings['analysis_mode'] == 'K05_PARTITION_ASSIGNED' and
            settings['partition_ids'] == FAMILY, 'Scientific mode/family changed')


def remaining(deadline, clock=time.time):
    value = deadline-clock()
    require(value > 0, 'Absolute worker deadline reached; retain disk')
    return value


def admission(root, job, *, initial=False):
    kind = subprocess.check_output(['findmnt','-T',str(root),'-n','-o','FSTYPE'],text=True).strip()
    require(kind in {'ext4','xfs'}, 'Intermediates require local block filesystem')
    require(shutil.disk_usage(root).free >= job['min_free_bytes'], 'Disk reserve exhausted')
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:')))*1024
    require(available >= (56 if initial else 18)*GIB, 'Insufficient available host memory')
    image = subprocess.check_output(['docker','image','inspect',job['container'],'--format','{{.Id}}'],text=True).strip()
    require(image == job['container_id'], 'Image differs from pilot configuration')


def task_config(job, vm, partition_id):
    require(vm in ('A','B') and partition_id in set(FAMILY)-{REUSE}, 'Unsafe task identity')
    tag = job['campaign_label']+'-'+vm.lower()+'-'+partition_id
    return "\n".join([
        "process.executor='local'", "process.container="+repr(job['container']),
        'process.cpus=1', "process.memory='16 GB'", "process.time='2h'",
        "process.errorStrategy='terminate'", 'executor.queueSize=1', 'docker.enabled=true',
        "docker.runOptions="+repr('--network none --cpus 1 --memory 16g --label dnabr_run='+tag),
        ''])


def stop_containers(job, vm, partition_id=None):
    ids = [partition_id] if partition_id else [r['partition_id'] for s in job['slots'] if s['vm']==vm for r in s['tasks']]
    for item in ids:
        tag = job['campaign_label']+'-'+vm.lower()+'-'+item
        found = subprocess.run(['docker','ps','-q','--filter','label=dnabr_run='+tag],capture_output=True,text=True,timeout=20)
        require(found.returncode == 0, 'Cannot identify exact task containers')
        containers = found.stdout.split()
        if containers:
            subprocess.run(['docker','stop','-t','20',*containers],capture_output=True,text=True,timeout=45,check=True)


def validate_result(folder, task, job):
    folder = publication.safe_local(folder, directory=True)
    summary = read(folder/'summary.json')
    require(summary['analysis_mode'] == 'K05_PARTITION_ASSIGNED' and
            summary['partition_id'] == task['partition_id'] and summary['n_partitions'] == 1 and
            summary['n_analysis_people'] == task['assigned_people'] and summary['table_rows'] == 24,
            'Partition output support or scope differs')
    with (folder/'conditional_associations.tsv').open() as stream:
        rows = list(csv.DictReader(stream,delimiter='\t'))
    require(len(rows)==24 and all(r['criterion']=='K05' and r['family_size']=='57' and
            r['partition_id']==task['partition_id'] for r in rows), 'Exactly 24 K05/family57 rows required')
    # Full row identities are asserted against the frozen settings by the tool;
    # this also rejects duplicate result rows independently of their coefficients.
    keys = [tuple(r[k] for k in ('kinship_threshold','relation','call_scope','model')) for r in rows]
    require(len(set(keys)) == 24, 'Duplicate model strata')
    if task['geometrically_constant']:
        require(all(r['status']=='CONSTANT_OR_EXPLAINED_EXPOSURE' and r['coefficient'] in ('','NA','nan') for r in rows),
                'Geometrically constant partition cannot become an estimated zero')
    manifest = read(folder/'manifest.json')
    require(manifest['source_contract_sha256'] == job['contract_sha256'], 'Output belongs to another contract')
    expected_source={name:job['files_sha256']['frozen/bin/'+name] for name in manifest['source_sha256']}
    require(manifest['source_sha256']==expected_source, 'Output scientific implementation differs')
    publication.authenticated_bundle(folder, sha(folder/'manifest.json'))
    return summary, rows


def publish_task(folder, task, job, receipt_path):
    summary, rows = validate_result(folder, task, job)
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['manifest_sha256']==sha(folder/'manifest.json'), 'Published output changed')
        for record in receipt['files']:
            publication.verify_remote(record,pinned=True)
    else:
        receipt = publication.publish(folder,sha(folder/'manifest.json'),
            job['output_prefix']+'/results/'+task['partition_id'],receipt_path)
    return dict(partition_id=task['partition_id'],summary=summary,publication=receipt)


def run_task(root, job, vm, task, deadline, cancelled):
    item = task['partition_id'];folder=root/'tasks'/item
    folder.mkdir(parents=True,exist_ok=True)
    complete=folder/'completed.json'
    if complete.exists():
        value=read(complete)
        require(value['partition_id']==item, 'Completion belongs to another partition')
        for record in value['publication']['files']:publication.verify_remote(record,pinned=True)
        publish_completion(complete,folder,job,item)
        return value
    require(not cancelled.is_set(), 'Worker cancelling after failure/deadline')
    remaining(deadline);admission(root,job)
    params=dict(source_bin=str(root/'frozen/bin'),input_contract=str(root/'frozen/contract.json'),
        input_contract_sha256=job['contract_sha256'],input_dependency_manifest=str(root/'frozen/dependencies.json'),
        max_memory_mb=job['rss_guard_mib'],min_free_mib=job['min_free_bytes']//2**20,partition_id=item)
    save(folder/'params.json',params)
    configuration=task_config(job,vm,item);config=folder/'resources.config'
    if config.exists():require(config.read_text()==configuration,'Task resources changed')
    else:config.write_text(configuration)
    attempt=str(time.time_ns());trace=folder/('trace.'+attempt+'.tsv')
    command=[str(root/'runtime/nextflow'),'-C',str(config),'-log',str(folder/('nextflow.'+attempt+'.log')),
        'run',str(root/'frozen/workflow.nf'),'-params-file',str(folder/'params.json'),
        '-work-dir',str(folder/'work'),'-with-trace',str(trace),'-ansi-log','false','-resume']
    save(folder/('command.'+attempt+'.json'),dict(command=command,partition_id=item,deadline_epoch=deadline))
    env={**os.environ,'NXF_HOME':str(root/'runtime/.nextflow'),'NXF_OFFLINE':'true','NXF_DISABLE_CHECK_LATEST':'true',
         'NXF_SYNTAX_PARSER':'v1','NXF_OPTS':'-Xms64m -Xmx768m','PYTHONDONTWRITEBYTECODE':'1'}
    process=None;start=time.time()
    try:
        with (folder/('console.'+attempt+'.log')).open('x') as stream:
            process=subprocess.Popen(command,cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                require(not cancelled.is_set(),'Other slot failed; preserving compatible completed tasks')
                require(time.time()-start<job['max_task_seconds'],'Partition task deadline reached')
                remaining(deadline)
                time.sleep(2)
            require(process.returncode==0,'Nextflow failed; retain work and inputs')
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        stop_containers(job,vm,item)
    with trace.open() as stream:traces=list(csv.DictReader(stream,delimiter='\t'))
    require(len(traces)==1 and traces[0]['status'] in {'COMPLETED','CACHED'} and traces[0]['exit']=='0',
            'Exactly one successful Nextflow task required')
    outputs=list((folder/'work').glob('*/*/community_validation_'+item))
    require(len(outputs)==1,'Ambiguous partition output: retain work')
    result=publish_task(outputs[0],task,job,folder/'publication.json')
    result.update(status='PARTITION_PUBLISHED_VERIFIED_NOT_BIOLOGICAL_VALIDATION',
        finished_utc=pipeline.now(),trace=str(trace.relative_to(root)),output=str(outputs[0]),
        contract_sha256=job['contract_sha256'])
    save(complete,result)
    publish_completion(complete,folder,job,item)
    return result



def publish_completion(complete, folder, job, item):
    receipt=folder/'completed.publication.json'
    if receipt.exists():
        record=read(receipt)
        require(record['sha256']==sha(complete),'Completion marker changed')
        publication.verify_remote(record,pinned=True)
    else:
        record=dict(name='completed.json',path=str(complete),uri=job['output_prefix']+'/results/'+item+'/completed.json',
                    **pipeline.publication_digests(complete))
        publication.upload_one(record);publication.verify_remote(record,pinned=True)
        save(receipt,record)
    return record

def run_slots(slots, callback, cancelled):
    """Each slot is serial; a failure stops dispatch while other tasks may close."""
    def serial(slot):
        output=[]
        for task in slot['tasks']:
            if cancelled.is_set():break
            try:output.append(callback(task))
            except BaseException:raise
        return output
    require(0<len(slots)<=3,'At most three serial slots per worker')
    output=[];errors=[]
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures=[executor.submit(serial,slot) for slot in slots]
        for future in as_completed(futures):
            try:output.extend(future.result())
            except BaseException as exc:errors.append(str(exc))
    require(not errors,'Worker incomplete: '+'; '.join(errors))
    return output


def worker(root, job, vm, identity):
    validate_launch(job);verify_inputs(root,job);admission(root,job,initial=True)
    approved=read(root/'frozen/scientific-admission.json')
    require(approved['status']=='ADMITTED_K05_56_REMAINING_CONDITIONAL_DESCRIPTIONS' and
            approved['contract_sha256']==job['contract_sha256'] and
            approved['scientific_code_sha256']==sha(root/'frozen/bin/r02_community_validation.py'),
            'Scientific stability/scope admission missing or changed')
    require(identity['worker']==vm and identity['max_vm_seconds']==job['max_vm_seconds'], 'Worker identity differs')
    deadline=identity['started_epoch']+job['max_vm_seconds'];remaining(deadline)
    cancel=threading.Event()
    def cancel_signal(sig,frame):cancel.set()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancel_signal)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        slots=[s for s in job['slots'] if s['vm']==vm]
        try:
            results=run_slots(slots,lambda task:run_task(root,job,vm,task,deadline,cancel),cancel)
            expected={r['partition_id'] for s in slots for r in s['tasks']}
            require({r['partition_id'] for r in results}==expected,'Incomplete worker result coverage')
            final=dict(schema=SCHEMA,status='WORKER_PARTITIONS_PUBLISHED_AWAITING_RETENTION',worker=vm,
                       finished_utc=pipeline.now(),partition_ids=sorted(expected),pending_work=[],
                       contract_sha256=job['contract_sha256'],results=results)
            save(root/'worker-science-completed.json',final)
            return final
        except BaseException as exc:
            existing={p.parent.name for p in (root/'tasks').glob('*/completed.json')}
            save(root/('failure.'+str(time.time_ns())+'.json'),dict(utc=pipeline.now(),error=str(exc),
                 pending_work=sorted({t['partition_id'] for s in slots for t in s['tasks']}-existing),retain_disk=True))
            raise
        finally:stop_containers(job,vm)



def outside_inventory(root, job):
    known=set(job['dependencies']);rows={};excluded=[]
    for anchor in ('/opt','/home','/srv','/mnt'):
        base=Path(anchor)
        if not base.exists():continue
        for folder,dirs,names in os.walk(base,followlinks=False):
            parent=Path(folder)
            dirs[:]=[d for d in dirs if not (parent/d).is_relative_to(root) and
                     d not in {'.ssh','.config','.cache','lost+found'}]
            if parent.is_relative_to(root):continue
            for name in names:
                path=parent/name
                if str(path) in known or name in {'.bashrc','.profile','.bash_logout'}:continue
                if path.is_symlink():rows[str(path)]=dict(kind='SYMLINK',target=os.readlink(path))
                elif path.is_file():
                    v=path.stat();rows[str(path)]=dict(kind='FILE',bytes=v.st_size,mtime_ns=v.st_mtime_ns)
    return rows


def verify_outside(root, job):
    baseline=read(root/'outside-baseline.json');current=outside_inventory(root,job)
    changed=[p for p,v in current.items() if baseline.get(p)!=v]
    # Explicitly do not classify a newly discovered scientific file as OS noise.
    require(not changed,'Unclassified paths outside scientific roots: '+str(changed[:20]))
    result=dict(roots=['/opt','/home','/srv','/mnt'],outside_roots_reviewed=True,
                unclassified_paths=[],baseline_sha256=sha(root/'outside-baseline.json'),
                n_unchanged_paths=len(current),excluded_credential_roots=['.ssh','.config','.cache'],
                os_exclusions=['/usr','/bin','/sbin','/lib','/etc','/var/lib/docker'],
                log_policy='Task commands, Nextflow logs and traces retained. Bootstrap/OS journal is regenerable installation detail; no scientific outputs configured under /var/log.',
                standard_home_initialization_excluded=['.bashrc','.profile','.bash_logout'],
                no_credential_contents_read=True)
    save(root/'outside-reviewed.json',result)
    return result

def archive_state(root, job, vm, destination):
    """Capture only this bounded worker's scientific/restart state, not the OS.

    Frozen sources and the fifty inputs live in the verified deployment bundle.
    Task work/cache/logs and symlink text are preserved here, with archive-member
    SHA checks. Completion does not authorize deletion by itself.
    """
    verify_inputs(root,job)
    verify_outside(root,job)
    science=read(root/'worker-science-completed.json')
    require(science['pending_work']==[] and science['worker']==vm,'Do not retire incomplete worker')
    out=root/'retention';out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():
        result=read(out/'completed.json')
        for record in result['files']:publication.verify_remote(record,pinned=True)
        record=dict(name='completed.json',path=str(out/'completed.json'),uri=destination+'/completed.json',**pipeline.publication_digests(out/'completed.json'))
        publication.upload_one(record);publication.verify_remote(record,pinned=True)
        return result
    archive=out/'worker-state.tar.gz';inventory=out/'inventory.json'
    excluded={'package.tar.gz','inputs.tar.gz','worker.lock'}
    paths=[p for p in root.rglob('*') if not p.is_relative_to(out) and
           p.relative_to(root).as_posix() not in excluded and (p.is_file() or p.is_symlink())]
    paths=sorted(set(paths));require(sum(p.stat().st_size for p in paths if not p.is_symlink())<GIB,'Unexpected state exceeds 1GiB; retain disk')
    entries=[]
    if not archive.exists():
        with tarfile.open(archive,'x:gz',dereference=False) as tar:
            for path in paths:
                rel=path.relative_to(root).as_posix();require('credentials.db' not in rel and '/gcloud/' not in rel,'Credential path forbidden')
                if path.is_symlink():row=dict(path=rel,kind='SYMLINK',target=os.readlink(path))
                else:row=dict(path=rel,kind='FILE',bytes=path.stat().st_size,sha256=sha(path))
                entries.append(row);tar.add(path,arcname=rel,recursive=False)
        with tarfile.open(archive) as tar:
            for row in entries:
                if row['kind']=='FILE':
                    require(hashlib.file_digest(tar.extractfile(row['path']),'sha256').hexdigest()==row['sha256'],'Archive member changed')
        save(inventory,dict(schema='r02_useful_archive_inventory_v1',archive_sha256=sha(archive),entries=entries))
    else:require(inventory.exists() and read(inventory)['archive_sha256']==sha(archive),'Incomplete retention attempt; retain disk')
    files=[]
    for path in (archive,inventory):
        record=dict(name=path.name,path=str(path),uri=destination+'/'+path.name,**pipeline.publication_digests(path))
        publication.upload_one(record);publication.verify_remote(record,pinned=True);files.append(record)
    result=dict(schema=SCHEMA,status='WORKER_STATE_PUBLISHED_VERIFIED',worker=vm,files=files,
                scientific_completion= science,credentials_included=False,os_included=False,
                resume_mode='SCIENTIFIC_PRODUCTS_AND_PRESERVED_TASK_STATE',
                resume_limitation='Scientific products are reusable. Interrupted Python fits restart their task; cache restore requires the same authenticated paths/image/runtime.')
    save(out/'completed.json',result)
    record=dict(name='completed.json',path=str(out/'completed.json'),uri=destination+'/completed.json',**pipeline.publication_digests(out/'completed.json'))
    publication.upload_one(record);publication.verify_remote(record,pinned=True)
    return result


def consolidate(records, job):
    """Require all hypotheses; never rank supports or fill missing estimates."""
    validate_launch(job)
    require(len(records)==57 and {r['partition_id'] for r in records}==set(FAMILY), 'Exactly 57 unique results required')
    require(all(r['contract_sha256']==job['contract_sha256'] for r in records),'Scientific contract differs')
    rows=[]
    for entry in sorted(records,key=lambda x:x['partition_id']):
        current=entry['rows'];require(len(current)==24,'Missing partition strata')
        require(all(r['criterion']=='K05' and r['family_size']=='57' and r['partition_id']==entry['partition_id'] for r in current),
                'Wrong family or partition in aggregate')
        rows.extend(current)
    require(len(rows)==1368,'Family output incomplete')
    for entry in records:
        keys=[tuple(r[k] for k in ('kinship_threshold','relation','call_scope','model')) for r in entry['rows']]
        require(len(set(keys))==24,'Duplicate strata in consolidated partition')
    return dict(schema=SCHEMA,status='57_CONDITIONAL_DESCRIPTIONS_NOT_WINNER_NOT_POPULATION_VALIDATION',
                n_partitions=57,n_rows=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
                partition_ids=FAMILY,reused_partition_id=REUSE,rows=rows,no_ranking=True)


def retire_reviewed(manifest_path, expected_sha256, record_path, state_dir):
    # One shared gate, no alternate deletion path and no cloud-name fallback.
    return retirement.retire(manifest_path,expected_sha256,read(record_path),state_dir)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['worker','archive','stop','retire','baseline'])
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--launch-sha256',required=True)
    parser.add_argument('--worker',choices=['A','B']);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--manifest-sha256');parser.add_argument('--manifest-record',type=Path)
    args=parser.parse_args(argv);root=args.root.resolve();require(root==args.root,'Canonical root required')
    require(sha(root/'launch.json')==args.launch_sha256,'Launch changed');job=read(root/'launch.json');validate_launch(job)
    if args.mode=='baseline':save(root/'outside-baseline.json',outside_inventory(root,job))
    elif args.mode=='worker':worker(root,job,args.worker,read(root/'identity.json'))
    elif args.mode=='archive':archive_state(root,job,args.worker,job['output_prefix']+'/workers/'+args.worker+'/retention')
    elif args.mode=='stop':stop_containers(job,args.worker)
    else:
        require(args.manifest is not None and args.manifest_sha256 and args.manifest_record is not None,'Reviewed retirement manifest required')
        state=root/'retirement-state';state.mkdir(exist_ok=True)
        retire_reviewed(args.manifest,args.manifest_sha256,args.manifest_record,state)


if __name__=='__main__':main()
