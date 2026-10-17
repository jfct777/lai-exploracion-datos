#!/usr/bin/env python3
"""Serial chromosome staging, individual extraction and complete cache retention.

Reuses existing source VCFs and FIT selections, never trains or chooses model
parameters. One chromosome at a time limits disk. Completed outputs and the
Nextflow cache are authenticated inside one archive before staging is removed.
An unfinished chromosome stays on disk; the externally bounded VM stops.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
import time
import urllib.request

import r02_autosome_pipeline as pipeline
import r02_publish_evidence as publication
import r02_region_support_queue as staging

require, read, sha, save = staging.require, staging.read, staging.sha, staging.save_once
SCHEMA = 'r02_genomic_event_queue_v1'
OWNERSHIP_SCHEMA = 'r02_genomic_event_ownership_v1'


def runtime_identity():
    """Read the attached GCE identity; never trust a worker name supplied alone."""
    def metadata(key):
        request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/'+key,
                                         headers={'Metadata-Flavor':'Google'})
        with urllib.request.urlopen(request,timeout=10) as response:
            return response.read(4096).decode().strip()
    return dict(project=metadata('project/project-id'),zone=metadata('instance/zone').split('/')[-1],
                name=metadata('instance/name'),id=metadata('instance/id'))


def validate_ownership(root, job, path, expected, worker_id):
    """Authenticate a static partition, without rewriting the original launch.

    The fence is an operator attestation about the legacy process, not a claim
    that a JSON document can stop it. It must be issued only after that process
    and its startup shutdown path are safely retired. A paused legacy Python
    process is insufficient: it still owns queue.lock and could resume.
    """
    require(path.is_absolute() and path.resolve()==path and sha(path)==publication.validate_sha(expected),
            'Ownership receipt changed or unsafe')
    plan=publication.read_unique_json(path)
    require(plan.get('schema')==OWNERSHIP_SCHEMA and plan.get('status')=='ORIGINAL_QUEUE_FENCED',
            'Original queue must be fenced before transfer')
    for key in ('launch_sha256','event_contract_sha256','scientific_contract_sha256','output_prefix'):
        require(plan.get(key)==job[key], 'Ownership binding differs: '+key)
    require(plan.get('operator_sha256')==sha(Path(__file__)), 'Unbound amended operator')
    require(plan.get('project')=='uspbr-242713' and plan.get('zone')=='us-central1-a', 'Unexpected fleet scope')
    owners=plan.get('assignments',[])
    require(isinstance(owners,list) and 1<=len(owners)<=4, 'One to four explicit workers required')
    for owner in owners:
        require(re.fullmatch(r'dnabr-r02-events-[a-z0-9-]+',owner.get('name','')) and
                re.fullmatch(r'[1-9][0-9]*',str(owner.get('id',''))), 'Missing exact worker identity')
        require(isinstance(owner.get('chromosomes'),list) and owner['chromosomes'] and
                all(c in job['chromosomes'] for c in owner['chromosomes']), 'Invalid assigned chromosome')
        require(owner['chromosomes']==[c for c in job['chromosomes'] if c in owner['chromosomes']],
                'Assignments must be unique and preserve queue order')
    require(len({o['name'] for o in owners})==len(owners) and len({str(o['id']) for o in owners})==len(owners),
            'Repeated worker identity')
    owned=[o for o in owners if o['name']==worker_id]
    require(len(owned)==1,'Worker missing or repeated')
    preserved=plan.get('preserved_results',[])
    require(isinstance(preserved,list),'Preserved results must be explicit')
    combined=[c for owner in owners for c in owner['chromosomes']]+[r['chromosome'] for r in preserved]
    require(len(combined)==len(set(combined))==22 and set(combined)==set(job['chromosomes']),
            'Assignments and preserved results must partition all 22 chromosomes exactly')
    for item in preserved:
        receipt=item['receipt']
        require(receipt['uri']==job['output_prefix']+f"/chr{int(item['chromosome']):02d}/completed.json" and
                re.fullmatch(r'[1-9][0-9]*',str(receipt.get('generation',''))) and
                isinstance(receipt.get('bytes'),int) and receipt['bytes']>0,'Unpinned preserved completion')
        publication.validate_sha(receipt['sha256'])
        require(isinstance(receipt.get('md5_base64'),str) and receipt['md5_base64'],'Missing completion MD5')
    descriptor=plan['source_fence']
    fence_path=path.parent/publication.safe_relative(descriptor['path'])
    require(fence_path.resolve()==fence_path and sha(fence_path)==publication.validate_sha(descriptor['sha256']),
            'Source fence changed or unsafe')
    fence=publication.read_unique_json(fence_path)
    require(fence.get('schema')=='r02_genomic_event_fence_v1' and
            fence.get('status')=='ORIGINAL_QUEUE_REMOVED_AFTER_CHILD_COMPLETED' and
            fence.get('launch_sha256')==job['launch_sha256'] and
            fence.get('instance')=='dnabr-r02-events-1005j' and
            str(fence.get('instance_id'))=='2595569497300425472' and
            fence.get('original_queue_absent') is True and
            fence.get('startup_shutdown_path_disabled') is True and
            fence.get('scientific_autostart_disabled') is True and
            fence.get('inflight_nextflow_complete') is True and
            fence.get('inflight_chromosome') in job['chromosomes'], 'Incomplete legacy-process fence')
    require(isinstance(fence.get('queue_pid'),int) and fence['queue_pid']>1 and
            re.fullmatch(r'[1-9][0-9]*',str(fence.get('queue_start_ticks',''))) and
            re.fullmatch(r'[0-9a-f-]{36}',fence.get('boot_id','')) and
            isinstance(fence.get('evidence_sha256'),dict) and fence['evidence_sha256'],
            'Fence lacks process identity or evidence hashes')
    for digest in fence['evidence_sha256'].values():publication.validate_sha(digest)
    # A still-live chromosome stays on its original disk, never delegated.
    active_owner=next((o for o in owners if fence['inflight_chromosome'] in o['chromosomes']),None)
    require(active_owner is None or (active_owner['name']==fence['instance'] and
            str(active_owner['id'])==str(fence['instance_id'])), 'In-flight chromosome cannot leave its original VM')
    return dict(plan=plan,worker=owned[0],sha256=expected,path=path,fence=fence,fence_path=fence_path)


def verify_ownership_runtime(ownership):
    plan,worker,fence=ownership['plan'],ownership['worker'],ownership['fence']
    require(sha(ownership['path'])==ownership['sha256'] and
            sha(ownership['fence_path'])==plan['source_fence']['sha256'] and
            sha(Path(__file__))==plan['operator_sha256'], 'Ownership inputs changed after admission')
    require(runtime_identity()==dict(project=plan['project'],zone=plan['zone'],name=worker['name'],id=str(worker['id'])),
            'Attached VM does not own this queue')
    if worker['name']==fence['instance']:
        require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==fence['boot_id'],
                'Original VM rebooted after fence; re-audit startup')
        proc=Path('/proc')/str(fence['queue_pid'])/'stat'
        if proc.exists():
            fields=proc.read_text().rsplit(')',1)[1].split()
            require(fields[19]!=str(fence['queue_start_ticks']), 'Legacy supervisor is still alive or paused')


def publish_ownership(root, job, ownership):
    """One canonical, immutable partition; no expiry or automatic reassignment."""
    verify_ownership_runtime(ownership)
    records=[]
    for path,name in ((ownership['fence_path'],'source-fence.json'),(ownership['path'],'plan.json')):
        record=dict(name=name,path=str(path),uri=job['output_prefix']+'/ownership/'+name,
                    **pipeline.publication_digests(path))
        publication.upload_one(record);publication.verify_remote(record,pinned=True)
        records.append({k:v for k,v in record.items() if k!='operation'})
    for item in ownership['plan']['preserved_results']:
        record=item['receipt'];publication.verify_remote(record,pinned=True)
        raw=subprocess.check_output(['gcloud','storage','cat',record['uri']+'#'+str(record['generation'])],timeout=120)
        require(hashlib.sha256(raw).hexdigest()==record['sha256'],'Preserved completion content changed')
        completion=json.loads(raw)
        require(completion['schema']==SCHEMA and
                completion['status']=='INDIVIDUAL_GENOTYPES_AND_CACHE_PUBLISHED_VERIFIED' and
                completion['chromosome']==item['chromosome'] and completion['training_allowed'] is False and
                completion['event_contract_sha256']==job['event_contract_sha256'],'Invalid preserved completion')
        require({r['name'] for r in completion['files']}=={'individuals-and-nextflow.tar.gz','inventory.json','manifest.json'},
                'Preserved completion lacks products/cache')
        for product in completion['files']:
            require(product['uri']==job['output_prefix']+f"/chr{int(item['chromosome']):02d}/"+product['name'],
                    'Preserved product escaped campaign')
            publication.verify_remote(product,pinned=True)
    save(root/'ownership-accepted.json',dict(ownership_sha256=ownership['sha256'],worker=ownership['worker'],records=records))


def claim_chromosome(root,job,chrom,ownership):
    require(chrom in ownership['worker']['chromosomes'],'Chromosome not assigned to this worker')
    verify_ownership_runtime(ownership)
    accepted=read(root/'ownership-accepted.json')
    require(accepted['ownership_sha256']==ownership['sha256'] and accepted['worker']==ownership['worker'] and
            len(accepted['records'])==2 and {r['uri'] for r in accepted['records']}==
            {job['output_prefix']+'/ownership/'+name for name in ('source-fence.json','plan.json')},
            'Canonical ownership admission differs')
    for record in accepted['records']:publication.verify_remote(record,pinned=True)
    folder=root/'ownership';folder.mkdir(exist_ok=True)
    path=folder/f'chr{int(chrom):02d}.json'
    save(path,dict(schema='r02_genomic_event_claim_v1',chromosome=chrom,
        ownership_sha256=ownership['sha256'],launch_sha256=job['launch_sha256'],worker=ownership['worker'],
        no_automatic_expiry=True))
    record=dict(name=path.name,path=str(path),uri=job['output_prefix']+'/ownership/'+path.name,
                **pipeline.publication_digests(path))
    publication.upload_one(record);publication.verify_remote(record,pinned=True)
    return record


def validate(root, expected):
    require(root.is_absolute() and root.resolve() == root and sha(root/'launch.json') == expected, 'Runtime/launch changed')
    job = read(root/'launch.json')
    require(job['schema'] == SCHEMA and job['root'] == str(root) and job['training_allowed'] is False, 'Wrong runtime scope')
    require(job['output_prefix'].startswith(publication.PREFIX+'r02-genomic-event-real-'), 'Wrong private output scope')
    require(job['chromosomes'] == [str(n) for n in range(22,0,-1)] and
            set(job['sources']) == set(job['chromosomes']), 'Complete unique autosomal queue required')
    require(job['max_runtime_seconds'] == 27000 and job['min_free_bytes'] >= 20*2**30 and
            job['max_archive_bytes'] <= 80*2**30, 'Runtime/storage bounds differ')
    for name, digest in job['files_sha256'].items():
        require(sha(root/publication.safe_relative(name)) == digest, 'Frozen runtime changed: '+name)
    source = read(root/'bundle/source/materialize.json')
    require(sha(root/'bundle/source/materialize.json') == job['scientific_contract_sha256'], 'Scientific contract changed')
    require(sha(root/'bundle/event_inputs.json') == job['event_contract_sha256'], 'Event contract changed')
    event=read(root/'bundle/event_inputs.json')
    require(event['materialize_contract_sha256'] == job['scientific_contract_sha256'] and
            set(event['fit_caches']) == set(event['maps']) == set(job['fit_remote']) == set(job['sources']),
            'FIT/map/source inventory or inherited contract differs')
    source_root = Path(job['source_root'])
    require(source_root == Path('/opt/r02-structure-materialize-20261005f') and source_root != root, 'Explicit original source path required')
    for chrom in job['chromosomes']:
        require(set(job['sources'][chrom]) == {'common_vcf','common_index','rare_vcf','rare_index'},'Unexpected source kind')
        for kind, record in job['sources'][chrom].items():
            d = source['sources'][chrom][kind]
            require(record['sha256'] == d['sha256'] and Path(d['path']).parent == source_root/'sources'/f'chr{int(chrom):02d}',
                    'Source staging escaped authenticated path')
        require(set(job['fit_remote'][chrom]) == {'manifest','contract','eligible'},'Wrong FIT cache inventory')
        for kind,record in job['fit_remote'][chrom].items():
            normalized=staging.normalize_record(record);d=event['fit_caches'][chrom][kind]
            require(normalized['sha256'] == d['sha256'] and Path(d['path']).parent == root/'fit'/f'chr{int(chrom):02d}',
                    'FIT staging differs from authenticated event contract')
    job['launch_sha256'] = expected
    return job, source


def run_nextflow(root, job, chrom, folder, started):
    params = dict(bundle=str(root/'bundle'),code_dir=str(root/'frozen/bin'),output_dir=str(folder/'published'),
        contract_sha256=job['event_contract_sha256'],chromosomes=chrom,memory_gb=4,max_forks=1,
        task_time='7h',publish_mode='link')
    save(folder/'params.json',params)
    attempt = str(time.time_ns())
    trace = folder/('trace.'+attempt+'.tsv')
    command = ['/usr/local/bin/nextflow','-C',str(root/'resources.config'),'-log',str(folder/('nextflow.'+attempt+'.log')),
        'run',str(root/'frozen/workflows/r02_genomic_event_inputs.nf'),'-params-file',str(folder/'params.json'),
        '-work-dir',str(folder/'work'),'-with-trace',str(trace),'-ansi-log','false']
    if (folder/'.nextflow').exists():
        command.append('-resume')
    save(folder/('command.'+attempt+'.json'),dict(command=command,launch_sha256=job['launch_sha256']))
    env = {**os.environ,'NXF_OFFLINE':'true','NXF_DISABLE_CHECK_LATEST':'true','NXF_SYNTAX_PARSER':'v1',
           'NXF_OPTS':'-Xms64m -Xmx768m','PYTHONDONTWRITEBYTECODE':'1'}
    with (folder/('console.'+attempt+'.log')).open('x') as log:
        p = subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                staging.remaining(started,job);staging.check_disk(root,job);time.sleep(10)
            require(p.returncode == 0,'Individual Nextflow failed; preserve all work')
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
            raise
    with trace.open() as handle:
        rows = list(csv.DictReader(handle,delimiter='\t'))
    require(len(rows) == 2 and all(r['status'] in {'COMPLETED','CACHED'} and r['exit'] == '0' for r in rows),
            'Incomplete individual chromosome trace')
    save(folder/'nextflow-complete.json',dict(trace=str(trace),trace_sha256=sha(trace),
        launch_sha256=job['launch_sha256'],chromosome=chrom))


def archive_chromosome(root, job, chrom, folder):
    output = folder/'published'/('chr'+chrom)
    m = read(output/'manifest.json')
    require(m['stage'] == 'COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING' and m['training_allowed'] is False and
            m['chromosome'] == chrom and m['event_contract_sha256'] == job['event_contract_sha256'] and
            m['source_contract_sha256'] == job['scientific_contract_sha256'] and m['n_people'] == 2619 and
            m['role_counts'] == {'FIT':1460,'SELECT':476,'EVALUATE':683},'Wrong individual output contract/cohort')
    for name,digest in m['outputs_sha256'].items():
        require(sha(output/publication.safe_relative(name)) == digest,'Individual output changed')
    for name,digest in m['code_sha256'].items():
        require(Path(name).name == name and sha(root/'frozen/bin'/name) == digest,'Producer version changed')
    entries = []
    for p in sorted(folder.rglob('*')):
        name = p.relative_to(folder).as_posix()
        if name in {'completed.json','completed.publication.json'}:continue
        if p.is_symlink():
            require(p.resolve().is_relative_to(root) or p.resolve().is_relative_to(Path(job['source_root'])),
                    'Cache symlink escaped authenticated runtime')
            entries.append(dict(path=name,kind='SYMLINK',target=os.readlink(p)))
        elif p.is_file():
            entries.append(dict(path=name,kind='FILE',bytes=p.stat().st_size,sha256=sha(p)))
    total = sum(e.get('bytes',0) for e in entries)
    require(0 < total <= job['max_archive_bytes'],'Archive exceeds admitted bound; retain disk')
    staging.check_disk(root,job,total+2**30)
    archive = root/'archives'/f'chr{int(chrom):02d}.tar.gz'
    archive.parent.mkdir(exist_ok=True)
    inventory = root/'archives'/f'chr{int(chrom):02d}.inventory.json'
    if archive.exists():
        if inventory.exists():
            require(read(inventory)['archive_sha256'] == sha(archive) and
                    read(inventory)['entries'] == entries, 'Existing archive differs; preserve disk')
    else:
        # Incomplete attempts remain named separately and never masquerade as
        # a complete archive. Only this exact launch-owned temporary is reset.
        partial = archive.with_suffix(archive.suffix+'.partial')
        require(not partial.is_symlink(), 'Unsafe incomplete archive')
        if partial.exists():partial.unlink()
        with tarfile.open(partial,'x:gz',compresslevel=1,dereference=False) as tar:
            for entry in entries:
                tar.add(folder/entry['path'],arcname=entry['path'],recursive=False)
        os.link(partial,archive);partial.unlink()
    with tarfile.open(archive,'r:gz') as tar:
        require(set(tar.getnames()) == {e['path'] for e in entries},'Archive inventory differs')
        for entry in entries:
            if entry['kind'] == 'FILE':
                with tar.extractfile(entry['path']) as stream:
                    require(hashlib.file_digest(stream,'sha256').hexdigest() == entry['sha256'],'Archive content checksum differs')
    save(inventory,dict(chromosome=chrom,entries=entries,source_vcf_recoverable_from_job=True,
        nextflow_cache_preserved=True,launch_sha256=job['launch_sha256'],archive_sha256=sha(archive)))
    records = []
    for path,name in [(archive,'individuals-and-nextflow.tar.gz'),(inventory,'inventory.json'),(output/'manifest.json','manifest.json')]:
        record = dict(name=name,path=str(path),uri=job['output_prefix']+f'/chr{int(chrom):02d}/'+name,
                      **pipeline.publication_digests(path))
        publication.upload_one(record);publication.verify_remote(record,pinned=True);records.append(record)
    completion = folder/'completed.json'
    save(completion,dict(schema=SCHEMA,status='INDIVIDUAL_GENOTYPES_AND_CACHE_PUBLISHED_VERIFIED',chromosome=chrom,
        launch_sha256=job['launch_sha256'],training_allowed=False,files=records,event_contract_sha256=job['event_contract_sha256']))
    record = dict(name='completed.json',path=str(completion),uri=job['output_prefix']+f'/chr{int(chrom):02d}/completed.json',
                  **pipeline.publication_digests(completion))
    publication.upload_one(record);publication.verify_remote(record,pinned=True)
    save(folder/'completed.publication.json',record)
    return record


def clean_verified_copies(root,job,source,chrom,folder):
    """Retire only exact worker copies after source and output receipts verify."""
    publication.verify_remote(read(folder/'completed.publication.json'),pinned=True)
    for record in read(folder/'completed.json')['files']:
        publication.verify_remote(record,pinned=True)
    for kind,record in job['sources'][chrom].items():
        staging.remote_check(staging.normalize_record(record))
        path = Path(source['sources'][chrom][kind]['path'])
        require(path.parent == Path(job['source_root'])/'sources'/f'chr{int(chrom):02d}' and not path.is_symlink(),'Unsafe local source cleanup')
        path.unlink(missing_ok=True)
    require(folder == root/'chromosomes'/f'chr{int(chrom):02d}' and folder.resolve() == folder,'Unsafe chromosome cleanup')
    for name in ('work','published'):
        p = folder/name
        require(not p.is_symlink(),'Unsafe work link')
        if p.exists():shutil.rmtree(p)
    for suffix in ('.tar.gz','.inventory.json'):
        (root/'archives'/f'chr{int(chrom):02d}{suffix}').unlink(missing_ok=True)


def execute(root,job,source,ownership=None):
    started = time.monotonic();results = {}
    cfg = read(root/'bundle/event_inputs.json')
    if ownership is not None:publish_ownership(root,job,ownership)
    chromosomes=job['chromosomes'] if ownership is None else ownership['worker']['chromosomes']
    for chrom in chromosomes:
        staging.remaining(started,job)
        claim=claim_chromosome(root,job,chrom,ownership) if ownership is not None else None
        folder = root/'chromosomes'/f'chr{int(chrom):02d}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'completed.publication.json').exists():
            results[chrom]=read(folder/'completed.publication.json')
            clean_verified_copies(root,job,source,chrom,folder)
            continue
        size = sum(x['bytes'] for x in job['sources'][chrom].values())
        staging.check_disk(root,job,size+job['max_archive_bytes']+40*2**30)
        for kind,record in job['sources'][chrom].items():
            path=Path(source['sources'][chrom][kind]['path']);path.parent.mkdir(parents=True,exist_ok=True)
            staging.stage_one(staging.normalize_record(record),path,started,job)
        for kind,record in job['fit_remote'][chrom].items():
            path=Path(cfg['fit_caches'][chrom][kind]['path']);path.parent.mkdir(parents=True,exist_ok=True)
            staging.stage_one(staging.normalize_record(record),path,started,job)
        checkpoint=folder/'nextflow-complete.json'
        if checkpoint.exists():
            old=read(checkpoint)
            require(old['chromosome']==chrom and old['launch_sha256']==job['launch_sha256'] and
                    sha(old['trace'])==old['trace_sha256'],'Completed Nextflow checkpoint differs')
        else:
            run_nextflow(root,job,chrom,folder,started)
        if claim is not None:publication.verify_remote(claim,pinned=True)
        results[chrom]=archive_chromosome(root,job,chrom,folder)
        clean_verified_copies(root,job,source,chrom,folder)
    if ownership is None:
        p=root/'complete.json';status='ALL_22_INDIVIDUAL_CHROMOSOMES_PUBLISHED_VERIFIED';extra={}
    else:
        p=root/('complete.'+ownership['worker']['name']+'.json')
        status='ASSIGNED_INDIVIDUAL_CHROMOSOMES_PUBLISHED_VERIFIED'
        extra=dict(ownership_sha256=ownership['sha256'],worker=ownership['worker'],chromosomes=chromosomes)
    save(p,dict(schema=SCHEMA,status=status,launch_sha256=job['launch_sha256'],results=results,training_allowed=False,**extra))
    record=dict(name=p.name,path=str(p),uri=job['output_prefix']+'/'+p.name,**pipeline.publication_digests(p))
    publication.upload_one(record);publication.verify_remote(record,pinned=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--launch-sha256',required=True)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--ownership',type=Path);p.add_argument('--ownership-sha256');p.add_argument('--worker-id')
    args=p.parse_args();os.umask(0o077)
    require(all(v is not None for v in (args.ownership,args.ownership_sha256,args.worker_id)) or
            all(v is None for v in (args.ownership,args.ownership_sha256,args.worker_id)), 'Ownership flags are inseparable')
    with (args.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job,source=validate(args.root,args.launch_sha256)
        ownership=validate_ownership(args.root,job,args.ownership,args.ownership_sha256,args.worker_id) if args.ownership else None
        require(ownership is not None or not (args.root/'ownership-accepted.json').exists(),
                'Partitioned runtime cannot return to the serial queue')
        if not args.verify_only:execute(args.root,job,source,ownership)


if __name__ == '__main__':main()
