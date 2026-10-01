#!/usr/bin/env python3
"""Package and launch the twelve explicitly authorized, fixed R02 workers.

No IAM, firewall, personal credentials or scientific parameters are changed.
Preparation is local; upload and creation each require their explicit CLI mode.
Existing VM names are not adopted or overwritten. Failed workers retain disks.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

PROJECT = 'uspbr-242713'
ZONE = 'us-central1-a'
SERVICE_ACCOUNT = '653458115080-compute@developer.gserviceaccount.com'
PREFIX = 'gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/biologico/R02_20260930/r02-autosomes-20261001b'
CONTROL = 'gs://projects-usp/dnaBr-lai/datalake/transient/DNABR_QC/R02_20260930/r02-autosomes-20261001b/parallel-launch-20261001'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def prepare(run):
    folder = run/'repairs/parallel-v1'
    bootstrap = folder/'bootstrap'
    runtime = bootstrap/'runtime.tar.gz'
    if not runtime.is_file(): raise ValueError('Missing runtime archive')
    records = []
    assigned = []
    source_hashes = set()
    for index in range(1, 13):
        worker = f'worker{index:02d}'
        path = run/'parallel'/worker
        for relative, digest in read(path/'frozen.sha256.json').items():
            if sha(path/relative) != digest: raise ValueError('Changed worker: '+relative)
        config = read(path/'run.json')
        assigned.extend(config['processing_order'])
        source_hashes.add(sha(path/'source.sha256.json'))
        if config.get('analysis_protocol_version') != 2 or not config.get('m14_skip_legacy_windows'):
            raise ValueError('Worker does not use corrected protocol')
        archive = bootstrap/(worker+'.tar.gz')
        with tarfile.open(archive, 'x:gz') as tar:
            for child in sorted(path.iterdir()):
                tar.add(child, arcname=child.name, recursive=True)
        records.append(dict(worker=worker, name=f'dnabr-r02p-1001-w{index:02d}',
            chromosomes=config['processing_order'], run_dir=str(path),
            run_sha256=sha(path/'run.json'), bundle=str(archive), bundle_sha256=sha(archive),
            bundle_uri=CONTROL+'/'+archive.name,
            log_uri=PREFIX+'/parallel/'+worker+'/00_worker_logs',
            completion_uri=PREFIX+'/parallel/'+worker+'/00_worker_provenance/worker_completion.json'))
    if sorted(assigned) != list(range(1,21)) or len(source_hashes) != 1:
        raise ValueError('Assignments duplicate, omit or change chromosome sources')
    startup = bootstrap/'startup.sh'
    # Exact copy from the working script, then frozen by its digest.
    with startup.open('xb') as out:
        out.write((Path(__file__).parent/'r02_parallel_startup.sh').read_bytes())
    spec = dict(schema='r02_fleet_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        project=PROJECT, zone=ZONE, machine_type='n2-highmem-8', boot_disk_gib=600,
        boot_disk_type='pd-balanced', vm_runtime_limit_hours=72,
        termination_action='STOP', auto_restart=False,
        os_image='projects/debian-cloud/global/images/debian-12-bookworm-v20260921',
        scientific_source_manifest_sha256=next(iter(source_hashes)),
        runtime=str(runtime),runtime_sha256=sha(runtime),runtime_uri=CONTROL+'/runtime.tar.gz',
        startup=str(startup),startup_sha256=sha(startup), workers=records,
        hourly_vm_usd=.524056, hourly_disk_gib_usd=.000136986,
        hourly_external_ipv4_usd=.005,
        interpretation='Parallel execution of the recorded protocol; not biological validation')
    write_new(folder/'fleet.json',spec)
    print(json.dumps(spec,indent=2))

def upload_one(local, uri):
    local=Path(local)
    md5=hashlib.md5(usedforsecurity=False)
    with local.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''): md5.update(chunk)
    md5=base64.b64encode(md5.digest()).decode()
    result=subprocess.run(['gcloud','storage','cp','--if-generation-match=0',
        '--content-md5='+md5,str(local),uri],capture_output=True,text=True,timeout=3600,
        env={**os.environ,'CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED':'false'})
    if result.returncode and 'HTTPError 412:' not in result.stderr:
        raise RuntimeError(result.stderr)
    metadata=json.loads(subprocess.check_output(['gcloud','storage','objects','describe',uri,'--format=json'],text=True))
    if int(metadata['size']) != local.stat().st_size or metadata.get('md5_hash') != md5:
        raise ValueError('Upload content mismatch: '+uri)
    return dict(uri=uri,generation=metadata['generation'],sha256=sha(local),md5_base64=md5)

def validate(spec):
    if spec['project'] != PROJECT or spec['zone'] != ZONE or spec['machine_type'] != 'n2-highmem-8':
        raise ValueError('Unapproved fleet settings')
    for pathkey in ('startup','runtime'):
        if sha(spec[pathkey]) != spec[pathkey+'_sha256']: raise ValueError('Changed '+pathkey)
    for worker in spec['workers']:
        if sha(worker['bundle']) != worker['bundle_sha256']: raise ValueError('Changed bundle')

def launch(spec, indices, folder):
    validate(spec)
    if not (folder/'upload.json').exists(): raise ValueError('Upload receipt missing')
    def one(index):
        worker=spec['workers'][index-1]
        receipt=folder/(worker['worker']+'.creation.json')
        if receipt.exists(): raise ValueError('Worker already launched; inspect its receipt')
        metadata=','.join(f'{k}={v}' for k,v in {
            'r02-worker':worker['worker'],'r02-bundle-uri':worker['bundle_uri'],
            'r02-bundle-sha256':worker['bundle_sha256'],
            'r02-runtime-uri':spec['runtime_uri'],'r02-runtime-sha256':spec['runtime_sha256'],
            'r02-log-uri':worker['log_uri'],'block-project-ssh-keys':'true'}.items())
        command=['gcloud','compute','instances','create',worker['name'],
            '--project='+PROJECT,'--zone='+ZONE,'--machine-type='+spec['machine_type'],
            '--image='+spec['os_image'],'--boot-disk-size=600GB','--boot-disk-type=pd-balanced',
            '--boot-disk-auto-delete','--provisioning-model=STANDARD',
            '--max-run-duration=72h','--instance-termination-action=STOP',
            '--no-restart-on-failure','--service-account='+SERVICE_ACCOUNT,
            '--scopes=cloud-platform','--network=default','--subnet=default',
            '--metadata='+metadata,'--metadata-from-file=startup-script='+spec['startup'],
            '--labels=team=frank,owner=jtantalean,project=dnabr-lai,round=r02,role=worker',
            '--format=json','--quiet']
        result=subprocess.run(command,capture_output=True,text=True,timeout=300)
        write_new(receipt,dict(worker=worker['worker'],command=command,
            finished_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,
            response=json.loads(result.stdout) if result.returncode==0 else None,
            stderr=result.stderr))
        if result.returncode: raise RuntimeError(worker['name']+': '+result.stderr)
        return worker['name']
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(one,indices): print(result,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','upload','launch'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--workers',help='Explicit worker indices; required for launch')
    args=parser.parse_args()
    run=args.run_dir.resolve(); folder=run/'repairs/parallel-v1'
    if args.mode=='prepare': return prepare(run)
    spec=read(folder/'fleet.json'); validate(spec)
    if args.mode=='upload':
        receipts=[upload_one(spec['runtime'],spec['runtime_uri'])]
        for worker in spec['workers']:
            receipts.append(upload_one(worker['bundle'],worker['bundle_uri']))
        write_new(folder/'upload.json',receipts)
    else:
        indices=[int(x) for x in (args.workers or '').split(',') if x]
        if not indices or len(indices)!=len(set(indices)) or any(x<1 or x>12 for x in indices):
            parser.error('Specify distinct worker indices1–12')
        launch(spec,indices,folder)

if __name__=='__main__': main()
