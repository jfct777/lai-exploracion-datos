#!/usr/bin/env python3
"""Delete only authenticated, stopped R02 workers after verified publication.

The authorized temporary VM and its sole auto-delete boot disk are removed only
after every assigned chromosome was imported by the coordinator and its remote
results were rechecked. Failed/unpublished/running workers are retained. This
program never stops machines, deletes GCS objects, changes IAM, or deletes disks
separately. Preparation is local; cloud deletion requires explicit --run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import types


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def fixed(path, value):
    path = Path(path)
    if path.exists():
        require(not path.is_symlink() and read(path) == value, 'Immutable cleanup record changed')
        return
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def status(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.cleanup-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(dict(updated_utc=now(), **value), handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def stamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'An explicit timezone is required')
    return parsed.timestamp()


def instance_from_creation(receipt):
    require(receipt.get('returncode') == 0 and type(receipt.get('returncode')) is int,
            'Worker creation was not successful')
    response = receipt.get('response')
    require(isinstance(response, list) and len(response) == 1 and isinstance(response[0], dict),
            'Creation receipt must identify exactly one VM')
    return response[0]


def validate_instance(actual, worker, *, stopped=False):
    require(str(actual.get('id')) == worker['instance_id'], 'VM native identity changed')
    require(actual.get('name') == worker['name'], 'VM name changed')
    base = f'projects/{worker["project"]}/zones/{worker["zone"]}'
    require(str(actual.get('selfLink', '')).endswith('/'+base+'/instances/'+worker['name']),
            'VM project/zone changed')
    require(all(actual.get('labels', {}).get(k) == v for k, v in
                {'team': 'frank', 'round': 'r02', 'role': 'worker'}.items()), 'VM ownership labels changed')
    disks = actual.get('disks', [])
    require(len(disks) == 1 and disks[0].get('boot') is True and disks[0].get('autoDelete') is True,
            'Cleanup requires one auto-delete boot disk and no additional disks')
    require(disks[0].get('source') == worker['boot_disk_source'], 'Boot disk changed')
    require(str(disks[0].get('source', '')).endswith('/'+base+'/disks/'+worker['boot_disk_name']),
            'Boot disk project/zone/name changed')
    if stopped:
        require(actual.get('status') == 'TERMINATED', 'Running workers cannot be deleted')


def prepare(run, coordinator_source, *, coordinator_manifest=None, cleanup_directory=None):
    """Seal all twelve native identities after creation, without cloud mutations."""
    run = Path(run).resolve()
    parent = run/'repairs/parallel-v1'
    fleet_path = parent/'fleet.json'
    coordinator_path = Path(coordinator_manifest).resolve() if coordinator_manifest else parent/'manifest.json'
    legacy_folder = parent/'cleanup'
    amended_folder = run/'repairs/preprocess-v1/cleanup'
    repaired_folder = run/'repairs/preprocess-v2/cleanup'
    folder = Path(cleanup_directory).resolve() if cleanup_directory else legacy_folder
    require((coordinator_path == parent/'manifest.json' and folder == legacy_folder)
            or (coordinator_path == run/'repairs/preprocess-v1/coordinator/manifest.json'
                and folder == amended_folder)
            or (coordinator_path == run/'repairs/preprocess-v2/coordinator/manifest.json'
                and folder == repaired_folder), 'Coordinator and cleanup versions must use matching approved directories')
    fleet, coordinator = read(fleet_path), read(coordinator_path)
    require(fleet.get('schema') == 'r02_fleet_v1' and fleet['project'] == 'uspbr-242713'
            and fleet['zone'] == 'us-central1-a', 'Unexpected fleet')
    require(coordinator.get('schema') == 'r02_parallel_coordinator_v1'
            and coordinator['run_dir'] == str(run), 'Wrong coordinator manifest')
    require(sha(coordinator_source) == coordinator['coordinator_sha256'], 'Coordinator source differs')
    require(0 < stamp(coordinator['deadline_utc'])-time.time() <= 80*3600, 'Cleanup deadline must be within80 hours')
    remote = coordinator['remote_chromosomes']
    require(len(remote) == 20 and {entry['chromosome'] for entry in remote} == set(range(1,21)),
            'Coordinator chromosome coverage differs')
    require(len(fleet['workers']) == 12 and {entry['worker'] for entry in fleet['workers']}
            == {entry['worker_id'] for entry in remote}, 'Fleet and coordinator worker identities differ')
    workers = []
    for entry in fleet['workers']:
        worker = entry['worker']
        require(re.fullmatch(r'worker(?:0[1-9]|1[0-2])', worker), 'Unexpected worker ID')
        require(re.fullmatch(r'dnabr-r02p-1001-w(?:0[1-9]|1[0-2])', entry['name']), 'Unexpected VM name')
        assigned = [item for item in remote if item['worker_id'] == worker]
        require(sorted(entry['chromosomes']) == sorted(item['chromosome'] for item in assigned),
                'Chromosome assignment changed')
        require(all(item['worker_run_sha256'] == entry['run_sha256']
                    and item['completion_uri'] == entry['completion_uri'] for item in assigned),
                'Worker output/source identity differs')
        creation = parent/(worker+'.creation.json')
        require(creation.is_file() and not creation.is_symlink(), 'All twelve creation receipts are required')
        receipt = read(creation)
        require(receipt.get('worker') == worker, 'Creation worker identity differs')
        vm = instance_from_creation(receipt)
        disks = vm.get('disks', [])
        require(len(disks) == 1, 'Unexpected worker disks')
        disk_source = str(disks[0].get('source', ''))
        target = dict(worker_id=worker, name=entry['name'], project=fleet['project'], zone=fleet['zone'],
            instance_id=str(vm.get('id')), boot_disk_source=disk_source,
            boot_disk_name=disk_source.rsplit('/',1)[-1], creation_file=str(creation), creation_sha256=sha(creation),
            chromosomes=sorted(entry['chromosomes']), completion_uri=entry['completion_uri'])
        require(re.fullmatch(r'[0-9]+', target['instance_id']), 'Missing native VM ID')
        validate_instance(vm, target)
        workers.append(target)
    require(len({item['name'] for item in workers}) == 12 and len({item['instance_id'] for item in workers}) == 12,
            'Duplicated VM cleanup identity')
    require(not folder.exists(), 'Use a new cleanup directory; existing state is immutable')
    folder.mkdir(mode=0o700)
    for name, source in [('cleanup.py', Path(__file__)), ('coordinator.py', Path(coordinator_source))]:
        with (folder/name).open('xb') as handle:
            handle.write(source.read_bytes())
    manifest = dict(schema='r02_success_only_cleanup_v1', parent_run=str(run),
        cleanup_directory=str(folder),
        fleet_file=str(fleet_path), fleet_sha256=sha(fleet_path),
        coordinator_manifest=str(coordinator_path), coordinator_manifest_sha256=sha(coordinator_path),
        source_sha256=sha(folder/'cleanup.py'), coordinator_source_sha256=sha(folder/'coordinator.py'),
        deadline_utc=coordinator['deadline_utc'], poll_seconds=30, workers=workers,
        policy='Only stopped success-published/imported workers; boot auto-delete; no direct disk/GCS deletion')
    fixed(folder/'manifest.json', manifest)
    return folder/'manifest.json'


def validate(path, expected):
    path = Path(path).resolve()
    require(sha(path) == expected, 'Cleanup manifest changed')
    spec = read(path)
    require(spec.get('schema') == 'r02_success_only_cleanup_v1', 'Invalid cleanup schema')
    run = Path(spec['parent_run'])
    allowed = {run/'repairs/parallel-v1/cleanup', run/'repairs/preprocess-v1/cleanup',
               run/'repairs/preprocess-v2/cleanup'}
    require(path.parent in allowed and path.parent == Path(spec.get('cleanup_directory',
            str(run/'repairs/parallel-v1/cleanup'))), 'Wrong cleanup directory')
    expected_coordinator = (run/'repairs/preprocess-v2/coordinator/manifest.json'
                            if path.parent == run/'repairs/preprocess-v2/cleanup' else
                            run/'repairs/preprocess-v1/coordinator/manifest.json'
                            if path.parent == run/'repairs/preprocess-v1/cleanup'
                            else run/'repairs/parallel-v1/manifest.json')
    require(Path(spec['coordinator_manifest']) == expected_coordinator, 'Wrong version of cleanup coordinator')
    require(sha(__file__) == spec['source_sha256'], 'Cleanup source changed')
    for key in ('fleet', 'coordinator_manifest'):
        file_key = 'fleet_file' if key == 'fleet' else key
        require(sha(spec[file_key]) == spec[key+'_sha256'], 'Changed cleanup dependency: '+key)
    require(sha(path.parent/'coordinator.py') == spec['coordinator_source_sha256'], 'Coordinator helper changed')
    for worker in spec['workers']:
        require(sha(worker['creation_file']) == worker['creation_sha256'], 'Creation receipt changed')
        validate_instance(instance_from_creation(read(worker['creation_file'])), worker)
    helper = types.ModuleType('_r02_cleanup_coordinator')
    helper.__file__ = str(path.parent/'coordinator.py')
    exec(compile(Path(helper.__file__).read_bytes(), helper.__file__, 'exec'), helper.__dict__)
    coordinator = read(spec['coordinator_manifest'])
    require(coordinator['coordinator_sha256'] == spec['coordinator_source_sha256'], 'Wrong authenticated helper')
    helper.validate_operational_amendments(coordinator)
    return spec, coordinator, helper


class Compute:
    def describe(self, worker, resource='instances'):
        name = worker['name'] if resource == 'instances' else worker['boot_disk_name']
        result = subprocess.run(['gcloud','compute',resource,'describe',name,
            '--project='+worker['project'],'--zone='+worker['zone'],'--format=json'],
            capture_output=True, text=True, timeout=120)
        if result.returncode:
            if 'was not found' in result.stderr and (name in result.stderr) and ('resource' in result.stderr.lower()):
                return None
            raise RuntimeError('Cannot verify cloud resource: '+result.stderr[-1000:])
        return json.loads(result.stdout)

    def delete(self, worker):
        result = subprocess.run(['gcloud','compute','instances','delete',worker['name'],
            '--project='+worker['project'],'--zone='+worker['zone'],'--quiet'],
            capture_output=True,text=True,timeout=300)
        require(result.returncode == 0, 'VM deletion failed: '+result.stderr[-1000:])


def verified_success(worker, spec, coordinator, helper, cloud):
    imports = Path(spec['parent_run'])/'repairs/parallel-v1/imports'
    proofs = []
    for chromosome in worker['chromosomes']:
        path = imports/f'chr{chromosome:02d}_import.json'
        if not path.is_file():
            return None
        require(not path.is_symlink(), 'Symlinked import receipt')
        proof = read(path)
        require(proof.get('schema') == 'r02_remote_import_v1' and proof.get('chromosome') == chromosome
                and proof.get('worker_id') == worker['worker_id']
                and proof.get('local_compute_checkpoint_created') is False, 'Wrong import receipt')
        require(proof['completion']['uri'] == worker['completion_uri'], 'Completion URI changed')
        proofs.append((path, proof))
    require(all(proof['completion'] == proofs[0][1]['completion'] for _, proof in proofs),
            'Worker chromosomes came from different completion receipts')
    completion = proofs[0][1]['completion']
    helper.validate_metadata(cloud.metadata(worker['completion_uri']), completion)
    raw = cloud.read(worker['completion_uri'], completion['generation'])
    require(hashlib.sha256(raw).hexdigest() == completion['sha256'], 'Completion SHA changed')
    payload = json.loads(raw)
    for _, proof in proofs:
        entry = next(item for item in coordinator['remote_chromosomes'] if item['chromosome'] == proof['chromosome'])
        records = helper.validate_worker_completion(payload, entry, coordinator)
        helper.verify_operational_provenance(payload, entry, coordinator, cloud)
        amendment = payload.get('operational_amendment')
        require(proof.get('operational_amendment') == amendment
                and proof.get('operational_provenance') == payload.get('operational_provenance'),
                'Imported operational provenance differs from worker completion')
        if amendment is not None:
            expected_source = (amendment['source_manifest_sha256']
                               if proof['chromosome'] in amendment['new_preprocess_chromosomes']
                               else coordinator['source_manifest_sha256'])
            require(proof.get('preprocess_source_manifest_sha256') == expected_source,
                    'Imported per-chromosome preprocessing source differs')
        outputs = proof['outputs']
        require(len(outputs) == len(helper.ESSENTIAL)
                and {item['relative_path'] for item in outputs} == helper.ESSENTIAL, 'Incomplete imported files')
        for output in outputs:
            original = records[output['relative_path']]
            require(all(output[key] == original[key] for key in ('uri','generation','bytes','sha256','md5_base64')),
                    'Imported source identity differs')
            path = Path(output['path'])
            expected = Path(spec['parent_run'])/f'chr{proof["chromosome"]:02d}'/output['relative_path']
            require(path == expected and path.resolve() == path and path.is_file()
                    and path.stat().st_size == output['bytes'] and sha(path) == output['sha256'],
                    'Local imported result changed')
        for record in records.values():
            helper.validate_metadata(cloud.metadata(record['uri']), record)
    return dict(completion=completion, import_receipts={str(path):sha(path) for path,_ in proofs})


def cleanup_one(worker, spec, coordinator, helper, folder, cloud, compute):
    """One bounded attempt. No import proof means no cloud deletion."""
    success_path = folder/(worker['worker_id']+'.deleted.json')
    if success_path.exists():
        prior = read(success_path)
        require(prior.get('instance_id') == worker['instance_id'], 'Deletion receipt identity changed')
        return 'DELETED_VERIFIED'
    imports = Path(spec['parent_run'])/'repairs/parallel-v1/imports'
    if not all((imports/f'chr{c:02d}_import.json').is_file() for c in worker['chromosomes']):
        return 'RETAINED_WAITING_VERIFIED_IMPORTS'
    vm = compute.describe(worker)
    intent_path = folder/(worker['worker_id']+'.delete_intent.json')
    if vm is None:
        if not intent_path.exists():
            return 'ALREADY_ABSENT_NOT_DELETED_BY_CLEANUP'
        if compute.describe(worker, 'disks') is not None:
            return 'VM_ABSENT_BOOT_DISK_STILL_PRESENT'
        fixed(success_path, dict(worker_id=worker['worker_id'],instance_id=worker['instance_id'],
            name=worker['name'],state='DELETED_VERIFIED',intent_sha256=sha(intent_path),boot_disk_absent=True))
        return 'DELETED_VERIFIED'
    validate_instance(vm, worker)
    if vm.get('status') != 'TERMINATED':
        return 'RETAINED_WAITING_VM_STOP'
    proof = verified_success(worker, spec, coordinator, helper, cloud)
    if proof is None:
        return 'RETAINED_WAITING_VERIFIED_IMPORTS'
    intent = dict(worker_id=worker['worker_id'],instance_id=worker['instance_id'],name=worker['name'],
                  boot_disk_source=worker['boot_disk_source'],**proof)
    fixed(intent_path,intent)
    # Recheck immediately before the destructive operation; names alone do not
    # authorize deleting a replacement VM with the same name.
    current = compute.describe(worker)
    require(current is not None, 'VM disappeared before deletion; retry observation')
    validate_instance(current,worker,stopped=True)
    compute.delete(worker)
    require(compute.describe(worker) is None, 'VM still exists after deletion')
    if compute.describe(worker,'disks') is not None:
        return 'VM_ABSENT_BOOT_DISK_STILL_PRESENT'
    fixed(success_path,dict(worker_id=worker['worker_id'],instance_id=worker['instance_id'],
        name=worker['name'],state='DELETED_VERIFIED',intent_sha256=sha(intent_path),boot_disk_absent=True))
    return 'DELETED_VERIFIED'


def run_cleanup(path, expected):
    spec, coordinator, helper = validate(path, expected)
    folder = Path(path).resolve().parent
    deadline = stamp(spec['deadline_utc'])
    require(0 < deadline-time.time() <= 80*3600, 'Cleanup deadline expired or unbounded')
    states = {}
    def alarm(_number,_frame):
        status(folder/'status.json',dict(state='DEADLINE_RETAINED_UNVERIFIED_WORKERS',workers=states))
        raise TimeoutError('Temporary worker cleanup reached total deadline')
    signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    cloud, compute = helper.GCS(), Compute()
    with (folder/'.cleanup.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while time.time() < deadline:
            for worker in spec['workers']:
                try:
                    states[worker['worker_id']] = dict(state=cleanup_one(worker,spec,coordinator,helper,folder,cloud,compute))
                except Exception as error:
                    if isinstance(error,TimeoutError):
                        raise
                    states[worker['worker_id']] = dict(state='RETAINED_VERIFICATION_ERROR',error=str(error))
            complete = all(value['state']=='DELETED_VERIFIED' for value in states.values())
            status(folder/'status.json',dict(state='ALL_SUCCESS_WORKERS_REMOVED' if complete else 'MONITORING',workers=states))
            if complete:
                return
            time.sleep(min(30,max(0,deadline-time.time())))
    status(folder/'status.json',dict(state='DEADLINE_RETAINED_UNVERIFIED_WORKERS',workers=states))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','run'])
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--coordinator-source',type=Path)
    parser.add_argument('--coordinator-manifest',type=Path)
    parser.add_argument('--cleanup-directory',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--expected-manifest-sha256')
    args=parser.parse_args()
    os.umask(0o077)
    if args.mode=='prepare':
        if not args.run_dir or not args.coordinator_source:
            parser.error('prepare requires --run-dir and --coordinator-source')
        path=prepare(args.run_dir,args.coordinator_source,
                     coordinator_manifest=args.coordinator_manifest, cleanup_directory=args.cleanup_directory)
        print(json.dumps(dict(state='PREPARED_NOT_RUNNING',manifest=str(path),sha256=sha(path))))
    else:
        if not args.manifest or not args.expected_manifest_sha256:
            parser.error('verify/run requires explicit manifest and independent SHA256')
        if args.mode=='verify':
            validate(args.manifest,args.expected_manifest_sha256)
            print(json.dumps(dict(state='VERIFIED_NOT_RUNNING')))
        else:
            run_cleanup(args.manifest,args.expected_manifest_sha256)


if __name__=='__main__':
    main()
