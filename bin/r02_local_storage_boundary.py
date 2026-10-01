#!/usr/bin/env python3
"""Switch one authenticated worker's temporary storage after M01, without changing science.

Default is read-only preflight. --apply stops only its Nextflow JVM, lets the
existing M01 container finish, authenticates/copies its normalized outputs, then
bind-mounts local storage at the exact existing temporary path and resumes Java.
Original GCS objects are retained. This is not a new genomic analysis.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time


SCHEMA = 'r02_local_storage_boundary_v1'
MOUNT = Path('/home/jose.tantalean/gcs-dnabr')
BUCKET = 'gs://projects-usp/dnaBr-lai/datalake/'
GIB = 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Repeated JSON field')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def utc():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    require(type(pid) is int and pid > 1, 'Unsafe process ID')
    directory = Path('/proc')/str(pid)
    try:
        tail = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        command = (directory/'cmdline').read_bytes()
    except FileNotFoundError:
        return None
    return dict(pid=pid, state=tail[0], start_ticks=int(tail[19]),
                cmdline_sha256=hashlib.sha256(command).hexdigest(), command=command)


def authenticate(identity):
    require(set(identity) == {'pid', 'start_ticks', 'cmdline_sha256'}
            and type(identity['start_ticks']) is int and identity['start_ticks'] > 0
            and digest(identity['cmdline_sha256']), 'Invalid process identity')
    actual = process_identity(identity['pid'])
    require(actual is not None and actual['state'] != 'Z'
            and all(actual[key] == value for key, value in identity.items()),
            'Process exited or identity changed')
    return actual


def signal_owned(identity, number):
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'),
            'Race-safe pidfd signalling is required')
    handle = os.pidfd_open(identity['pid'])
    try:
        authenticate(identity)  # Recheck AFTER opening the kernel process handle.
        signal.pidfd_send_signal(handle, number)
    finally:
        os.close(handle)


def file_metadata(path):
    info = Path(path).lstat()
    require(stat.S_ISREG(info.st_mode), 'Only regular files can be copied')
    return dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns, mode=stat.S_IMODE(info.st_mode),
                uid=info.st_uid, gid=info.st_gid)


def mount_info(path):
    result = json.loads(subprocess.check_output(
        ['findmnt', '-J', '-T', str(path), '-o', 'TARGET,FSTYPE,SOURCE'], text=True, timeout=30))
    entries = result['filesystems']
    require(len(entries) == 1, 'Ambiguous filesystem mount')
    return entries[0]


def active_containers(run_id):
    return subprocess.check_output(['docker', 'ps', '-q', '--filter', 'label=dnabr-r02='+run_id],
                                   text=True, timeout=60).split()


def cloud_metadata(path):
    uri = BUCKET + str(Path(path).relative_to(MOUNT))
    record = json.loads(subprocess.check_output(
        ['gcloud', 'storage', 'objects', 'describe', uri, '--format=json'], text=True, timeout=120))
    md5 = record.get('md5_hash', record.get('md5Hash'))
    require(str(record.get('generation', '')).isdigit() and md5,
            'GCS object needs a generation and MD5 before copying')
    return dict(uri=uri, generation=str(record['generation']), bytes=int(record['size']), md5_base64=md5)


def copy_verified(source, destination):
    """copy2 preserves timestamps; verify GCS identity and local SHA256/MD5 afterward."""
    source, destination = Path(source), Path(destination)
    require(not destination.exists(), 'Refuse to overwrite an existing local copy')
    before, cloud = file_metadata(source), cloud_metadata(source)
    require(before['bytes'] == cloud['bytes'], 'FUSE size differs from GCS object')
    handle, temporary_name = tempfile.mkstemp(prefix=destination.name+'.copying-', dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    shutil.copy2(source, temporary, follow_symlinks=False)
    os.chown(temporary, before['uid'], before['gid'], follow_symlinks=False)
    local_sha, local_md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with temporary.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            local_sha.update(block); local_md5.update(block)
    require(file_metadata(source) == before and cloud_metadata(source) == cloud,
            'Source changed during transfer')
    require(file_metadata(temporary) == before, 'Copied file metadata differs')
    require(base64.b64encode(local_md5.digest()).decode() == cloud['md5_base64'],
            'Copied bytes differ from GCS MD5')
    os.link(temporary, destination)  # Exclusive installation: fails if destination appeared.
    temporary.unlink()
    return {**before, **cloud, 'source':str(source), 'destination':str(destination),
            'sha256':local_sha.hexdigest()}


class Boundary:
    def __init__(self, spec_path, expected_sha256):
        self.spec_path = Path(spec_path).resolve()
        require(digest(expected_sha256) and sha(self.spec_path) == expected_sha256, 'Spec hash mismatch')
        self.spec_sha256 = expected_sha256
        self.spec = read(self.spec_path)
        require(self.spec.get('schema') == SCHEMA, 'Wrong storage transition schema')
        require(self.spec.get('helper_sha256') == sha(__file__), 'Helper differs from frozen specification')
        self.run = Path(self.spec['run_dir'])
        require(self.run.is_absolute() and self.run.resolve() == self.run and self.run.is_dir(), 'Unsafe run directory')
        require(sha(self.run/'frozen.sha256.json') == self.spec['worker_frozen_sha256']
                and sha(self.run/'run.json') == self.spec['worker_run_sha256'], 'Worker settings changed')
        frozen = read(self.run/'frozen.sha256.json')
        for relative, expected in frozen.items():
            p = Path(relative)
            require(not p.is_absolute() and '..' not in p.parts and not (self.run/p).is_symlink()
                    and sha(self.run/p) == expected, 'Worker frozen input changed')
        self.config = read(self.run/'run.json')
        self.worker = self.config['parallel_worker']['worker_id']
        require(re.fullmatch(r'worker[0-9]{2}', self.worker) is not None
                and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{1,100}',
                                 self.config['parallel_worker']['parent_run_id']) is not None,
                'Unsafe worker or parent-run identifier')
        self.chrom = self.spec['chromosome']
        require(type(self.chrom) is int and self.chrom in self.config['processing_order']
                and 1 <= self.chrom <= 20, 'Chromosome not assigned to this worker')
        self.target = Path(self.config['bulk'])
        expected_target = MOUNT/'transient/DNABR_QC/R02_20260930'/self.config['parallel_worker']['parent_run_id']/'parallel'/self.worker
        require(self.target == expected_target and not self.target.is_symlink(), 'Bulk target escaped worker prefix')
        self.backing = self.run/'local_bulk'
        self.task = Path(self.spec['task_dir'])
        work = self.run/f'chr{self.chrom:02d}/work'
        require(self.task.is_absolute() and self.task.resolve() == self.task
                and self.task.parent.parent == work
                and re.fullmatch(r'[a-f0-9]{2}', self.task.parent.name)
                and re.fullmatch(r'[a-f0-9]{6,64}', self.task.name), 'Unexpected Nextflow task directory')
        require(sha(self.task/'.command.sh') == self.spec['task_command_sha256'], 'M01 command changed')
        for name in ('parameters.json', 'runtime.config'):
            require(sha(self.run/f'chr{self.chrom:02d}'/name) == self.spec[name.replace('.', '_')+'_sha256'],
                    'Current workflow settings changed: '+name)
        self.timeout = self.spec.get('timeout_seconds', 72*3600)
        self.poll = self.spec.get('poll_seconds', 15)
        require(type(self.timeout) is int and 0 < self.timeout <= 72*3600, 'Invalid bounded timeout')
        require(type(self.poll) is int and 0 < self.poll <= 30, 'Invalid polling interval')
        raw = str(Path(self.config['raw_dir'])/f'dnabr.hg38.2723.chr{self.chrom}.vcf.gz')
        raw_record = read(self.run/'input_objects.json')[raw]
        require(int(raw_record['size']) > 0, 'Invalid authenticated compressed input size')
        self.required_bytes = 3*int(raw_record['size']) + 32*GIB
        self.receipt_dir = self.run/'storage_transition'/self.spec_sha256[:16]
        self.events = self.receipt_dir/'events.jsonl'
        self.owned_stop = False
        self.mounted = False
        self.committed = False
        self.deadline = None

    def event(self, state, **data):
        with self.events.open('a') as handle:
            handle.write(json.dumps(dict(utc=utc(), state=state, **data), sort_keys=True, allow_nan=False)+'\n')
            handle.flush(); os.fsync(handle.fileno())

    def no_downstream(self):
        work = self.task.parent.parent
        require(not any(p.parent != self.task for p in work.glob('*/*/.command.sh')),
                'Another workflow task has been prepared; boundary is no longer unambiguous')
        require(not (self.run/'checkpoints'/f'chr{self.chrom:02d}_M01_M02_M021.json').exists(),
                'Preprocessing already finished')

    def preflight(self):
        java = authenticate(self.spec['nextflow'])
        runner = authenticate(self.spec['runner'])
        require(b'nextflow' in java['command'] and str(self.run).encode() in java['command'],
                'JVM command does not identify this worker workflow')
        require(b'worker.py' in runner['command'] and str(self.run).encode() in runner['command'],
                'Runner command does not identify this worker')
        require(java['state'] not in ('T', 't'), 'JVM was already stopped by another action')
        status = read(self.run/'status.json')
        require(status.get('stage') == f'chr{self.chrom:02d}_M01_M02_M021'
                and status.get('state') == 'RUNNING', 'Worker is not in its declared preprocessing workflow')
        self.no_downstream()
        require(self.target.is_dir() and mount_info(self.target)['fstype'].startswith('fuse'),
                'Expected existing GCS FUSE temporary directory')
        require(not self.backing.exists(), 'Local backing directory already exists; inspect previous attempt')
        free = shutil.disk_usage(self.run).free
        require(free >= self.required_bytes, 'Insufficient local disk for 3×compressed input +32GiB')
        return dict(state='PREFLIGHT_VERIFIED_NOT_APPLIED', chromosome=self.chrom,
                    target=str(self.target), backing=str(self.backing), free_bytes=free,
                    required_free_bytes=self.required_bytes, scientific_parameters_changed=False)

    def wait_finished(self):
        while True:
            require(time.monotonic() < self.deadline, 'Timed out waiting for M01 completion')
            authenticate(self.spec['runner'])
            require(authenticate(self.spec['nextflow'])['state'] in ('T', 't'), 'JVM no longer paused')
            self.no_downstream()
            exit_path = self.task/'.exitcode'
            if exit_path.is_file():
                value = exit_path.read_text().strip()
                require(re.fullmatch(r'-?\d+', value), 'Malformed M01 exit code')
                require(int(value) == 0, 'M01 failed; no storage transition permitted')
                if not active_containers(self.config['run_id']):
                    return
            time.sleep(self.poll)

    def copy_outputs(self):
        base = f'dnabr.hg38.2723.chr{self.chrom}'
        marker = self.task/(base+'.m01.large_temp_dir.txt')
        source = Path(marker.read_text().strip())
        require(source.parent == self.target and re.fullmatch(fr'm01_chr{self.chrom}\.[A-Za-z0-9]+', source.name)
                and source.is_dir() and not source.is_symlink(), 'Unexpected M01 temporary directory')
        require(list(self.target.iterdir()) == [source], 'Unexpected directories at temporary target')
        copied_names = {base+'.norm.vcf.gz', base+'.norm.vcf.gz.tbi', base+'.norm.log'}
        omitted_names = {base+'.original.bcf'}
        entries = list(source.iterdir())
        require({p.name for p in entries} == copied_names | omitted_names,
                'Missing or unexpected M01 temporary files')
        for p in entries:
            file_metadata(p)
        self.backing.mkdir(mode=0o700)
        destination = self.backing/source.name
        destination.mkdir(mode=0o700)
        records = [copy_verified(source/name, destination/name) for name in sorted(copied_names)]
        retained = [dict(path=str(source/name), **cloud_metadata(source/name)) for name in sorted(omitted_names)]
        shutil.copystat(source, destination, follow_symlinks=False)
        shutil.copystat(self.target, self.backing, follow_symlinks=False)
        os.chown(destination, source.stat().st_uid, source.stat().st_gid, follow_symlinks=False)
        os.chown(self.backing, self.target.stat().st_uid, self.target.stat().st_gid, follow_symlinks=False)
        require(shutil.disk_usage(self.run).free >= 12*GIB, 'Disk below existing 12GiB safety limit after copying')
        self.event('COPIED_VERIFIED', files=records, retained_gcs_objects=retained,
                   original_gcs_objects_deleted=False,
                   cleanup_note='Later cleanup paths are local bind-mounted files; recorded GCS URIs do not prove GCS deletion')
        return records

    def validate_mount(self, records):
        info = mount_info(self.target)
        require(info['target'] == str(self.target) and not info['fstype'].startswith('fuse')
                and os.path.samefile(self.target, self.backing), 'Bind mount is not the expected local backing directory')
        for item in records:
            target = self.target/Path(item['destination']).relative_to(self.backing)
            require(file_metadata(target) == {key: item[key] for key in ('bytes', 'mtime_ns', 'mode', 'uid', 'gid')}
                    and sha(target) == item['sha256'], 'Bound output differs from verified copy')
        return info

    def apply(self):
        require(os.geteuid() == 0, 'Bind-mount transition requires root')
        self.receipt_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        with (self.receipt_dir/'lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not self.events.exists(), 'A previous transition attempt exists; inspect its append-only record')
            checked = self.preflight()
            self.deadline = time.monotonic()+self.timeout
            identity = process_identity(os.getpid())
            self.event('STARTED', spec_sha256=self.spec_sha256, helper_sha256=sha(__file__),
                       helper_identity={k: identity[k] for k in ('pid', 'start_ticks', 'cmdline_sha256')},
                       preflight=checked)
            try:
                signal_owned(self.spec['nextflow'], signal.SIGSTOP)
                self.owned_stop = True
                for _ in range(100):
                    if authenticate(self.spec['nextflow'])['state'] in ('T', 't'):
                        break
                    time.sleep(.05)
                require(authenticate(self.spec['nextflow'])['state'] in ('T', 't'), 'JVM did not stop')
                self.event('JVM_STOPPED_M01_CONTAINER_CONTINUES', nextflow=self.spec['nextflow'])
                self.wait_finished()
                self.event('M01_COMPLETED_ZERO_NO_ACTIVE_CONTAINERS')
                records = self.copy_outputs()
                self.no_downstream()
                require(not active_containers(self.config['run_id']), 'Unexpected active worker container')
                require(authenticate(self.spec['nextflow'])['state'] in ('T', 't'), 'JVM resumed before bind mount')
                subprocess.run(['mount', '--bind', str(self.backing), str(self.target)], check=True, timeout=60)
                self.mounted = True
                info = self.validate_mount(records)
                self.event('LOCAL_BIND_VERIFIED', mount=info, original_gcs_objects_deleted=False)
                signal_owned(self.spec['nextflow'], signal.SIGCONT)
                self.owned_stop = False
                self.committed = True
                self.event('COMPLETE_JVM_RESUMED', scientific_parameters_changed=False)
            except BaseException as error:
                try:
                    self.event('FAILED', error_type=type(error).__name__, error=str(error))
                except OSError:
                    pass  # Logging failure must not prevent unpausing a safely stopped JVM.
                # A mount command can time out after the kernel performed it.
                if self.backing.exists() and self.target.exists() and os.path.samefile(self.target, self.backing):
                    self.mounted = True
                if self.mounted and not self.committed:
                    try:
                        subprocess.run(['umount', str(self.target)], check=True, timeout=60)
                        require(mount_info(self.target)['fstype'].startswith('fuse'), 'Original FUSE path not restored')
                        self.mounted = False
                        self.event('OWN_BIND_ROLLED_BACK_ORIGINAL_GCS_RESTORED')
                    except BaseException as rollback_error:
                        self.event('BLOCKED_ROLLBACK_FAILED_JVM_REMAINS_PAUSED', error=str(rollback_error))
                        raise RuntimeError('Own bind rollback failed; JVM remains paused for explicit recovery') from rollback_error
                raise
            finally:
                if self.owned_stop and not self.mounted:
                    signal_owned(self.spec['nextflow'], signal.SIGCONT)
                    self.owned_stop = False
                    self.event('ABORT_JVM_RESUMED_ORIGINAL_STORAGE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--apply', action='store_true', help='Default is read-only preflight')
    args = parser.parse_args()
    os.umask(0o077)
    boundary = Boundary(args.spec, args.spec_sha256)
    if not args.apply:
        print(json.dumps(boundary.preflight(), sort_keys=True))
        return
    def interrupted(number, _frame):
        raise InterruptedError(f'Helper received signal {number}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, boundary.timeout)
    try:
        boundary.apply()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
