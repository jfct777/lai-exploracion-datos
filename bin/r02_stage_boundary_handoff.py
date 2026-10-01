#!/usr/bin/env python3
"""Finish one authenticated R02 subprocess, then hand off without rerunning it.

Linux-only operational barrier: only the existing Python supervisor receives a
soft RLIMIT_NPROC of zero. Its already-running Nextflow child retains its limits.
The old supervisor can reap that child and write its normal checkpoint, but its
next Popen fails with EAGAIN. No frozen source is edited and no ptrace is used.

This program changes live process state ONLY when its CLI is explicitly run.
The request hash must be supplied independently of the request file. A separate
watchdog is essential because the old supervisor's docker-based cleanup also
requires fork. On an emergency, authenticated processes are terminated, never
resumed with restored limits into an old biological stage.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, once=False):
    """Durable atomic status writes; immutable records use create-only linking."""
    path = Path(path)
    if once and path.exists():
        if read_json(path) != value:
            raise ValueError('Immutable handoff record differs: ' + str(path))
        return
    payload = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if once:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def member(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Unsafe manifest member')
    path = root / relative
    if path.resolve() != path or not path.is_file():
        raise ValueError('Missing or symlinked manifest member: ' + str(path))
    return path


def verify_manifest(root, path):
    manifest = read_json(path)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Empty source/frozen manifest')
    for relative, expected in manifest.items():
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid manifest digest')
        if sha(member(root, relative)) != expected:
            raise ValueError('Manifest hash mismatch: ' + relative)
    return manifest


def process_info(pid):
    """Read identity without exposing argv/environment in operational records."""
    root = Path('/proc') / str(pid)
    try:
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        status = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines())
        command = (root / 'cmdline').read_bytes()
        after = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    if stat[19] != after[19]:
        raise RuntimeError('PID changed while reading its identity')
    return dict(pid=int(pid), start_ticks=int(stat[19]), state=after[0],
                ppid=int(after[1]), pgid=int(after[2]), sid=int(after[3]),
                cmdline_sha256=hashlib.sha256(command).hexdigest(),
                uids=[int(x) for x in status['Uid'].split()],
                gids=[int(x) for x in status['Gid'].split()],
                threads=int(status['Threads']),
                cap_eff=int(status['CapEff'].strip(), 16),
                cap_prm=int(status['CapPrm'].strip(), 16))


def authenticated(spec, *, live=True):
    current = process_info(spec['pid'])
    if current is None:
        return None
    if current['start_ticks'] != spec['start_ticks']:
        raise RuntimeError('PID reuse: refusing to signal or adopt')
    if current['state'] in ('Z', 'X'):
        return None if live else current
    if current['cmdline_sha256'] != spec['cmdline_sha256']:
        raise RuntimeError('Process command identity changed')
    return current


def send(spec, sig):
    if authenticated(spec) is None:
        return False
    # An open pidfd removes the final PID-reuse race when supported by Python.
    fd = os.pidfd_open(spec['pid'])
    try:
        if authenticated(spec) is None:
            return False
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)
    return True


def nproc(pid, limits=None):
    if limits is None:
        return resource.prlimit(pid, resource.RLIMIT_NPROC)
    return resource.prlimit(pid, resource.RLIMIT_NPROC, tuple(limits))


def containers(run_id):
    result = subprocess.run(['docker', 'ps', '-q', '--no-trunc', '--filter',
                             'label=dnabr-r02=' + run_id], check=True,
                            capture_output=True, text=True, timeout=30)
    ids = result.stdout.split()
    if any(not re.fullmatch('[0-9a-f]{64}', item) for item in ids):
        raise ValueError('Unexpected Docker container identity')
    return ids


def stop_containers(run_id):
    ids = containers(run_id)
    for container in ids:
        result = subprocess.run(['docker', 'inspect', '--format', '{{json .Config.Labels}}', container],
                                check=True, capture_output=True, text=True, timeout=30)
        if json.loads(result.stdout).get('dnabr-r02') != run_id:
            raise RuntimeError('Container label changed; refusing stop')
        subprocess.run(['docker', 'stop', '--time', '30', container],
                       check=True, capture_output=True, text=True, timeout=60)


def checkpoint_inventory(run):
    return {p.name: sha(p) for p in sorted((run / 'checkpoints').glob('*.json'))}


def wait_stopped(spec):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = authenticated(spec)
        if current is None:
            raise RuntimeError('Supervisor exited while installing barrier')
        if current['state'] == 'T':
            return current
        time.sleep(.02)
    raise RuntimeError('Supervisor did not enter stopped state')


class Handoff:
    def __init__(self, request, expected_sha256):
        self.request_path = Path(request).resolve()
        self.expected_sha256 = expected_sha256
        self.request = None
        self.armed = None
        self.interference_started = False
        self.aborted = False

    def validate(self):
        if not re.fullmatch('[0-9a-f]{64}', self.expected_sha256):
            raise ValueError('Expected request SHA256 must be a lowercase SHA256')
        raw = self.request_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.expected_sha256:
            raise ValueError('Request hash mismatch')
        request = json.loads(raw)
        if request.get('schema_version') != 1:
            raise ValueError('Unsupported request schema')
        run = Path(request['run_dir'])
        amendment = Path(request['amendment_dir'])
        if (not run.is_absolute() or run.resolve() != run or not amendment.is_absolute()
                or amendment.resolve() != amendment or amendment == run
                or not amendment.is_relative_to(run)
                or self.request_path != amendment / 'request.json'):
            raise ValueError('Request/amendment must have explicit local paths within the same run')
        template = request['amendment_template']
        if template.get('schema_version') != 1 or template.get('run_dir') != str(run):
            raise ValueError('Amendment template does not identify the original run')
        boundary = template['boundary']
        chrom = boundary['chromosome']
        if isinstance(chrom, bool) or not isinstance(chrom, int) or not 1 <= chrom <= 22:
            raise ValueError('Invalid boundary chromosome')
        stage = f'chr{chrom:02d}_M01_M02_M021'
        if request['boundary_stage'] != stage or boundary['stage'] != stage:
            raise ValueError('Only an authenticated preprocessing boundary is supported')
        for name, expected in [('frozen.sha256.json', template['original_frozen_sha256']),
                               ('source.sha256.json', template['original_source_manifest_sha256'])]:
            if sha(run / name) != expected:
                raise ValueError('Original frozen provenance changed: ' + name)
        verify_manifest(run, run / 'frozen.sha256.json')
        verify_manifest(run / 'source', run / 'source.sha256.json')
        if sha(amendment / 'source.sha256.json') != request['amendment_source_manifest_sha256']:
            raise ValueError('Amendment source manifest hash mismatch')
        sources = verify_manifest(amendment / 'source', amendment / 'source.sha256.json')
        for required in ('bin/r02_apply_amendment.py', 'bin/r02_stage_boundary_handoff.py'):
            if required not in sources:
                raise ValueError('Required handoff executable is not frozen: ' + required)
        folder = run / f'chr{chrom:02d}'
        for name, key in [('parameters.json', 'parameters_sha256'), ('runtime.config', 'runtime_config_sha256')]:
            if sha(folder / name) != boundary[key]:
                raise ValueError('Current preprocessing parameters changed: ' + name)
        for role in ('supervisor', 'child'):
            spec = request[role]
            if (not isinstance(spec['pid'], int) or spec['pid'] <= 1
                    or not isinstance(spec['start_ticks'], int) or spec['start_ticks'] <= 0
                    or not re.fullmatch('[0-9a-f]{64}', spec['cmdline_sha256'])):
                raise ValueError('Invalid process identity')
        if request['supervisor']['pid'] == request['child']['pid']:
            raise ValueError('Supervisor and child must differ')
        limits = request['original_limits']
        if (len(limits) != 2 or any(not isinstance(n, int) or isinstance(n, bool) for n in limits)
                or limits[0] <= 0 or (limits[1] != -1 and limits[1] < limits[0])):
            raise ValueError('Invalid original nproc limits')
        if request['free_disk_min_gib'] != 12:
            raise ValueError('This handoff preserves the original 12 GiB safety threshold')
        config = read_json(run / 'run.json')
        if config['run_id'] != run.name or config['free_disk_min_gib'] != 12:
            raise ValueError('Original disk/run identity contract changed')
        self.request = request
        self.run, self.amendment, self.run_id = run, amendment, config['run_id']
        return self

    def state(self, state, **extra):
        write_json(self.amendment / 'boundary_status.json',
                   dict(state=state, updated_utc=now(), request_sha256=self.expected_sha256, **extra))
        print(json.dumps(dict(state=state, **extra)), flush=True)

    def eligible(self, *, already_armed=False):
        parent = authenticated(self.request['supervisor'])
        child = authenticated(self.request['child'])
        if parent is None or child is None:
            raise RuntimeError('Both original processes must be active when arming')
        if (parent['uids'] != [os.getuid()] * 4 or parent['gids'] != [os.getgid()] * 4
                or parent['uids'][0] == 0 or parent['cap_eff'] or parent['cap_prm']
                or parent['threads'] != 1):
            raise RuntimeError('Supervisor is not an eligible unprivileged single-threaded process')
        if (child['ppid'] != parent['pid'] or child['pgid'] != child['pid']
                or child['sid'] != child['pid'] or child['uids'] != parent['uids']):
            raise RuntimeError('Nextflow is not the authenticated direct child in its own session')
        children = (Path('/proc') / str(parent['pid']) / 'task' / str(parent['pid']) / 'children').read_text().split()
        if children != [str(child['pid'])]:
            raise RuntimeError('Supervisor has unexpected children')
        status = read_json(self.run / 'status.json')
        if status.get('stage') != self.request['boundary_stage'] or status.get('state') != 'RUNNING' or status.get('pid') != child['pid']:
            raise RuntimeError('Live run status does not match the requested boundary')
        expected = (0, self.request['original_limits'][1]) if already_armed else tuple(self.request['original_limits'])
        if nproc(parent['pid']) != expected:
            raise RuntimeError('Supervisor nproc limits differ from request')
        if shutil.disk_usage(self.run).free / 1024**3 < 12:
            raise RuntimeError('Insufficient free disk before handoff')
        return child

    def arm(self):
        armed_path = self.amendment / 'armed.json'
        if armed_path.exists():
            self.armed = read_json(armed_path)
            if (self.armed['request_sha256'] != self.expected_sha256
                    or self.armed['original_limits'] != self.request['original_limits']):
                raise ValueError('Armed record does not match immutable request')
            self.interference_started = True
            parent = authenticated(self.request['supervisor'])
            if parent is not None:
                if nproc(parent['pid']) != (0, self.request['original_limits'][1]):
                    raise RuntimeError('Restarted watcher found a removed barrier')
                if parent['state'] == 'T':
                    send(self.request['supervisor'], signal.SIGCONT)
            self.state('ARMED_RECOVERED')
            return
        intent_path = self.amendment / 'arm_intent.json'
        if intent_path.exists():
            intent = read_json(intent_path)
            if intent['request_sha256'] != self.expected_sha256:
                raise ValueError('Arming intent belongs to another request')
            self.interference_started = True
            if checkpoint_inventory(self.run) != intent['checkpoint_baseline']:
                raise RuntimeError('Checkpoint changed during incomplete arming; manual review required')
            already_armed = nproc(self.request['supervisor']['pid']) == (0, self.request['original_limits'][1])
            self.eligible(already_armed=already_armed)
        else:
            self.eligible()
            baseline = checkpoint_inventory(self.run)
            if self.request['boundary_stage'] + '.json' in baseline:
                raise RuntimeError('Boundary checkpoint already exists; refusing to arm')
            intent = dict(request_sha256=self.expected_sha256, created_utc=now(),
                          original_limits=self.request['original_limits'],
                          child_limits=list(nproc(self.request['child']['pid'])), checkpoint_baseline=baseline)
            write_json(intent_path, intent, once=True)
        self.state('ARMING')
        stopped = False
        try:
            self.interference_started = True
            if not send(self.request['supervisor'], signal.SIGSTOP):
                raise RuntimeError('Supervisor exited before stop')
            stopped = True
            wait_stopped(self.request['supervisor'])
            self.eligible(already_armed=nproc(self.request['supervisor']['pid'])[0] == 0)
            nproc(self.request['supervisor']['pid'], (0, self.request['original_limits'][1]))
            if nproc(self.request['supervisor']['pid']) != (0, self.request['original_limits'][1]):
                raise RuntimeError('Failed to install process-creation barrier')
            if list(nproc(self.request['child']['pid'])) != intent['child_limits']:
                raise RuntimeError('Existing Nextflow child limits changed')
            self.armed = dict(intent, armed_utc=now(), barrier='SUPERVISOR_RLIMIT_NPROC_SOFT_ZERO')
            write_json(armed_path, self.armed, once=True)
        except BaseException as error:
            if stopped:
                # If installation never changed the limit and the current child
                # is still running, preserve that work and explicitly report
                # NOT_ARMED. This is not a protected continuation or an adoption.
                parent = authenticated(self.request['supervisor'])
                child = authenticated(self.request['child'])
                if (parent is not None and child is not None
                        and child['ppid'] == parent['pid']
                        and nproc(parent['pid']) == tuple(self.request['original_limits'])
                        and not (self.run / 'checkpoints' / (self.request['boundary_stage'] + '.json')).exists()):
                    send(self.request['supervisor'], signal.SIGCONT)
                    stopped = False
                    self.interference_started = False
                    self.state('NOT_ARMED', reason=str(error), child_active=True)
                    raise
                self.emergency('Failed to establish durable boundary barrier: ' + str(error))
            raise
        finally:
            if stopped and not self.aborted and authenticated(self.request['supervisor']) is not None:
                # Never resume an unprotected old controller after a partial failure.
                if nproc(self.request['supervisor']['pid']) == (0, self.request['original_limits'][1]):
                    send(self.request['supervisor'], signal.SIGCONT)
                else:
                    self.emergency('Barrier disappeared before supervisor continuation')
        self.state('ARMED')

    def emergency(self, reason):
        self.aborted = True
        errors = []
        try:
            self.state('EMERGENCY_ABORT', reason=reason)
        except Exception as error:
            errors.append('status write: ' + str(error))
        try:
            parent = authenticated(self.request['supervisor'])
        except Exception as error:
            parent = None
            errors.append('parent identity: ' + str(error))
        parent_stopped = False
        if parent is not None:
            try:
                send(self.request['supervisor'], signal.SIGSTOP)
                wait_stopped(self.request['supervisor'])
                parent_stopped = True
            except Exception as error:
                errors.append('parent stop: ' + str(error))
        try:
            stop_containers(self.run_id)
        except Exception as error:
            errors.append('container cleanup: ' + str(error))
        try:
            child = authenticated(self.request['child'])
        except Exception as error:
            child = None
            errors.append('child identity: ' + str(error))
        if child is not None:
            if child['pgid'] != child['pid'] or child['sid'] != child['pid']:
                errors.append('child session identity changed; no group signal sent')
            else:
                # Authenticate immediately before sending only this child's group.
                try:
                    if authenticated(self.request['child']) is not None:
                        os.killpg(child['pid'], signal.SIGTERM)
                except Exception as error:
                    errors.append('child termination: ' + str(error))
        if parent is not None:
            # Parent remains stopped while restored. SIGTERM is pending before
            # CONT, so it cannot launch an old next stage even if Nextflow exited 0.
            try:
                if parent_stopped and authenticated(self.request['supervisor']) is not None:
                    nproc(parent['pid'], self.request['original_limits'])
            except Exception as error:
                errors.append('parent limit restoration: ' + str(error))
            # Cleanup errors above must not strand an authenticated stopped parent.
            terminated = False
            try:
                terminated = send(self.request['supervisor'], signal.SIGTERM)
            except Exception as error:
                errors.append('parent termination: ' + str(error))
            try:
                if terminated:
                    send(self.request['supervisor'], signal.SIGCONT)
            except Exception as error:
                errors.append('parent termination continuation: ' + str(error))
        try:
            self.state('ABORTED', reason=reason, cleanup_errors=errors)
        except Exception as error:
            errors.append('abort status write: ' + str(error))
        raise RuntimeError(reason + ('; ' + '; '.join(errors) if errors else ''))

    def fail_closed(self, error):
        """Unexpected monitor errors cannot silently abandon an armed live job."""
        cleanup = None
        if self.interference_started and not self.aborted:
            try:
                self.emergency('Unexpected handoff failure: ' + str(error))
            except Exception as cleanup_error:
                cleanup = str(cleanup_error)
        self.state('FAILED_CLOSED' if self.interference_started else 'NOT_ARMED', error=str(error), cleanup=cleanup,
                   note='No automatic old-stage retry or unprotected supervisor continuation')

    def validate_completion(self):
        if authenticated(self.request['supervisor']) is not None or authenticated(self.request['child']) is not None:
            raise RuntimeError('Cannot adopt while either original process remains active')
        stage = self.request['boundary_stage']
        inventory = checkpoint_inventory(self.run)
        baseline = self.armed['checkpoint_baseline']
        if (set(inventory) != set(baseline) | {stage + '.json'}
                or any(inventory.get(name) != digest for name, digest in baseline.items())):
            raise RuntimeError('Unexpected new, changed, or missing checkpoints across boundary')
        checkpoint = read_json(self.run / 'checkpoints' / (stage + '.json'))
        if checkpoint.get('returncode') != 0 or not checkpoint.get('outputs'):
            raise RuntimeError('Boundary did not produce an authenticated successful checkpoint')
        chrom = self.request['amendment_template']['boundary']['chromosome']
        folder = self.run / f'chr{chrom:02d}'
        command = ['/usr/local/bin/nextflow', '-log', str(folder / 'nextflow.log'),
                   '-C', str(folder / 'runtime.config'), 'run',
                   str(self.run / 'source/workflows/r02_preprocess_autosome.nf'),
                   '-params-file', str(folder / 'parameters.json'), '-work-dir',
                   str(folder / 'work'), '-ansi-log', 'false', '-with-trace',
                   str(folder / 'trace.tsv'), '-resume']
        if checkpoint.get('command') != command:
            raise ValueError('Boundary checkpoint command is not the original preprocessing command')
        prefix = folder / 'preprocess/lai_rare' / f'dnabr.hg38.2723.chr{chrom}.rare.minor'
        expected_outputs = {str(prefix) + suffix for suffix in ('.vcf.gz', '.vcf.gz.tbi', '.contract.json', '.counts.tsv')}
        if {item['path'] for item in checkpoint['outputs']} != expected_outputs or len(checkpoint['outputs']) != 4:
            raise ValueError('Boundary checkpoint does not authenticate exactly the four durable outputs')
        for output in checkpoint['outputs']:
            path = Path(output['path'])
            if (not path.is_absolute() or not path.resolve().is_relative_to(self.run)
                    or not path.is_file() or path.stat().st_size != output['bytes'] or sha(path) != output['sha256']):
                raise ValueError('Boundary output changed or escaped the original run')
        status = read_json(self.run / 'status.json')
        if (status.get('state') != 'FAILED' or status.get('failed_stage') != f'chr{chrom:02d}_rare_J'
                or '[Errno 11]' not in status.get('error', '')):
            raise RuntimeError('Supervisor did not exit at the expected blocked next fork')
        if containers(self.run_id):
            raise RuntimeError('Run-labelled containers remain active; refusing adoption')
        return inventory[stage + '.json']

    def activate(self):
        self.validate()  # Reauthenticate immutable code/settings after a possibly long wait.
        checkpoint_sha256 = self.validate_completion()
        amendment = copy.deepcopy(self.request['amendment_template'])
        amendment['boundary']['checkpoint_sha256'] = checkpoint_sha256
        write_json(self.amendment / 'amendment.json', amendment, once=True)
        seal = {name: sha(self.amendment / name)
                for name in ('amendment.json', 'source.sha256.json', 'request.json')}
        write_json(self.amendment / 'frozen.sha256.json', seal, once=True)
        adapter = self.amendment / 'source/bin/r02_apply_amendment.py'
        command = [sys.executable, str(adapter), '--run-dir', str(self.run),
                   '--amendment-dir', str(self.amendment), '--run']
        self.state('BOUNDARY_VERIFIED_ACTIVATING', checkpoint_sha256=checkpoint_sha256)
        activation = dict(request_sha256=self.expected_sha256, checkpoint_sha256=checkpoint_sha256,
                          command=command, interpretation='Operational continuation, not a new scientific round')
        write_json(self.amendment / 'boundary_activation.json', activation, once=True)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.execv(sys.executable, command)

    def monitor(self, poll_seconds=10):
        while True:
            free = shutil.disk_usage(self.run).free / 1024**3
            parent = authenticated(self.request['supervisor'])
            child = authenticated(self.request['child'])
            if free < self.request['free_disk_min_gib']:
                self.emergency('Free disk below original 12 GiB guard')
            if parent is None:
                if child is not None:
                    self.emergency('Supervisor exited before its authenticated Nextflow child')
                self.activate()
                return
            if nproc(parent['pid']) != (0, self.request['original_limits'][1]):
                self.emergency('Process-creation barrier was removed')
            if child is not None and list(nproc(child['pid'])) != self.armed['child_limits']:
                self.emergency('Existing Nextflow limits changed')
            if parent['state'] == 'T':
                send(self.request['supervisor'], signal.SIGCONT)
            self.state('WAITING_CURRENT_WORKFLOW', free_disk_gib=round(free, 2), child_active=child is not None)
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--expected-request-sha256', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    handoff = Handoff(args.request, args.expected_request_sha256).validate()
    sources = read_json(handoff.amendment / 'source.sha256.json')
    if sha(__file__) != sources['bin/r02_stage_boundary_handoff.py']:
        raise ValueError('Running watcher is not the authenticated snapshot version')
    with (handoff.amendment / '.boundary.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            handoff.arm()
            handoff.monitor()
        except BaseException as error:
            handoff.fail_closed(error)
            raise


if __name__ == '__main__':
    main()
