#!/usr/bin/env python3
"""Authenticated, single-use replacement of one temporary R02 worker controller.

Preflight is read-only. --apply pauses only the root startup shell (preventing its
EXIT shutdown trap), installs RLIMIT_NPROC=0 only on the old unprivileged Python
controller, and lets its existing Nextflow child finish unchanged. A genuine
preprocessing checkpoint and the expected EAGAIN at the next stage are required
before the separately authenticated successor is run. Version 2 can instead
adopt an already armed boundary, without arming again. Its only additional
completion case is the exact legacy source-count guard failure, after a genuine
checkpoint and a successful authenticated successor count preflight. No scientific source,
original manifest, checkpoint or existing output is edited by this helper.

Run as an independent root systemd service, Restart=no, KillMode=process,
PrivateMounts=no, TimeoutStopSec >= 90. The helper itself watches deadline/disk;
the VM's pre-existing maximum-run-duration remains the external final bound.
As in the original startup policy, this VM is STOPPED on failure too; its disk
and unpublished outputs are retained. This helper never deletes a VM or disk.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import resource
import select
import shutil
import signal
import subprocess
import time
import urllib.request


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def write_once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(),
            'Expected explicit regular nonsymlink file: ' + str(path))
    return path


def check_file(path, expected):
    require(isinstance(expected, str) and re.fullmatch('[a-f0-9]{64}', expected), 'Invalid SHA256')
    require(sha(regular(path)) == expected, 'File hash mismatch: ' + str(path))


def process_info(pid):
    proc = Path('/proc') / str(pid)
    try:
        first = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines())
        command = (proc / 'cmdline').read_bytes()
        last = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    require(first[19] == last[19], 'PID changed while reading identity')
    return dict(pid=int(pid), start_ticks=int(first[19]), state=last[0], ppid=int(last[1]),
                pgid=int(last[2]), sid=int(last[3]), cmdline_sha256=hashlib.sha256(command).hexdigest(),
                uids=list(map(int, status['Uid'].split())), gids=list(map(int, status['Gid'].split())),
                cap_eff=int(status['CapEff'], 16), cap_prm=int(status['CapPrm'], 16),
                threads=int(status['Threads']))


def identity(info):
    return {key: info[key] for key in ('pid', 'start_ticks', 'cmdline_sha256')}


def same_incarnation(spec):
    """Reject PID reuse even when the replacement is already a zombie."""
    current = process_info(spec['pid'])
    if current is not None:
        require(current['start_ticks'] == spec['start_ticks'], 'PID reused; refusing adoption or signal')
    return current


def pidfd_exited(fd, timeout_ms=0):
    """Kernel exit notification, without reaping another controller's child."""
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    events = poller.poll(timeout_ms)
    require(not any(mask & (select.POLLERR | select.POLLNVAL) for _, mask in events),
            'Invalid process exit notification')
    return any(mask & (select.POLLIN | select.POLLHUP) for _, mask in events)


def authenticated(spec):
    current = same_incarnation(spec)
    if current is None or current['state'] in ('Z', 'X'):
        return None
    if current['cmdline_sha256'] == spec['cmdline_sha256']:
        return current
    # exit_mm() clears cmdline before exit_notify() changes the task to Z.
    # Empty argv alone is NOT proof of exit: an exec/argv change can also
    # alter it. Only a bounded kernel notification may resolve this ambiguity.
    empty = hashlib.sha256(b'').hexdigest()
    message = (f"Process command changed: pid={spec['pid']} state={current['state']} "
               f"expected_sha256={spec['cmdline_sha256']} observed_sha256={current['cmdline_sha256']}")
    require(current['cmdline_sha256'] == empty, message)
    try:
        fd = os.pidfd_open(spec['pid'])
    except ProcessLookupError:
        require(same_incarnation(spec) is None, 'Process reappeared after exit lookup')
        return None
    try:
        # Binding a pidfd can race with exit/reuse; authenticate again before
        # interpreting its notification. A nonempty changed command still fails.
        current = same_incarnation(spec)
        if current is None or current['state'] in ('Z', 'X'):
            return None
        require(current['cmdline_sha256'] in (empty, spec['cmdline_sha256']), message)
        exited = pidfd_exited(fd, timeout_ms=1000)
        current = same_incarnation(spec)
        if current is not None and current['state'] not in ('Z', 'X'):
            require(current['cmdline_sha256'] in (empty, spec['cmdline_sha256']), message)
        require(exited, message + '; empty command without confirmed process exit')
        return None
    finally:
        os.close(fd)


def authenticated_runuser(spec):
    current = authenticated(spec)
    if current is None:
        return None
    command = (Path('/proc') / str(spec['pid']) / 'cmdline').read_bytes().split(b'\0')
    require(command and Path(os.fsdecode(command[0])).name == 'runuser'
            and current['uids'] == [0] * 4, 'Stopped ancestor is not the authenticated root runuser')
    # Recheck after reading argv, before returning a process to the caller.
    return authenticated(spec)


def send(spec, sig):
    if authenticated(spec) is None:
        return False
    try:
        fd = os.pidfd_open(spec['pid'])
    except ProcessLookupError:
        return False
    try:
        if authenticated(spec) is None or pidfd_exited(fd):
            return False
        try:
            signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            return False
    finally:
        os.close(fd)
    return True


def nproc(pid, limits=None):
    if limits is None:
        return resource.prlimit(pid, resource.RLIMIT_NPROC)
    return resource.prlimit(pid, resource.RLIMIT_NPROC, tuple(limits))


def stopped(spec):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = authenticated(spec)
        require(current is not None, 'Process exited while establishing boundary')
        if current['state'] == 'T':
            return
        time.sleep(.02)
    raise RuntimeError('Process did not stop within five seconds')


def checkpoint_inventory(run):
    return {p.name: sha(p) for p in sorted((run / 'checkpoints').glob('*.json'))}


def run_checked(argv, timeout=30):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout)


def containers(run_id):
    rows = run_checked(['docker', 'ps', '-q', '--no-trunc', '--filter', 'label=dnabr-r02=' + run_id]).stdout.split()
    require(all(re.fullmatch('[a-f0-9]{64}', value) for value in rows), 'Unexpected container identity')
    return rows


def stop_containers(run_id):
    for container in containers(run_id):
        labels = json.loads(run_checked(['docker', 'inspect', '--format', '{{json .Config.Labels}}', container]).stdout)
        require(labels.get('dnabr-r02') == run_id, 'Container does not belong to this worker')
        run_checked(['docker', 'stop', '--time', '30', container], timeout=60)


def descendants(specifications):
    """Pin currently observable descendants of authenticated roots, not names."""
    roots = [p for p in (authenticated(s) for s in specifications) if p is not None]
    found = {p['pid']: p for p in roots}
    all_processes = []
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                info = process_info(int(proc.name))
                if info is not None:
                    all_processes.append(info)
            except (FileNotFoundError, ProcessLookupError, ValueError):
                continue
    while True:
        additions = [p for p in all_processes if p['ppid'] in found and p['pid'] not in found]
        if not additions:
            return [identity(p) for p in found.values()]
        found.update({p['pid']: p for p in additions})


def metadata_identity():
    def value(key):
        request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/' + key,
                                         headers={'Metadata-Flavor': 'Google'})
        with urllib.request.urlopen(request, timeout=5) as response:
            require(response.headers.get('Metadata-Flavor') == 'Google', 'Invalid metadata response')
            return response.read(4096).decode().strip()
    return dict(id=value('instance/id'), name=value('instance/name'),
                zone=value('instance/zone').rsplit('/', 1)[-1], project=value('project/project-id'))


def build_spec(inventory, worker_manifest_path, worker_manifest_sha256, helperpath, deadline):
    """Construct a specification; does not signal, write or inspect a remote VM.

    Paths refer to the existing remote run. The inventory must be freshly
    captured; preflight reauthenticates every process and file before action.
    """
    run = Path(inventory['run'])
    match = re.fullmatch(r'chr(\d{2})_M01_M02_M021', inventory['status']['stage'])
    require(match is not None, 'Inventory is not at a preprocessing boundary')
    chrom = int(match[1])
    ancestors = inventory['ancestors']
    startup_index = next((i for i, p in enumerate(ancestors)
                          if Path(p['args'][0]).name == 'bash' and len(p['args']) == 2
                          and p['args'][1].startswith('/tmp/metadata-scripts')
                          and p['args'][1].endswith('/startup-script')), None)
    require(startup_index is not None, 'Inventory lacks original root startup shell')
    manifest_path = Path(worker_manifest_path)
    # The deployed script is stored beside the successor's immutable manifest.
    script_path = manifest_path.parent / 'r02_optimized_worker.py'
    instance = inventory['instance']
    folder = run / f'chr{chrom:02d}'
    command = ['/usr/local/bin/nextflow', '-log', str(folder / 'nextflow.log'), '-C', str(folder / 'runtime.config'),
               'run', str(run / 'source/workflows/r02_preprocess_autosome.nf'), '-params-file', str(folder / 'parameters.json'),
               '-work-dir', str(folder / 'work'), '-ansi-log', 'false', '-with-trace', str(folder / 'trace.tsv'), '-resume']
    child_args = inventory['child']['args']
    require(child_args[child_args.index('-jar') + 2:] == command[1:], 'Inventory Java command is not expected original Nextflow')
    replacement = dict(script_path=str(script_path), script_sha256=sha(script_path), manifest_path=str(manifest_path),
                       manifest_sha256=worker_manifest_sha256,
                       command=['/usr/bin/python3', str(script_path), '--manifest', str(manifest_path),
                                '--manifest-sha256', worker_manifest_sha256, '--run'],
                       completion_path=str(manifest_path.parent / 'completed.json'))
    destination = read(run / 'run.json')['destination']
    mount = Path('/home/jose.tantalean/gcs-dnabr')
    return dict(schema='r02_optimization_boundary_v1', run_dir=str(run),
                worker_run_sha256=inventory['config_sha256'], frozen_sha256=inventory['frozen_sha256'],
                source_manifest_sha256=inventory['source_manifest_sha256'], helper_sha256=sha(helperpath),
                startup=identity(ancestors[startup_index]), ancestors=[identity(p) for p in ancestors[:startup_index]],
                supervisor=identity(inventory['supervisor']), child=identity(inventory['child']),
                original_limits=inventory['original_limits'], uid=inventory['uid'], gid=inventory['gid'], user='jose.tantalean',
                chromosome=chrom, boundary_command=command, next_failed_stage=f'chr{chrom:02d}_rare_J',
                parameters_sha256=inventory['parameters_sha256'], runtime_config_sha256=inventory['runtime_config_sha256'],
                deadline_utc=deadline, free_disk_min_gib=12, replacement=replacement,
                instance=dict(id=instance['instance/id'], name=instance['instance/name'],
                              zone=instance['instance/zone'].rsplit('/', 1)[-1], project=instance['project/project-id']),
                publication_prefix='gs://projects-usp/dnaBr-lai/datalake/' + str(Path(destination).relative_to(mount))
                                   + '/00_worker_provenance/optimization_boundary/' + worker_manifest_sha256)


class Boundary:
    def __init__(self, path, expected):
        self.path, self.expected = Path(path).absolute(), expected
        self.interfered = False
        self.successor = None
        self.successor_identity = None

    def validate(self):
        check_file(self.path, self.expected)
        self.s = read(self.path)
        s = self.s
        require(s.get('schema') in ('r02_optimization_boundary_v1', 'r02_optimization_boundary_v2'),
                'Unsupported boundary schema')
        require(('previous_boundary' in s) == (s['schema'] == 'r02_optimization_boundary_v2'),
                'Boundary continuation requires an explicit version-2 predecessor')
        self.run = Path(s['run_dir'])
        require(self.run.is_absolute() and self.run.resolve() == self.run and self.run.is_dir(), 'Unsafe run directory')
        require(self.path.parent.is_relative_to(self.run / 'repairs'), 'Boundary specification must be under run/repairs')
        self.directory = self.path.parent / ('boundary-' + self.expected[:16])
        require(not self.directory.is_symlink(), 'Unsafe boundary evidence directory')
        check_file(__file__, s['helper_sha256'])
        for name, key in [('run.json', 'worker_run_sha256'), ('frozen.sha256.json', 'frozen_sha256'),
                          ('source.sha256.json', 'source_manifest_sha256')]:
            check_file(self.run / name, s[key])
        for root, name in [(self.run, 'frozen.sha256.json'), (self.run / 'source', 'source.sha256.json')]:
            manifest = read(self.run / name)
            require(bool(manifest), 'Empty frozen manifest')
            for relative, expected in manifest.items():
                member = Path(relative)
                require(not member.is_absolute() and '..' not in member.parts, 'Unsafe frozen manifest member')
                check_file(root / member, expected)
        config = read(self.run / 'run.json')
        self.run_id = config['run_id']
        require(re.fullmatch('[A-Za-z0-9._-]+', self.run_id), 'Unsafe container label')
        require(s['free_disk_min_gib'] == config['free_disk_min_gib'] == 12, 'Disk guard must remain 12 GiB')
        self.deadline = datetime.fromisoformat(s['deadline_utc'])
        require(self.deadline.tzinfo is not None and 0 < (self.deadline - datetime.now(timezone.utc)).total_seconds() <= 80 * 3600,
                'Deadline must be future, timezone-aware and bounded by 80 hours')
        require(type(s['chromosome']) is int and s['chromosome'] in config['parallel_worker']['assigned_chromosomes'], 'Chromosome not assigned')
        require(type(s['uid']) is int and s['uid'] > 0 and type(s['gid']) is int and s['gid'] > 0, 'Unprivileged identity required')
        account = pwd.getpwnam(s['user'])
        require((account.pw_uid, account.pw_gid) == (s['uid'], s['gid']), 'Account UID/GID mismatch')
        roles = [s['startup'], s['supervisor'], s['child'], *s['ancestors']]
        require(len({p['pid'] for p in roles}) == len(roles), 'Repeated process identity')
        for p in roles:
            require(type(p['pid']) is int and p['pid'] > 1 and type(p['start_ticks']) is int and p['start_ticks'] > 0
                    and re.fullmatch('[a-f0-9]{64}', p['cmdline_sha256']), 'Malformed process identity')
        limits = s['original_limits']
        require(len(limits) == 2 and all(type(x) is int for x in limits) and limits[0] > 0
                and (limits[1] == -1 or limits[1] >= limits[0]), 'Invalid original limits')
        self.stage = f"chr{s['chromosome']:02d}_M01_M02_M021"
        require(s['next_failed_stage'] == f"chr{s['chromosome']:02d}_rare_J", 'Unexpected subsequent blocked stage')
        folder = self.run / f"chr{s['chromosome']:02d}"
        check_file(folder / 'parameters.json', s['parameters_sha256'])
        check_file(folder / 'runtime.config', s['runtime_config_sha256'])
        self.command = ['/usr/local/bin/nextflow', '-log', str(folder / 'nextflow.log'), '-C', str(folder / 'runtime.config'),
                        'run', str(self.run / 'source/workflows/r02_preprocess_autosome.nf'), '-params-file', str(folder / 'parameters.json'),
                        '-work-dir', str(folder / 'work'), '-ansi-log', 'false', '-with-trace', str(folder / 'trace.tsv'), '-resume']
        require(s['boundary_command'] == self.command, 'Boundary command differs from original preprocessing')
        replacement = s['replacement']
        for key in ('manifest', 'script'):
            path = regular(replacement[key + '_path'])
            require(path.is_relative_to(self.run / 'repairs'), 'Replacement escaped worker repairs')
            check_file(path, replacement[key + '_sha256'])
        require(Path(replacement['script_path']).name == 'r02_optimized_worker.py', 'Unexpected successor script')
        require(replacement['command'] == ['/usr/bin/python3', replacement['script_path'], '--manifest', replacement['manifest_path'],
                                           '--manifest-sha256', replacement['manifest_sha256'], '--run'], 'Replacement command must be fixed and explicit')
        completion = Path(replacement['completion_path'])
        require(completion == Path(replacement['manifest_path']).parent / 'completed.json'
                and completion.resolve() == completion, 'Unsafe completion receipt')
        require(not completion.exists(), 'Replacement completion already exists; manual reconciliation required')
        require(metadata_identity() == s['instance'], 'This is not the authorized temporary VM')
        destination = Path(config['destination'])
        mount = Path('/home/jose.tantalean/gcs-dnabr')
        require(destination.is_relative_to(mount), 'Invalid existing publication destination')
        prefix = 'gs://projects-usp/dnaBr-lai/datalake/' + str(destination.relative_to(mount))
        require(s['publication_prefix'].startswith(prefix + '/00_worker_provenance/optimization_boundary/')
                and '..' not in s['publication_prefix'].split('/'), 'Publication escaped worker operational evidence')
        if 'previous_boundary' in s:
            self.validate_previous_boundary()
        return self

    def validate_previous_boundary(self):
        """Authenticate an already armed boundary; never re-arm or edit its evidence."""
        require(self.path == self.run / 'repairs/preprocess-v2/boundary.json',
                'Version-2 continuation requires its own immutable directory')
        previous = self.s['previous_boundary']
        require(isinstance(previous, dict) and set(previous) == {
            'spec_path', 'spec_sha256', 'intent_sha256', 'armed_sha256', 'controller'},
            'Invalid previous boundary reference')
        old_path = regular(previous['spec_path'])
        require(old_path.parent == self.run / 'repairs/preprocess-v1', 'Unexpected previous boundary directory')
        check_file(old_path, previous['spec_sha256'])
        old = read(old_path)
        require(old.get('schema') == 'r02_optimization_boundary_v1' and 'previous_boundary' not in old,
                'Only the original armed version-1 boundary may be adopted')
        changed = {'schema', 'helper_sha256', 'replacement', 'publication_prefix', 'previous_boundary'}
        require({k: v for k, v in old.items() if k not in changed}
                == {k: v for k, v in self.s.items() if k not in changed},
                'Continuation changed original identities, computation, resources or deadline')
        old_directory = old_path.parent / ('boundary-' + previous['spec_sha256'][:16])
        require(old_directory.resolve() == old_directory and old_directory.is_dir(),
                'Unsafe previous armed evidence directory')
        check_file(old_directory / 'spec.json', previous['spec_sha256'])
        check_file(old_directory / 'helper.py', old['helper_sha256'])
        for filename, field in [('intent.json', 'intent_sha256'), ('armed.json', 'armed_sha256')]:
            check_file(old_directory / filename, previous[field])
        intent, armed = read(old_directory / 'intent.json'), read(old_directory / 'armed.json')
        require(intent.get('spec_sha256') == armed.get('spec_sha256') == previous['spec_sha256']
                and armed.get('startup_paused') is True and armed.get('supervisor_fork_blocked') is True,
                'Previous boundary was not genuinely armed')
        baseline, limits = armed.get('checkpoint_baseline'), armed.get('child_limits')
        require(isinstance(baseline, dict) and baseline == intent.get('checkpoint_baseline')
                and self.stage + '.json' not in baseline
                and all(re.fullmatch(r'[A-Za-z0-9_.-]+\.json', k)
                        and isinstance(v, str) and re.fullmatch('[a-f0-9]{64}', v)
                        for k, v in baseline.items()), 'Invalid inherited checkpoint baseline')
        require(limits == intent.get('child_limits') and isinstance(limits, list) and len(limits) == 2
                and all(type(value) is int for value in limits) and limits[0] > 0
                and (limits[1] == -1 or limits[1] >= limits[0]), 'Invalid inherited Nextflow limits')
        require(not any((old_directory / name).exists() for name in
                        ('boundary_verified.json', 'successor_started.json', 'publication.json')),
                'Previous boundary advanced beyond waiting; manual reconciliation required')
        controller = previous['controller']
        require(isinstance(controller, dict) and set(controller) == {'pid', 'start_ticks', 'cmdline_sha256'}
                and type(controller['pid']) is int and controller['pid'] > 1
                and type(controller['start_ticks']) is int and controller['start_ticks'] > 0
                and isinstance(controller['cmdline_sha256'], str)
                and re.fullmatch('[a-f0-9]{64}', controller['cmdline_sha256'])
                and controller['pid'] not in {p['pid'] for p in
                    [self.s['startup'], self.s['supervisor'], self.s['child'], *self.s['ancestors']]},
                'Invalid previous boundary controller identity')
        require(authenticated(controller) is None, 'Previous boundary controller is still active')
        self.baseline, self.child_limits = baseline, limits
        self.previous_directory = old_directory

    def continuation_eligible(self):
        """Read-only adoption check for live children or a genuine finished checkpoint."""
        self.validate_previous_boundary()
        self.health()
        startup = authenticated(self.s['startup'])
        require(startup is not None and startup['state'] == 'T' and startup['uids'] == [0] * 4,
                'Inherited startup shutdown guard must remain stopped')
        inventory = checkpoint_inventory(self.run)
        require(set(self.baseline) <= set(inventory) <= set(self.baseline) | {self.stage + '.json'}
                and all(inventory[k] == v for k, v in self.baseline.items()),
                'Inherited checkpoint baseline changed')
        parent, child = authenticated(self.s['supervisor']), authenticated(self.s['child'])
        if parent is None:
            require(child is None, 'Supervisor exited before Nextflow')
            self.completion(allow_stopped_runuser=True)
            return
        require(parent['state'] != 'T' and parent['uids'] == [self.s['uid']] * 4
                and parent['gids'] == [self.s['gid']] * 4
                and parent['cap_eff'] == parent['cap_prm'] == 0 and parent['threads'] == 1,
                'Inherited supervisor identity or state changed')
        require(nproc(parent['pid']) == (0, self.s['original_limits'][1]),
                'Inherited supervisor fork barrier disappeared')
        current = parent
        for ancestor_spec in [*self.s['ancestors'], self.s['startup']]:
            ancestor = authenticated(ancestor_spec)
            require(ancestor is not None and current['ppid'] == ancestor['pid'],
                    'Inherited startup ancestry changed')
            current = ancestor
        if child is not None:
            require(child['ppid'] == parent['pid'] and child['uids'] == parent['uids']
                    and child['pgid'] == child['sid'] == child['pid']
                    and list(nproc(child['pid'])) == self.child_limits, 'Inherited Nextflow identity or limits changed')

    def adopt(self):
        self.continuation_eligible()
        # No process has been changed by this version; only now take responsibility
        # for the inherited barrier and its original bounded failure policy.
        write_once(self.directory / 'intent.json', dict(spec_sha256=self.expected, utc=now(),
                   checkpoint_baseline=self.baseline, child_limits=self.child_limits,
                   previous_boundary=self.s['previous_boundary'], existing_barrier_adopted=True))
        self.interfered = True
        self.event('ARMED_BOUNDARY_ADOPTED_WITHOUT_SIGNALS', previous_spec_sha256=
                   self.s['previous_boundary']['spec_sha256'], original_deadline_preserved=True)

    def event(self, state, **fields):
        record = dict(state=state, utc=now(), spec_sha256=self.expected, **fields)
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)

    def health(self):
        require(datetime.now(timezone.utc) < self.deadline, 'Original deadline reached')
        require(shutil.disk_usage(self.run).free >= 12 * 1024**3, 'Free disk below 12 GiB')

    def eligible(self):
        self.health()
        p = authenticated(self.s['supervisor'])
        child = authenticated(self.s['child'])
        startup = authenticated(self.s['startup'])
        require(p is not None and child is not None and startup is not None, 'All original processes must still be alive')
        require(startup['uids'] == [0] * 4, 'Startup shell must be root')
        require(p['uids'] == [self.s['uid']] * 4 and p['gids'] == [self.s['gid']] * 4
                and p['cap_eff'] == p['cap_prm'] == 0 and p['threads'] == 1, 'Supervisor must be unprivileged and single-threaded')
        require(child['ppid'] == p['pid'] and child['uids'] == p['uids'] and child['pgid'] == child['sid'] == child['pid'], 'Nextflow child/session identity mismatch')
        current = p
        for ancestor_spec in [*self.s['ancestors'], self.s['startup']]:
            ancestor = authenticated(ancestor_spec)
            require(ancestor is not None and current['ppid'] == ancestor['pid'], 'Startup ancestry mismatch')
            current = ancestor
        children = (Path('/proc') / str(p['pid']) / 'task' / str(p['pid']) / 'children').read_text().split()
        require(children == [str(child['pid'])], 'Supervisor has unexpected children')
        require(nproc(p['pid']) == tuple(self.s['original_limits']), 'Supervisor limits changed')
        status = read(self.run / 'status.json')
        require(status.get('stage') == self.stage and status.get('state') == 'RUNNING' and status.get('pid') == child['pid'], 'Current stage is not requested preprocessing')
        require(not (self.run / 'checkpoints' / (self.stage + '.json')).exists(), 'Boundary already completed; do not arm stale processes')
        # Independent service must not be within the startup process tree.
        me = process_info(os.getpid())
        while me and me['ppid'] > 1:
            require(me['pid'] != startup['pid'], 'Helper is inside old startup process tree')
            me = process_info(me['ppid'])
        return child

    def arm(self):
        child = self.eligible()
        self.baseline = checkpoint_inventory(self.run)
        self.child_limits = list(nproc(child['pid']))
        write_once(self.directory / 'intent.json', dict(spec_sha256=self.expected, utc=now(),
                   checkpoint_baseline=self.baseline, child_limits=self.child_limits))
        self.interfered = True
        require(send(self.s['startup'], signal.SIGSTOP), 'Startup shell exited before barrier')
        stopped(self.s['startup'])
        require(send(self.s['supervisor'], signal.SIGSTOP), 'Supervisor exited before barrier')
        stopped(self.s['supervisor'])
        self.eligible()
        nproc(self.s['supervisor']['pid'], (0, self.s['original_limits'][1]))
        require(nproc(self.s['supervisor']['pid']) == (0, self.s['original_limits'][1]), 'Barrier installation failed')
        require(list(nproc(self.s['child']['pid'])) == self.child_limits, 'Existing Nextflow limits changed')
        write_once(self.directory / 'armed.json', dict(spec_sha256=self.expected, utc=now(), child_limits=self.child_limits,
                   checkpoint_baseline=self.baseline, startup_paused=True, supervisor_fork_blocked=True))
        require(send(self.s['supervisor'], signal.SIGCONT), 'Protected supervisor exited before continuation')
        self.event('ARMED_EXISTING_NEXTFLOW_UNCHANGED')

    def validate_legacy_count(self):
        """Do not trust an error string alone: verify trace and the sealed count fix."""
        folder = self.run / f"chr{self.s['chromosome']:02d}"
        trace = regular(folder / 'trace.tsv')
        with trace.open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        names = {f'{name} (chr{self.s["chromosome"]})' for name in (
            'PREPROCESS_NORM_LEFTALIGN', 'PREPROCESS_FILTER_SNV_BIALLELIC_PASS', 'LAI_RARE_BIALELIC_ONLY')}
        require(len(rows) == 3 and {row.get('name') for row in rows} == names
                and all(row.get('status') in ('COMPLETED', 'CACHED') and row.get('exit') == '0'
                        and re.fullmatch(r'[a-f0-9]{2}/[a-f0-9]{6,}', row.get('hash', '')) for row in rows),
                'Legacy count recovery requires three genuine successful preprocessing trace tasks')
        replacement = self.s['replacement']
        check_file(replacement['script_path'], replacement['script_sha256'])
        check_file(replacement['manifest_path'], replacement['manifest_sha256'])
        command = [*replacement['command'][:-1], '--validate-legacy-count', str(self.s['chromosome'])]
        require(replacement['command'][-1] == '--run', 'Unexpected successor execution command')
        if os.geteuid() == 0:
            command = ['/usr/sbin/runuser', '-u', self.s['user'], '--', *command]
        else:
            require(os.geteuid() == self.s['uid'], 'Count preflight requires the authenticated worker account')
        # This mode validates only. It must not activate a worker or create an
        # execution checkpoint; its command is fixed by authenticated source.
        result = run_checked(command, timeout=60)
        evidence = json.loads(result.stdout)
        require(isinstance(evidence, dict) and evidence.get('legacy_count_validated') is True
                and type(evidence.get('chromosome')) is int
                and evidence['chromosome'] == self.s['chromosome'], 'Successor did not validate the legacy count')
        return dict(trace_sha256=sha(trace), successor_manifest_sha256=replacement['manifest_sha256'],
                    successor_script_sha256=replacement['script_sha256'], count_validation=evidence)

    def completion(self, *, allow_stopped_runuser=False):
        require(authenticated(self.s['supervisor']) is None and authenticated(self.s['child']) is None,
                'Original processes must exit before adoption')
        inventory = checkpoint_inventory(self.run)
        require(set(inventory) == set(self.baseline) | {self.stage + '.json'}
                and all(inventory.get(k) == v for k, v in self.baseline.items()), 'Unexpected checkpoint changes')
        checkpoint = read(self.run / 'checkpoints' / (self.stage + '.json'))
        require(checkpoint.get('returncode') == 0 and checkpoint.get('command') == self.command, 'Original preprocessing did not complete successfully')
        folder = self.run / f"chr{self.s['chromosome']:02d}"
        base = f"dnabr.hg38.2723.chr{self.s['chromosome']}.rare.minor"
        prefix = folder / 'preprocess/lai_rare' / base
        paths = {str(prefix) + suffix for suffix in ('.vcf.gz', '.vcf.gz.tbi', '.contract.json', '.counts.tsv')}
        outputs = checkpoint.get('outputs', [])
        require(len(outputs) == 4 and {x['path'] for x in outputs} == paths, 'Wrong preprocessing output contract')
        for record in outputs:
            path = regular(record['path'])
            require(path.stat().st_size == record['bytes'], 'Output size changed')
            check_file(path, record['sha256'])
        status = read(self.run / 'status.json')
        expected_fork_failure = (status.get('state') == 'FAILED'
            and status.get('failed_stage') == self.s['next_failed_stage']
            and '[Errno 11]' in status.get('error', ''))
        legacy_count_failure = (self.s['schema'] == 'r02_optimization_boundary_v2'
            and status.get('state') == 'FAILED' and status.get('failed_stage') == self.stage
            and status.get('error') == f"chr{self.s['chromosome']}: source index count does not match completed sequential M01 annotation")
        require(expected_fork_failure or legacy_count_failure,
                'Worker did not stop at expected next-stage EAGAIN or an explicitly validated legacy count guard')
        require(not containers(self.run_id), 'Worker containers still active at adoption boundary')
        for ancestor in self.s['ancestors']:
            current = authenticated(ancestor)
            if current is not None and allow_stopped_runuser:
                current = authenticated_runuser(ancestor)
                require(len(self.s['ancestors']) == 1 and current is not None and current['state'] == 'T'
                        and current['ppid'] == self.s['startup']['pid'], 'Unexpected active ancestor after preprocessing')
            else:
                require(current is None, 'Intermediate original startup child still active')
        if legacy_count_failure:
            self.legacy_count_evidence = self.validate_legacy_count()
        return inventory[self.stage + '.json']

    def resume_stopped_runuser(self):
        """Allow runuser to reap its child without unblocking the old controller.

        util-linux waits with WUNTRACED. On child SIGSTOP, runuser stops itself;
        continuing only Python does not resume runuser. This is checked on every
        poll because runuser can consume the child-stop event after arming.
        Startup never receives SIGCONT and Nextflow is never signalled here.
        """
        pending = [s for s in self.s['ancestors']
                   if (current := authenticated(s)) is not None and current['state'] == 'T']
        if not pending:
            return False
        require(len(self.s['ancestors']) == len(pending) == 1, 'Unexpected stopped ancestor chain')
        ancestor_spec = pending[0]
        ancestor = authenticated_runuser(ancestor_spec)
        require(ancestor is not None and ancestor['state'] == 'T'
                and ancestor['ppid'] == self.s['startup']['pid'], 'runuser ancestry changed')
        startup = authenticated(self.s['startup'])
        require(startup is not None and startup['state'] == 'T', 'Startup shutdown guard must remain stopped')
        parent, child = authenticated(self.s['supervisor']), authenticated(self.s['child'])
        completion_checked = False
        if parent is not None:
            require(parent['ppid'] == ancestor['pid'] and parent['state'] != 'T', 'Protected supervisor ancestry/state changed')
            require(nproc(parent['pid']) == (0, self.s['original_limits'][1]), 'Supervisor fork barrier disappeared')
            if child is not None:
                require(child['ppid'] == parent['pid'] and list(nproc(child['pid'])) == self.child_limits,
                        'Existing Nextflow identity or limits changed')
        else:
            require(child is None, 'Supervisor exited before Nextflow')
            # If Python already exited, only its genuine successful checkpoint
            # and expected EAGAIN authorize releasing the stopped reaper.
            self.completion(allow_stopped_runuser=True)
            completion_checked = True
        # Reauthenticate the two protections immediately before pidfd signalling.
        startup = authenticated(self.s['startup'])
        require(startup is not None and startup['state'] == 'T', 'Startup shutdown guard changed before continuation')
        parent = authenticated(self.s['supervisor'])
        if parent is not None:
            require(nproc(parent['pid']) == (0, self.s['original_limits'][1]), 'Barrier changed before runuser continuation')
        elif not completion_checked:
            self.completion(allow_stopped_runuser=True)
        changed = send(ancestor_spec, signal.SIGCONT)
        self.event('RUNUSER_REAPER_CONTINUED', ancestor_pid=ancestor_spec['pid'], signal_sent=changed,
                   startup_resumed=False, nextflow_signalled=False, supervisor_barrier_preserved=True)
        return changed

    def wait_boundary(self):
        while True:
            self.health()
            startup = authenticated(self.s['startup'])
            require(startup is not None and startup['state'] == 'T', 'Startup shutdown guard disappeared')
            self.resume_stopped_runuser()
            p, child = authenticated(self.s['supervisor']), authenticated(self.s['child'])
            if p is None:
                require(child is None, 'Supervisor exited before Nextflow; not a safe boundary')
                # runuser can still be reaping the supervisor for a brief interval.
                if any(authenticated(x) is not None for x in self.s['ancestors']):
                    time.sleep(1)
                    continue
                return self.completion()
            require(nproc(p['pid']) == (0, self.s['original_limits'][1]), 'Supervisor fork barrier disappeared')
            if child is not None:
                require(list(nproc(child['pid'])) == self.child_limits, 'Nextflow resource limits changed')
            require(p['state'] != 'T', 'Protected supervisor unexpectedly stopped')
            self.event('WAITING_GENUINE_PREPROCESSING_CHECKPOINT', child_active=child is not None)
            time.sleep(10)

    def run_successor(self, checkpoint_sha256):
        self.validate()  # Recheck frozen scripts/configuration after the wait.
        self.health()
        self.completion()
        require(send(self.s['startup'], signal.SIGKILL), 'Startup shell disappeared before replacement')
        deadline = time.monotonic() + 5
        while authenticated(self.s['startup']) is not None:
            require(time.monotonic() < deadline, 'Killed startup shell did not exit')
            time.sleep(.02)
        write_once(self.directory / 'boundary_verified.json', dict(spec_sha256=self.expected, utc=now(),
                   checkpoint_sha256=checkpoint_sha256, command=self.command,
                   replacement_manifest_sha256=self.s['replacement']['manifest_sha256'],
                   legacy_count_recovery=getattr(self, 'legacy_count_evidence', None),
                   historical_blocked_fork_status_fabricated=False))
        replacement = self.s['replacement']
        env = dict(os.environ, NXF_VER='26.04.6', NXF_OFFLINE='true', NXF_DISABLE_CHECK_LATEST='true',
                   PYTHONDONTWRITEBYTECODE='1')
        command = ['/usr/sbin/runuser', '-u', self.s['user'], '--', *replacement['command']]
        with (self.directory / 'successor.log').open('x') as log:
            self.successor = subprocess.Popen(command, cwd=self.run, env=env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
            info = process_info(self.successor.pid)
            require(info is not None, 'Successor wrapper exited before authentication')
            self.successor_identity = identity(info)
            write_once(self.directory / 'successor_started.json', dict(utc=now(), process=self.successor_identity,
                       command=command, manifest_sha256=replacement['manifest_sha256']))
            while self.successor.poll() is None:
                self.health()
                if authenticated(self.successor_identity) is None:
                    # It may exit after poll() but before /proc authentication.
                    # Reap only our own Popen child and retain its true status.
                    self.successor.wait(timeout=5)
                    break
                time.sleep(10)
            require(self.successor.returncode == 0, 'Successor failed; no automatic retry')
        self.validate_successor_completion()
        require(not containers(self.run_id), 'Containers remain after successor returned')
        self.event('SUCCESSOR_COMPLETE_PUBLISHED', completion_sha256=sha(replacement['completion_path']))

    def validate_successor_completion(self):
        done = read(regular(self.s['replacement']['completion_path']))
        require(done.get('manifest_sha256') == self.s['replacement']['manifest_sha256']
                and done.get('published') is True, 'Successor has not certified its publication')
        path = regular(self.run / 'worker_provenance/worker_completion.json')
        receipt = read(path)
        config = read(self.run / 'run.json')
        worker = config['parallel_worker']
        require(receipt.get('operational_amendment', {}).get('manifest_sha256') == self.s['replacement']['manifest_sha256']
                and receipt.get('worker_id') == worker['worker_id']
                and receipt.get('chromosomes') == worker['assigned_chromosomes']
                and receipt.get('aggregate_executed') is False and receipt.get('biological_validation_complete') is False,
                'Successor lacks authenticated publication receipt')
        publication = read(self.run / 'checkpoints/publish_00_worker_provenance.json')
        require(done.get('worker_completion_sha256') == sha(path)
                and done.get('publication_checkpoint_sha256') == sha(self.run / 'checkpoints/publish_00_worker_provenance.json'),
                'Successor completion hashes do not bind published records')
        records = [x for x in publication.get('files', []) if x.get('path') == str(path)]
        require(len(records) == 1, 'Worker completion lacks exactly one publication record')
        record = records[0]
        require(record['sha256'] == sha(path) and record['bytes'] == path.stat().st_size,
                'Published completion differs from local receipt')
        mount = Path('/home/jose.tantalean/gcs-dnabr')
        expected_uri = 'gs://projects-usp/dnaBr-lai/datalake/' + str(Path(config['destination']).relative_to(mount))
        require(record['uri'] == expected_uri + '/00_worker_provenance/worker_completion.json', 'Completion published outside worker prefix')
        remote = json.loads(run_checked(['gcloud', 'storage', 'objects', 'describe', record['uri'], '--format=json'], timeout=60).stdout)
        require(str(remote.get('generation')) == str(record['generation'])
                and int(remote.get('size', -1)) == record['bytes']
                and remote.get('md5_hash', remote.get('md5Hash')) == record['md5_base64'], 'Remote completion generation/checksum changed')
        prefix = expected_uri + '/00_worker_provenance/operational_amendments/' + self.s['replacement']['manifest_sha256']
        expected_operational = {
            prefix + '/manifest.json': self.s['replacement']['manifest_sha256'],
            prefix + '/source.sha256.json': sha(Path(self.s['replacement']['manifest_path']).parent / 'source.sha256.json'),
            prefix + '/optimized_worker.py': self.s['replacement']['script_sha256'],
        }
        for uri, digest in expected_operational.items():
            found = [x for x in receipt.get('operational_provenance', []) if x.get('uri') == uri]
            require(len(found) == 1 and found[0].get('sha256') == digest, 'Missing or inconsistent published successor provenance')

    def emergency(self, error):
        errors = []
        try:
            self.event('FAILED_CLOSED', error=str(error), automatic_retry=False)
        except Exception as failure:
            errors.append('Event write failed: ' + str(failure))
        tree = []
        try:
            roots = [self.s['supervisor'], self.s['child']]
            if self.successor_identity is not None:
                roots.append(self.successor_identity)
            # Stop roots first so no new Nextflow work is scheduled while
            # enumerating descendants, including their separate sessions.
            for spec in roots:
                send(spec, signal.SIGSTOP)
            tree = descendants(roots)
            for spec in tree:
                send(spec, signal.SIGSTOP)
        except Exception as failure:
            errors.append('Process tree freeze: ' + str(failure))
        # Disable the original root EXIT trap first. Signals only use pinned PIDs.
        for spec in [self.s['startup'], self.s['supervisor']]:
            try:
                send(spec, signal.SIGKILL)
            except Exception as failure:
                errors.append(str(failure))
        try:
            stop_containers(self.run_id)
        except Exception as failure:
            errors.append(str(failure))
        for spec in [*reversed(tree), self.s['child'], *self.s['ancestors']]:
            try:
                send(spec, signal.SIGKILL)
            except Exception as failure:
                errors.append(str(failure))
        if self.successor_identity is not None:
            try:
                current = authenticated(self.successor_identity)
                if current is not None:
                    require(current['pid'] == current['pgid'] == current['sid'], 'Successor session changed')
                    os.killpg(current['pgid'], signal.SIGTERM)
                    self.successor.wait(timeout=60)
            except Exception as failure:
                errors.append(str(failure))
        try:
            self.event('EMERGENCY_STOP_FINISHED', errors=errors, originals_preserved=True,
                       failure_policy='STOP_VM_RETAIN_DISK_NO_DELETE')
        except Exception:
            pass  # Evidence failure must not bypass outer publication/shutdown.

    def publish(self):
        # Publish only bounded operational evidence, not raw successor logs/data.
        records = []
        for path in sorted(self.directory.iterdir()):
            if (path.suffix not in ('.json', '.jsonl') and path.name != 'helper.py') or path.name == 'publication.json':
                continue
            require(path.stat().st_size <= 8 * 1024**2, 'Operational record unexpectedly large')
            data = path.read_bytes()
            md5 = base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode()
            uri = self.s['publication_prefix'].rstrip('/') + '/' + path.name
            result = subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0', '--content-md5=' + md5,
                                     str(path), uri], capture_output=True, text=True, timeout=300)
            require(result.returncode == 0 or 'HTTPError 412:' in result.stderr, 'Operational evidence publication failed')
            remote = json.loads(run_checked(['gcloud', 'storage', 'objects', 'describe', uri, '--format=json'], timeout=60).stdout)
            require(int(remote.get('size', -1)) == len(data) and remote.get('md5_hash', remote.get('md5Hash')) == md5
                    and str(remote.get('generation', '')).isdigit(), 'Operational publication verification failed')
            records.append(dict(uri=uri, generation=str(remote['generation']), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        write_once(self.directory / 'publication.json', dict(utc=now(), files=records, spec_sha256=self.expected))

    def apply(self):
        require(os.geteuid() == 0, 'Application requires independent root service')
        self.directory.mkdir(mode=0o700, exist_ok=True)
        with (self.run / '.optimization_boundary.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not (self.directory / 'intent.json').exists(), 'Interrupted or used boundary requires manual reconciliation')
            for name, source in [('spec.json', self.path), ('helper.py', Path(__file__))]:
                with (self.directory / name).open('xb') as destination, source.open('rb') as original:
                    shutil.copyfileobj(original, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
            try:
                if 'previous_boundary' in self.s:
                    self.adopt()
                else:
                    self.arm()
                checkpoint_sha256 = self.wait_boundary()
                self.run_successor(checkpoint_sha256)
            except BaseException as error:
                if self.interfered:
                    self.emergency(error)
                raise
            finally:
                if self.interfered:
                    try:
                        self.publish()
                    finally:
                        # Stop only this authenticated temporary VM. Never delete it.
                        require(metadata_identity() == self.s['instance'], 'VM identity changed; shutdown refused')
                        run_checked(['/usr/sbin/shutdown', '-h', 'now'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    boundary = Boundary(args.spec, args.spec_sha256).validate()
    if 'previous_boundary' in boundary.s:
        boundary.continuation_eligible()
    else:
        boundary.eligible()
    if args.apply:
        def terminate(signum, frame):
            raise RuntimeError('Boundary service termination signal ' + str(signum))
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, terminate)
        boundary.apply()
    else:
        print(json.dumps(dict(state='PREFLIGHT_ONLY_NO_PROCESS_CHANGES', run_dir=str(boundary.run),
                              spec_sha256=args.spec_sha256, chromosome=boundary.s['chromosome'])))


if __name__ == '__main__':
    main()
