#!/usr/bin/env python3
"""One-shot R02 recovery on an explicitly authenticated new worker boot.

Runs the unchanged, hash-bound preprocess-v2 worker. No old PID is signalled,
no fork barrier is replayed, no checkpoint is manufactured, and no VM/disk is
deleted. Requires an independent root service with Restart=no, KillMode=process,
PrivateMounts=no and TimeoutStopSec>=90. The caller must first quarantine the
old startup/services and preserve the VM's original absolute termination time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import signal
import subprocess
import time
import types


def load_verified_module(path, expected, name):
    path = Path(path)
    if not (path.is_absolute() and path.resolve() == path and path.is_file()
            and isinstance(expected, str) and re.fullmatch('[a-f0-9]{64}', expected)):
        raise ValueError('Unsafe module path or digest')
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError('Module hash differs')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


class RebootRecovery:
    def __init__(self, path, expected):
        self.path, self.expected = Path(path).absolute(), expected
        if not (self.path.resolve() == self.path and self.path.is_file()
                and hashlib.sha256(self.path.read_bytes()).hexdigest() == expected):
            raise ValueError('Recovery specification hash/path differs')
        envelope = json.loads(self.path.read_text())
        # Only this fixed sibling is loaded, after authenticating its bytes.
        self.b = load_verified_module(self.path.parent / 'r02_optimization_boundary.py',
                                      envelope['helper_sha256'], '_reboot_boundary_utilities')
        self.s = self.b.read(self.path)
        self.successor = None

    def validate(self):
        b, s = self.b, self.s
        b.check_file(self.path, self.expected)
        b.check_file(__file__, s['wrapper_sha256'])
        b.require(s.get('schema') == 'r02_worker_reboot_resume_v1', 'Unsupported reboot recovery schema')
        self.run = Path(s['run_dir'])
        b.require(self.run.is_absolute() and self.run.resolve() == self.run and self.run.is_dir(),
                  'Unsafe original run directory')
        b.require(self.path.parent.is_relative_to(self.run / 'repairs')
                  and self.path.parent.name.startswith('reboot-'), 'Recovery must use a separate reboot repair directory')
        self.directory = self.path.parent / ('recovery-' + self.expected[:16])
        b.require(not self.directory.is_symlink(), 'Unsafe recovery evidence directory')
        b.require(b.metadata_identity() == s['instance'], 'This is not the authorized temporary VM')
        b.require(re.fullmatch(r'[a-f0-9-]{36}', s['boot_id'])
                  and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == s['boot_id'],
                  'Recovery boot identity changed')
        started = datetime.fromisoformat(s['original_start_utc'])
        self.deadline = datetime.fromisoformat(s['deadline_utc'])
        b.require(started.tzinfo is not None and self.deadline.tzinfo is not None
                  and self.deadline - started == timedelta(hours=72)
                  and started <= datetime.now(timezone.utc) < self.deadline,
                  'Recovery must retain the original 72-hour absolute deadline')
        account = pwd.getpwnam(s['user'])
        b.require(type(s['uid']) is int and s['uid'] > 0 and type(s['gid']) is int and s['gid'] > 0
                  and (account.pw_uid, account.pw_gid) == (s['uid'], s['gid']), 'Unprivileged account changed')
        replacement = s['replacement']
        directory = self.run / 'repairs/preprocess-v2'
        for key, name in [('manifest', 'manifest.json'), ('script', 'r02_optimized_worker.py')]:
            b.require(Path(replacement[key + '_path']) == directory / name, 'Worker path escaped frozen preprocess-v2')
            b.check_file(replacement[key + '_path'], replacement[key + '_sha256'])
        b.require(replacement['command'] == ['/usr/bin/python3', str(directory / 'r02_optimized_worker.py'),
            '--manifest', str(directory / 'manifest.json'), '--manifest-sha256', replacement['manifest_sha256'], '--run'],
            'Worker command changed')
        b.require(Path(replacement['completion_path']) == directory / 'completed.json'
                  and not Path(replacement['completion_path']).exists(), 'Worker completion exists or escaped its directory')
        self.worker = load_verified_module(replacement['script_path'], replacement['script_sha256'], '_frozen_reboot_worker')
        self.worker_spec, self.config, _, _, _ = self.worker.validate_spec(
            Path(replacement['manifest_path']), replacement['manifest_sha256'])
        b.require(self.worker_spec['schema'] == self.worker.COUNT_SCHEMA
                  and self.worker_spec['run_dir'] == str(self.run), 'Not the original version-2 worker')
        self.run_id = self.config['run_id']
        b.require(re.fullmatch('[A-Za-z0-9._-]+', self.run_id), 'Unsafe run/container identity')
        b.require(Path(s['local_bulk']) == self.run / 'local_bulk'
                  and s['bulk'] == self.config['bulk'], 'Original scratch paths changed')
        destination = Path(self.config['destination'])
        mount = Path('/home/jose.tantalean/gcs-dnabr')
        b.require(destination.is_relative_to(mount), 'Worker publication outside original datalake')
        prefix = 'gs://projects-usp/dnaBr-lai/datalake/' + str(destination.relative_to(mount))
        b.require(s['publication_prefix'].startswith(prefix + '/00_worker_provenance/reboot_recovery/')
                  and '..' not in s['publication_prefix'].split('/'), 'Recovery publication escaped worker prefix')
        failure = b.regular(s['failure_events_path'])
        b.require(failure.is_relative_to(self.run / 'repairs/preprocess-v2'), 'Failure evidence escaped original repair')
        b.check_file(failure, s['failure_events_sha256'])
        b.require(failure.stat().st_size <= 8 * 1024**2, 'Oversized failure evidence')
        events = [json.loads(line) for line in failure.read_text().splitlines() if line.strip()]
        b.require(any(event.get('state') == 'FAILED_CLOSED' for event in events), 'Missing original failed-boundary evidence')
        # The error text is provenance, never a substitute for scientific checks.
        self.validate_checkpoint_baseline()
        self.health()
        self.idle()
        return self

    def validate_checkpoint_baseline(self):
        baseline = self.s['checkpoint_baseline']
        self.b.require(isinstance(baseline, dict) and baseline
                       and all(re.fullmatch(r'[A-Za-z0-9_.-]+\.json', name)
                               and isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value)
                               for name, value in baseline.items()), 'Invalid recovery checkpoint inventory')
        self.b.require(self.b.checkpoint_inventory(self.run) == baseline, 'Pre-recovery checkpoint inventory changed')

    def health(self):
        b = self.b
        b.require(datetime.now(timezone.utc) < self.deadline, 'Original absolute deadline reached')
        local, bulk = Path(self.s['local_bulk']), Path(self.s['bulk'])
        b.require(local.is_dir() and bulk.is_dir() and local.samefile(bulk), 'Original local scratch bind is absent')
        for path in (local, bulk):
            filesystem = b.run_checked(['findmnt', '--noheadings', '--output', 'FSTYPE', '--target', str(path)],
                                       timeout=10).stdout.strip()
            b.require(filesystem == 'ext4', 'Recovery scratch is not ext4 local disk')
        b.require(shutil.disk_usage(local).free >= 12 * 1024**3, 'Free scratch disk below 12 GiB')

    def idle(self):
        self.worker.validate_idle(self.worker_spec, self.config)
        self.b.require(not self.b.run_checked(['docker', 'ps', '-q'], timeout=30).stdout.strip(),
                       'Temporary worker has active containers')
        # Read-only probe: never truncate the original runner's lock/evidence.
        lock_path = self.run / '.runner.lock'
        if lock_path.exists():
            with lock_path.open('r') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def stop_owned_work(self):
        errors = []
        if self.successor is not None and self.successor.poll() is None:
            try:
                # Our own unreaped Popen child pins this process group; no
                # boot-old PID lookup or cmdline-based lifecycle guess is used.
                os.killpg(self.successor.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append(str(error))
        try:
            self.b.stop_containers(self.run_id)
        except Exception as error:
            errors.append(str(error))
        if self.successor is not None:
            try:
                self.successor.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.successor.pid, signal.SIGKILL)
                    self.successor.wait(timeout=5)
                except Exception as error:
                    errors.append(str(error))
        return errors

    def apply(self):
        b = self.b
        b.require(os.geteuid() == 0, 'Recovery application requires independent root service')
        with (self.run / '.optimization_boundary.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.validate()
            self.directory.mkdir(mode=0o700, exist_ok=False)
            for name, source in [('spec.json', self.path), ('helper.py', Path(__file__))]:
                with (self.directory / name).open('xb') as output, source.open('rb') as incoming:
                    shutil.copyfileobj(incoming, output)
                    output.flush()
                    os.fsync(output.fileno())
            b.write_once(self.directory / 'intent.json', dict(utc=b.now(), spec_sha256=self.expected,
                         boot_id=self.s['boot_id'], checkpoint_baseline=self.s['checkpoint_baseline'],
                         deadline_utc=self.s['deadline_utc'], original_worker_unchanged=True))
            try:
                replacement = self.s['replacement']
                command = ['/usr/sbin/runuser', '-u', self.s['user'], '--', *replacement['command']]
                env = dict(os.environ, NXF_VER='26.04.6', NXF_OFFLINE='true',
                           NXF_DISABLE_CHECK_LATEST='true', PYTHONDONTWRITEBYTECODE='1')
                with (self.directory / 'worker.log').open('x') as log:
                    self.successor = subprocess.Popen(command, cwd=self.run, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    b.write_once(self.directory / 'worker_started.json', dict(utc=b.now(),
                        pid=self.successor.pid, boot_id=self.s['boot_id'], command=command,
                        manifest_sha256=replacement['manifest_sha256']))
                    while self.successor.poll() is None:
                        self.health()
                        try:
                            self.successor.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                    b.require(self.successor.returncode == 0, 'Frozen worker failed; no automatic retry')
                b.Boundary.validate_successor_completion(self)
                b.require(not b.containers(self.run_id), 'Containers remain after worker completion')
                b.write_once(self.directory / 'completed.json', dict(utc=b.now(), spec_sha256=self.expected,
                    manifest_sha256=replacement['manifest_sha256'], original_worker_unchanged=True,
                    completion_sha256=b.sha(replacement['completion_path']), published=True))
            except BaseException as error:
                errors = self.stop_owned_work()
                b.write_once(self.directory / 'failed.json', dict(utc=b.now(), spec_sha256=self.expected,
                    error=str(error), cleanup_errors=errors, automatic_retry=False,
                    failure_policy='STOP_VM_RETAIN_DISK_NO_DELETE'))
                raise
            finally:
                try:
                    b.Boundary.publish(self)
                finally:
                    b.require(b.metadata_identity() == self.s['instance'], 'VM identity changed; shutdown refused')
                    b.run_checked(['/usr/sbin/shutdown', '-h', 'now'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    recovery = RebootRecovery(args.spec, args.spec_sha256).validate()
    if args.apply:
        def terminate(signum, _frame):
            raise RuntimeError('Recovery service received signal ' + str(signum))
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, terminate)
        recovery.apply()
    else:
        print(json.dumps(dict(state='VERIFIED_NOT_EXECUTED', spec_sha256=args.spec_sha256,
                              boot_id=recovery.s['boot_id'], deadline_utc=recovery.s['deadline_utc'])))


if __name__ == '__main__':
    main()
