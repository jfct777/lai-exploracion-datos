#!/usr/bin/env python3
"""Freeze a parallel coordinator request without changing existing jobs."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    folder = run/'repairs/parallel-v1'
    fleet = json.loads((folder/'fleet.json').read_text())
    source = Path(__file__).parent/'r02_parallel_coordinator.py'
    if sha(source) != 'e1dbeb276fe687274f6344906b15e386af1c52e4e27598ed494d522adec68f4a':
        raise ValueError('Coordinator differs from the tested version')
    service = 'dnabr-r02-autosomes-20261001b-boundary-v2.service'
    pid = int(subprocess.check_output(['systemctl', '--user', 'show', service, '-p', 'MainPID', '--value']))
    command = Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'r02_stage_boundary_handoff.py' not in command:
        raise ValueError('Unexpected old watcher')
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    if Path(f'/proc/{pid}/task/{pid}/children').read_text().split():
        raise ValueError('Old watcher has children')
    if (run/'repairs/boundary-v2/activation.json').exists():
        raise ValueError('Previous controller already activated')
    frozen = folder/'coordinator.py'
    with frozen.open('xb') as stream:
        stream.write(source.read_bytes())
    entries = []
    for worker in fleet['workers']:
        for chromosome in worker['chromosomes']:
            entries.append(dict(chromosome=chromosome, worker_id=worker['worker'],
                publication_prefix=worker['completion_uri'].rsplit('/00_worker_provenance/',1)[0],
                completion_uri=worker['completion_uri'], worker_run_sha256=worker['run_sha256']))
    request = run/'repairs/boundary-v2/request.json'
    manifest = dict(schema='r02_parallel_coordinator_v1', run_dir=str(run), request=str(request),
        request_sha256=sha(request), coordinator_sha256=sha(frozen),
        source_manifest_sha256=fleet['scientific_source_manifest_sha256'],
        samples_sha256=sha(run/'samples.txt'), deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=79.9)).isoformat(),
        poll_seconds=30, max_import_bytes=50*1024**3,
        old_watcher=dict(pid=pid,start_ticks=int(fields[19]),cmdline_sha256=hashlib.sha256(command).hexdigest()),
        remote_chromosomes=entries)
    target = folder/'manifest.json'
    with target.open('x') as stream:
        json.dump(manifest,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(manifest=str(target),sha256=sha(target),old_watcher=manifest['old_watcher'])))


if __name__ == '__main__':
    main()
