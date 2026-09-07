#!/usr/bin/env python3
"""Chain two frozen Nextflow training stages under one wall-time and cost ceiling.

Only development artifacts are mounted into the local audit container. GPU jobs
remain Batch-owned; this controller never deletes a resource outside its exact
run labels. Cost checks are conservative estimates, not billing-account totals.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

from m39_gpu_launch import (cancel_owned_jobs, native_auth, native_env_prefix,
                            observed_job_ids, validate_target, watcher_command, write_json)
from m39_launch_ordered_training import sha256
from m39_ordered_gpu_launch import SOURCES, prepare
from m39_ordered_gpu_manifest import load_plan, require

SCHEMA = 'm39-ordered-gpu-campaign-v1'
CAMPAIGN_SOURCES = (*SOURCES, 'bin/m39_ordered_campaign.py', 'bin/m39_ordered_training_sweep.py')


def private_path(value: str, repository: Path, *, exists: bool = True) -> Path:
    path = Path(value).absolute()
    require(re.fullmatch(r'[A-Za-z0-9_./-]+', str(path)) is not None
            and not any(part.is_symlink() for part in (path, *path.parents)), 'unsafe campaign path')
    path = path.resolve(strict=exists)
    require(path.is_relative_to(repository / '.claude/runs') and path != repository / '.claude/runs',
            'campaign paths must remain in private project runs')
    return path


def load_campaign(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    required = {'schema_version', 'repository', 'run_dir', 'source_commit', 'source_sha256',
                'input_sha256', 'native_auth_dir', 'service_account', 'gpu_image', 'cpu_image',
                'wall_timeout_seconds', 'costs', 'screen', 'followup'}
    require(set(cfg) == required and cfg['schema_version'] == SCHEMA, 'campaign schema differs')
    repo = Path(cfg['repository']).resolve(strict=True)
    private_path(str(path), repo)
    run = private_path(cfg['run_dir'], repo)
    require(run.is_dir(), 'campaign run directory required')
    require(re.fullmatch(r'[0-9a-f]{40}', cfg['source_commit']) is not None, 'full source commit required')
    require(set(CAMPAIGN_SOURCES) <= set(cfg['source_sha256']), 'campaign source closure incomplete')
    for relative, digest in cfg['source_sha256'].items():
        source = (repo / relative).resolve(strict=True)
        require(source.is_relative_to(repo) and not (repo / relative).is_symlink()
                and sha256(source) == digest, 'campaign source changed')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
            == cfg['source_commit'], 'campaign source commit changed')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=repo, text=True).strip(), 'tracked source is dirty')
    require(set(cfg['screen']) == {'run_dir', 'train_store', 'select_store', 'development', 'plan'},
            'screen input inventory differs')
    require(set(cfg['followup']) == {'run_dir', 'train_store', 'select_store', 'development',
                                  'base_config', 'resources_file', 'seeds'}, 'followup input inventory differs')
    for stage in (cfg['screen'], cfg['followup']):
        for key, value in stage.items():
            if key != 'seeds':
                private_path(value, repo)
        validate_target(Path(stage['run_dir']).name, cfg['gpu_image'])
    require(cfg['screen']['run_dir'] != cfg['followup']['run_dir'], 'stages require distinct run prefixes')
    required_seals = {cfg['screen']['plan'], cfg['followup']['base_config'], cfg['followup']['resources_file']}
    require(required_seals <= set(cfg['input_sha256']), 'campaign decision-input seals missing')
    for value, digest in cfg['input_sha256'].items():
        require(sha256(private_path(value, repo)) == digest, 'frozen campaign input changed')
    require(re.fullmatch(r'us-central1-docker\.pkg\.dev/uspbr-242713/dnabr-lai/[a-z0-9-]+@sha256:[a-f0-9]{64}',
                         cfg['cpu_image']) is not None, 'CPU image must be a project-owned digest')
    require(type(cfg['wall_timeout_seconds']) is int and 60 <= cfg['wall_timeout_seconds'] <= 39600,
            'campaign deadline must not exceed eleven hours')
    costs = cfg['costs']
    require(set(costs) == {'max_usd', 'worker_hourly_usd', 'worker_count_ceiling', 'reserve_usd',
                          'screen_forecast_usd', 'followup_forecast_usd'},
            'cost fields differ')
    require(all(type(costs[key]) in (int, float) and costs[key] > 0 for key in costs)
            and costs['max_usd'] <= 20 and costs['worker_count_ceiling'] == 2, 'invalid campaign cost ceilings')
    maximum = cfg['wall_timeout_seconds'] / 3600 * costs['worker_hourly_usd'] * costs['worker_count_ceiling']
    require(maximum + costs['reserve_usd'] <= costs['max_usd'], 'wall-time ceiling exceeds frozen cost envelope')
    require(load_plan(Path(cfg['screen']['plan']))['stage'] == 'exploratory_screen', 'screen plan stage differs')
    native_auth(Path(cfg['native_auth_dir']), cfg['service_account'], repo)
    return cfg


class Campaign:
    def __init__(self, path: Path):
        self.path, self.cfg = path, load_campaign(path)
        self.config_hash = sha256(path)
        self.repo, self.run = Path(self.cfg['repository']), Path(self.cfg['run_dir'])
        self.auth = native_auth(Path(self.cfg['native_auth_dir']), self.cfg['service_account'], self.repo)
        self.started = time.monotonic()
        self.active_receipt = None
        self.events = 0

    def remaining(self) -> float:
        return self.cfg['wall_timeout_seconds'] - (time.monotonic() - self.started)

    def event(self, name: str, **values) -> None:
        self.events += 1
        item = {'event': name, 'elapsed_seconds': time.monotonic() - self.started, **values}
        write_json(self.run / f'event-{self.events:03d}-{name}.json', item)
        print(json.dumps(item), flush=True)

    def check_sources(self) -> None:
        require(sha256(self.path) == self.config_hash, 'campaign config changed')
        require(load_campaign(self.path) == self.cfg, 'campaign binding changed')

    def execute(self, command: list[str], log: Path, *, timeout: float | None = None,
                gpu_watcher: bool = False, cidfile: Path | None = None) -> None:
        remaining = self.remaining()
        require(remaining > 0, 'campaign wall-time ceiling reached')
        budget = min(remaining, timeout) if timeout is not None else remaining
        process = None
        try:
            with log.open('x') as stream:
                process = subprocess.Popen(command, cwd=self.run, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + budget
                while process.poll() is None:
                    wait = deadline - time.monotonic()
                    if wait <= 0:
                        raise subprocess.TimeoutExpired(command, budget)
                    try:
                        process.wait(timeout=min(30, wait))
                    except subprocess.TimeoutExpired:
                        continue
                require(process.returncode == 0, f'child failed with exit code {process.returncode}; see {log.name}')
        finally:
            if process is not None and process.poll() is None:
                # Let the GPU watcher cancel its exact jobs before forcefully ending it.
                process.terminate()
                try:
                    process.wait(timeout=180 if gpu_watcher else 10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            if cidfile is not None and cidfile.is_file():
                identifier = cidfile.read_text().strip()
                if re.fullmatch(r'[0-9a-f]{64}', identifier):
                    # A cidfile is created by this docker invocation only; --rm usually
                    # already removed the container. Never target a shared name.
                    subprocess.run(['docker', 'rm', '-f', identifier], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=30, check=False)

    def gcloud(self) -> list[str]:
        return [*native_env_prefix(self.auth), 'gcloud', f"--account={self.auth['service_account']}"]

    def ensure_retired(self, receipt: Path) -> None:
        record = json.loads(receipt.read_text())
        jobs = observed_job_ids(receipt.parent, record['run_id'], process_name=record['process_name'],
                                max_jobs=record['resources']['max_jobs'])
        require(jobs, 'no native job IDs available to verify retirement')
        for name, uid in jobs.items():
            job = json.loads(subprocess.check_output([*self.gcloud(), 'batch', 'jobs', 'describe', name,
                '--project=uspbr-242713', '--location=us-central1', '--format=json'], timeout=30))
            require(job.get('uid') == uid and job.get('labels', {}).get('team') == 'frank'
                    and job['labels'].get('m39_run') == record['run_id']
                    and job.get('status', {}).get('state') in ('SUCCEEDED', 'FAILED'), 'worker is not terminal or owned')
        deadline = time.monotonic() + min(180, max(0, self.remaining()))
        while True:
            machines = json.loads(subprocess.check_output([*self.gcloud(), 'compute', 'instances', 'list',
                '--project=uspbr-242713', f"--filter=labels.m39_run={record['run_id']}",
                '--format=json(name,status,labels)'], timeout=30))
            require(all(vm.get('labels', {}).get('m39_run') == record['run_id']
                        and vm['labels'].get('team') == 'frank' for vm in machines), 'VM label lookup differs')
            if not any(vm.get('status') != 'TERMINATED' for vm in machines):
                return
            require(time.monotonic() + 10 < deadline, 'GPU workers are not retired; do not start next stage')
            time.sleep(10)

    def stage(self, name: str, stage: dict) -> Path:
        self.check_sources()
        args = argparse.Namespace(**{key: Path(value) for key, value in stage.items()},
            image=self.cfg['gpu_image'], native_auth_dir=Path(self.cfg['native_auth_dir']),
            service_account=self.cfg['service_account'])
        receipt = prepare(args, self.repo)
        self.active_receipt = receipt
        self.event('stage-' + name + '-start', receipt=str(receipt))
        self.execute(watcher_command(receipt, self.auth), self.run / f'stage-{name}.controller.log', gpu_watcher=True)
        self.ensure_retired(receipt)
        record = json.loads(receipt.read_text())
        destination = self.run / ('primary-' + name)
        destination.mkdir(mode=0o700, exist_ok=False)
        for attempt in range(1, 4):
            try:
                self.execute([*self.gcloud(), 'storage', 'rsync', '--recursive', record['outputs'], str(destination)],
                    self.run / f'download-{name}-{attempt}.log', timeout=600)
                break
            except (OSError, ValueError, subprocess.SubprocessError):
                if attempt == 3:
                    raise
        self.active_receipt = None
        self.event('stage-' + name + '-downloaded', outputs=str(destination))
        return destination

    def cpu(self, name: str, arguments: list[str], mounts: dict[str, Path]) -> Path:
        self.check_sources()
        root = self.run / ('cpu-' + name)
        root.mkdir(mode=0o700, exist_ok=False)
        cidfile = root / 'container.cid'
        command = ['docker', 'run', '--rm', '--cidfile', str(cidfile), '--network', 'none',
            '--cpus', '2', '--memory', '4g', '--user', f'{os.getuid()}:{os.getgid()}',
            '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONPATH=/code',
            '--env', 'OMP_NUM_THREADS=2', '--env', 'OPENBLAS_NUM_THREADS=2']
        for target, source in {'/code': self.repo / 'bin', **mounts}.items():
            command.extend(('--mount', f'type=bind,src={source},dst={target},readonly'))
        command.extend(('--mount', f'type=bind,src={root},dst=/output', self.cfg['cpu_image'],
                        'python3', '/code/m39_ordered_training_sweep.py', *arguments,
                        '--outdir', '/output/result'))
        self.execute(command, root / 'controller.log', timeout=900, cidfile=cidfile)
        return root / 'result'

    def measured_screen_cost(self, outputs: Path) -> float:
        """Use completed arm wall times, including SELECT, to expose a slow screen."""
        plan = load_plan(Path(self.cfg['screen']['plan']))
        seconds = 0.0
        for group in plan['groups']:
            for spec in group['configs']:
                config = json.loads((Path(self.cfg['screen']['plan']).parent / spec['file']).read_text())
                path = outputs / ('training-' + group['id']) / config['arm'] / 'training.receipt.json'
                receipt = json.loads(path.read_text())
                require(receipt.get('config') == config, 'screen timing belongs to a different config')
                duration = receipt.get('elapsed_seconds')
                require(type(duration) in (int, float) and duration > 0, 'missing completed arm wall time')
                seconds += duration
        return seconds / 3600 * self.cfg['costs']['worker_hourly_usd']

    def budget_allows_followup(self) -> dict:
        costs = self.cfg['costs']
        elapsed = time.monotonic() - self.started
        spent = elapsed / 3600 * costs['worker_hourly_usd'] * costs['worker_count_ceiling']
        screen_actual = getattr(self, 'screen_compute_cost_usd', 0.0)
        ratio = max(1.0, screen_actual / costs['screen_forecast_usd'])
        forecast = costs['followup_forecast_usd'] * ratio
        projected = spent + forecast + costs['reserve_usd']
        return {'elapsed_cost_upper_estimate_usd': spent, 'followup_projected_usd': projected,
                'screen_arm_time_cost_estimate_usd': screen_actual, 'followup_forecast_usd': forecast,
                'observed_slowdown_multiplier': ratio,
                'ceiling_usd': costs['max_usd'], 'allowed': projected <= costs['max_usd'],
                'actual_invoice': False}

    def run_all(self) -> dict:
        summary = {'schema_version': 'm39-ordered-campaign-completion-v1', 'status': 'FAILED',
                   'campaign_sha256': self.config_hash, 'source_commit': self.cfg['source_commit'],
                   'SCORE_opened': False}
        try:
            self.event('campaign-start', wall_timeout_seconds=self.cfg['wall_timeout_seconds'])
            outputs_a = self.stage('a', self.cfg['screen'])
            plan_a = Path(self.cfg['screen']['plan'])
            post_a = self.cpu('audit-a', ['audit', '--plan', '/plan/' + plan_a.name, '--outputs', '/primary'],
                              {'/plan': plan_a.parent, '/primary': outputs_a})
            self.screen_compute_cost_usd = self.measured_screen_cost(outputs_a)
            self.event('screen-audited', comparison=str(post_a / 'comparison.json'))
            budget = self.budget_allows_followup()
            self.event('followup-budget', **budget)
            summary['followup_budget'] = budget
            if not budget['allowed']:
                summary['status'] = 'SCREEN_AUDITED_FOLLOWUP_STOPPED_BY_BUDGET'
                return summary
            self.check_sources()
            followup = self.cfg['followup']
            plan_b = self.cpu('prepare-b', ['prepare-followup', '--base-config', '/base/config.json',
                '--resources-file', '/resources/resources.json', '--screen-plan', '/plan/' + plan_a.name,
                '--screen-outputs', '/primary', '--seeds', *map(str, followup['seeds'])],
                {'/base/config.json': Path(followup['base_config']),
                 '/resources/resources.json': Path(followup['resources_file']),
                 '/plan': plan_a.parent, '/primary': outputs_a}) / 'plan.json'
            require(load_plan(plan_b)['stage'] == 'controlled_followup', 'followup plan stage differs')
            # Audit/preparation time is charged conservatively before creating Stage B.
            budget = self.budget_allows_followup()
            require(budget['allowed'], 'followup estimate crossed cost ceiling during preparation')
            stage_b = {key: followup[key] for key in ('run_dir', 'train_store', 'select_store', 'development')}
            stage_b['plan'] = str(plan_b)
            outputs_b = self.stage('b', stage_b)
            post_b = self.cpu('audit-b', ['audit', '--plan', '/plan/plan.json', '--outputs', '/primary'],
                              {'/plan': plan_b.parent, '/primary': outputs_b})
            summary.update(status='TWO_STAGES_AUDITED_EXPLORATORY_ONLY',
                           comparison_a=str(post_a / 'comparison.json'), comparison_b=str(post_b / 'comparison.json'))
            self.event('campaign-audited', **summary)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            summary.update(failure_type=type(exc).__name__, failure=str(exc))
        finally:
            if self.active_receipt is not None:
                receipt = json.loads(self.active_receipt.read_text())
                summary['cleanup'] = cancel_owned_jobs(receipt['run_id'], self.active_receipt.parent,
                    auth=self.auth, process_name=receipt['process_name'], max_jobs=receipt['resources']['max_jobs'])
            summary['elapsed_seconds'] = time.monotonic() - self.started
            write_json(self.run / 'campaign.completion.json', summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--watch', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    cfg = load_campaign(args.campaign)
    run = Path(cfg['run_dir'])
    require(not (run / 'campaign.completion.json').exists(), 'campaign already completed')
    if args.watch:
        def interrupt(signum, frame):
            raise InterruptedError(f'Campaign received signal {signum}')
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        result = Campaign(args.campaign).run_all()
        raise SystemExit(0 if result['status'] != 'FAILED' else 1)
    marker = run / 'campaign.launch.json'
    command = [*native_env_prefix(native_auth(Path(cfg['native_auth_dir']), cfg['service_account'],
                            Path(cfg['repository']))), sys.executable, str(Path(__file__).resolve()),
               '--watch', '--campaign', str(args.campaign.resolve())]
    write_json(marker, {'schema_version': SCHEMA, 'campaign_sha256': sha256(args.campaign), 'argv': command})
    # Redirect a known private log; the tmux server owns this command after SSH exits.
    shell_command = shlex.join(command) + ' > ' + shlex.quote(str(run / 'campaign.controller.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', run.name, '-c', str(run), shell_command], check=True)
    print(json.dumps({'status': 'DETACHED_CAMPAIGN_STARTED', 'receipt': str(marker)}))


if __name__ == '__main__':
    main()
