#!/usr/bin/env python3
"""Replace the local R02 controller after its authentic preprocessing command.

Only --apply changes process state. The running Nextflow child keeps its limits
and completes normally. An independent root service watches free disk and time
while the old, unprivileged Python controller cannot launch its next command.
No data are deleted, no checkpoint is synthesized, and this main VM is not shut
down. The successor separately verifies the genuine recovery and task evidence.
"""
from __future__ import annotations
import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time
import types

SCHEMA='r02_controller_fork_barrier_v1'

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def verify(spec_path,expected):
    path=Path(spec_path).resolve()
    require(sha(path)==expected,'Controller barrier manifest changed')
    spec=read(path);run=Path(spec['run_dir'])
    require(spec['schema']==SCHEMA and spec['helper_sha256']==sha(__file__),'Wrong controller barrier')
    require(run.resolve()==run and path.parent==run/'repairs/preprocess-v1/controller-boundary',
            'Unexpected controller barrier directory')
    require(sha(run/'run.json')==spec['run_sha256'],'Original run changed')
    frozen=run/'repairs/boundary-v2';hashes=read(frozen/'source.sha256.json')
    require(sha(frozen/'source.sha256.json')==spec['original_source_manifest_sha256'],'Original source changed')
    helper=frozen/'source/bin/r02_stage_boundary_handoff.py'
    require(sha(helper)==hashes['bin/r02_stage_boundary_handoff.py'],'Identity verifier changed')
    module=types.ModuleType('_authenticated_boundary');module.__file__=str(helper)
    exec(compile(helper.read_bytes(),str(helper),'exec'),module.__dict__)
    for role in ('supervisor','child'):
        info=module.authenticated(spec[role])
        require(info is not None,'Expected live process is absent: '+role)
    supervisor=module.authenticated(spec['supervisor']);child=module.authenticated(spec['child'])
    require(child['ppid']==supervisor['pid'],'Child is not owned by this controller')
    require(supervisor['uids'][0]!=0 and supervisor['cap_eff']==0,'Controller must be unprivileged')
    require(tuple(resource.prlimit(supervisor['pid'],resource.RLIMIT_NPROC))==tuple(spec['original_limits']),
            'Original controller limits changed')
    successor=spec['successor']
    require(Path(successor['script']).resolve().is_relative_to(run/'repairs/preprocess-v1'),
            'Successor escaped amendment')
    require(sha(successor['script'])==successor['script_sha256'] and
            sha(successor['manifest'])==successor['manifest_sha256'],'Successor changed')
    require(spec['free_disk_min_gib']==12 and spec['user']=='jose.tantalean','Unexpected execution scope')
    require(datetime.fromisoformat(spec['deadline_utc']).timestamp()>time.time(),'Deadline already reached')
    coordinator=read(run/'repairs/preprocess-v1/coordinator/manifest.json')
    require(spec['deadline_utc']==coordinator['deadline_utc'],'Handoff cannot extend the recorded deadline')
    state=read(run/'status.json')
    require(state['stage']=='chr21_M01_M02_M021' and state['pid']==spec['child']['pid'] and state['state']=='RUNNING',
            'Controller is no longer at the recorded preprocessing command')
    return spec,module

def apply(spec,path,module):
    with (path.parent/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return apply_locked(spec,path,module)

def apply_locked(spec,path,module):
    run=Path(spec['run_dir']);directory=path.parent
    def event(state,**extra):
        value=dict(utc=datetime.now(timezone.utc).isoformat(),state=state,**extra)
        with (directory/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        module.write_json(directory/'status.json',value)
        print(json.dumps(value),flush=True)
    require(os.geteuid()==0,'Applying a fork barrier requires the independent root service')
    require(not (directory/'armed.json').exists(),'Controller barrier already armed')
    stopped=False;restricted=False
    try:
        child_limits=resource.prlimit(spec['child']['pid'],resource.RLIMIT_NPROC)
        module.send(spec['supervisor'],signal.SIGSTOP);stopped=True
        module.wait_stopped(spec['supervisor'])
        require(module.authenticated(spec['child']) is not None,'Child exited before arming')
        resource.prlimit(spec['supervisor']['pid'],resource.RLIMIT_NPROC,(0,spec['original_limits'][1]));restricted=True
        require(resource.prlimit(spec['child']['pid'],resource.RLIMIT_NPROC)==child_limits,'Child limits unexpectedly changed')
        module.write_json(directory/'armed.json',dict(spec_sha256=sha(path),supervisor=spec['supervisor'],
            child=spec['child'],original_limits=spec['original_limits']),once=True)
        module.send(spec['supervisor'],signal.SIGCONT);stopped=False
        event('ARMED_CHILD_CONTINUES')
        deadline=datetime.fromisoformat(spec['deadline_utc']).timestamp()
        while module.authenticated(spec['supervisor']) is not None:
            require(time.time()<deadline,'Controller handoff deadline reached')
            require(shutil.disk_usage(run).free>=12*1024**3,'Controller disk below 12 GiB')
            time.sleep(10)
        require(module.authenticated(spec['child']) is None,'Child remains after controller exit')
        require(not module.containers(read(run/'run.json')['run_id']),'Old controller still owns a container')
        require((run/'repairs/local-preprocess-v1/completed.json').is_file(),
                'Preprocessing did not produce genuine recovery evidence')
        successor=spec['successor']
        require(sha(successor['script'])==successor['script_sha256'] and
                sha(successor['manifest'])==successor['manifest_sha256'],'Successor changed before execution')
        remaining=max(1,int(deadline-time.time()))
        command=['systemd-run','--wait','--collect','--unit=dnabr-r02-optimized-controller-'+run.name,
                 '--property=User='+spec['user'],'--property=Group='+spec['user'],
                 '--property=KillMode=control-group','--property=RuntimeMaxSec='+str(remaining),
                 '--property=TimeoutStopSec=90','--property=WorkingDirectory='+str(run),
                 '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=NXF_VER=26.04.6',
                 '--setenv=NXF_OFFLINE=true','--setenv=NXF_DISABLE_CHECK_LATEST=true',
                 '--setenv=CLOUDSDK_CONFIG='+os.environ['CLOUDSDK_CONFIG'],
                 '/usr/bin/python3',successor['script'],'--spec',successor['manifest'],
                 '--expected-spec-sha256',successor['manifest_sha256'],'--run']
        event('STARTING_VERIFIED_SUCCESSOR',command=command)
        result=subprocess.run(command,check=False,timeout=remaining+120)
        require(result.returncode==0,'Successor failed: '+str(result.returncode))
        final=read(run/'status.json')
        require(final.get('state')=='COMPLETE' and final.get('stage')=='COMPLETE_PUBLISHED',
                'Successor exited without published-completion status')
        require(read(run/'repairs/preprocess-v1/coordinator/resume_status.json').get('state')
                =='PARALLEL_AGGREGATION_COMPLETE','Coordinator did not record complete aggregation')
        event('SUCCESSOR_COMPLETED')
    except BaseException as error:
        try:event('FAILED_CLOSED',error=str(error))
        except OSError:pass
        if restricted:
            module.stop_containers(read(run/'run.json')['run_id'])
            if module.authenticated(spec['child']) is not None:module.send(spec['child'],signal.SIGTERM)
            # Do not release the old controller into unapproved downstream commands.
            if module.authenticated(spec['supervisor']) is not None:module.send(spec['supervisor'],signal.SIGTERM)
        elif stopped:
            module.send(spec['supervisor'],signal.SIGCONT)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--spec-sha256',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();os.umask(0o077)
    spec,module=verify(args.spec,args.spec_sha256)
    if args.apply:apply(spec,args.spec,module)
    else:print(json.dumps(dict(state='VERIFIED_NOT_APPLIED',schema=SCHEMA)))

if __name__=='__main__':main()
