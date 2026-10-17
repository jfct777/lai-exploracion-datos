#!/usr/bin/env python3
"""Run sequential authenticated Nextflow stages under one original deadline.

The caller owns scientific validation and host/VM admission. This module owns
session separation, complete trace checks, bounded execution and owned cleanup.
It neither renews a deadline nor edits input parameters or frozen workflows.
"""
import csv,json,os,signal,subprocess,time
from pathlib import Path
import r02_publish_evidence as pub


def terminate_group(process):
    if process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def run_stage(root, *, stage, nextflow, workflow, params, config_files, expected_process_counts,
              deadline_epoch, check_space, stop_owned, env=None, poll_seconds=2):
    pub.safe_relative(stage)
    pub.require('/' not in stage and expected_process_counts,'One named bounded stage required')
    pub.require(time.time()<deadline_epoch,'Original stage deadline exhausted')
    root=Path(root);folder=root/'nextflow'/stage;folder.mkdir(parents=True,exist_ok=True)
    attempt=str(time.time_ns());session='r02_'+stage+'_'+attempt
    workflow=pub.safe_local(workflow);params=pub.safe_local(params)
    configs=[pub.safe_local(x) for x in config_files]
    pins={str(p):pub.pipeline.sha(p) for p in [workflow,params,*configs]}
    contract=dict(workflow=str(workflow),params=str(params),pins=pins,expected_process_counts=expected_process_counts)
    contract_path=folder/'stage-contract.json'
    if contract_path.exists():pub.require(pub.read_unique_json(contract_path)==contract,'Stage source/parameters changed before resume')
    else:pub.write_receipt_new(contract_path,contract)
    trace=folder/('trace.'+attempt+'.tsv');log=folder/('nextflow.'+attempt+'.log');console=folder/('console.'+attempt+'.log')
    previous=sorted(folder.glob('session.*.json'))
    command=[str(nextflow),'-C',','.join(map(str,configs)),'-log',str(log),'run',str(workflow),
             '-params-file',str(params),'-work-dir',str(folder/'work'),'-with-trace',str(trace),'-ansi-log','false','-name',session]
    if previous:command+=['-resume',pub.read_unique_json(previous[-1])['name']]
    receipt=folder/('session.'+attempt+'.json')
    pub.write_receipt_new(receipt,dict(name=session,command=command,deadline_epoch=deadline_epoch,
                                    started_utc=pub.pipeline.now(),pins=pins))
    execution_env={**os.environ,'NXF_OFFLINE':'true','NXF_DISABLE_CHECK_LATEST':'true','NXF_SYNTAX_PARSER':'v1',
                   'NXF_OPTS':'-Xms64m -Xmx768m','PYTHONDONTWRITEBYTECODE':'1',**(env or {})}
    check_space()
    with console.open('x') as output:
        process=subprocess.Popen(command,cwd=folder,env=execution_env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                pub.require(time.time()<deadline_epoch,'Original stage deadline reached')
                check_space();time.sleep(min(poll_seconds,max(.01,deadline_epoch-time.time())))
            pub.require(process.returncode==0,'Nextflow stage failed; preserve work and completed products')
        except BaseException:
            terminate_group(process);stop_owned();raise
    pub.require(all(pub.pipeline.sha(path)==digest for path,digest in pins.items()),'Stage source changed while running')
    with trace.open() as handle:rows=list(csv.DictReader(handle,delimiter='\t'))
    pub.require(len(rows)==sum(expected_process_counts.values()) and
                all(x['status'] in ('COMPLETED','CACHED') and x['exit']=='0' for x in rows),'Incomplete stage trace')
    names=[x['name'].split(' (')[0] for x in rows]
    pub.require(set(names)==set(expected_process_counts) and all(names.count(k)==v for k,v in expected_process_counts.items()),'Unexpected process family/count')
    result=dict(status='COMPLETE_AUTHENTICATED_NEXTFLOW_STAGE',stage=stage,session=session,
                trace=str(trace),trace_sha256=pub.pipeline.sha(trace),tasks=rows,pins=pins,deadline_epoch=deadline_epoch)
    pub.write_receipt_new(folder/('completed.'+attempt+'.json'),result)
    return result
