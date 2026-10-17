#!/usr/bin/env python3
"""Build the portable, hash-inventoried K02/K07 FIT/SELECT deployment package."""
from __future__ import annotations
import argparse,hashlib,json,shutil,stat
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def run(repo,out,nextflow_jar,nextflow_launcher):
    repo=Path(repo).resolve();out=Path(out).resolve();assert not out.exists();out.mkdir(parents=True)
    sources={
      'workflow/main.nf':'workflows/r02_structure_fullcommon_pair_select.nf',
      'bin/r02_pair_stage_view.py':'bin/r02_pair_stage_view.py',
      'bin/r02_pair_stage_manifest.py':'bin/r02_pair_stage_manifest.py',
      'bin/assemble_fullcommon_v4.py':'bin/r02_pair_assemble_fullcommon.py',
      'bin/r02_structure_increment.py':'bin/r02_structure_increment.py',
      'bin/r02_pair_operator.py':'bin/r02_pair_operator.py',
      'bin/r02_pair_preconditioner.py':'bin/r02_pair_preconditioner.py',
      'bin/r02_structure_pair_reduce.py':'bin/r02_structure_pair_reduce.py',
      'config/assembly-tasks.json':'.claude/runs/r02-corrections-20261006j/parallel-next-20261008an/rare-increment/prepared/assembly-tasks.json',
      'config/staged-source-objects.json':'.claude/runs/r02-corrections-20261006j/common-search-territories-20261008al/runtime/staged-source-objects.json',
      'config/shard-manifest-objects.json':'.claude/runs/r02-corrections-20261006j/parallel-continuation-20261009aq/pair/shard-manifest-objects.json',
      'config/evaluation-BLOCK_AND_LD_MARGIN.json':'.claude/runs/r02-corrections-20261006j/common-territories-20261008ak/evaluation-BLOCK_AND_LD_MARGIN.json',
      'config/evaluation-WHOLE_CHROMOSOME.json':'.claude/runs/r02-corrections-20261006j/common-territories-20261008ak/evaluation-WHOLE_CHROMOSOME.json',
      'config/pair-numerics.json':'.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/pair-backend/pair-numerics.json',
      'config/RUN_SCOPE_AMENDMENT.json':'.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/pair-backend/RUN_SCOPE_AMENDMENT.json',
      'config/ADMISSION.template.json':'.claude/runs/r02-corrections-20261006j/parallel-continuation-20261009aq/pair/ADMISSION-v3.template.json',
      'data/reused-chr01-part01-whole.npz':'.claude/runs/r02-corrections-20261006j/family-completion-20261008ae/common/real-source/target_chr01_part01__WHOLE_CHROMOSOME/continuous.private.npz'}
    for relative,source in sources.items():
        destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/source,destination)
    for relative,source in [('nextflow-home/framework/26.04.6/nextflow-26.04.6-one.jar',nextflow_jar),
                            ('nextflow/nextflow',nextflow_launcher)]:
        destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        if relative.endswith('/nextflow'):destination.chmod(destination.stat().st_mode|stat.S_IXUSR)
    base='/opt/r02-pair/package'
    params=dict(tasks=f'{base}/config/assembly-tasks.json',common_objects=f'{base}/config/staged-source-objects.json',
        shard_manifest_objects=f'{base}/config/shard-manifest-objects.json',
        reused_common=f'{base}/data/reused-chr01-part01-whole.npz',
        evaluation_block=f'{base}/config/evaluation-BLOCK_AND_LD_MARGIN.json',
        evaluation_whole=f'{base}/config/evaluation-WHOLE_CHROMOSOME.json',
        stager=f'{base}/bin/r02_pair_stage_view.py',manifest_stager=f'{base}/bin/r02_pair_stage_manifest.py',
        assembler=f'{base}/bin/assemble_fullcommon_v4.py',
        evaluator=f'{base}/bin/r02_structure_increment.py',reducer=f'{base}/bin/r02_structure_pair_reduce.py',
        pair_operator=f'{base}/bin/r02_pair_operator.py',pair_preconditioner=f'{base}/bin/r02_pair_preconditioner.py',
        run_scope_amendment=f'{base}/config/RUN_SCOPE_AMENDMENT.json',
        numerical_contract=f'{base}/config/pair-numerics.json',admission=f'{base}/config/ADMISSION.template.json',
        output='/opt/r02-pair/output',container='us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/dnabr-qc@sha256:3e9381d165b1e664e75093d885a62d8c180711c1c3f495324e8c08bdb5bd2af2',
        max_forks=1,select_cpus=2,select_memory_gb=8,select_disk_gb=12)
    keys={'stager':'bin/r02_pair_stage_view.py','manifest_stager':'bin/r02_pair_stage_manifest.py',
      'shard_manifest_objects':'config/shard-manifest-objects.json','assembler':'bin/assemble_fullcommon_v4.py',
      'common_objects':'config/staged-source-objects.json','reused_common':'data/reused-chr01-part01-whole.npz',
      'evaluation_block':'config/evaluation-BLOCK_AND_LD_MARGIN.json','evaluation_whole':'config/evaluation-WHOLE_CHROMOSOME.json',
      'evaluator':'bin/r02_structure_increment.py','reducer':'bin/r02_structure_pair_reduce.py',
      'pair_operator':'bin/r02_pair_operator.py','pair_preconditioner':'bin/r02_pair_preconditioner.py',
      'run_scope_amendment':'config/RUN_SCOPE_AMENDMENT.json',
      'numerical_contract':'config/pair-numerics.json','admission':'config/ADMISSION.template.json'}
    for key,relative in keys.items():params[key+'_sha256']=sha(out/relative)
    params['workflow_sha256']=sha(out/'workflow/main.nf')
    (out/'config/runtime-params.json').write_text(json.dumps(params,indent=2,sort_keys=True)+'\n')
    (out/'config/nextflow.config').write_text(
        "process.executor='local'\ndocker.enabled=true\nprocess.maxForks=1\nexecutor.queueSize=2\n")
    command="""#!/bin/bash
set -euo pipefail
test $# -eq 2
admission=$(readlink -f "$1")
output=$(readlink -m "$2")
test -f "$admission"
admission_sha=$(sha256sum "$admission" | cut -d ' ' -f1)
export NXF_HOME=/opt/r02-pair/package/nextflow-home NXF_OFFLINE=true NXF_VER=26.04.6
exec /opt/r02-pair/package/nextflow/nextflow -C /opt/r02-pair/package/config/nextflow.config \
  run /opt/r02-pair/package/workflow/main.nf \
  -params-file /opt/r02-pair/package/config/runtime-params.json \
  --admission "$admission" --admission_sha256 "$admission_sha" --output "$output" \
  -work-dir /opt/r02-pair/work -resume
"""
    (out/'run_after_review.sh').write_text(command);(out/'run_after_review.sh').chmod(0o700)
    inventory={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(out.rglob('*')) if p.is_file()}
    manifest=dict(schema='r02_k02_k07_pair_deployment_package_v1',status='COMPLETE_AWAITING_ROOT_ADMISSION',
        runtime_root=base,view_count=88,evaluate_allowed=False,files=inventory,
        total_bytes=sum(x['bytes'] for x in inventory.values()),nextflow_version='26.04.6',
        container=params['container'])
    (out/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    p.add_argument('--nextflow-jar',required=True);p.add_argument('--nextflow-launcher',required=True);a=p.parse_args()
    print(json.dumps(run(a.repo,a.out,a.nextflow_jar,a.nextflow_launcher),sort_keys=True))
