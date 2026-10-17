nextflow.enable.dsl=2

params.packet = null
params.store_request_index = null
params.repo = null
params.run_root = null
params.gcloud_config = null
params.prior_full22_gate = null
params.prior_full22_gate_sha256 = null
params.training_container = null
params.target_root = null
params.eligibility_manifest = null
params.eligibility_manifest_sha256 = null
params.stage_time = '30m'
params.assembly_time = '30m'
params.training_time = '2h'
// Technical numerical implementation. Existing behavior remains the default.
params.adam_state_precision = 'fp32'

def required(name, value) {
    if (value == null || value.toString().trim() == '') error "Missing required parameter --${name}"
    value
}

process STAGE_AUTHENTICATED_STORE {
    tag "chr${chrom}"
    cpus 1
    memory '2 GB'
    time params.stage_time
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'

    input:
    tuple val(chrom), path(request, stageAs: 'request.json'), val(request_sha256),
          path(manifest, stageAs: 'store-manifest.json'), val(manifest_sha256)

    output:
    tuple val(chrom), path("chr${chrom}.staged.json"), emit: receipts

    script:
    """
    set -euo pipefail
    export CLOUDSDK_CONFIG='${params.gcloud_config}'
    PYTHONPATH='${params.repo}/bin' python3 '${params.repo}/bin/r02_neural_structure_stage.py' \
      --request '${request}' --request-sha256 '${request_sha256}' \
      --manifest-override '${manifest}' \
      --prior-full22-gate '${params.prior_full22_gate}' \
      --prior-full22-gate-sha256 '${params.prior_full22_gate_sha256}' \
      --output 'chr${chrom}.staged.json'
    """
}

process ASSEMBLE_ADMISSION {
    tag 'full22-one-view-admission'
    cpus 1
    memory '4 GB'
    time params.assembly_time
    cache 'deep'
    errorStrategy 'terminate'

    input:
    path receipts

    output:
    path 'assembly.complete.json', emit: marker

    script:
    """
    set -euo pipefail
    DEPS='${params.repo}/.claude/runs/r02-corrections-20261006j/parallel-integral-20261009au/neural-structure/deployment/python-deps'
    PYRUN=(docker run --rm --user "\$(id -u):\$(id -g)" --cpus 1 --memory 4g \
      -v '${params.repo}:${params.repo}:ro' \
      -v '${params.repo}/.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/neural-inputs/runtime/stage:${params.repo}/.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/neural-inputs/runtime/stage:rw' \
      -v '${params.target_root}:${params.target_root}:ro' -e "PYTHONPATH=\${DEPS}:${params.repo}/bin" \
      -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
      --entrypoint python3 '${params.training_container}')
    test "\$(find . -name 'chr*.staged.json' -type f | wc -l)" -eq 22
    mkdir -p '${params.run_root}/assembly'
    PACKET_SHA=\$(sha256sum '${params.packet}' | cut -d' ' -f1)
    if [[ ! -f '${params.run_root}/assembly/build-spec.json' ]]; then
      "\${PYRUN[@]}" '${params.repo}/bin/r02_neural_structure_admission.py' materialize-spec \
        --packet '${params.packet}' --packet-sha256 "\$PACKET_SHA" \
        --staged-root '${params.repo}/.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/neural-inputs/runtime/stage' \
        --output '${params.run_root}/assembly/build-spec.json'
    fi
    SPEC_SHA=\$(sha256sum '${params.run_root}/assembly/build-spec.json' | cut -d' ' -f1)
    if [[ ! -f '${params.run_root}/assembly/person-batches/integration-manifest.json' ]]; then
      if [[ -e '${params.run_root}/assembly/person-batches' ]]; then
        mv '${params.run_root}/assembly/person-batches' '${params.run_root}/assembly/person-batches.incomplete.'\$(date -u +%Y%m%dT%H%M%SZ)
      fi
      "\${PYRUN[@]}" '${params.repo}/.claude/runs/r02-corrections-20261006j/parallel-launch-20261008ac/person_batching/build_human_autosomal_person_batches.py' \
        --spec '${params.run_root}/assembly/build-spec.json' --spec-sha256 "\$SPEC_SHA" \
        --output '${params.run_root}/assembly/person-batches'
    fi
    if [[ ! -f '${params.run_root}/final/FINAL.json' ]]; then
      if [[ -e '${params.run_root}/final' ]]; then
        mv '${params.run_root}/final' '${params.run_root}/final.incomplete.'\$(date -u +%Y%m%dT%H%M%SZ)
      fi
      "\${PYRUN[@]}" '${params.repo}/bin/r02_neural_structure_admission.py' finalize \
        --packet '${params.packet}' --packet-sha256 "\$PACKET_SHA" \
        --build-manifest '${params.run_root}/assembly/person-batches/integration-manifest.json' \
        --eligibility-manifest '${params.eligibility_manifest}' \
        --eligibility-manifest-sha256 '${params.eligibility_manifest_sha256}' \
        --output '${params.run_root}/final'
    fi
    FINAL_SHA=\$(sha256sum '${params.run_root}/final/FINAL.json' | cut -d' ' -f1)
    python3 -c 'import json,sys; json.dump({"schema":"r02_neural_structure_assembly_marker_v1","final_path":sys.argv[1],"final_sha256":sys.argv[2]},open("assembly.complete.json","w"),sort_keys=True,indent=2)' '${params.run_root}/final/FINAL.json' "\$FINAL_SHA"
    """
}

process TRAIN_INSTRUMENTATION_ARM {
    tag "${arm}"
    cpus 1
    memory '30 GB'
    time params.training_time
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'

    input:
    tuple val(arm), path(marker)

    output:
    path "${arm}.measurement.json", emit: measurements

    script:
    def recovery = arm == 'RARE' ? '--force-recovery-step1' : ''
    """
    set -euo pipefail
    DEPS='${params.repo}/.claude/runs/r02-corrections-20261006j/parallel-integral-20261009au/neural-structure/deployment/python-deps'
    PYRUN=(docker run --rm --user "\$(id -u):\$(id -g)" --cpus 1 --memory 30g \
      -v '${params.repo}:${params.repo}:ro' \
      -v '${params.repo}/.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/neural-inputs/runtime/stage:${params.repo}/.claude/runs/r02-corrections-20261006j/parallel-launch-20261009ap/neural-inputs/runtime/stage:rw' \
      -v "\$PWD:\$PWD:rw" -w "\$PWD" \
      -v '${params.target_root}:${params.target_root}:ro' -e "PYTHONPATH=\${DEPS}:${params.repo}/bin" \
      -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
      --entrypoint python3 '${params.training_container}')
    if [[ -f '${params.run_root}/${arm}.measurement.json' ]]; then
      python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="COMPLETE" and d["arm"]==sys.argv[2] and d["final_confirmed_step"]==2 and d["adam_state_precision"]==sys.argv[3]' '${params.run_root}/${arm}.measurement.json' '${arm}' '${params.adam_state_precision}'
      cp '${params.run_root}/${arm}.measurement.json' '${arm}.measurement.json'
    else
      CONTRACT='${params.run_root}/final/contract.json'
      RESOLVED=\$(python3 -c 'import json; print(json.load(open("${params.run_root}/final/FINAL.json"))["resolved_config"]["path"])')
      CONTRACT_SHA=\$(sha256sum "\$CONTRACT" | cut -d' ' -f1)
      RESOLVED_SHA=\$(sha256sum "\$RESOLVED" | cut -d' ' -f1)
      "\${PYRUN[@]}" '${params.repo}/bin/r02_neural_structure_run.py' \
        --contract "\$CONTRACT" --contract-sha256 "\$CONTRACT_SHA" \
        --resolved "\$RESOLVED" --resolved-sha256 "\$RESOLVED_SHA" \
        --arm '${arm}' --seed 1701 --output '${params.run_root}/training/${arm}' \
        --checkpoint-dir '${params.run_root}/checkpoints/${arm}' \
        --adam-state-precision '${params.adam_state_precision}' \
        --measurement '${arm}.measurement.json' --recover-owned-run ${recovery}
      cp '${arm}.measurement.json' '${params.run_root}/${arm}.measurement.json'
    fi
    """
}

process VERIFY_INSTRUMENTATION {
    tag 'three-arm-gate'
    cpus 1
    memory '1 GB'
    time '30m'
    cache 'deep'

    input:
    path measurements

    output:
    path 'INSTRUMENTATION_COMPLETE.json'

    script:
    """
    python3 - <<'PY'
import glob,json
rows=[json.load(open(p)) for p in glob.glob('*.measurement.json')]
assert len(rows)==3 and {x['arm'] for x in rows}=={'NONE','COUNTS','RARE'}
assert all(x['status']=='COMPLETE' and x['final_confirmed_step']==2 for x in rows)
assert all(x['adam_state_precision']=='${params.adam_state_precision}' for x in rows)
rare=next(x for x in rows if x['arm']=='RARE')
assert rare['forced_interruption_after_step']==1 and rare['restored_steps']==1
json.dump({'schema':'r02_neural_structure_instrumentation_gate_v1','status':'PASS_ENGINEERING_ONLY',
           'evaluate_opened':False,'biological_evidence_obtained':False,'measurements':rows},
          open('INSTRUMENTATION_COMPLETE.json','w'),sort_keys=True,indent=2)
PY
    cp INSTRUMENTATION_COMPLETE.json '${params.run_root}/INSTRUMENTATION_COMPLETE.json'
    """
}

workflow {
    packet_file = file(required('packet', params.packet), checkIfExists: true)
    index_file = file(required('store_request_index', params.store_request_index), checkIfExists: true)
    required('repo', params.repo); required('run_root', params.run_root)
    required('gcloud_config', params.gcloud_config)
    required('prior_full22_gate', params.prior_full22_gate)
    required('prior_full22_gate_sha256', params.prior_full22_gate_sha256)
    required('training_container', params.training_container)
    required('target_root', params.target_root)
    required('eligibility_manifest', params.eligibility_manifest)
    required('eligibility_manifest_sha256', params.eligibility_manifest_sha256)
    if (!(params.adam_state_precision in ['fp32','fp64'])) error 'Invalid Adam state precision'
    index = new groovy.json.JsonSlurper().parse(index_file)
    if (index.schema != 'r02_neural_store_request_index_v1' || index.requests.size() != 22) error 'Exact FULL22 index required'
    rows = index.requests.sort { a,b -> a.chromosome.toInteger() <=> b.chromosome.toInteger() }.collect { row ->
        tuple(row.chromosome.toString(), file(row.request.path, checkIfExists:true), row.request.sha256,
              file(row.manifest.path, checkIfExists:true), row.manifest.sha256)
    }
    STAGE_AUTHENTICATED_STORE(Channel.fromList(rows))
    ASSEMBLE_ADMISSION(STAGE_AUTHENTICATED_STORE.out.receipts.map{ it[1] }.collect())
    tasks = ASSEMBLE_ADMISSION.out.marker.flatMap { m -> ['NONE','COUNTS','RARE'].collect { a -> tuple(a,m) } }
    TRAIN_INSTRUMENTATION_ARM(tasks)
    VERIFY_INSTRUMENTATION(TRAIN_INSTRUMENTATION_ARM.out.measurements.collect())
}
