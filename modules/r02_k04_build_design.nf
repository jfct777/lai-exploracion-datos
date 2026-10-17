process K04_BUILD_DESIGN {
    tag "K04-build:${task_id}"
    cpus params.build_cpus
    memory "${params.build_memory_gb} GB"
    time params.build_time
    maxForks params.build_max_forks
    cache 'deep'
    container params.container

    input:
    tuple val(task_id), path(request), val(request_sha256), path(inputs, stageAs: 'input/??/*'),
          path(bridge_code, stageAs: 'code/*')

    output:
    tuple val(task_id), path("model_inputs_${task_id}")

    script:
    """
    set -euo pipefail
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code
    export OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    python3 code/r02_k04_model_bridge.py --request '${request}' \
      --request-sha256 '${request_sha256}' --staged-root input --output 'model_inputs_${task_id}'
    """

    stub:
    """
    mkdir 'model_inputs_${task_id}'
    printf '%s\n' '{"schema":"r02_k04_build_stub_v1","scientific_result":false}' > 'model_inputs_${task_id}/manifest.json'
    """
}
