nextflow.enable.dsl=2

params.batch = null
params.batch_sha256 = null
params.code_dir = null
params.output_dir = null
params.container = null
params.task_cpus = null
params.memory_gb = null
params.task_time = null
params.max_forks = null

process K04_MODEL_INPUTS {
    tag "K04-model-inputs:${task_id}"
    cpus params.task_cpus
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    container params.container
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple val(task_id), path(request), val(request_sha256),
          path(inputs, stageAs: 'input/??/*'), path(code, stageAs: 'code/*')

    output:
    tuple val(task_id), path("model_inputs_${task_id}")

    script:
    """
    set -euo pipefail
    umask 077
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH=code
    export OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    python3 code/r02_k04_model_bridge.py \
      --request '${request}' --request-sha256 '${request_sha256}' \
      --staged-root input --output 'model_inputs_${task_id}'
    """

    stub:
    """
    mkdir 'model_inputs_${task_id}'
    printf '%s\n' '{"schema":"r02_k04_ccqcqr_stub_v1","scientific_result":false}' \
      > 'model_inputs_${task_id}/manifest.json'
    """
}

def safePath(value) {
    value instanceof String && value.startsWith('/') &&
        !value.tokenize('/').contains('..') &&
        !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

def sha256(path) {
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream { stream ->
        new java.security.DigestInputStream(stream, digest).transferTo(java.io.OutputStream.nullOutputStream())
    }
    digest.digest().encodeHex().toString()
}

def pinned(path, expected) {
    if (!safePath(path) || !(expected ==~ /[0-9a-f]{64}/))
        error 'Safe absolute path and lowercase SHA256 required'
    def value = file(path, checkIfExists: true)
    if (!value.toFile().isFile() ||
        value.toFile().toPath().toRealPath() != value.toFile().toPath().toAbsolutePath().normalize())
        error 'Pinned K04 input must be a canonical regular file'
    if (sha256(value) != expected) error "K04 input differs before cache lookup: ${path}"
    value
}

workflow {
    if (![params.batch, params.code_dir, params.output_dir].every { safePath(it) } ||
        !(params.batch_sha256 ==~ /[0-9a-f]{64}/) ||
        !(params.container instanceof String) || !params.container.contains('@sha256:') ||
        ![params.task_cpus, params.memory_gb, params.max_forks].every {
            it?.toString() ==~ /[1-9][0-9]*/
        } || !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/))
        error 'Authenticated batch/container and explicit positive resource limits required'

    def batch = pinned(params.batch, params.batch_sha256)
    def value = new groovy.json.JsonSlurper().parse(batch.toFile())
    def expectedCode = ['r02_k04_model_bridge.py', 'r02_k04_predictor_inputs.py',
                        'r02_pair_operator.py', 'r02_interval_ridge.py'] as Set
    if (value.schema != 'r02_k04_ccqcqr_batch_v1' ||
        !(value.tasks instanceof List) || value.tasks.isEmpty() ||
        !(value.implementation_sha256 instanceof Map) ||
        value.implementation_sha256.keySet() as Set != expectedCode)
        error 'Nonempty reviewed K04 batch and exact implementation inventory required'
    def code = value.implementation_sha256.collect { name, digest ->
        pinned(params.code_dir + '/' + name, digest)
    }
    def seen = [] as Set
    def jobs = value.tasks.collect { task ->
        if (!(task instanceof Map) || task.keySet() as Set !=
            ['id', 'request', 'request_sha256', 'inputs'] as Set ||
            !(task.id ==~ /[A-Za-z0-9_]+/) || !seen.add(task.id) ||
            !(task.inputs instanceof List) || task.inputs.isEmpty())
            error 'Unique safe task and explicit input inventory required'
        def request = pinned(task.request, task.request_sha256)
        def specification = new groovy.json.JsonSlurper().parse(request.toFile())
        if (specification.schema != 'r02_k04_ccqcqr_request_v1' ||
            specification.operation != 'MATERIALIZE_REVIEWED_C_CQ_CQR_FIT_SELECT' ||
            specification.implementation_sha256 != value.implementation_sha256 ||
            specification.targets.keySet() as Set != ['FIT', 'SELECT'] as Set)
            error 'Reviewed request schema/code/FIT-SELECT roles differ'
        def staged = task.inputs.collect { descriptor ->
            if (!(descriptor instanceof Map) || descriptor.keySet() as Set != ['path', 'sha256'] as Set)
                error 'Exact staged input descriptor required'
            pinned(descriptor.path, descriptor.sha256)
        }
        tuple(task.id, request, task.request_sha256, staged, code)
    }
    K04_MODEL_INPUTS(channel.fromList(jobs))
}
