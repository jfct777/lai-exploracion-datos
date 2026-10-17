nextflow.enable.dsl=2

// Source matrices are private immutable local paths pinned in request.json.
// The uncached auditor rereads byte hashes before a completed materialization
// may be reused. No training, community selection or cloud staging is implied.
process K03_REUSE_AUDIT {
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache false
    stageInMode 'copy'
    publishDir "${params.output_dir}/audit", mode:'copy', overwrite:false
    input:
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    path request, stageAs:'request.json'
    val pin
    output:
    path 'audit', emit: result
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_structure_discrete_reuse.py --bundle bundle --request request.json \
      --request-sha256 '${pin}' --stage audit --output-dir audit
    """
}

process K03_REUSE_MATERIALIZE {
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    stageInMode 'copy'
    publishDir "${params.output_dir}/affinities", mode:'copy', overwrite:false
    input:
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    path request, stageAs:'request.json'
    val pin
    val admitted
    val block
    output:
    path 'affinities', emit: result
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_structure_discrete_reuse.py --bundle bundle --request request.json \
      --request-sha256 '${pin}' --stage materialize --block-id '${block}' --output-dir affinities
    """
}

workflow {
    def safe = { p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.bundle, params.code_dir, params.request, params.output_dir].every(safe)) error 'Absolute safe paths required'
    if (!(params.request_sha256 ==~ /[0-9a-f]{64}/)) error 'Request SHA256 required'
    if (!(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)) error 'Positive memory_gb required'
    if (!(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Bounded task_time required'
    if (!(params.stage in ['audit','materialize'])) error 'Explicit audit or materialize stage required'
    if (params.stage == 'materialize' && !(params.block_id ==~ /[A-Za-z][A-Za-z0-9_]*/)) error 'Safe block_id required'
    def bundle = file(params.bundle,checkIfExists:true)
    def code = file(params.code_dir,checkIfExists:true)
    def request = file(params.request,checkIfExists:true)
    K03_REUSE_AUDIT(bundle, code, request, params.request_sha256)
    if (params.stage == 'materialize') {
        def admitted = K03_REUSE_AUDIT.out.result.map { dir ->
            def receipt = new groovy.json.JsonSlurper().parseText(file("${dir}/manifest.json").text)
            if (receipt.request_sha256 != params.request_sha256 || receipt.training_allowed != false)
                error 'Audit receipt changed'
            return receipt.request_sha256
        }
        K03_REUSE_MATERIALIZE(bundle, code, request, params.request_sha256, admitted, params.block_id)
    }
}
