nextflow.enable.dsl=2

// Hash every used source even on -resume. Source directories are immutable,
// explicit local paths in the pinned request; this workflow downloads nothing.
process K03_COMMON_CONTINUOUS_AUDIT {
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
    python3 code/r02_structure_continuous.py --bundle bundle --request request.json \
      --request-sha256 '${pin}' --stage audit --output-dir audit
    """
}

process K03_COMMON_CONTINUOUS_FIT {
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    stageInMode 'copy'
    publishDir "${params.output_dir}/representation", mode:'copy', overwrite:false
    input:
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    path request, stageAs:'request.json'
    val pin
    val audited
    output:
    path 'continuous', emit: result
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_structure_continuous.py --bundle bundle --request request.json \
      --request-sha256 '${pin}' --stage fit --output-dir continuous
    """
}

workflow {
    def safe = { p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.bundle,params.code_dir,params.request,params.output_dir].every(safe)) error 'Absolute safe paths required'
    if (!(params.request_sha256 ==~ /[0-9a-f]{64}/)) error 'Request SHA256 required'
    if (!(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)) error 'Positive memory_gb required'
    if (!(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Bounded task time required'
    if (!(params.stage in ['audit','fit'])) error 'Explicit audit or fit stage required'
    def bundle=file(params.bundle,checkIfExists:true)
    def code=file(params.code_dir,checkIfExists:true)
    def request=file(params.request,checkIfExists:true)
    K03_COMMON_CONTINUOUS_AUDIT(bundle,code,request,params.request_sha256)
    if (params.stage=='fit') {
        def audited=K03_COMMON_CONTINUOUS_AUDIT.out.result.map { dir ->
            def m=new groovy.json.JsonSlurper().parseText(file("${dir}/manifest.json").text)
            if (m.stage!='AUDITED_NO_SCORES' || m.request_sha256!=params.request_sha256 || m.targets_read!=false)
                error 'Unexpected continuous audit receipt'
            return m.request_sha256
        }
        K03_COMMON_CONTINUOUS_FIT(bundle,code,request,params.request_sha256,audited)
    }
}
