#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Reserved scoring is independent of Optuna/train; never selects a model or refits.
params.plan = null
params.plan_sha256 = null
params.outdir = null
params.code_dir = new File("${projectDir}/../bin").canonicalPath
params.python = 'python3'
params.python_sha256 = ''
params.task_cpus = 1
params.task_memory = '1 GB'
params.task_time = '30m'

process VERIFY_RESERVED_CONTEXT {
    stageInMode 'copy'
    cache false
    cpus 1
    memory params.task_memory
    time params.task_time
    input:
    path plan
    val digest
    path sources
    output:
    path 'verified.json'
    script:
    """
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    '${params.python}' r02_context_train.py --stage verify-evaluation \\
      --evaluation-plan '${plan}' --evaluation-plan-sha256 '${digest}' --output verified.json
    """
}

process SCORE_RESERVED_CONTEXT {
    stageInMode 'copy'
    cache 'deep'
    cpus params.task_cpus
    memory params.task_memory
    time params.task_time
    publishDir params.outdir, mode: 'copy', overwrite: false
    input:
    path plan
    val digest
    path verified
    path sources
    output:
    path 'evaluation'
    script:
    """
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    '${params.python}' r02_context_train.py --stage evaluate \\
      --evaluation-plan '${plan}' --evaluation-plan-sha256 '${digest}' --output evaluation
    """
}

workflow {
    ['plan','plan_sha256','outdir'].each { if (!params[it]) error "Missing --${it}" }
    def safe = {p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if (![params.plan,params.outdir,params.code_dir].every {safe.call(it)}) error 'Absolute safe paths required'
    if (!(params.plan_sha256 ==~ /[0-9a-f]{64}/)) error 'Plan SHA256 required'
    def hash = {p -> java.security.MessageDigest.getInstance('SHA-256').digest(p.toFile().bytes).encodeHex().toString()}
    if (!(params.python in ['python3','python'])) {
        if (!safe.call(params.python) || !(params.python_sha256 ==~ /[0-9a-f]{64}/)) error 'Python wrapper must be an absolute executable pinned by SHA256, not a shell command'
        def executable=file(params.python,checkIfExists:true)
        if (!executable.toFile().canExecute() || hash.call(executable)!=params.python_sha256) error 'Python executable differs'
    }
    if ((params.task_cpus as int) < 1 || (params.task_cpus as int) > 8)
        error 'task_cpus must be in 1..8; match the frozen CPU thread budget'
    // Only read resource metadata here. VERIFY authenticates both JSON files
    // before SCORE can consume its gate or access cached scientific outputs.
    def document = new groovy.json.JsonSlurper().parse(file(params.plan,checkIfExists:true).toFile())
    if (!safe.call(document.development_contract.path) || !(document.development_contract.sha256 ==~ /[0-9a-f]{64}/))
        error 'Development descriptor requires an absolute safe path and SHA256'
    def development = new groovy.json.JsonSlurper().parse(file(document.development_contract.path,checkIfExists:true).toFile())
    if ((development.threads as int) > (params.task_cpus as int))
        error 'Frozen Torch threads exceed task_cpus; never oversubscribe silently'
    def plan = Channel.value(file(params.plan, checkIfExists:true))
    def sources = Channel.value(['r02_context_train.py','genomic_context_runtime.py'].collect {
        file("${params.code_dir}/${it}",checkIfExists:true) })
    VERIFY_RESERVED_CONTEXT(plan,params.plan_sha256,sources)
    SCORE_RESERVED_CONTEXT(plan,params.plan_sha256,VERIFY_RESERVED_CONTEXT.out,sources)
}
