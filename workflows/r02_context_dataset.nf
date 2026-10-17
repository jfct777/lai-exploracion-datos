#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// One explicitly declared bounded view. No cohort split, target or window choice.
// Absolute authenticated input paths require a local/shared admitted filesystem.
params.request = null
params.request_sha256 = null
params.reserved_admission = ''
params.reserved_admission_sha256 = ''
params.outdir = null
params.code_dir = new File("${projectDir}/../bin").canonicalPath
params.python = 'python3'
params.python_sha256 = ''
params.task_memory = '1 GB'
params.task_time = '30m'

process VERIFY_CONTEXT_DATASET {
    stageInMode 'copy'
    cache false
    cpus 1
    memory params.task_memory
    time params.task_time
    input:
    path request
    val digest
    val admission
    val admission_sha
    path sources
    output:
    path 'verified.json'
    script:
    def extra = admission ? "--reserved-admission '${admission}' --reserved-admission-sha256 '${admission_sha}'" : ''
    """
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    '${params.python}' r02_context_dataset.py --verify-only \\
      --request '${request}' --request-sha256 '${digest}' --output verified.json ${extra}
    """
}

process PREPARE_CONTEXT_DATASET {
    stageInMode 'copy'
    cache 'deep'
    cpus 1
    memory params.task_memory
    time params.task_time
    publishDir params.outdir, mode: 'copy', overwrite: false
    input:
    path request
    val digest
    val admission
    val admission_sha
    path verified
    path sources
    output:
    path 'dataset'
    script:
    def extra = admission ? "--reserved-admission '${admission}' --reserved-admission-sha256 '${admission_sha}'" : ''
    """
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    '${params.python}' r02_context_dataset.py \\
      --request '${request}' --request-sha256 '${digest}' --output dataset --verified '${verified}' ${extra}
    """
}

workflow {
    ['request','request_sha256','outdir'].each { if (!params[it]) error "Missing --${it}" }
    def safe = {p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if (![params.request,params.outdir,params.code_dir].every {safe.call(it)}) error 'Absolute safe paths required'
    if (!(params.request_sha256 ==~ /[0-9a-f]{64}/)) error 'Request SHA256 required'
    if (params.reserved_admission && (!safe.call(params.reserved_admission) || !(params.reserved_admission_sha256 ==~ /[0-9a-f]{64}/)))
        error 'Reserved admission requires an absolute safe path and SHA256'
    def hash = {p -> java.security.MessageDigest.getInstance('SHA-256').digest(p.toFile().bytes).encodeHex().toString()}
    if (!(params.python in ['python3','python'])) {
        if (!safe.call(params.python) || !(params.python_sha256 ==~ /[0-9a-f]{64}/)) error 'Python wrapper must be an absolute executable pinned by SHA256, not a shell command'
        def executable=file(params.python,checkIfExists:true)
        if (!executable.toFile().canExecute() || hash.call(executable)!=params.python_sha256) error 'Python executable differs'
    }
    if (!!params.reserved_admission != !!params.reserved_admission_sha256)
        error 'Both reserved admission path and SHA256 are required together'
    def request = Channel.value(file(params.request, checkIfExists:true))
    def sources = Channel.value(['r02_context_dataset.py','r02_genomic_event_inputs.py','r02_genomic_event_cache.py',
        'r02_structure_materialize.py','r02_structure_increment.py','r02_structure_input_plan.py',
        'r02_genomic_pair_evidence.py','r02_structure_regions.py'].collect {
            file("${params.code_dir}/${it}",checkIfExists:true) })
    VERIFY_CONTEXT_DATASET(request,params.request_sha256,params.reserved_admission,params.reserved_admission_sha256,sources)
    PREPARE_CONTEXT_DATASET(request,params.request_sha256,params.reserved_admission,params.reserved_admission_sha256,
                            VERIFY_CONTEXT_DATASET.out,sources)
}
