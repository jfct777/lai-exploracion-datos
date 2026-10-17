nextflow.enable.dsl=2

def safePath(p) {
    return p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

process R02_EVENT_SOURCE {
    tag "chr${chromosome}"
    cpus 2
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    input:
    val chromosome
    path bundle
    path code, stageAs:'code'
    val contractHash
    output:
    tuple val(chromosome), path('verified'), emit: checked
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_genomic_event_inputs.py --bundle ${bundle} --contract-sha256 ${contractHash} \
      --chromosome ${chromosome} --stage verify-source --output-dir verified
    """
}

process R02_EVENT_EXTRACT {
    tag "chr${chromosome}"
    cpus 2
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:params.publish_mode ?: 'copy', overwrite:false
    input:
    tuple val(chromosome), path(verified)
    path bundle
    path code, stageAs:'code'
    val contractHash
    output:
    tuple val(chromosome), path("chr${chromosome}"), emit: individual_inputs
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_genomic_event_inputs.py --bundle ${bundle} --contract-sha256 ${contractHash} \
      --chromosome ${chromosome} --verified ${verified}/manifest.json \
      --stage extract --output-dir chr${chromosome}
    """
}

workflow {
    if (params.publish_mode && !(params.publish_mode in ['copy','link'])) error 'Only copy or local hard-link publication allowed'
    if (!safePath(params.bundle) || !safePath(params.code_dir) || !safePath(params.output_dir)) error 'Absolute safe paths required'
    if (!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Explicit event contract SHA256 required'
    if (!(params.memory_gb?.toString() ==~ /[1-9][0-9]*/) || !(params.max_forks?.toString() ==~ /[1-9][0-9]*/)) error 'Explicit positive memory_gb/max_forks required'
    if (!(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smhd]/)) error 'Explicit bounded task time required'
    def chromosomes = params.chromosomes?.toString()?.tokenize(',')
    if (!chromosomes || chromosomes.toSet().size()!=chromosomes.size() || !chromosomes.every { it ==~ /([1-9]|1[0-9]|2[0-2])/ }) error 'Unique explicit autosomal chromosome list required'
    def bundle = file(params.bundle,checkIfExists:true)
    def code = file(params.code_dir,checkIfExists:true)
    R02_EVENT_SOURCE(Channel.fromList(chromosomes),bundle,code,params.contract_sha256)
    R02_EVENT_EXTRACT(R02_EVENT_SOURCE.out.checked,bundle,code,params.contract_sha256)
}
