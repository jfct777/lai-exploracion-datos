nextflow.enable.dsl=2

process COMMON_TARGET {
    tag "K04:${chromosome}"
    cpus params.task_cpus
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(chromosome), path(request), val(pin), path(sources, stageAs:'sources/??/*')
    path code
    output:
    path "chr${chromosome}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    python3 r02_k04_common_target.py --request '${request}' --request-sha256 '${pin}' --output chr${chromosome} --scratch .
    """
}

def safe(s) {
    s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') && !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}
def hash(p) {
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {s -> new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())}
        md.digest().encodeHex().toString()
}
def pinned(name,pin) {
        if(!safe(name) || !(pin ==~ /[0-9a-f]{64}/)) error 'Safe absolute source/SHA256 required'
        def p=file(name,checkIfExists:true)
        if(hash(p)!=pin) error "K04 source differs before cache: ${p}"
        p
}
workflow {
    if(![params.batch,params.code_dir,params.output_dir].every{safe(it)} ||
       ![params.task_cpus,params.memory_gb,params.max_forks].every{it?.toString() ==~ /[1-9][0-9]*/} ||
       !(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit K04 inputs/resources required'
    def batch=pinned(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(b.schema!='r02_k04_common_target_batch_v1' || !b.tasks) error 'K04 batch schema differs'
    def code=b.implementation_sha256.collect{name,digest -> pinned("${params.code_dir}/${name}",digest)}
    def tasks=b.tasks.collect {r ->
        def request=pinned(r.request,r.sha256)
        def sources=r.sources.collect{v -> pinned(v.path,v.sha256)}
        tuple(r.chromosome,request,r.sha256,sources)
    }
    COMMON_TARGET(Channel.fromList(tasks),code)
}
