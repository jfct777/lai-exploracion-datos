nextflow.enable.dsl=2

process LAI_REFERENCE_DETAIL {
    tag chromosome
    cpus 1
    memory params.task_memory
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    container params.container
    publishDir params.output_dir, mode: 'link', overwrite: false
    input:
    tuple val(chromosome), path(request), val(pin)
    path code
    output:
    path "chr${chromosome}"
    script:
    """
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    python3 r02_lai_reference_detail.py --request ${request} --request-sha256 ${pin} --out chr${chromosome}
    """
}

def sha(p) {
    def d=java.security.MessageDigest.getInstance('SHA-256')
    p.toFile().withInputStream { s -> new java.security.DigestInputStream(s,d).transferTo(java.io.OutputStream.nullOutputStream()) }
    d.digest().encodeHex().toString()
}

workflow {
    def plan=file(params.plan,checkIfExists:true)
    if(sha(plan)!=params.plan_sha256) error 'Plan changed'
    def p=new groovy.json.JsonSlurper().parse(plan.toFile())
    if(p.schema!='r02_lai_reference_detail_batch_v1' || p.training_allowed!=false ||
       p.tasks.collect{it.chromosome as Integer}.toSet()!=(1..22).toSet() || p.tasks.size()!=22)
        error 'All22 unique reference chromosomes required; no training'
    if(!(params.max_forks?.toString() ==~ /[1-2]/) || !(params.task_time ==~ /[1-9][0-9]*[mh]/) ||
       !(params.container ==~ /[^\s]+@sha256:[a-f0-9]{64}/)) error 'Bounded resources and pinned image required'
    def code=file(params.code,checkIfExists:true)
    if(sha(code)!=p.code_sha256) error 'Code changed'
    def jobs=p.tasks.collect { t ->
        def q=file(t.path,checkIfExists:true)
        if(sha(q)!=t.sha256) error 'Request changed'
        tuple(t.chromosome,q,t.sha256)
    }
    LAI_REFERENCE_DETAIL(Channel.fromList(jobs),code)
}
