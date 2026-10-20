nextflow.enable.dsl=2

process K03_COMPLETE_BASE {
    tag id
    cpus 1
    memory '2 GB'
    time params.base_time
    maxForks params.max_forks
    cache 'deep'
    container params.container
    input:
    tuple val(chromosome), val(id), path(request)
    path code
    output:
    tuple val(chromosome), path("${id}")
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_k03_complete_views.py produce --request ${request} --output ${id}
    """
    stub:
    """
    mkdir ${id}
    touch ${id}/STUB_NOT_HUMAN_RESULT
    """
}

process K03_COMPLETE_EXCLUSION {
    tag id
    cpus 1
    memory '2 GB'
    time params.derive_time
    maxForks 2
    cache 'deep'
    container params.container
    input:
    tuple val(chromosome), val(id), path(request), path(base, stageAs:'base/*')
    path code
    output:
    path "${id}"
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_k03_complete_views.py produce --request ${request} --base ${base} --output ${id}
    """
    stub:
    """
    mkdir ${id}
    touch ${id}/STUB_NOT_HUMAN_RESULT
    """
}

process K03_COMPLETE_ASSEMBLY {
    cpus 1
    memory '2 GB'
    time params.assembly_time
    cache 'deep'
    container params.container
    publishDir params.output_dir, mode:'link', overwrite:false
    input:
    path plan
    path code
    path products
    output:
    path 'views'
    script:
    def args=products.collect { "--product ${it}" }.join(' ')
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_k03_complete_views.py assemble --plan ${plan} ${args} --output views --storage-root ${params.storage_root} --storage-alias ${params.storage_alias}
    """
    stub:
    """
    mkdir views
    touch views/STUB_NOT_HUMAN_RESULT
    """
}

def sha(p) {
    def h=java.security.MessageDigest.getInstance('SHA-256')
    p.toFile().withInputStream { s -> new java.security.DigestInputStream(s,h).transferTo(java.io.OutputStream.nullOutputStream()) }
    h.digest().encodeHex().toString()
}

workflow {
    def admission=file(params.admission,checkIfExists:true)
    if(sha(admission)!=params.admission_sha256) error 'Continuation inventory changed'
    def a=new groovy.json.JsonSlurper().parse(admission.toFile())
    if(a.original_writers_closed!=true || a.scientific_models_admitted!=false || a.bases.size()!=22 || a.jobs.size()!=88)
        error 'Closed original execution and all planned tasks required'
    if(!(params.container ==~ /[^\s]+@sha256:[a-f0-9]{64}/) || !(params.max_forks.toString() ==~ /[1-4]/))
        error 'Pinned image and bounded concurrency required'
    def code=a.code.collect { name,pin ->
        if(!(name ==~ /r02_[a-z0-9_]+\.py/)) error 'Invalid code name'
        def p=file(params.code_dir+'/'+name,checkIfExists:true)
        if(sha(p)!=pin) error 'Frozen code changed'
        p
    }
    def jobs=a.jobs.collect { j ->
        def p=file(j.request,checkIfExists:true)
        if(sha(p)!=j.sha256 || !(j.id ==~ /chr[0-9]{2}_[a-f0-9]{16}/) || !(j.chromosome in 1..22)) error 'Invalid job'
        tuple(j.chromosome,j.id,p)
    }
    def plan=file(a.plan,checkIfExists:true)
    if(sha(plan)!=a.plan_sha256) error 'Original all-view plan changed'
    K03_COMPLETE_BASE(Channel.fromList(jobs.findAll { a.bases.contains(it[1]) }),code)
    K03_COMPLETE_EXCLUSION(Channel.fromList(jobs).combine(K03_COMPLETE_BASE.out,by:0),code)
    K03_COMPLETE_ASSEMBLY(plan,code,K03_COMPLETE_EXCLUSION.out.collect())
}
