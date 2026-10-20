nextflow.enable.dsl=2

process K03_SOURCE_MINOR_CHROMOSOME {
    tag id
    cpus 1
    memory '2 GB'
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    container params.container
    input:
    tuple val(id), val(chromosome), path(campaign)
    path code
    output:
    path "${id}"
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_k03_event_alt_chunks.py chromosome --campaign ${campaign} --chromosome ${chromosome} --output ${id}
    """
    stub:
    """
    mkdir ${id}
    touch ${id}/STUB_NOT_HUMAN_RESULT
    """
}

process K03_ASSEMBLE_ALL_FIXED_VIEWS {
    cpus 1
    memory '2 GB'
    time params.assembly_time
    cache 'deep'
    container params.container
    publishDir params.output_dir, mode: 'link', overwrite: false
    input:
    path plan
    path code
    path products
    output:
    path 'views'
    script:
    def productArgs = products.collect { "--product ${it}" }.join(' ')
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_k03_event_view_plan.py assemble --plan ${plan} ${productArgs} --output views
    """
    stub:
    """
    mkdir views
    touch views/STUB_NOT_HUMAN_RESULT
    """
}

def hash(p) {
    def digest=java.security.MessageDigest.getInstance('SHA-256')
    p.toFile().withInputStream { stream ->
        new java.security.DigestInputStream(stream,digest).transferTo(java.io.OutputStream.nullOutputStream())
    }
    digest.digest().encodeHex().toString()
}

workflow {
    if (!(params.container ==~ /[^\s]+@sha256:[a-f0-9]{64}/) ||
        !(params.max_forks?.toString() ==~ /[1-4]/) ||
        !(params.task_time ==~ /[1-9][0-9]*[smh]/) ||
        !(params.assembly_time ==~ /[1-9][0-9]*[smh]/))
        error 'Pinned image and explicit bounded resources required'
    def plan=file(params.plan,checkIfExists:true)
    if(hash(plan)!=params.plan_sha256) error 'All-view plan changed'
    def cfg=new groovy.json.JsonSlurper().parse(plan.toFile())
    if(cfg.schema!='r02_k03_fixed_source_all_views_v1' || cfg.views.size()!=88 ||
       cfg.scientific_models_admitted!=false || cfg.jobs.collect { it.id }.toSet().size()!=cfg.jobs.size())
        error 'Complete distinct all-view input plan required; models remain closed'
    def dependencies=cfg.deployment_sha256
    if(!(dependencies instanceof Map) || !dependencies ||
       dependencies['r02_k03_event_view_plan.py']!=cfg.planner_sha256)
        error 'Complete transitive Python deployment required'
    def code=dependencies.collect { name,pin ->
        if(!(name ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe dependency name'
        def p=file(params.code_dir+'/'+name,checkIfExists:true)
        if(hash(p)!=pin) error 'Frozen implementation changed'
        p
    }
    def jobs=cfg.jobs.collect { j ->
        if(!(j.id ==~ /chr[0-9]{2}_[a-f0-9]{16}/) || !(j.chromosome in 1..22)) error 'Invalid chromosome task'
        def p=file(j.campaign,checkIfExists:true)
        if(hash(p)!=j.campaign_sha256) error 'Source campaign changed'
        tuple(j.id,j.chromosome,p)
    }
    K03_SOURCE_MINOR_CHROMOSOME(Channel.fromList(jobs),code)
    K03_ASSEMBLE_ALL_FIXED_VIEWS(plan,code,K03_SOURCE_MINOR_CHROMOSOME.out.collect())
}
