nextflow.enable.dsl=2

process COMMON_PARTIAL {
    tag "common:${taskId}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/partials", mode:'copy', overwrite:false
    input:
    tuple val(taskId), path(request), val(pin), path(inputs, stageAs:'inputs/??/*')
    path metadata, stageAs:'metadata/*'
    val bundle
    path code
    output:
    path "common_${taskId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_common_inputs.py --bundle '${bundle}' --request ${request} \
      --request-sha256 '${pin}' --stage partial --output common_${taskId}
    """
}

process COMMON_PLAN_MERGES {
    tag 'freeze_common_target_membership'
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir "${params.output_dir}/merge-plan", mode:'copy', overwrite:false
    input:
    path partials, stageAs:'partials/*'
    path requests, stageAs:'requests/*'
    path metadata, stageAs:'metadata/*'
    val bundle
    path batch
    val pin
    path code
    output:
    path 'merge_plan'
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_common_inputs.py --bundle '${bundle}' --request ${batch} \
      --request-sha256 '${pin}' --stage plan-merges --partials partials/* --output merge_plan
    """
}

process COMMON_TARGET {
    tag "common_target:${targetId}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/targets", mode:'copy', overwrite:false
    input:
    tuple val(targetId), path(request), val(pin), path(partials, stageAs:'partials/*')
    path requests, stageAs:'requests/*'
    path metadata, stageAs:'metadata/*'
    val bundle
    path code
    output:
    path "target_${targetId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_common_inputs.py --bundle '${bundle}' --request ${request} \
      --request-sha256 '${pin}' --stage merge --output target_${targetId}
    """
}

workflow {
    def safe={s -> s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') && !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {s -> new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())}
        md.digest().encodeHex().toString()
    }
    def pinned={name,pin ->
        if(!safe(name) || !(pin ==~ /[0-9a-f]{64}/)) error 'Safe absolute common source and SHA256 required'
        def p=file(name,checkIfExists:true)
        if(hash(p)!=pin) error 'Common source changed before cache lookup'
        p
    }
    if(![params.batch,params.code_dir,params.output_dir].every{safe(it)} || !(params.batch_sha256 ==~ /[0-9a-f]{64}/) ||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/) || !(params.max_forks?.toString() ==~ /[1-9][0-9]*/) ||
       !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit pinned common batch and bounded resources required'
    def batch=pinned(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(b.schema!='r02_structure_materialized_common_batch_v2') error 'Unsupported common batch'
    def material=pinned(b.bundle.path+'/materialize.json',b.bundle.sha256)
    def source=new groovy.json.JsonSlurper().parse(material.toFile())
    def metadata=[material]+source.metadata_sha256.collect {n,h -> pinned(b.bundle.path+'/'+n,h)}
    def code=b.implementation_sha256.collect {n,h ->
        if(!(n ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe common consumer source name'
        pinned(params.code_dir+'/'+n,h)
    }
    def requests=[]
    def tasks=b.partials.collect {d ->
        def request=pinned(d.request,d.request_sha256);requests.add(request)
        def r=new groovy.json.JsonSlurper().parse(request.toFile())
        if(r.schema!='r02_structure_materialized_common_v2' || r.implementation_sha256!=b.implementation_sha256 ||
           r.source_contract_sha256!=b.bundle.sha256 || !(r.task_id ==~ /features_chr[0-9]+_[a-f0-9]{16}/)) error 'Partial source/code/task differs'
        def selection=pinned(r.selection.directory+'/manifest.json',r.selection.manifest_sha256)
        def s=new groovy.json.JsonSlurper().parse(selection.toFile())
        def markers=pinned(r.selection.directory+'/common_markers.tsv',s.outputs_sha256['common_markers.tsv'])
        def audit=pinned(r.selection.directory+'/selection.json',s.outputs_sha256['selection.json'])
        def events=pinned(r.events.directory+'/manifest.json',r.events.manifest_sha256)
        def e=new groovy.json.JsonSlurper().parse(events.toFile())
        def positions=markers.readLines().drop(1).collect {it.split('\t')[2].toLong()}
        def chunks=e.channels.common.findAll {c -> positions.any{it>=c.start_bp && it<=c.end_bp}}.collect {c ->
            if(c.path.startsWith('/') || c.path.tokenize('/').contains('..')) error 'Unsafe common chunk path'
            pinned(r.events.directory+'/'+c.path,c.sha256)
        }
        def supplement=[]
        if(r.supplement!=null) {
            def manifest=pinned(r.supplement.directory+'/manifest.json',r.supplement.manifest_sha256)
            def sm=new groovy.json.JsonSlurper().parse(manifest.toFile())
            if(sm.array.path!='supplement.npz') error 'Unsafe common supplement filename'
            supplement=[manifest,pinned(r.supplement.directory+'/'+sm.array.path,sm.array.sha256)]
        }
        tuple(r.task_id,request,d.request_sha256,[selection,markers,audit,events]+chunks+supplement)
    }
    if(tasks.collect{it[0]}.toSet().size()!=tasks.size() || tasks.isEmpty()) error 'Empty/duplicate common partials'
    COMMON_PARTIAL(Channel.fromList(tasks),metadata,b.bundle.path,code)
    COMMON_PLAN_MERGES(COMMON_PARTIAL.out.collect(),requests,metadata,b.bundle.path,batch,params.batch_sha256,code)
    def targets=COMMON_PLAN_MERGES.out.flatMap {dir ->
        def m=new groovy.json.JsonSlurper().parse(dir.resolve('manifest.json').toFile())
        if(m.batch_sha256!=params.batch_sha256 || m.arrays_read!=false) error 'Common merge plan provenance differs'
        m.targets.collect {t -> tuple(t.name,pinned(t.request,t.request_sha256),t.request_sha256,t.directories.collect{file(it,checkIfExists:true)})}
    }
    COMMON_TARGET(targets,requests,metadata,b.bundle.path,code)
}
