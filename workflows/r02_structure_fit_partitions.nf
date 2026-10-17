nextflow.enable.dsl=2

process FIT_COMMUNITY {
    tag "fit:${candidateId}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(candidateId), path(request), val(pin), path(inputs, stageAs:'inputs/??/*'), val(entrypoint)
    path code
    output:
    path "candidate_${candidateId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl
    python3 ${entrypoint} --request ${request} --request-sha256 '${pin}' --output candidate_${candidateId}
    """
}

workflow {
    def safe={s -> s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') &&
        !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {s -> new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())}
        md.digest().encodeHex().toString()
    }
    def pinned={name,pin ->
        if(!safe(name) || !(pin ==~ /[0-9a-f]{64}/)) error 'Safe absolute FIT source and SHA256 required'
        def p=file(name,checkIfExists:true)
        if(hash(p)!=pin) error 'FIT source changed before cache lookup'
        p
    }
    if(![params.batch,params.code_dir,params.output_dir].every{safe(it)} ||
       !(params.batch_sha256 ==~ /[0-9a-f]{64}/) ||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
       !(params.max_forks?.toString() ==~ /[1-9][0-9]*/) ||
       !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/))
        error 'Explicit FIT batch, paths and operational limits required'
    def batch=pinned(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(!(b.schema in ['r02_fit_learning_batch_v1','r02_fit_profile_batch_v2']) || !b.candidates ||
       b.candidates.collect{it.id}.toSet().size()!=b.candidates.size()) error 'Explicit unique FIT candidates required'
    def profile=b.schema=='r02_fit_profile_batch_v2'
    def requestSchema=profile ? 'r02_fit_profile_request_v2' : 'r02_fit_learning_request_v1'
    def entrypoint=profile ? 'r02_resolution_profile.py' : 'r02_structure_fit_partitions.py'
    def code=b.implementation_sha256.collect {name,pin ->
        if(!(name ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe FIT source filename'
        pinned(params.code_dir+'/'+name,pin)
    }
    def tasks=b.candidates.collect {entry ->
        if(!(entry.id ==~ /[a-zA-Z0-9_-]+/)) error 'Safe FIT candidate ID required'
        def request=pinned(entry.request,entry.sha256)
        def r=new groovy.json.JsonSlurper().parse(request.toFile())
        if(r.schema!=requestSchema || r.implementation_sha256!=b.implementation_sha256)
            error 'FIT request/current code differs'
        def inputs=r.inputs.collect {name,record -> pinned(record.path,record.sha256)}
        tuple(entry.id,request,entry.sha256,inputs,entrypoint)
    }
    FIT_COMMUNITY(Channel.fromList(tasks),code)
}
