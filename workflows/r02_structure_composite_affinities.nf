nextflow.enable.dsl=2

def safeAbsolute(value) {
    value instanceof String && value.startsWith('/') && !value.tokenize('/').contains('..') &&
        !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

def fileHash(path) {
    def md=java.security.MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream { stream ->
        new java.security.DigestInputStream(stream,md).transferTo(java.io.OutputStream.nullOutputStream())
    }
    md.digest().encodeHex().toString()
}

def pinned(name,digest) {
    if(!safeAbsolute(name)||!(digest==~/[0-9a-f]{64}/)) error 'Safe absolute lazy-composite source and SHA256 required'
    def path=file(name,checkIfExists:true)
    if(fileHash(path)!=digest) error 'Lazy-composite source changed before cache lookup'
    path
}

process FREEZE_LAZY_COMPOSITE {
    tag taskId
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(taskId), path(request,stageAs:'request.json'), val(requestPin), path(sources,stageAs:'sources/??/*')
    path code
    output:
    path "composite_${taskId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_composite_affinities.py --request ${request} \
      --request-sha256 '${requestPin}' --output composite_${taskId}
    """
}

workflow {
    if(![params.batch,params.code_dir,params.output_dir].every{safeAbsolute(it)}||
       !(params.batch_sha256==~/[0-9a-f]{64}/)||!(params.memory_gb?.toString()==~/[1-9][0-9]*/)||
       !(params.max_forks?.toString()==~/[1-9][0-9]*/)||!(params.task_time==~/[1-9][0-9]*[smh]/))
        error 'Explicit lazy-composite paths/resources required'
    def batch=pinned(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(b.schema!='r02_structure_lazy_composite_batch_v1'||!b.tasks||b.tasks.collect{it.id}.toSet().size()!=b.tasks.size())
        error 'Explicit unique lazy-composite tasks required'
    def code=b.implementation_sha256.collect{name,pin ->
        if(!(name==~/r02_[a-z0-9_]+\.py/))error 'Unsafe implementation filename'
        pinned(params.code_dir+'/'+name,pin)
    }
    def tasks=b.tasks.collect{entry ->
        if(!(entry.id==~/[A-Za-z0-9_-]+/))error 'Unsafe lazy-composite task ID'
        def request=pinned(entry.request,entry.sha256);def r=new groovy.json.JsonSlurper().parse(request.toFile())
        if(r.schema!='r02_structure_lazy_composite_request_v1'||r.implementation_sha256!=b.implementation_sha256||
           r.normalization_policy!='COMPONENT_PAIR_EQUAL_POSITIVE_MEAN')error 'Lazy-composite request/code/policy differ'
        def sources=[]
        [r.common_inputs,r.rare_inputs].each{group -> group.each{name,d -> sources.add(pinned(d.path,d.sha256))}}
        tuple(entry.id,request,entry.sha256,sources)
    }
    FREEZE_LAZY_COMPOSITE(Channel.fromList(tasks),code)
}
