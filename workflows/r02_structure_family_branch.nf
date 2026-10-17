nextflow.enable.dsl=2

def safeAbsolute(value) {
    value instanceof String && value.startsWith('/') && !value.tokenize('/').contains('..') &&
        !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}
def fileHash(path) {
    def md=java.security.MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream {s -> new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())}
    md.digest().encodeHex().toString()
}
def pinned(name,digest) {
    if(!safeAbsolute(name)||!(digest==~/[0-9a-f]{64}/))error 'Safe absolute family source and SHA256 required'
    def path=file(name,checkIfExists:true);if(fileHash(path)!=digest)error 'Family source changed before cache lookup';path
}

process MATERIALIZE_C {
    tag taskId; cpus 1; memory "${params.memory_gb} GB"; time params.task_time
    maxForks params.max_forks; cache 'deep'; publishDir params.output_dir,mode:'copy',overwrite:false
    input:
    tuple val(taskId),path(request,stageAs:'request.json'),val(requestPin),path(sources,stageAs:'sources/??/*')
    path code
    output:
    path "affinity_${taskId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_common_affinities.py --request ${request} --request-sha256 '${requestPin}' --output affinity_${taskId}
    """
}

process FREEZE_RC {
    tag taskId; cpus 1; memory "${params.memory_gb} GB"; time params.task_time
    maxForks params.max_forks; cache 'deep'; publishDir params.output_dir,mode:'copy',overwrite:false
    input:
    tuple val(taskId),path(requestSet,stageAs:'request-set.json'),val(requestSetPin),path(sources,stageAs:'sources/??/*')
    path code
    output:
    path "composite_*"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_composite_affinities.py --request-set ${requestSet} --request-set-sha256 '${requestSetPin}' --output .
    """
}

workflow {
    if(!(params.family_stage in ['C','RC'])||
       ![params.batch,params.code_dir,params.output_dir].every{safeAbsolute(it)}||
       !(params.batch_sha256==~/[0-9a-f]{64}/)||!(params.memory_gb?.toString()==~/[1-9][0-9]*/)||
       !(params.max_forks?.toString()==~/[1-9][0-9]*/)||
       !(params.expected_tasks?.toString()==~/[1-9][0-9]*/)||!(params.task_time==~/[1-9][0-9]*[smh]/))
        error 'Explicit family stage, paths and operational limits required'
    def batch=pinned(params.batch,params.batch_sha256);def body=new groovy.json.JsonSlurper().parse(batch.toFile())
    def expected=params.family_stage=='C'?'r02_structure_common_family_batch_v1':'r02_structure_lazy_composite_batch_v1'
    if(body.schema!=expected||!body.tasks||body.tasks.size()!=params.expected_tasks.toInteger()||
       body.tasks.collect{it.id}.toSet().size()!=body.tasks.size())
        error 'Family batch schema/task inventory differs'
    def code=body.implementation_sha256.collect{name,digest ->
        if(!(name==~/r02_[a-z0-9_]+\.py/))error 'Unsafe family implementation filename'
        pinned(params.code_dir+'/'+name,digest)
    }
    def tasks=body.tasks.collect{entry ->
        if(!(entry.id==~/[A-Za-z0-9_-]+/))error 'Unsafe family task identifier'
        def request=params.family_stage=='C'?pinned(entry.request,entry.sha256):pinned(entry.request_set,entry.sha256)
        def r=new groovy.json.JsonSlurper().parse(request.toFile());def sources=[]
        if(params.family_stage=='C') {
            if(r.implementation_sha256!=body.implementation_sha256)error 'Family request/code differ'
            if(r.schema!='r02_structure_common_family_request_v1'||r.family!='C'||r.mix_weight!=null||r.rare!=null)
                error 'Exact C request required'
            if(r.resources.max_memory_bytes > params.memory_gb.toBigInteger()*1024L*1024L*1024L)
                error 'Declared C workspace exceeds allocated task memory'
            def manifest=pinned(r.common.directory+'/manifest.json',r.common.manifest_sha256);sources.add(manifest)
            def m=new groovy.json.JsonSlurper().parse(manifest.toFile())
            m.outputs_sha256.each{name,digest -> sources.add(pinned(r.common.directory+'/'+name,digest))}
            sources.add(pinned(r.source_contract.manifest.path,r.source_contract.manifest.sha256))
            sources.add(pinned(r.source_contract.persons.path,r.source_contract.persons.sha256))
        } else {
            if(r.schema!='r02_structure_lazy_composite_request_set_v1'||r.requests.size()!=3)
                error 'Exact grouped lazy R+C request set required'
            r.requests.each{descriptor ->
                def one=pinned(descriptor.path,descriptor.sha256);def q=new groovy.json.JsonSlurper().parse(one.toFile())
                sources.add(one)
                if(q.schema!='r02_structure_lazy_composite_request_v1'||q.implementation_sha256!=body.implementation_sha256||
                   q.normalization_policy!='COMPONENT_PAIR_EQUAL_POSITIVE_MEAN')error 'Exact lazy R+C request required'
                [q.common_inputs,q.rare_inputs].each{group -> group.each{name,d -> sources.add(pinned(d.path,d.sha256))}}
            }
        }
        tuple(entry.id,request,entry.sha256,sources)
    }
    if(params.family_stage=='C') MATERIALIZE_C(Channel.fromList(tasks),code)
    else FREEZE_RC(Channel.fromList(tasks),code)
}
