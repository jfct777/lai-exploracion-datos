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
    if(!safeAbsolute(name) || !(digest ==~ /[0-9a-f]{64}/)) error 'Safe absolute path and SHA256 required'
    def path=file(name,checkIfExists:true)
    if(fileHash(path)!=digest) error 'Common-family source changed before cache lookup'
    path
}

process COMMON_FAMILY_AFFINITIES {
    tag "${family}:${taskId}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(taskId), val(family), path(request, stageAs:'request.json'), val(requestPin),
          path(sources, stageAs:'sources/??/*')
    path code
    output:
    path "affinity_${taskId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_common_affinities.py --request ${request} \
      --request-sha256 '${requestPin}' --output affinity_${taskId}
    """
}

workflow {
    if(![params.batch, params.code_dir, params.output_dir].every{safeAbsolute(it)} ||
       !(params.batch_sha256 ==~ /[0-9a-f]{64}/) ||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
       !(params.max_forks?.toString() ==~ /[1-9][0-9]*/) ||
       !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/))
        error 'Explicit common-family paths and operational limits required'
    def batch = pinned(params.batch, params.batch_sha256)
    def spec = new groovy.json.JsonSlurper().parse(batch.toFile())
    if(spec.schema != 'r02_structure_common_family_batch_v1' || !spec.tasks ||
       spec.tasks.collect{it.id}.toSet().size() != spec.tasks.size())
        error 'Explicit unique common-family tasks required'
    def code = spec.implementation_sha256.collect { name, digest ->
        if(!(name ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe common-family implementation filename'
        pinned(params.code_dir+'/'+name, digest)
    }
    def tasks = spec.tasks.collect { entry ->
        if(!(entry.id ==~ /[A-Za-z0-9_-]+/)) error 'Unsafe common-family task identifier'
        def request = pinned(entry.request, entry.sha256)
        def r = new groovy.json.JsonSlurper().parse(request.toFile())
        if(r.schema != 'r02_structure_common_family_request_v1' ||
           !(r.family in ['C','R_PLUS_C']) || r.implementation_sha256 != spec.implementation_sha256)
            error 'Common-family request/code differ'
        if(r.resources.max_memory_bytes > params.memory_gb.toBigInteger()*1024L*1024L*1024L)
            error 'Declared workspace exceeds allocated task memory'
        def sources = [
            pinned(r.common.directory+'/manifest.json', r.common.manifest_sha256),
            pinned(r.source_contract.manifest.path, r.source_contract.manifest.sha256),
            pinned(r.source_contract.persons.path, r.source_contract.persons.sha256)
        ]
        def commonManifest = new groovy.json.JsonSlurper().parse(sources[0].toFile())
        commonManifest.outputs_sha256.each { name, digest ->
            if(name != 'continuous.private.npz') error 'Unexpected common representation artifact'
            sources.add(pinned(r.common.directory+'/'+name, digest))
        }
        if(r.family == 'R_PLUS_C') {
            if(r.mix_weight == null || r.rare == null ||
               r.normalization_policy != 'COMPONENT_PAIR_EQUAL_POSITIVE_MEAN')
                error 'R+C weight, rare source and FIT normalization are explicit'
            r.rare.each { name, descriptor -> sources.add(pinned(descriptor.path, descriptor.sha256)) }
        } else if(r.mix_weight != null || r.rare != null || r.normalization_policy != null) {
            error 'C request contains mixture inputs'
        }
        tuple(entry.id, r.family, request, entry.sha256, sources)
    }
    COMMON_FAMILY_AFFINITIES(Channel.fromList(tasks), code)
}
