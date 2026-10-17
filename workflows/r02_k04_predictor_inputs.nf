nextflow.enable.dsl=2

process K04_PREDICTOR_INPUTS {
    tag "K04-predictors:${task_id}"
    cpus params.task_cpus
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(task_id), path(request), val(request_sha256), path(inputs, stageAs:'input/??/*')
    path code
    output:
    tuple val(task_id), path("predictors_${task_id}")
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    python3 r02_k04_predictor_inputs.py \\
      --request '${request}' --request-sha256 '${request_sha256}' \\
      --output 'predictors_${task_id}' --allow-legacy-compatibility
    """
}

def safePath(value) {
    value instanceof String && value.startsWith('/') &&
        !value.tokenize('/').contains('..') &&
        !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

def sha256(path) {
    def digest=java.security.MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream { stream ->
        new java.security.DigestInputStream(stream,digest).transferTo(java.io.OutputStream.nullOutputStream())
    }
    digest.digest().encodeHex().toString()
}

def pinned(path, expected) {
    if(!safePath(path) || !(expected ==~ /[0-9a-f]{64}/))
        error 'Safe absolute path and lowercase SHA256 required'
    def value=file(path,checkIfExists:true)
    if(!value.toFile().isFile() || value.toFile().toPath().toRealPath()!=value.toFile().toPath().toAbsolutePath().normalize())
        error 'Pinned K04 predictor input must be a canonical regular file'
    if(sha256(value)!=expected) error "K04 predictor input differs before cache lookup: ${path}"
    value
}

workflow {
    if(![params.batch,params.code,params.output_dir].every{ value -> safePath(value) } ||
       !(params.batch_sha256 ==~ /[0-9a-f]{64}/) || !(params.code_sha256 ==~ /[0-9a-f]{64}/) ||
       ![params.task_cpus,params.memory_gb,params.max_forks].every{ value -> value?.toString() ==~ /[1-9][0-9]*/} ||
       !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/) ||
       params.allow_legacy_compatibility?.toString() != 'true')
        error 'Explicit authenticated predictor inputs and positive resources required'
    def batch=pinned(params.batch,params.batch_sha256)
    def code=pinned(params.code,params.code_sha256)
    def value=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(value.schema!='r02_k04_legacy_b012_predictor_inputs_batch_v1' || !(value.tasks instanceof List) || value.tasks.isEmpty())
        error 'Nonempty K04 predictor batch required'
    def seen=[] as Set
    def jobs=value.tasks.collect { task ->
        if(!(task instanceof Map) || task.keySet() as Set != ['id','request','request_sha256'] as Set ||
           !(task.id ==~ /[A-Za-z0-9_]+/) || !seen.add(task.id)) error 'Unique safe predictor task required'
        def request=pinned(task.request,task.request_sha256)
        def specification=new groovy.json.JsonSlurper().parse(request.toFile())
        if(specification.schema!='r02_k04_legacy_b012_predictor_inputs_request_v1' ||
           specification.operation!='MATERIALIZE_LEGACY_B0_B1_B2_COMPATIBILITY_ONLY' ||
           specification.implementation_sha256!=params.code_sha256 ||
           specification.targets.keySet() as Set != ['FIT','SELECT'] as Set)
            error 'Predictor request schema/code/roles differ'
        def descriptors=[]
        descriptors.add(specification.source_contract)
        descriptors.add(specification.persons)
        specification.targets.each { _role, descriptor ->
            descriptors.add([path:descriptor.directory+'/manifest.json',sha256:descriptor.manifest_sha256])
            def manifest=new groovy.json.JsonSlurper().parse(pinned(descriptor.directory+'/manifest.json',descriptor.manifest_sha256).toFile())
            ['person_i.npy','person_j.npy'].each { name ->
                descriptors.add([path:descriptor.directory+'/'+name,sha256:manifest.outputs_sha256[name]])
            }
        }
        specification.partials.each { _chrom, descriptor ->
            descriptors.add([path:descriptor.directory+'/manifest.json',sha256:descriptor.manifest_sha256])
            def manifest=new groovy.json.JsonSlurper().parse(pinned(descriptor.directory+'/manifest.json',descriptor.manifest_sha256).toFile())
            descriptors.add([path:descriptor.directory+'/data.npz',sha256:manifest.outputs_sha256['data.npz']])
        }
        tuple(task.id,request,task.request_sha256,descriptors.collect{ descriptor -> pinned(descriptor.path,descriptor.sha256)})
    }
    K04_PREDICTOR_INPUTS(channel.fromList(jobs),code)
}
