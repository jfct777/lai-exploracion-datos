nextflow.enable.dsl=2

process K04_INTERVAL_TARGET {
    tag "K04-target:${chromosome}:${ancestry}:${role}"
    cpus params.task_cpus
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(task_id), val(chromosome), val(ancestry), val(role),
          path(request), val(request_sha256), path(source_files, stageAs:'source/??/*')
    path code
    output:
    tuple val(task_id), path("target_${task_id}")
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    python3 r02_k04_interval_target.py \\
      --request '${request}' --request-sha256 '${request_sha256}' \\
      --output 'target_${task_id}'
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
        error 'Pinned K04 input must be a canonical regular file'
    if(sha256(value)!=expected) error "K04 interval-target input differs before cache lookup: ${path}"
    value
}

workflow {
    if(![params.batch,params.code_dir,params.output_dir].every{safePath(it)} ||
       !(params.batch_sha256 ==~ /[0-9a-f]{64}/) ||
       ![params.task_cpus,params.memory_gb,params.max_forks,params.max_output_bytes].every{
           it?.toString() ==~ /[1-9][0-9]*/
       } || !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/))
        error 'Explicit authenticated inputs and positive CPU/memory/time/output/concurrency limits required'

    def batch=pinned(params.batch,params.batch_sha256)
    def value=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(value.schema!='r02_k04_interval_target_batch_v1' || !(value.tasks instanceof List) || value.tasks.isEmpty())
        error 'Nonempty K04 interval-target batch required'
    def expectedCode=[
        'r02_genomic_pair_evidence.py','r02_ibd_compatibility.py','r02_ibd_local_ancestry.py',
        'r02_ibd_prepare.py','r02_k04_common_target.py',
        'r02_k04_identified_comparison.py','r02_k04_interval_target.py',
        'r02_rare_ancestry_chromosome.py','r02_rare_ancestry_identifiability.py',
        'r02_rare_ancestry_pair_bounds.py','r02_segment_evidence.py',
        'r02_structure_common_inputs.py','r02_structure_continuous.py',
        'r02_structure_discrete.py','r02_structure_increment.py',
        'r02_structure_input_plan.py','r02_structure_materialize.py'
    ] as Set
    if(!(value.implementation_sha256 instanceof Map) ||
       value.implementation_sha256.keySet() as Set != expectedCode)
        error 'Exact K04 adapter dependency inventory required'
    def code=value.implementation_sha256.collect { name,digest ->
        pinned(params.code_dir+'/'+name,digest)
    }
    def seen=[] as Set
    def jobs=value.tasks.collect { descriptor ->
        if(!(descriptor instanceof Map) || descriptor.keySet() as Set !=
           ['id','chromosome','ancestry','role','request','request_sha256'] as Set ||
           !(descriptor.id ==~ /[A-Za-z0-9_]+/) ||
           !(descriptor.chromosome.toString() ==~ /[1-9]|1[0-9]|2[0-2]/) ||
           !(descriptor.ancestry in ['AFR','EUR','NAM']) || !(descriptor.role in ['FIT','SELECT']))
            error 'Exact safe chromosome/ancestry/FIT-or-SELECT task required'
        def identity="${descriptor.chromosome}:${descriptor.ancestry}:${descriptor.role}"
        if(!seen.add(descriptor.id) || !seen.add('identity:'+identity))
            error 'Duplicate task id or chromosome/ancestry/role target'
        def request=pinned(descriptor.request,descriptor.request_sha256)
        def specification=new groovy.json.JsonSlurper().parse(request.toFile())
        if(specification.schema!='r02_k04_interval_target_input_v1' ||
           specification.operation!='MATERIALIZE_SOLVER_TARGET_ONLY' ||
           specification.scientific_scope!='CONDITIONAL_GNOMIX_COMMON_DISCORDANCE_OUTER_INTERVAL' ||
           specification.target.chromosome.toString()!=descriptor.chromosome.toString() ||
           specification.ancestry!=descriptor.ancestry || specification.role!=descriptor.role ||
           !(specification.resources.max_output_bytes instanceof Number) ||
           specification.resources.max_output_bytes.longValue()>params.max_output_bytes.toString().toLong())
            error 'Task descriptor/request/resource binding differs'
        def directory=specification.target.directory
        if(!safePath(directory)) error 'Safe absolute K04 producer directory required'
        def manifest=pinned(directory+'/manifest.json',specification.target.manifest_sha256)
        def producer=new groovy.json.JsonSlurper().parse(manifest.toFile())
        if(producer.schema!='r02_k04_common_target_v1' ||
           producer.status!='COMPLETE_CONDITIONAL_COMMON_TARGET_NOT_MODEL_COMPARISON' ||
           producer.chromosome.toString()!=descriptor.chromosome.toString() ||
           producer.outputs_sha256.keySet() as Set !=
               ['counts.npz','persons.tsv','coordinate-preflight.json'] as Set)
            error 'Authenticated completed common target required'
        def inputs=[manifest]
        producer.outputs_sha256.each { name,digest ->
            if(!(name in ['counts.npz','persons.tsv','coordinate-preflight.json']))
                error 'Unexpected producer output'
            inputs.add(pinned(directory+'/'+name,digest))
        }
        tuple(descriptor.id,descriptor.chromosome.toString(),descriptor.ancestry,
              descriptor.role,request,descriptor.request_sha256,inputs)
    }
    K04_INTERVAL_TARGET(Channel.fromList(jobs),code)
}
