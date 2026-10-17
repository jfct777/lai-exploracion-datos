nextflow.enable.dsl=2

params.batch = null
params.batch_sha256 = null
params.code_dir = null
params.output_dir = null
params.container = null
params.build_cpus = null
params.build_memory_gb = null
params.build_time = null
params.build_max_forks = null
params.reserve_gb = null
params.publisher_sha256 = null
params.mode = 'FULL'

include { K04_BUILD_DESIGN } from '../modules/r02_k04_build_design'

process K04_PUBLISH_BUILD {
    tag "K04-publish:${task_id}"
    cpus 1
    memory '1 GB'
    time '30m'
    maxForks 1
    cache 'deep'
    container params.container
    containerOptions { "-v ${params.output_dir}:${params.output_dir}" }
    input:
    tuple val(task_id), path(model_inputs), val(request_sha256), path(publisher, stageAs:'code/r02_k04_build_publish.py')
    output:
    path("publication_${task_id}.json")
    script:
    """
    set -euo pipefail
    umask 077
    python3 code/r02_k04_build_publish.py publish --task-id '${task_id}' \
      --request-sha256 '${request_sha256}' --product '${model_inputs}' \
      --output-root '${params.output_dir}' --reserve-bytes ${params.reserve_gb * 1024L * 1024L * 1024L} \
      --receipt 'publication_${task_id}.json'
    """
}

process K04_BUILD_GATE {
    tag 'K04-build-gate:66'
    cpus 1
    memory '1 GB'
    time '30m'
    cache 'deep'
    container params.container
    containerOptions { "-v ${params.output_dir}:${params.output_dir}" }
    input:
    path(receipts, stageAs:'receipts/*')
    path(publisher, stageAs:'code/r02_k04_build_publish.py')
    output:
    path('COMPLETE.json')
    script:
    """
    set -euo pipefail
    umask 077
    python3 code/r02_k04_build_publish.py gate --receipts receipts \
      --output-root '${params.output_dir}' --batch-sha256 '${params.batch_sha256}' --complete COMPLETE.json
    """
}

process K04_VERIFY_REUSED_BUILD {
    tag "K04-reuse:${task_id}"
    cpus 1
    memory '1 GB'
    time '30m'
    maxForks 1
    cache 'deep'
    container params.container
    containerOptions { "-v ${params.output_dir}:${params.output_dir}" }
    input:
    tuple val(task_id), val(request_sha256), path(publisher, stageAs:'code/r02_k04_build_publish.py')
    output:
    path("publication_${task_id}.json")
    script:
    """
    set -euo pipefail
    umask 077
    python3 code/r02_k04_build_publish.py verify --task-id '${task_id}' \
      --request-sha256 '${request_sha256}' --output-root '${params.output_dir}' \
      --receipt 'publication_${task_id}.json'
    """
}

def safePath(value) {
    value instanceof String && value.startsWith('/') && !value.tokenize('/').contains('..') &&
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
    if (!safePath(path) || !(expected ==~ /[0-9a-f]{64}/)) error 'Safe absolute path and SHA256 required'
    def value=file(path,checkIfExists:true)
    if (!value.toFile().isFile() || sha256(value)!=expected) error "Pinned K04 file differs: ${path}"
    value
}

workflow {
    if (![params.batch,params.code_dir,params.output_dir].every { safePath(it) } ||
        !(params.batch_sha256 ==~ /[0-9a-f]{64}/) || !(params.publisher_sha256 ==~ /[0-9a-f]{64}/) ||
        !(params.container instanceof String) || !params.container.contains('@sha256:') ||
        !(params.mode in ['FULL','BENCHMARK_MAX']) ||
        ![params.build_cpus,params.build_memory_gb,params.build_max_forks,params.reserve_gb].every {
            it?.toString() ==~ /[1-9][0-9]*/ } || !(params.build_time instanceof String) ||
        !(params.build_time ==~ /[1-9][0-9]*[smh]/)) error 'Pinned inputs and explicit positive technical limits required'
    def outputRoot=java.nio.file.Paths.get(params.output_dir).normalize()
    if (!outputRoot.isAbsolute() || !java.nio.file.Files.isDirectory(outputRoot) ||
        java.nio.file.Files.isSymbolicLink(outputRoot)) error 'Existing real publication root required'
    def batch=pinned(params.batch,params.batch_sha256)
    def value=new groovy.json.JsonSlurper().parse(batch.toFile())
    def codeNames=['r02_k04_model_bridge.py','r02_k04_predictor_inputs.py','r02_pair_operator.py','r02_interval_ridge.py'] as Set
    if (value.schema!='r02_k04_build_batch_v1' || value.tasks?.size()!=66 ||
        value.implementation_sha256.keySet() as Set != codeNames || value.evaluate_allowed!=false ||
        value.models_fitted!=0) error 'Exact complete BUILD-only batch required'
    def code=value.implementation_sha256.collectEntries { name,digest -> [(name):pinned(params.code_dir+'/'+name,digest)] }
    def publisher=pinned(params.code_dir+'/r02_k04_build_publish.py',params.publisher_sha256)
    def seen=[] as Set; def builds=[]; def metadata=[]; def reuse=[]; def requiredReuse=[] as Set
    value.tasks.each { task ->
        if (!(task.id ==~ /chr(0[1-9]|1[0-9]|2[0-2])_(AFR|EUR|NAM)/) || !seen.add(task.id))
            error 'Unique 22x3 task identity required'
        if (!(task.reuse_required instanceof Boolean)) error 'Explicit per-task reuse policy required'
        if (task.reuse_required) requiredReuse.add(task.id)
        if (params.mode=='FULL' || task.id=='chr06_EUR') {
            def receiptPath=outputRoot.resolve('receipts').resolve(task.id+'.json')
            def productPath=outputRoot.resolve('model_inputs_'+task.id)
            def hasReceipt=java.nio.file.Files.isRegularFile(receiptPath)
            def hasProduct=java.nio.file.Files.isDirectory(productPath)
            if (hasReceipt != hasProduct) error "Partial prior K04 publication: ${task.id}"
            if (task.reuse_required && !hasReceipt)
                error "Required authenticated AL publication is absent; refusing rebuild: ${task.id}"
            if (hasReceipt) {
                reuse << tuple(task.id,task.request_sha256,publisher)
            } else {
                def request=pinned(task.request,task.request_sha256)
                def inputs=task.inputs.collect { pinned(it.path,it.sha256) }
                builds << tuple(task.id,request,task.request_sha256,inputs,code.values() as List)
                metadata << tuple(task.id,task.request_sha256,publisher)
            }
        }
    }
    if (seen.size()!=66 || requiredReuse != (['chr06_EUR'] as Set))
        error 'Complete 22x3 set and exact required AL reuse identity required'
    K04_BUILD_DESIGN(channel.fromList(builds))
    K04_PUBLISH_BUILD(K04_BUILD_DESIGN.out.join(channel.fromList(metadata),by:0))
    K04_VERIFY_REUSED_BUILD(channel.fromList(reuse))
    def publications=K04_PUBLISH_BUILD.out.mix(K04_VERIFY_REUSED_BUILD.out)
    if (params.mode=='FULL') K04_BUILD_GATE(publications.collect(),publisher)
}
