nextflow.enable.dsl=2

def safeAbsolute(value) {
    value instanceof String && value.startsWith('/') && !value.tokenize('/').contains('..') &&
        !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

def fileHash(path) {
    def digest=java.security.MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream { stream ->
        new java.security.DigestInputStream(stream,digest).transferTo(java.io.OutputStream.nullOutputStream())
    }
    digest.digest().encodeHex().toString()
}

def pinned(name, digest) {
    if(!safeAbsolute(name) || !(digest ==~ /[a-f0-9]{64}/)) error 'Safe absolute pinned adapter input required'
    def path=file(name,checkIfExists:true)
    if(fileHash(path)!=digest) error 'Adapter input changed before cache lookup'
    path
}

def readJson(path) { new groovy.json.JsonSlurper().parse(path.toFile()) }

process K03_NEURAL_TARGET_ADAPTER {
    tag 'FIT_SELECT_only'
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path request
    val requestHash
    path source, stageAs:'source/*'
    path targetContract
    path code
    output:
    path 'neural-targets'
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 ${code} --request '${request}' --request-sha256 '${requestHash}' \
      --source-directory source --target-contract '${targetContract}' --output neural-targets
    """
}

workflow {
    if(![params.request,params.code,params.output_dir].every{safeAbsolute(it)} ||
       !(params.request_sha256 ==~ /[a-f0-9]{64}/) || !(params.code_sha256 ==~ /[a-f0-9]{64}/) ||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
       !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/))
        error 'Explicit safe adapter paths, hash, memory and bounded time required'
    def request=pinned(params.request,params.request_sha256)
    def body=readJson(request)
    if(body.schema!='r02_k03_neural_target_adapter_request_v1' ||
       body.status!='PREPARE_TARGETS_NOT_TRAINING_ADMISSION') error 'Exact non-admitting adapter request required'
    def sourceRoot=body.source.directory
    def manifest=pinned(sourceRoot+'/manifest.json',body.source.manifest_sha256)
    def sourceManifest=readJson(manifest)
    if(sourceManifest.evaluate_targets_computed!=false ||
       sourceManifest.outputs_sha256.keySet()!=['persons.tsv','arrays.private.npz'].toSet())
        error 'Exact K03 FIT/SELECT-only source required'
    def source=[manifest]
    sourceManifest.outputs_sha256.each { name,digest ->
        if(!(name in ['persons.tsv','arrays.private.npz'])) error 'Unexpected K03 source product'
        source.add(pinned(sourceRoot+'/'+name,digest))
    }
    def contract=pinned(body.target_contract.path,body.target_contract.sha256)
    K03_NEURAL_TARGET_ADAPTER(request,params.request_sha256,source,contract,
                              pinned(params.code,params.code_sha256))
}
