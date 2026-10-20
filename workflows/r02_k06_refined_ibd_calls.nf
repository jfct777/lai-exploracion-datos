nextflow.enable.dsl=2

params.requests = null
params.operator = "${projectDir}/../bin/r02_k06_refined_ibd_calls.py"
params.outdir = null
params.reuse_calls = null
params.java_stack_mib = null

def requireValue(condition, message) {
    if (!condition)
        throw new IllegalArgumentException(message)
}

def sha256File(path) {
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    digest.digest(java.nio.file.Files.readAllBytes(path)).encodeHex().toString()
}

def authenticateCompleted(callDir, item, request, requestSha256) {
    def receiptPath = callDir.resolve('receipt.json')
    requireValue(java.nio.file.Files.isRegularFile(receiptPath),
                 "Partial reused result lacks receipt: ${item.case_id}")
    def receipt = new groovy.json.JsonSlurper().parse(receiptPath.toFile())
    requireValue(receipt.schema == 'r02_k06_refined_ibd_receipt_v1' && receipt.status == 'COMPLETE',
                 "Reused result is not COMPLETE: ${item.case_id}")
    requireValue(receipt.case_id == item.case_id && receipt.chromosome == request.chromosome,
                 "Reused case identity differs: ${item.case_id}")
    if (receipt.request_sha256 != null)
        requireValue(receipt.request_sha256 == requestSha256,
                     "Reused request hash differs: ${item.case_id}")
    requireValue(receipt.detector_sha256 == request.detector_jar.sha256 &&
                 receipt.genotypes_sha256 == request.genotypes.sha256 &&
                 receipt.genetic_map_sha256 == request.genetic_map.sha256 &&
                 receipt.positive_territory_sha256 == request.positive_territory.sha256 &&
                 receipt.parameter_spec_sha256 == request.parameter_spec.sha256 &&
                 receipt.sample_count == request.expected_samples &&
                 receipt.marker_count == request.expected_markers,
                 "Reused result inputs differ from authenticated request ${requestSha256}: ${item.case_id}")
    def outputs = [
        calls_sha256: 'refined-ibd.detected.private.tsv.gz',
        native_calls_sha256: 'refined-ibd.native.ibd.gz',
        native_hbd_calls_sha256: 'refined-ibd.native.hbd.gz',
        effective_plink_map_sha256: 'refined-ibd.effective.plink.map',
        log_sha256: 'refined-ibd.log'
    ]
    outputs.each { hashName, fileName ->
        def output = callDir.resolve(fileName)
        requireValue(java.nio.file.Files.isRegularFile(output) && receipt[hashName] != null &&
                     sha256File(output) == receipt[hashName],
                     "Reused output hash differs (${fileName}): ${item.case_id}")
    }
    def terminal = callDir.resolve('refined-ibd.log').toFile().text.trim()
    requireValue(terminal.endsWith('refined-ibd.17Jan20.102.jar finished'),
                 "Reused Refined IBD log lacks terminal marker: ${item.case_id}")
    true
}

process REFINED_IBD_CALLS {
    tag "${case_id}"
    cpus 2
    memory '2 GB'
    time '10m'
    maxForks 1
    cache 'deep'
    publishDir params.outdir, mode: 'copy', overwrite: false
    input:
    tuple val(case_id), path(request), val(request_sha256), val(java_stack_mib), path(operator_code), path(inputs, stageAs: 'inputs/*')
    output:
    tuple val(case_id), path("${case_id}.calls")
    script:
    """
    python3 '${operator_code}' --request '${request}' --request-sha256 '${request_sha256}' \
      --staged-input-dir inputs --java-stack-mib '${java_stack_mib}' --output '${case_id}.calls'
    """
    stub:
    """
    mkdir '${case_id}.calls'
    """
}

workflow {
    if (!params.requests || !params.outdir || !params.reuse_calls || params.java_stack_mib == null)
        error 'Explicit request manifest, output/reuse directory, and JVM stack MiB required'
    def javaStackMib = params.java_stack_mib as Integer
    assert javaStackMib >= 1 && javaStackMib <= 16
    def operator = file(params.operator, checkIfExists: true)
    def reuseRoot = file(params.reuse_calls, checkIfExists: true)
    assert java.nio.file.Files.isDirectory(reuseRoot)
    def manifest = new groovy.json.JsonSlurper().parse(file(params.requests).toFile())
    assert manifest.schema == 'r02_k06_refined_ibd_request_manifest_v1'
    assert manifest.requests.size() == 12
    assert manifest.requests && manifest.requests.collect { it.case_id }.unique().size() == manifest.requests.size()
    def remaining = []
    def reused = []
    manifest.requests.each { item ->
        assert item.case_id ==~ /[A-Za-z0-9_.-]+/
        def request = file(item.path, checkIfExists: true)
        def pin = sha256File(request)
        assert pin == item.sha256
        def value = new groovy.json.JsonSlurper().parse(request.toFile())
        assert value.case_id == item.case_id
        def completed = reuseRoot.resolve("${item.case_id}.calls")
        if (java.nio.file.Files.exists(completed)) {
            requireValue(java.nio.file.Files.isDirectory(completed),
                         "Reused result is not a directory: ${item.case_id}")
            authenticateCompleted(completed, item, value, pin)
            reused << item.case_id
        }
        else {
            remaining << item
        }
    }
    log.info "K06 authenticated resume: ${reused.size()} reused COMPLETE; ${remaining.size()} remaining; universe 12"
    def tasks = Channel.fromList(remaining).map { item ->
        def request = file(item.path, checkIfExists: true)
        def pin = sha256File(request)
        def value = new groovy.json.JsonSlurper().parse(request.toFile())
        def inputs = ['detector_jar', 'genotypes', 'genetic_map', 'positive_territory', 'parameter_spec'].collect { name ->
            def path = file(value[name].path, checkIfExists: true)
            assert sha256File(path) == value[name].sha256
            path
        }
        tuple(item.case_id, request, pin, javaStackMib, operator, inputs)
    }
    REFINED_IBD_CALLS(tasks)
}
