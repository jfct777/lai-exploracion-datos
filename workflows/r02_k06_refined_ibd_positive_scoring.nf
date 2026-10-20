nextflow.enable.dsl=2

params.requests = null
params.calls_dir = null
params.operator = "${projectDir}/../bin/r02_k06_score_refined_ibd_positive.py"
params.outdir = null

process SCORE_POSITIVE_GEOMETRY {
    tag "${case_id}"
    cpus 1
    memory '2 GB'
    time '5m'
    maxForks 1
    cache 'deep'
    publishDir params.outdir, mode: 'copy', overwrite: false
    input:
    tuple val(case_id), path(request), val(request_sha256), path(calls_dir),
          path(operator_code), path(inputs, stageAs: 'inputs/*')
    output:
    tuple val(case_id), path("${case_id}.positive-geometry")
    script:
    """
    python3 '${operator_code}' --request '${request}' --request-sha256 '${request_sha256}' \
      --calls-dir '${calls_dir}' --staged-input-dir inputs \
      --output '${case_id}.positive-geometry'
    """
}

workflow {
    if (!params.requests || !params.calls_dir || !params.outdir)
        error 'Explicit request manifest, completed calls directory and output directory required'
    def operator = file(params.operator, checkIfExists: true)
    def manifest = new groovy.json.JsonSlurper().parse(file(params.requests).toFile())
    assert manifest.schema == 'r02_k06_refined_ibd_request_manifest_v1'
    assert manifest.requests.size() == 12
    assert manifest.requests.collect { it.case_id }.unique().size() == 12
    def tasks = Channel.fromList(manifest.requests).map { item ->
        def request = file(item.path, checkIfExists: true)
        assert java.security.MessageDigest.getInstance('SHA-256').digest(request.bytes).encodeHex().toString() == item.sha256
        def value = new groovy.json.JsonSlurper().parse(request.toFile())
        assert value.case_id == item.case_id
        def calls = file("${params.calls_dir}/${item.case_id}.calls", checkIfExists: true)
        def inputs = ['genetic_map', 'positive_territory', 'parameter_spec'].collect { name ->
            def path = file(value[name].path, checkIfExists: true)
            assert java.security.MessageDigest.getInstance('SHA-256').digest(path.bytes).encodeHex().toString() == value[name].sha256
            path
        }
        tuple(item.case_id, request, item.sha256, calls, operator, inputs)
    }
    SCORE_POSITIVE_GEOMETRY(tasks)
}
