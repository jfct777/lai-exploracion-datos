nextflow.enable.dsl=2

/*
 * Materialize truth-blind LAI query overlays or consumer-ready batches.
 * The batch manifest pins every request and the adapter; no training occurs.
 */

params.batch_manifest = null
params.output_dir = null
params.adapter = "${projectDir}/../bin/r02_lai_truthblind_adapter.py"
params.adapter_sha256 = null
params.container = null

process PREPARE_LAI_CONTEXT_QUERY {
    tag "${task_id}"
    container params.container
    stageInMode 'copy'
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple val(task_id), path(request), val(request_sha256), path(query_grid), path(query_grid_receipt), \
          path(persons), path(roles), path(truth), path(adapter), val(adapter_sha256)

    output:
    tuple val(task_id), path("${task_id}"), emit: prepared

    script:
    """
    test "\$(sha256sum ${request} | cut -d' ' -f1)" = "${request_sha256}"
    test "\$(sha256sum ${adapter} | cut -d' ' -f1)" = "${adapter_sha256}"
    python3 ${adapter} --request ${request} --staged-inputs . --output-dir ${task_id}
    """
}

workflow {
    if (!params.batch_manifest || !params.output_dir || !params.container || !params.adapter_sha256) {
        error 'Required: --batch_manifest, --output_dir, --container (digest), --adapter_sha256'
    }
    def manifest = new groovy.json.JsonSlurper().parse(file(params.batch_manifest))
    if (manifest.schema != 'r02_lai_truthblind_context_batch_v2' || !manifest.tasks) {
        error 'invalid LAI truth-blind batch manifest'
    }
    requests = Channel.fromList(manifest.tasks.collect { row ->
        def requestPath = file(row.request_path as String)
        def request = new groovy.json.JsonSlurper().parse(requestPath)
        tuple(row.task_id as String, requestPath, row.request_sha256 as String,
              file(request.query_grid_path as String), file(request.query_grid_receipt_path as String),
              file(request.persons_path as String), file(request.roles_path as String),
              file(request.truth_path as String), file(params.adapter as String),
              params.adapter_sha256 as String)
    })
    PREPARE_LAI_CONTEXT_QUERY(requests)
}
