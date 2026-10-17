nextflow.enable.dsl=2

params.request_index = null
params.code_dir = null
params.stage_root = null
params.validation_time = null
params.output_dir = null
params.container = null
params.gcloud_config = null
params.only_chromosome = null
params.completed_manifest = null

def required(name, value) {
    if (value == null || value.toString().trim() == '') error "Missing required parameter --${name}"
    value
}

process VALIDATE_ONE_NEURAL_STORE {
    tag "chr${chrom}"
    cpus 1
    memory '3 GB'
    time params.validation_time
    maxForks 1
    cache 'deep'
    errorStrategy 'terminate'
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple val(chrom), path(request, stageAs: 'request.json'), val(request_sha256),
          path(manifest, stageAs: 'store-manifest.json'), val(manifest_sha256),
          path(code_dir, stageAs: 'code')

    output:
    tuple val(chrom), path("chr${chrom}.validation.local.json"), emit: receipts

    script:
    """
    set -euo pipefail
    test "\$(sha256sum '${request}' | cut -d' ' -f1)" = '${request_sha256}'
    test "\$(sha256sum '${manifest}' | cut -d' ' -f1)" = '${manifest_sha256}'
    export CLOUDSDK_CONFIG='${params.gcloud_config}'
    PYTHONPATH='${code_dir}' python3 '${code_dir}/r02_neural_store_validate.py' \\
      --request '${request}' \\
      --request-sha256 '${request_sha256}' \\
      --manifest-override '${manifest}' \\
      --recover-owned-stage \\
      --validation-container '${params.container}' \\
      --output 'chr${chrom}.validation.local.json'
    """
}

workflow {
    index_file = file(required('request_index', params.request_index), checkIfExists: true)
    code_dir_file = file(required('code_dir', params.code_dir), checkIfExists: true)
    required('stage_root', params.stage_root)
    required('validation_time', params.validation_time)
    required('output_dir', params.output_dir)
    required('container', params.container)
    required('gcloud_config', params.gcloud_config)
    index = new groovy.json.JsonSlurper().parse(index_file)
    if (index.schema != 'r02_neural_store_request_index_v1' || index.requests.size() != 22) {
        error 'Exact 22-store request index required'
    }
    completed = [] as Set
    if (params.completed_manifest != null) {
        completed_file = file(params.completed_manifest, checkIfExists: true)
        completed_data = new groovy.json.JsonSlurper().parse(completed_file)
        if (completed_data.schema != 'r02_neural_input_validation_completed_inputs_v1') {
            error 'Completed-input manifest schema differs'
        }
        completed_data.receipts.each { receipt ->
            def row = index.requests.find { candidate ->
                candidate.chromosome.toString() == receipt.chromosome.toString()
            }
            if (row == null || row.request.sha256 != receipt.request_sha256 ||
                receipt.status != 'COMPLETE_INPUT_VALIDATION_NO_TRAINING' ||
                !receipt.derived_store_released || receipt.publication_generation == null) {
                error 'Completed-input receipt differs from the 22-store index'
            }
            if (!completed.add(receipt.chromosome.toString())) error 'Duplicate completed chromosome'
        }
    }
    selected = index.requests.findAll { row ->
        !completed.contains(row.chromosome.toString()) &&
        (params.only_chromosome == null || row.chromosome.toString() == params.only_chromosome.toString())
    }
    if (params.only_chromosome != null && selected.size() != 1) error 'Requested chromosome is absent or duplicated'
    rows = selected.sort { a, b -> a.chromosome.toInteger() <=> b.chromosome.toInteger() }.collect { row ->
        def label = String.format('chr%02d.json', row.chromosome.toInteger())
        tuple(row.chromosome.toString(), file("${index_file.parent}/store-requests/${label}", checkIfExists: true), row.request.sha256,
              file("${index_file.parent}/store-manifests/${label}", checkIfExists: true), row.manifest.sha256,
              code_dir_file)
    }
    VALIDATE_ONE_NEURAL_STORE(Channel.fromList(rows))
}
