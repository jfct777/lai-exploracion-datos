#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.request_index = null
params.request_index_sha256 = null
params.code_root = null
params.code_manifest = null
params.code_manifest_sha256 = null
params.prior_full22_gate = null
params.prior_full22_gate_sha256 = null
params.amendment = null
params.amendment_sha256 = null
params.stage_root = null
params.output_dir = null
params.gcloud_config = null
params.expected_account = '653458115080-compute@developer.gserviceaccount.com'
params.stage_time = '2h'

process RESTORE_AUTHENTICATED_STORE {
    tag "chr${chromosome}"
    cpus 1
    memory '2 GB'
    time params.stage_time
    maxForks 2
    cache 'deep'
    errorStrategy 'terminate'

    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple val(chromosome), path(request, stageAs: 'request.json'), val(request_sha256),
          path(manifest, stageAs: 'store-manifest.json'), val(manifest_sha256)
    path code_root, stageAs: 'frozen-code'
    path code_manifest, stageAs: 'restore-code-manifest.json'
    path prior_gate, stageAs: 'prior-full22-gate.json'
    path amendment, stageAs: 'stage-recovery-amendment.json'

    output:
    tuple val(chromosome), path("chr${chromosome}.restored.json"), emit: receipts

    script:
    """
    set -euo pipefail
    test "\$(sha256sum '${request}' | cut -d' ' -f1)" = '${request_sha256}'
    test "\$(sha256sum '${manifest}' | cut -d' ' -f1)" = '${manifest_sha256}'
    python3 '${code_root}/bin/r02_structure_restore_code_verify.py' \
      --root '${code_root}' --manifest '${code_manifest}' \
      --manifest-sha256 '${params.code_manifest_sha256}'
    export CLOUDSDK_CONFIG='${params.gcloud_config}'
    export CLOUDSDK_CORE_ACCOUNT='${params.expected_account}'
    test "\$(gcloud config get account 2>/dev/null)" = '${params.expected_account}'
    PYTHONPATH='${code_root}/bin' python3 '${code_root}/bin/r02_neural_structure_stage_v3.py' \
      --request '${request}' --request-sha256 '${request_sha256}' \
      --manifest-override '${manifest}' \
      --prior-full22-gate '${prior_gate}' \
      --prior-full22-gate-sha256 '${params.prior_full22_gate_sha256}' \
      --amendment '${amendment}' --amendment-sha256 '${params.amendment_sha256}' \
      --output 'chr${chromosome}.restored.json'
    """

    stub:
    """
    echo '{"schema":"stub","chromosome":"${chromosome}"}' > 'chr${chromosome}.restored.json'
    """
}

process CLOSE_RESTORED_STORE_INDEX {
    tag '22_authenticated_store_paths'
    cpus 1
    memory '512 MB'
    time '10m'
    errorStrategy 'terminate'

    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    path receipts
    path request_index
    path code_root, stageAs: 'frozen-code'
    path code_manifest, stageAs: 'restore-code-manifest.json'
    path prior_gate, stageAs: 'prior-full22-gate.json'
    path amendment, stageAs: 'stage-recovery-amendment.json'

    output:
    path 'RESTORED_STORES_22.json'

    script:
    """
    set -euo pipefail
    python3 '${code_root}/bin/r02_structure_restore_code_verify.py' \
      --root '${code_root}' --manifest '${code_manifest}' \
      --manifest-sha256 '${params.code_manifest_sha256}'
    python3 '${code_root}/bin/r02_structure_store_stage_index.py' \
      --request-index '${request_index}' --request-index-sha256 '${params.request_index_sha256}' \
      --prior-full22-gate '${prior_gate}' \
      --prior-full22-gate-sha256 '${params.prior_full22_gate_sha256}' \
      --amendment '${amendment}' --amendment-sha256 '${params.amendment_sha256}' \
      --stage-root '${params.stage_root}' --receipts ${receipts} \
      --output RESTORED_STORES_22.json
    """

    stub:
    """
    echo '{"schema":"stub","chromosomes":22}' > RESTORED_STORES_22.json
    """
}

workflow {
    ['request_index', 'request_index_sha256', 'code_root', 'code_manifest', 'code_manifest_sha256',
     'prior_full22_gate', 'prior_full22_gate_sha256', 'amendment', 'amendment_sha256',
     'stage_root', 'output_dir', 'gcloud_config', 'expected_account'].each {
        if (!params[it]) error "Missing --${it}"
    }
    def indexFile = file(params.request_index, checkIfExists: true)
    def codeRoot = file(params.code_root, checkIfExists: true)
    def codeManifest = file(params.code_manifest, checkIfExists: true)
    def priorGate = file(params.prior_full22_gate, checkIfExists: true)
    def amendmentFile = file(params.amendment, checkIfExists: true)
    def index = new groovy.json.JsonSlurper().parse(indexFile)
    if (index.schema != 'r02_neural_store_request_index_v1' || index.requests.size() != 22) {
        error 'Exact 22-store request index required'
    }
    def rows = index.requests.sort { a, b -> a.chromosome.toInteger() <=> b.chromosome.toInteger() }.collect { row ->
        tuple(row.chromosome.toString(), file(row.request.path, checkIfExists: true), row.request.sha256,
              file(row.manifest.path, checkIfExists: true), row.manifest.sha256)
    }
    RESTORE_AUTHENTICATED_STORE(Channel.fromList(rows), Channel.value(codeRoot),
        Channel.value(codeManifest),
        Channel.value(priorGate), Channel.value(amendmentFile))
    CLOSE_RESTORED_STORE_INDEX(RESTORE_AUTHENTICATED_STORE.out.receipts.map { it[1] }.collect(),
        Channel.value(indexFile), Channel.value(codeRoot), Channel.value(codeManifest), Channel.value(priorGate),
        Channel.value(amendmentFile))
}
