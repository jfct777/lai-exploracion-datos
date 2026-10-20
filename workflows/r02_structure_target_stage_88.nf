#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.catalogue = null
params.catalogue_sha256 = null
params.stager = null
params.stager_sha256 = null
params.verifier = "${projectDir}/../bin/r02_structure_target_stage_verify.py"
params.verifier_sha256 = null
params.target_root = null
params.receipt_root = null
params.gcloud_config = null
params.expected_account = '653458115080-compute@developer.gserviceaccount.com'
params.python = 'python3'
params.stage_time = '30m'
params.reserve_bytes = 10737418240

process STAGE_STRUCTURE_TARGET {
    tag "${view_id}"
    cpus 1
    memory '2 GB'
    maxForks 2
    time params.stage_time
    errorStrategy 'terminate'

    publishDir params.receipt_root, mode: 'copy', overwrite: false

    input:
    val view_id
    path catalogue
    path stager
    path verifier

    output:
    path "${view_id}.stage.json", emit: receipts

    script:
    def targetDir = "${params.target_root}/${view_id}"
    """
    set -euo pipefail
    ${params.python} -c "import shutil; assert shutil.disk_usage('${params.target_root}').free >= ${params.reserve_bytes}, 'insufficient free space for verified target staging'"
    test "\$(sha256sum ${stager} | cut -d' ' -f1)" = '${params.stager_sha256}'
    test "\$(sha256sum ${verifier} | cut -d' ' -f1)" = '${params.verifier_sha256}'
    active_account=\$(CLOUDSDK_CONFIG='${params.gcloud_config}' gcloud auth list \
      --filter='status:ACTIVE' --format='value(account)')
    test "\${active_account}" = '${params.expected_account}'
    CLOUDSDK_CONFIG='${params.gcloud_config}' ${params.python} ${stager} \
      --catalogue ${catalogue} --catalogue-sha256 '${params.catalogue_sha256}' \
      --view-id '${view_id}' --out '${targetDir}'
    ${params.python} ${verifier} bundle \
      --catalogue ${catalogue} --catalogue-sha256 '${params.catalogue_sha256}' \
      --stager ${stager} --stager-sha256 '${params.stager_sha256}' \
      --view-id '${view_id}' --target-dir '${targetDir}' \
      --output '${view_id}.stage.json'
    """

    stub:
    """
    echo '{"schema":"stub","view_id":"${view_id}"}' > '${view_id}.stage.json'
    """
}

process VERIFY_STRUCTURE_TARGET_CAMPAIGN {
    tag '88_authenticated_fit_select_targets'
    cpus 1
    memory '512 MB'
    time '10m'
    errorStrategy 'terminate'

    publishDir params.receipt_root, mode: 'copy', overwrite: false

    input:
    path receipts
    path catalogue
    path verifier

    output:
    path 'STAGE_88_COMPLETE.json'

    script:
    """
    set -euo pipefail
    test "\$(sha256sum ${verifier} | cut -d' ' -f1)" = '${params.verifier_sha256}'
    ${params.python} ${verifier} campaign \
      --catalogue ${catalogue} --catalogue-sha256 '${params.catalogue_sha256}' \
      --receipts ${receipts} --output STAGE_88_COMPLETE.json
    """

    stub:
    """
    echo '{"schema":"stub","views":88}' > STAGE_88_COMPLETE.json
    """
}

workflow {
    ['catalogue', 'catalogue_sha256', 'stager', 'stager_sha256', 'verifier_sha256',
     'target_root', 'receipt_root', 'gcloud_config', 'expected_account'].each {
        if (!params[it]) error "Missing --${it}"
    }
    def catalogueFile = file(params.catalogue, checkIfExists: true)
    def catalogueValue = new groovy.json.JsonSlurper().parse(catalogueFile)
    if (catalogueValue.schema != 'r02_structure_88_target_catalogue_v1' ||
        catalogueValue.status != 'AUTHENTICATED_88_FIT_SELECT_TARGETS_NO_TRAINING' ||
        catalogueValue.views != 88 || catalogueValue.targets.size() != 88) {
        error 'The target catalogue is not the authenticated 88-view FIT/SELECT catalogue'
    }
    def viewIds = catalogueValue.targets.collect { it.view_id }
    if (viewIds.toSet().size() != 88) error 'Target view IDs are not unique'

    STAGE_STRUCTURE_TARGET(
        Channel.fromList(viewIds),
        Channel.value(catalogueFile),
        Channel.value(file(params.stager, checkIfExists: true)),
        Channel.value(file(params.verifier, checkIfExists: true)))
    VERIFY_STRUCTURE_TARGET_CAMPAIGN(
        STAGE_STRUCTURE_TARGET.out.receipts.collect(),
        Channel.value(catalogueFile),
        Channel.value(file(params.verifier, checkIfExists: true)))
}
