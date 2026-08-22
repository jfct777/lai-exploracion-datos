nextflow.enable.dsl=2

process M33_I0_STAGE_REAL_SOURCE {
    tag "m33_i0_stage_${root_label}"
    cache false
    cpus 1
    memory '256 MB'
    time '5m'
    stageInMode 'copy'

    input:
    val root_label
    path real_script
    path helper_script
    path authorization
    path contract
    path source_auth
    val repo_root

    output:
    tuple val(root_label), path("${root_label}.flare.anc.vcf.gz"), path("${root_label}.source.receipt.json")

    script:
    """
    set -euo pipefail
    python3 ${real_script} stage \
      --root-label ${root_label} \
      --output-vcf ${root_label}.flare.anc.vcf.gz \
      --receipt ${root_label}.source.receipt.json \
      --authorization ${authorization} \
      --contract ${contract} \
      --source-auth ${source_auth} \
      --helper-script ${helper_script} \
      --repo-root ${repo_root} \
      --run-id ${params.m33_i0_real_run_id}
    """
}

process M33_I0_DERIVE_REAL_INDEX {
    tag "m33_i0_index_${root_label}"
    cache false
    cpus 1
    memory '512 MB'
    time '5m'
    stageInMode 'copy'
    container params.m33_i0_real_tabix_image
    containerOptions '--network none'
    publishDir { "${params.m33_i0_real_local_results}/${params.m33_i0_real_run_id}/${root_label}" },
        mode: 'copy', overwrite: false,
        saveAs: { filename -> filename.endsWith('.vcf.gz') ? null : filename }

    input:
    tuple val(root_label), path(source_vcf), path(source_receipt)
    path real_script
    path helper_script
    path authorization
    path contract
    path source_auth

    output:
    tuple val(root_label), path(source_vcf),
        path("${root_label}.flare.anc.vcf.gz.tbi"),
        path("${root_label}.i0_real.receipt.json"),
        path("${root_label.toUpperCase()}_I0_REAL_PASS_NON_CONSUMABLE")

    script:
    """
    set -euo pipefail
    chmod 0400 ${source_vcf}
    python3 ${real_script} index \
      --root-label ${root_label} \
      --source ${source_vcf} \
      --source-receipt ${source_receipt} \
      --output-tbi ${root_label}.flare.anc.vcf.gz.tbi \
      --receipt ${root_label}.i0_real.receipt.json \
      --marker ${root_label.toUpperCase()}_I0_REAL_PASS_NON_CONSUMABLE \
      --authorization ${authorization} \
      --contract ${contract} \
      --source-auth ${source_auth} \
      --helper-script ${helper_script} \
      --run-id ${params.m33_i0_real_run_id} \
      --container-image '${task.container}'
    """
}

process M33_I0_AGGREGATE_REAL {
    tag 'm33_i0_aggregate_real'
    cache false
    cpus 1
    memory '512 MB'
    time '5m'
    stageInMode 'copy'
    container params.m33_i0_real_tabix_image
    containerOptions '--network none'
    publishDir { "${params.m33_i0_real_local_results}/${params.m33_i0_real_run_id}" },
        mode: 'copy', overwrite: false

    input:
    path receipts
    path markers
    path sources
    path indexes
    path real_script
    path helper_script
    path authorization
    path contract
    path source_auth

    output:
    path 'm33_i0_real.manifest.json'
    path 'I0_REAL_PASS_NON_CONSUMABLE'

    script:
    """
    set -euo pipefail
    python3 ${real_script} aggregate \
      --receipts ${receipts.join(' ')} \
      --markers ${markers.join(' ')} \
      --sources ${sources.join(' ')} \
      --indexes ${indexes.join(' ')} \
      --manifest m33_i0_real.manifest.json \
      --completion-marker I0_REAL_PASS_NON_CONSUMABLE \
      --authorization ${authorization} \
      --contract ${contract} \
      --source-auth ${source_auth} \
      --helper-script ${helper_script} \
      --run-id ${params.m33_i0_real_run_id} \
      --container-image '${task.container}'
    """
}
