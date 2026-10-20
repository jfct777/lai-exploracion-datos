#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.package_root = null
params.recovery_root = null
params.recovery_receipt = null
params.recovery_receipt_sha256 = null
params.run_root = null

def need(name) {
    def value = params[name]
    if (value == null || value.toString().trim() == '') error "Missing --${name}"
    value
}

process FINALIZE_RECOVERED_V8_PREPARATION {
    tag 'authenticated_v8_products_only'
    cpus 1
    memory '512 MB'
    time '10m'
    cache 'deep'
    errorStrategy 'terminate'
    stageInMode 'copy'
    publishDir params.run_root, mode: 'copy', overwrite: false
    input:
    path recovery
    path recovery_receipt
    output:
    path 'FINAL.json'
    path 'prepared-final'
    path 'materialized-final'
    path 'measurements-final'
    script:
    """
    set -euo pipefail
    python3 '${params.package_root}/bin/r02_structure_finalize_preparation.py' recover \
      --recovery-root '${recovery}' --recovery-receipt '${recovery_receipt}' \
      --recovery-receipt-sha256 '${params.recovery_receipt_sha256}' \
      --published-root '${params.run_root}' --output finalized
    mv finalized/FINAL.json finalized/prepared-final finalized/materialized-final \
      finalized/measurements-final .
    rmdir finalized
    """
}

workflow {
    ['package_root','recovery_root','recovery_receipt','recovery_receipt_sha256','run_root'].each { need(it) }
    FINALIZE_RECOVERED_V8_PREPARATION(
        file(params.recovery_root, checkIfExists:true),
        file(params.recovery_receipt, checkIfExists:true))
}
