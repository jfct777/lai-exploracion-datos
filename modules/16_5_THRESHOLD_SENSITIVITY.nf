nextflow.enable.dsl=2

process WRITE_M16_5_THRESHOLD_SENSITIVITY_PROVENANCE {
    tag "m16_5_threshold_sensitivity_provenance"
    publishDir params.m16_5_sensitivity_results_dir, mode: 'copy', overwrite: false
    container params.m16_5_sensitivity_container_image
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    val provenance_b64

    output:
    path "run_provenance.json", emit: provenance

    script:
    """
    set -euo pipefail
    printf '%s' '${provenance_b64}' | base64 -d > run_provenance.json
    """
}


process RUN_M16_5_THRESHOLD_SENSITIVITY {
    tag "m16_5_threshold_sensitivity"
    publishDir params.m16_5_sensitivity_results_dir, mode: 'copy', overwrite: false
    container params.m16_5_sensitivity_container_image
    cpus params.m16_5_sensitivity_cpus
    memory params.m16_5_sensitivity_memory
    time params.m16_5_sensitivity_time

    input:
    path pair_summary
    path individual_summary
    path global_summary
    path metadata
    path burden_table
    path pcrelate
    path preregistration
    path evaluator_py
    path core_py
    path run_provenance

    output:
    path "configuration_summary.tsv", emit: configuration_summary
    path "resolution_summary.tsv", emit: resolution_summary
    path "assignments.tsv.gz", emit: assignments
    path "neighbor_comparisons.tsv", emit: neighbor_comparisons
    path "pcrelate_community_concentration.tsv", emit: pcrelate_community_concentration
    path "identity_control.json", emit: identity_control
    path "decision.json", emit: decision
    path "artifact_inventory.tsv", emit: artifact_inventory
    path "plots", emit: plots
    path "m16_5_threshold_sensitivity.log", emit: log

    script:
    """
    set -euo pipefail
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    export NUMBA_CACHE_DIR="\$PWD/.numba_cache"
    export HOME="\$PWD/.home"
    mkdir -p "\$MPLCONFIGDIR" "\$NUMBA_CACHE_DIR" "\$HOME"

    python3 ${evaluator_py} \
      --pair-summary ${pair_summary} \
      --individual-summary ${individual_summary} \
      --global-summary ${global_summary} \
      --metadata ${metadata} \
      --burden-table ${burden_table} \
      --pcrelate ${pcrelate} \
      --preregistration ${preregistration} \
      --core-script ${core_py} \
      --outdir . \
      > m16_5_threshold_sensitivity.log 2>&1
    """
}
