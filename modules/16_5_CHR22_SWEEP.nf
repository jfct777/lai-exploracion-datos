nextflow.enable.dsl=2

process PREPARE_M165_CHR22_SWEEP {
    tag 'm165_chr22_prepare'
    label 'm165_chr22_sweep'
    cpus 1
    memory '4 GB'

    input:
    path pair_summary
    path configuration_summary
    path sample_ids
    val settings
    path adapter

    output:
    path 'prepared/L*', emit: configurations
    path 'prepared/preparation.json', emit: receipt

    script:
    def encoded = groovy.json.JsonOutput.toJson(settings).bytes.encodeBase64().toString()
    """
    set -euo pipefail
    python3 ${adapter} prepare \\
      --pair-summary '${pair_summary}' \\
      --configuration-summary '${configuration_summary}' \\
      --sample-ids '${sample_ids}' \\
      --settings-base64 '${encoded}' --output prepared
    """
}

process RUN_M165_CHR22_SWEEP {
    tag "${config_id}"
    label 'm165_chr22_sweep'
    cpus 2
    memory '8 GB'
    publishDir params.m165_chr22_sweep_results_dir, mode: 'copy', overwrite: false

    input:
    tuple val(config_id), path(configuration_dir, stageAs: 'input_configuration')
    path adapter
    path core_script

    output:
    tuple val(config_id), path("${config_id}"), emit: results

    script:
    """
    set -euo pipefail
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    export NUMBA_CACHE_DIR="\$PWD/.numba_cache"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1
    python3 ${adapter} run --configuration-dir '${configuration_dir}' \\
      --core-script '${core_script}' --output '${config_id}'
    """
}
