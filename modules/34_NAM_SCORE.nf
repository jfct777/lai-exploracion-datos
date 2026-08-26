nextflow.enable.dsl=2

process M34_NAM_SCORE_VALID {
    tag { "m34_score_${family}_${configId}_${arm}" }
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/metrics/${family}/${configId}/${arm}"
    }, mode: 'copy', overwrite: false
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_score_cpus }
    memory { params.m34_inputs_score_memory }
    time { params.m34_inputs_score_time }
    maxForks params.m34_inputs_score_max_forks

    input:
    tuple val(family), val(configId), val(arm), path(prediction)
    path validTruth
    path scorerPy

    output:
    tuple val(family), val(configId), val(arm),
          path("${family}.${configId}.${arm}.metrics.json"), emit: metrics

    script:
    """
    set -euo pipefail
    python3 ${scorerPy} \
      --prediction ${prediction} \
      --truth ${validTruth} \
      --output ${family}.${configId}.${arm}.metrics.json
    """

    stub:
    """
    set -euo pipefail
    touch ${family}.${configId}.${arm}.metrics.json
    """
}
