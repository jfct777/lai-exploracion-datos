nextflow.enable.dsl=2

process M34_NAM_AGGREGATE_TRIAGE {
    tag 'm34_aggregate_triage_metrics'
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/triage_summary"
    }, mode: 'copy', overwrite: false
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_plan_cpus }
    memory { params.m34_inputs_plan_memory }
    time { params.m34_inputs_plan_time }

    input:
    path adaptiveContract
    path triagePlan
    path factorizedManifest
    path baselineMetrics
    path(candidateMetrics, stageAs: 'candidate_metrics/*')
    path(trainingReceipts, stageAs: 'training_receipts/receipt??/*')
    path(transformerBatchingReceipts, stageAs: 'transformer_batching/receipt??/*')
    path aggregatorPy
    path adaptiveSweepPy

    output:
    tuple path('m34_triage.records.json'),
          path('m34_triage.comparison.tsv'),
          path('m34_triage.aggregate.receipt.json'), emit: aggregate

    script:
    def metricArgs = candidateMetrics.collect {
        "--candidate-metric '${it}'"
    }.join(' ')
    def receiptArgs = trainingReceipts.collect {
        "--train-receipt '${it}'"
    }.join(' ')
    def batchingArgs = transformerBatchingReceipts.collect {
        "--transformer-batching-receipt '${it}'"
    }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${aggregatorPy} staged/bin/m34_aggregate_triage.py
    cp ${adaptiveSweepPy} staged/bin/m34_adaptive_sweep.py
    PYTHONPATH=staged/bin python3 staged/bin/m34_aggregate_triage.py \
      --contract ${adaptiveContract} \
      --plan ${triagePlan} \
      --factorized-manifest ${factorizedManifest} \
      --baseline-metrics ${baselineMetrics} \
      ${metricArgs} \
      ${receiptArgs} \
      ${batchingArgs} \
      --records m34_triage.records.json \
      --table m34_triage.comparison.tsv \
      --receipt m34_triage.aggregate.receipt.json
    """

    stub:
    """
    set -euo pipefail
    touch \
      m34_triage.records.json \
      m34_triage.comparison.tsv \
      m34_triage.aggregate.receipt.json
    """
}
