nextflow.enable.dsl=2

process M35_FLARE2_PAIRED_BASELINE {
    tag { "m35_flare2_paired_${params.m35_run_id}" }
    publishDir { "${params.m35_results_dir}/${params.m35_run_id}/inference" }, mode: 'copy', overwrite: false
    container params.m35_container_image
    containerOptions { "--network none --user ${params.m35_container_user}" }
    cpus { params.m35_cpus }
    memory { params.m35_memory }
    time { params.m35_time }
    maxForks params.m35_max_forks

    input:
    path contract
    path referenceVcf
    path referenceTbi
    path targetVcf
    path targetTbi
    path sampleMap
    path panelMacroMap
    path geneticMap
    path runnerPy

    output:
    path 'm35_paired/m35_paired.delta_manifest.json', emit: delta_manifest
    path 'm35_paired/m35_paired.resource_estimate.json', emit: resource_estimate
    path 'm35_paired/m35_paired.receipt.json', emit: receipt
    path 'm35_paired/m35.flare2.cluster_assignment.evidence.json', optional: true, emit: cluster_assignment_evidence
    path 'm35_paired/m35.flare060.anc.vcf.gz', optional: true, emit: flare060_prediction
    path 'm35_paired/m35.flare2.anc.vcf.gz', optional: true, emit: flare2_prediction

    script:
    def preflightFlag = params.m35_preflight_only ? '--preflight-only' : ''
    """
    set -euo pipefail
    python3 ${runnerPy} \\
      --contract ${contract} \\
      --reference-vcf ${referenceVcf} --reference-tbi ${referenceTbi} \\
      --target-vcf ${targetVcf} --target-tbi ${targetTbi} \\
      --sample-map ${sampleMap} --panel-macro-map ${panelMacroMap} --genetic-map ${geneticMap} \\
      --flare-jar /opt/flare/flare.jar \\
      --flare2-model-builder /opt/flare/m35_create_model_wrapper.py \\
      --flare2-upstream-model-builder /opt/flare/create_model_file.py \\
      --outdir m35_paired ${preflightFlag}
    """
}

process M35_PACK_FLARE_PREDICTION {
    tag { "m35_pack_${method}" }
    publishDir { "${params.m35_results_dir}/${params.m35_run_id}/predictions/${method}" }, mode: 'copy', overwrite: false
    container params.m35_scoring_image
    containerOptions { "--network none --user ${params.m35_container_user}" }
    cpus { params.m35_score_cpus }
    memory { params.m35_score_memory }
    time { params.m35_score_time }
    maxForks params.m35_score_max_forks

    input:
    tuple val(method), path(flareAnc)
    path geneticMap
    path parserPy
    path mosaicPy
    path bridgeCorePy
    path packerPy

    output:
    tuple val(method), path("m35_${method}.prediction.npz"), emit: prediction

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin f0
    cp ${parserPy} staged/bin/m34_parse_flare_truth.py
    cp ${mosaicPy} staged/bin/m34_generate_mosaics.py
    cp ${bridgeCorePy} staged/bin/m33_safe_bridge_core.py
    cp ${packerPy} staged/bin/m34_pack_f0_prediction.py
    PYTHONPATH=staged/bin python3 staged/bin/m34_parse_flare_truth.py f0 \\
      --flare-anc ${flareAnc} --genetic-map ${geneticMap} \\
      --ancestry-order AFR,EUR,NAM --flare-id-map 0=AFR,1=EUR,2=NAM --outdir f0
    PYTHONPATH=staged/bin python3 staged/bin/m34_pack_f0_prediction.py \\
      --f0 f0/m34_f0.npz --marker-cm f0/marker_cM.npz --output m35_${method}.prediction.npz
    """
}

process M35_SCORE_PAIRED {
    tag { "m35_score_paired_${params.m35_run_id}" }
    publishDir { "${params.m35_results_dir}/${params.m35_run_id}/paired_score" }, mode: 'copy', overwrite: false
    container params.m35_scoring_image
    containerOptions { "--network none --user ${params.m35_container_user}" }
    cpus { params.m35_score_cpus }
    memory { params.m35_score_memory }
    time { params.m35_score_time }
    maxForks params.m35_score_max_forks

    input:
    tuple val(directMethod), path(directPrediction)
    tuple val(flare2Method), path(flare2Prediction)
    path m34Truth
    path canonicalF0Metrics
    path scorerPy
    path verifyDirectPy
    path summaryPy

    output:
    path 'm35.flare_0_6.metrics.json', emit: flare060_metrics
    path 'm35.flare2.metrics.json', emit: flare2_metrics
    path 'm35.direct_f0.verification.json', emit: direct_f0_verification
    path 'm35.paired.summary.json', emit: summary

    script:
    """
    set -euo pipefail
    test '${directMethod}' = 'FLARE_0_6'
    test '${flare2Method}' = 'FLARE2'
    python3 ${scorerPy} --prediction ${directPrediction} --truth ${m34Truth} --output m35.flare_0_6.metrics.json
    python3 ${scorerPy} --prediction ${flare2Prediction} --truth ${m34Truth} --output m35.flare2.metrics.json
    python3 ${verifyDirectPy} --direct-metrics m35.flare_0_6.metrics.json \\
      --canonical-f0-metrics ${canonicalF0Metrics} --output m35.direct_f0.verification.json
    python3 ${summaryPy} --flare-0-6-metrics m35.flare_0_6.metrics.json \\
      --flare2-metrics m35.flare2.metrics.json --direct-f0-gate m35.direct_f0.verification.json \\
      --output m35.paired.summary.json
    """
}
