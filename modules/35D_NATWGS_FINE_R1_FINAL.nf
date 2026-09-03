nextflow.enable.dsl=2

process M35D_RUN_R1_FINAL_PAIR {
    tag { "m35d_r1_final" }
    publishDir { "${params.m35d_results_dir}/${params.m35d_run_id}/final_inference" },
        mode: 'copy', overwrite: false
    container params.m35d_flare2_image
    containerOptions { "--network none --user ${params.m35d_container_user}" }
    cpus 2
    memory '5 GB'
    time '2h'
    maxForks 1

    input:
    path contract
    path gate
    path token
    path screenDir
    path referenceVcf
    path targetVcf
    path coarseSampleMap
    path coarsePanelMacroMap
    path runnerPy
    path sourceCommonPy
    path balancedCommonPy
    path m35Py
    path flareCommonPy

    output:
    path 'm35d_final/m35d.direct.anc.vcf.gz', emit: direct
    path 'm35d_final/m35d.flare2.anc.vcf.gz', emit: flare2
    path 'm35d_final/m35d.final_inference_receipt.json', emit: receipt
    path 'm35d_final/m35d.excluded_monomorphic_reference_loci.tsv', emit: excluded_loci

    script:
    """
    set -euo pipefail
    mkdir -p staged_bin
    cp ${runnerPy} staged_bin/m35d_natwgs_fine_r1.py
    cp ${sourceCommonPy} staged_bin/m35c_prepare_source_comparison.py
    cp ${balancedCommonPy} staged_bin/m35b_prepare_balanced_reference.py
    cp ${m35Py} staged_bin/m35_flare2_paired.py
    cp ${flareCommonPy} staged_bin/m34_run_flare.py
    python3 staged_bin/m35d_natwgs_fine_r1.py final \
      --contract ${contract} --gate ${gate} --go-token ${token} \
      --screen-dir ${screenDir} --reference-vcf ${referenceVcf} \
      --target-vcf ${targetVcf} --coarse-sample-map ${coarseSampleMap} \
      --coarse-panel-macro-map ${coarsePanelMacroMap} \
      --flare-jar /opt/flare/flare.jar --outdir m35d_final
    """
}

process M35D_PACK_R1_PREDICTION {
    tag { "m35d_pack_${method}" }
    publishDir { "${params.m35d_results_dir}/${params.m35d_run_id}/packed/${method}" },
        mode: 'copy', overwrite: false
    container params.m35d_scoring_image
    containerOptions { "--network none --user ${params.m35d_container_user}" }
    cpus 1
    memory '4 GB'
    time '1h'
    maxForks 2

    input:
    tuple val(method), path(flareAnc)
    path geneticMap
    path parserPy
    path mosaicPy
    path bridgeCorePy
    path packerPy

    output:
    tuple val(method), path("m35d_${method}.prediction.npz"), emit: prediction

    script:
    """
    set -euo pipefail
    mkdir -p staged_bin parsed
    cp ${parserPy} staged_bin/m34_parse_flare_truth.py
    cp ${mosaicPy} staged_bin/m34_generate_mosaics.py
    cp ${bridgeCorePy} staged_bin/m33_safe_bridge_core.py
    cp ${packerPy} staged_bin/m34_pack_f0_prediction.py
    PYTHONPATH=staged_bin python3 staged_bin/m34_parse_flare_truth.py f0 \
      --flare-anc ${flareAnc} --genetic-map ${geneticMap} \
      --ancestry-order AFR,EUR,NAM --flare-id-map 0=AFR,1=EUR,2=NAM --outdir parsed
    PYTHONPATH=staged_bin python3 staged_bin/m34_pack_f0_prediction.py \
      --f0 parsed/m34_f0.npz --marker-cm parsed/marker_cM.npz \
      --output m35d_${method}.prediction.npz
    """
}

process M35D_SCORE_R1_PAIR {
    tag { "m35d_score_r1" }
    publishDir { "${params.m35d_results_dir}/${params.m35d_run_id}/score" },
        mode: 'copy', overwrite: false
    container params.m35d_scoring_image
    containerOptions { "--network none --user ${params.m35d_container_user}" }
    cpus 1
    memory '4 GB'
    time '1h'
    maxForks 1

    input:
    tuple val(directMethod), path(directPrediction)
    tuple val(flare2Method), path(flare2Prediction)
    path truth
    path canonicalMetrics
    path inferenceReceipt
    path contract
    path scorerPy
    path runnerPy
    path subsetTruthPy
    path parserPy
    path mosaicPy
    path bridgeCorePy
    path sourceCommonPy
    path balancedCommonPy
    path m35Py
    path flareCommonPy

    output:
    path 'm35d.FLARE_F0_SAME_69.metrics.json', emit: direct_metrics
    path 'm35d.FLARE2_NATWGS_FINE_SAME_69.metrics.json', emit: flare2_metrics
    path 'm35d.r1_paired_summary.json', emit: summary
    path 'm35d.r1_common_axis_truth.npz', emit: common_truth
    path 'm35d.r1_common_axis_truth.receipt.json', emit: common_truth_receipt

    script:
    """
    set -euo pipefail
    test '${directMethod}' = 'FLARE_F0_SAME_69'
    test '${flare2Method}' = 'FLARE2_NATWGS_FINE_SAME_69'
    mkdir -p staged_bin
    cp ${subsetTruthPy} staged_bin/m35d_subset_truth.py
    cp ${parserPy} staged_bin/m34_parse_flare_truth.py
    cp ${mosaicPy} staged_bin/m34_generate_mosaics.py
    cp ${bridgeCorePy} staged_bin/m33_safe_bridge_core.py
    PYTHONPATH=staged_bin python3 staged_bin/m35d_subset_truth.py \
      --direct-prediction ${directPrediction} --flare2-prediction ${flare2Prediction} \
      --truth ${truth} --output m35d.r1_common_axis_truth.npz \
      --receipt m35d.r1_common_axis_truth.receipt.json
    python3 ${scorerPy} --prediction ${directPrediction} \
      --truth m35d.r1_common_axis_truth.npz \
      --output m35d.FLARE_F0_SAME_69.metrics.json
    python3 ${scorerPy} --prediction ${flare2Prediction} \
      --truth m35d.r1_common_axis_truth.npz \
      --output m35d.FLARE2_NATWGS_FINE_SAME_69.metrics.json
    cp ${runnerPy} staged_bin/m35d_natwgs_fine_r1.py
    cp ${sourceCommonPy} staged_bin/m35c_prepare_source_comparison.py
    cp ${balancedCommonPy} staged_bin/m35b_prepare_balanced_reference.py
    cp ${m35Py} staged_bin/m35_flare2_paired.py
    cp ${flareCommonPy} staged_bin/m34_run_flare.py
    python3 staged_bin/m35d_natwgs_fine_r1.py summarize \
      --contract ${contract} \
      --direct-metrics m35d.FLARE_F0_SAME_69.metrics.json \
      --flare2-metrics m35d.FLARE2_NATWGS_FINE_SAME_69.metrics.json \
      --canonical-metrics ${canonicalMetrics} --inference-receipt ${inferenceReceipt} \
      --truth-subset-receipt m35d.r1_common_axis_truth.receipt.json \
      --output m35d.r1_paired_summary.json
    """
}
