nextflow.enable.dsl=2

process M35B_PREPARE_BALANCED_REFERENCE {
    tag { "m35b_balance_s${selectionSeed}" }
    publishDir { "${params.m35b_results_dir}/${params.m35b_run_id}/balanced_reference/s${selectionSeed}" },
        mode: 'copy', overwrite: false
    container params.m35b_tabix_image
    containerOptions { "--network none --user ${params.m35b_container_user}" }
    cpus 1
    memory '2 GB'
    time '30m'
    maxForks params.m35b_prepare_max_forks

    input:
    val selectionSeed
    path roles
    path referenceVcf
    path referenceTbi
    path targetVcf
    path targetTbi
    path selectorPy

    output:
    tuple val(selectionSeed),
        path("m35b_s${selectionSeed}.ref.vcf.gz"),
        path("m35b_s${selectionSeed}.ref.vcf.gz.tbi"),
        path("m35b_s${selectionSeed}.coarse.sample_panel.tsv"),
        path("m35b_s${selectionSeed}.coarse.panel_macro.tsv"),
        path("m35b_s${selectionSeed}.fine.sample_panel.tsv"),
        path("m35b_s${selectionSeed}.fine.panel_macro.tsv"),
        path("m35b_s${selectionSeed}.prepare_receipt.json"),
        emit: balanced_reference

    script:
    """
    set -euo pipefail
    python3 ${selectorPy} \
      --roles ${roles} \
      --reference-vcf ${referenceVcf} --reference-tbi ${referenceTbi} \
      --target-vcf ${targetVcf} --target-tbi ${targetTbi} \
      --selection-seed ${selectionSeed} --per-ancestry 25 \
      --expected-markers 42986 --chromosome 22 \
      --output-prefix m35b_s${selectionSeed}
    bgzip -@ 1 -c m35b_s${selectionSeed}.ref.vcf > m35b_s${selectionSeed}.ref.vcf.gz
    tabix -f -p vcf m35b_s${selectionSeed}.ref.vcf.gz
    rm m35b_s${selectionSeed}.ref.vcf
    """
}

process M35B_CLUSTER_SCREEN {
    tag { "m35b_${granularity}_s${selectionSeed}_g${gmmSeed}" }
    publishDir { "${params.m35b_results_dir}/${params.m35b_run_id}/cluster_screen/${granularity}/s${selectionSeed}_g${gmmSeed}" },
        mode: 'copy', overwrite: false
    container params.m35b_flare2_image
    containerOptions { "--network none --user ${params.m35b_container_user}" }
    cpus params.m35b_screen_cpus
    memory params.m35b_screen_memory
    time params.m35b_screen_time
    maxForks params.m35b_screen_max_forks

    input:
    tuple val(selectionSeed), val(granularity), val(gmmSeed),
        path(referenceVcf), path(referenceTbi), path(sampleMap), path(panelMacroMap),
        path(prepareReceipt)
    path contract
    path targetVcf
    path targetTbi
    path geneticMap
    path screenPy
    path m35Py
    path flareCommonPy
    path modelWrapperPy

    output:
    tuple val(selectionSeed), val(granularity), val(gmmSeed),
        path("m35b_screen_s${selectionSeed}_${granularity}_g${gmmSeed}"),
        emit: screen

    script:
    """
    set -euo pipefail
    mkdir -p staged_bin
    cp ${screenPy} staged_bin/m35b_cluster_screen.py
    cp ${m35Py} staged_bin/m35_flare2_paired.py
    cp ${flareCommonPy} staged_bin/m34_run_flare.py
    cp ${modelWrapperPy} staged_bin/m35b_create_model_wrapper.py
    python3 staged_bin/m35b_cluster_screen.py \
      --contract ${contract} \
      --selection-seed ${selectionSeed} --gmm-seed ${gmmSeed} --granularity ${granularity} \
      --reference-vcf ${referenceVcf} --reference-tbi ${referenceTbi} \
      --target-vcf ${targetVcf} --target-tbi ${targetTbi} \
      --sample-map ${sampleMap} --panel-macro-map ${panelMacroMap} \
      --prepare-receipt ${prepareReceipt} --genetic-map ${geneticMap} \
      --flare-jar /opt/flare/flare.jar \
      --model-wrapper staged_bin/m35b_create_model_wrapper.py \
      --upstream-builder /opt/flare/create_model_file.py \
      --outdir m35b_screen_s${selectionSeed}_${granularity}_g${gmmSeed}
    """
}

process M35B_AGGREGATE_CLUSTER_GATE {
    tag { "m35b_gate_${params.m35b_run_id}" }
    publishDir { "${params.m35b_results_dir}/${params.m35b_run_id}/gate" },
        mode: 'copy', overwrite: false
    container params.m35b_scoring_image
    containerOptions { "--network none --user ${params.m35b_container_user}" }
    cpus 1
    memory '2 GB'
    time '30m'

    input:
    path screenDirs
    path contract
    path aggregatorPy

    output:
    path 'm35b.cluster_gate.json', emit: gate_receipt
    path 'm35b.go_final.token.json', optional: true, emit: go_token

    script:
    def screenArgs = screenDirs.collect { "--screen-dir ${it}" }.join(' ')
    """
    set -euo pipefail
    python3 ${aggregatorPy} --contract ${contract} ${screenArgs} \
      --output m35b.cluster_gate.json --go-token m35b.go_final.token.json
    """
}

process M35B_RUN_PREASSIGNED_FINAL {
    tag { "m35b_final_s${selectionSeed}_g${gmmSeed}" }
    publishDir { "${params.m35b_results_dir}/${params.m35b_run_id}/final_inference" },
        mode: 'copy', overwrite: false
    container params.m35b_flare2_image
    containerOptions { "--network none --user ${params.m35b_container_user}" }
    cpus params.m35b_final_cpus
    memory params.m35b_final_memory
    time params.m35b_final_time
    maxForks 1

    input:
    tuple val(selectionSeed), val(granularity), val(gmmSeed), path(screenDir),
        path(referenceVcf), path(referenceTbi), path(prepareReceipt)
    path goToken
    path gateReceipt
    path contract
    path targetVcf
    path targetTbi
    path finalPy
    path screenPy
    path m35Py
    path flareCommonPy

    output:
    path 'm35b_final/m35b.direct.anc.vcf.gz', emit: direct_prediction
    path 'm35b_final/m35b.flare2.anc.vcf.gz', emit: flare2_prediction
    path 'm35b_final/m35b.final_inference_receipt.json', emit: inference_receipt

    script:
    """
    set -euo pipefail
    test '${granularity}' = 'coarse'
    mkdir -p staged_bin
    cp ${finalPy} staged_bin/m35b_run_final_pair.py
    cp ${screenPy} staged_bin/m35b_cluster_screen.py
    cp ${m35Py} staged_bin/m35_flare2_paired.py
    cp ${flareCommonPy} staged_bin/m34_run_flare.py
    python3 staged_bin/m35b_run_final_pair.py \
      --contract ${contract} --go-token ${goToken} --gate-receipt ${gateReceipt} \
      --screen-dir ${screenDir} --reference-vcf ${referenceVcf} \
      --target-vcf ${targetVcf} --flare-jar /opt/flare/flare.jar \
      --outdir m35b_final
    """
}

process M35B_PACK_PREDICTION {
    tag { "m35b_pack_${method}" }
    publishDir { "${params.m35b_results_dir}/${params.m35b_run_id}/packed/${method}" },
        mode: 'copy', overwrite: false
    container params.m35b_scoring_image
    containerOptions { "--network none --user ${params.m35b_container_user}" }
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
    tuple val(method), path("m35b_${method}.prediction.npz"), emit: prediction

    script:
    """
    set -euo pipefail
    mkdir -p staged_bin f0
    cp ${parserPy} staged_bin/m34_parse_flare_truth.py
    cp ${mosaicPy} staged_bin/m34_generate_mosaics.py
    cp ${bridgeCorePy} staged_bin/m33_safe_bridge_core.py
    cp ${packerPy} staged_bin/m34_pack_f0_prediction.py
    PYTHONPATH=staged_bin python3 staged_bin/m34_parse_flare_truth.py f0 \
      --flare-anc ${flareAnc} --genetic-map ${geneticMap} \
      --ancestry-order AFR,EUR,NAM --flare-id-map 0=AFR,1=EUR,2=NAM --outdir f0
    PYTHONPATH=staged_bin python3 staged_bin/m34_pack_f0_prediction.py \
      --f0 f0/m34_f0.npz --marker-cm f0/marker_cM.npz \
      --output m35b_${method}.prediction.npz
    """
}

process M35B_SCORE_PAIRED {
    tag { "m35b_score_${params.m35b_run_id}" }
    publishDir { "${params.m35b_results_dir}/${params.m35b_run_id}/score" },
        mode: 'copy', overwrite: false
    container params.m35b_scoring_image
    containerOptions { "--network none --user ${params.m35b_container_user}" }
    cpus 1
    memory '4 GB'
    time '1h'
    maxForks 1

    input:
    tuple val(directMethod), path(directPrediction)
    tuple val(flare2Method), path(flare2Prediction)
    path truth
    path canonicalF0Metrics
    path inferenceReceipt
    path scorerPy
    path summaryPy

    output:
    path 'm35b.FLARE_0_6_BALANCED.metrics.json', emit: direct_metrics
    path 'm35b.FLARE2_BALANCED.metrics.json', emit: flare2_metrics
    path 'm35b.paired_summary.json', emit: summary

    script:
    """
    set -euo pipefail
    test '${directMethod}' = 'FLARE_0_6_BALANCED'
    test '${flare2Method}' = 'FLARE2_BALANCED'
    python3 ${scorerPy} --prediction ${directPrediction} --truth ${truth} \
      --output m35b.FLARE_0_6_BALANCED.metrics.json
    python3 ${scorerPy} --prediction ${flare2Prediction} --truth ${truth} \
      --output m35b.FLARE2_BALANCED.metrics.json
    python3 ${summaryPy} \
      --direct-metrics m35b.FLARE_0_6_BALANCED.metrics.json \
      --flare2-metrics m35b.FLARE2_BALANCED.metrics.json \
      --canonical-f0-metrics ${canonicalF0Metrics} \
      --inference-receipt ${inferenceReceipt} \
      --output m35b.paired_summary.json
    """
}
