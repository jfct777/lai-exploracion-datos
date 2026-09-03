nextflow.enable.dsl=2

process M35D_PREPARE_R1_NATWGS_REFERENCE {
    tag { "m35d_prepare_s${selectionSeed}" }
    publishDir { "${params.m35d_results_dir}/${params.m35d_run_id}/reference/s${selectionSeed}" },
        mode: 'copy', overwrite: false
    container params.m35d_tabix_image
    containerOptions { "--network none --user ${params.m35d_container_user}" }
    cpus 1
    memory '3 GB'
    time '45m'
    maxForks params.m35d_prepare_max_forks

    input:
    val selectionSeed
    path contract
    path roles
    path phasedScaffoldVcf
    path targetVcf
    path targetTbi
    path m27dManifest
    path m27dStrata
    path m27dTrainingSet
    path m27dRelatedPairs
    path r1DonorAudit
    path r1MosaicReceipt
    path runnerPy
    path sourceCommonPy
    path balancedCommonPy
    path m35Py
    path flareCommonPy

    output:
    tuple val(selectionSeed),
        path("m35d_s${selectionSeed}.natwgs.ref.vcf.gz"),
        path("m35d_s${selectionSeed}.natwgs.ref.vcf.gz.tbi"),
        path("m35d_s${selectionSeed}.natwgs.coarse.sample_panel.tsv"),
        path("m35d_s${selectionSeed}.natwgs.coarse.panel_macro.tsv"),
        path("m35d_s${selectionSeed}.natwgs.fine.sample_panel.tsv"),
        path("m35d_s${selectionSeed}.natwgs.fine.panel_macro.tsv"),
        path("m35d_s${selectionSeed}.natwgs.selected_samples.txt"),
        path("m35d_s${selectionSeed}.prepare_receipt.json"),
        emit: reference

    script:
    """
    set -euo pipefail
    mkdir -p staged_bin
    cp ${runnerPy} staged_bin/m35d_natwgs_fine_r1.py
    cp ${sourceCommonPy} staged_bin/m35c_prepare_source_comparison.py
    cp ${balancedCommonPy} staged_bin/m35b_prepare_balanced_reference.py
    cp ${m35Py} staged_bin/m35_flare2_paired.py
    cp ${flareCommonPy} staged_bin/m34_run_flare.py
    python3 staged_bin/m35d_natwgs_fine_r1.py prepare \
      --contract ${contract} --roles ${roles} \
      --phased-scaffold-vcf ${phasedScaffoldVcf} \
      --target-vcf ${targetVcf} --target-tbi ${targetTbi} \
      --m27d-manifest ${m27dManifest} --m27d-strata ${m27dStrata} \
      --m27d-training-set ${m27dTrainingSet} --m27d-related-pairs ${m27dRelatedPairs} \
      --m34-r1-donor-audit ${r1DonorAudit} \
      --m34-r1-mosaic-receipt ${r1MosaicReceipt} \
      --selection-seed ${selectionSeed} --output-prefix m35d_s${selectionSeed}
    bgzip -@ 1 -c m35d_s${selectionSeed}.natwgs.ref.vcf \
      > m35d_s${selectionSeed}.natwgs.ref.vcf.gz
    tabix -f -p vcf m35d_s${selectionSeed}.natwgs.ref.vcf.gz
    rm m35d_s${selectionSeed}.natwgs.ref.vcf m35d_s${selectionSeed}.external_nam.ref.vcf
    """
}

process M35D_R1_CLUSTER_SCREEN {
    tag { "m35d_${granularity}_s${selectionSeed}_g${gmmSeed}" }
    publishDir { "${params.m35d_results_dir}/${params.m35d_run_id}/cluster_screen/${granularity}/s${selectionSeed}_g${gmmSeed}" },
        mode: 'copy', overwrite: false
    container params.m35d_flare2_image
    containerOptions { "--network none --user ${params.m35d_container_user}" }
    cpus params.m35d_screen_cpus
    memory params.m35d_screen_memory
    time params.m35d_screen_time
    maxForks params.m35d_screen_max_forks

    input:
    tuple val(selectionSeed), val(granularity), val(gmmSeed),
        path(referenceVcf), path(referenceTbi), path(sampleMap), path(panelMacroMap),
        path(prepareReceipt)
    path contract
    path targetVcf
    path targetTbi
    path geneticMap
    path runnerPy
    path sourceCommonPy
    path balancedCommonPy
    path m35Py
    path flareCommonPy
    path modelWrapperPy

    output:
    tuple val(selectionSeed), val(granularity), val(gmmSeed),
        path("m35d_screen_s${selectionSeed}_${granularity}_g${gmmSeed}"), emit: screen

    script:
    """
    set -euo pipefail
    mkdir -p staged_bin
    cp ${runnerPy} staged_bin/m35d_natwgs_fine_r1.py
    cp ${sourceCommonPy} staged_bin/m35c_prepare_source_comparison.py
    cp ${balancedCommonPy} staged_bin/m35b_prepare_balanced_reference.py
    cp ${m35Py} staged_bin/m35_flare2_paired.py
    cp ${flareCommonPy} staged_bin/m34_run_flare.py
    cp ${modelWrapperPy} staged_bin/m35b_create_model_wrapper.py
    python3 staged_bin/m35d_natwgs_fine_r1.py screen \
      --contract ${contract} --selection-seed ${selectionSeed} \
      --gmm-seed ${gmmSeed} --granularity ${granularity} \
      --reference-vcf ${referenceVcf} --reference-tbi ${referenceTbi} \
      --target-vcf ${targetVcf} --target-tbi ${targetTbi} \
      --sample-map ${sampleMap} --panel-macro-map ${panelMacroMap} \
      --prepare-receipt ${prepareReceipt} --genetic-map ${geneticMap} \
      --flare-jar /opt/flare/flare.jar \
      --model-wrapper staged_bin/m35b_create_model_wrapper.py \
      --upstream-builder /opt/flare/create_model_file.py \
      --outdir m35d_screen_s${selectionSeed}_${granularity}_g${gmmSeed}
    """
}

process M35D_AGGREGATE_R1_GATE {
    tag { "m35d_gate_${params.m35d_run_id}" }
    publishDir { "${params.m35d_results_dir}/${params.m35d_run_id}/gate" },
        mode: 'copy', overwrite: false
    container params.m35d_scoring_image
    containerOptions { "--network none --user ${params.m35d_container_user}" }
    cpus 1
    memory '2 GB'
    time '30m'

    input:
    path screenDirs
    path contract
    path runnerPy
    path sourceCommonPy
    path balancedCommonPy
    path m35Py
    path flareCommonPy

    output:
    path 'm35d.r1_cluster_gate.json', emit: gate
    path 'm35d.go_r1_final.token.json', optional: true, emit: token

    script:
    def screenArgs = screenDirs.collect { "--screen-dir ${it}" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged_bin
    cp ${runnerPy} staged_bin/m35d_natwgs_fine_r1.py
    cp ${sourceCommonPy} staged_bin/m35c_prepare_source_comparison.py
    cp ${balancedCommonPy} staged_bin/m35b_prepare_balanced_reference.py
    cp ${m35Py} staged_bin/m35_flare2_paired.py
    cp ${flareCommonPy} staged_bin/m34_run_flare.py
    python3 staged_bin/m35d_natwgs_fine_r1.py aggregate --contract ${contract} \
      ${screenArgs} --output m35d.r1_cluster_gate.json \
      --go-token m35d.go_r1_final.token.json
    """
}
