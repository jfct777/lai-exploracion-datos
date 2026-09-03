nextflow.enable.dsl=2

process M35C_PREPARE_SOURCE_REFERENCES {
    tag { "m35c_prepare_s${selectionSeed}" }
    publishDir { "${params.m35c_results_dir}/${params.m35c_run_id}/reference/s${selectionSeed}" },
        mode: 'copy', overwrite: false
    container params.m35c_tabix_image
    containerOptions { "--network none --user ${params.m35c_container_user}" }
    cpus 1
    memory '3 GB'
    time '45m'
    maxForks params.m35c_prepare_max_forks

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
    path m34DonorAudit
    path m34MosaicReceipt
    path preparePy
    path m35bPreparePy

    output:
    tuple val(selectionSeed),
        path("m35c_s${selectionSeed}.external_nam.ref.vcf.gz"),
        path("m35c_s${selectionSeed}.external_nam.ref.vcf.gz.tbi"),
        path("m35c_s${selectionSeed}.external_nam.coarse.sample_panel.tsv"),
        path("m35c_s${selectionSeed}.external_nam.coarse.panel_macro.tsv"),
        path("m35c_s${selectionSeed}.external_nam.fine.sample_panel.tsv"),
        path("m35c_s${selectionSeed}.external_nam.fine.panel_macro.tsv"),
        path("m35c_s${selectionSeed}.external_nam.selected_samples.txt"),
        path("m35c_s${selectionSeed}.natwgs.ref.vcf.gz"),
        path("m35c_s${selectionSeed}.natwgs.ref.vcf.gz.tbi"),
        path("m35c_s${selectionSeed}.natwgs.coarse.sample_panel.tsv"),
        path("m35c_s${selectionSeed}.natwgs.coarse.panel_macro.tsv"),
        path("m35c_s${selectionSeed}.natwgs.fine.sample_panel.tsv"),
        path("m35c_s${selectionSeed}.natwgs.fine.panel_macro.tsv"),
        path("m35c_s${selectionSeed}.natwgs.selected_samples.txt"),
        path("m35c_s${selectionSeed}.prepare_receipt.json"),
        emit: source_references

    script:
    """
    set -euo pipefail
    mkdir -p staged_bin
    cp ${preparePy} staged_bin/m35c_prepare_source_comparison.py
    cp ${m35bPreparePy} staged_bin/m35b_prepare_balanced_reference.py
    python3 staged_bin/m35c_prepare_source_comparison.py \
      --contract ${contract} --roles ${roles} \
      --phased-scaffold-vcf ${phasedScaffoldVcf} \
      --target-vcf ${targetVcf} --target-tbi ${targetTbi} \
      --m27d-manifest ${m27dManifest} --m27d-strata ${m27dStrata} \
      --m27d-training-set ${m27dTrainingSet} --m27d-related-pairs ${m27dRelatedPairs} \
      --m34-donor-audit ${m34DonorAudit} --m34-mosaic-receipt ${m34MosaicReceipt} \
      --selection-seed ${selectionSeed} --output-prefix m35c_s${selectionSeed}
    bgzip -@ 1 -c m35c_s${selectionSeed}.external_nam.ref.vcf \
      > m35c_s${selectionSeed}.external_nam.ref.vcf.gz
    tabix -f -p vcf m35c_s${selectionSeed}.external_nam.ref.vcf.gz
    rm m35c_s${selectionSeed}.external_nam.ref.vcf
    bgzip -@ 1 -c m35c_s${selectionSeed}.natwgs.ref.vcf \
      > m35c_s${selectionSeed}.natwgs.ref.vcf.gz
    tabix -f -p vcf m35c_s${selectionSeed}.natwgs.ref.vcf.gz
    rm m35c_s${selectionSeed}.natwgs.ref.vcf
    """
}

process M35C_CLUSTER_SCREEN {
    tag { "m35c_${arm}_${granularity}_s${selectionSeed}_g${gmmSeed}" }
    publishDir { "${params.m35c_results_dir}/${params.m35c_run_id}/cluster_screen/${arm}/${granularity}/s${selectionSeed}_g${gmmSeed}" },
        mode: 'copy', overwrite: false
    container params.m35c_flare2_image
    containerOptions { "--network none --user ${params.m35c_container_user}" }
    cpus params.m35c_screen_cpus
    memory params.m35c_screen_memory
    time params.m35c_screen_time
    maxForks params.m35c_screen_max_forks

    input:
    tuple val(arm), val(selectionSeed), val(granularity), val(gmmSeed),
        path(referenceVcf), path(referenceTbi), path(sampleMap), path(panelMacroMap),
        path(prepareReceipt)
    path contract
    path targetVcf
    path targetTbi
    path geneticMap
    path screenPy
    path preparePy
    path m35bPreparePy
    path m35Py
    path flareCommonPy
    path modelWrapperPy

    output:
    tuple val(arm), val(selectionSeed), val(granularity), val(gmmSeed),
        path("m35c_screen_${arm.toLowerCase()}_s${selectionSeed}_${granularity}_g${gmmSeed}"),
        emit: screen

    script:
    """
    set -euo pipefail
    mkdir -p staged_bin
    cp ${screenPy} staged_bin/m35c_cluster_screen.py
    cp ${preparePy} staged_bin/m35c_prepare_source_comparison.py
    cp ${m35bPreparePy} staged_bin/m35b_prepare_balanced_reference.py
    cp ${m35Py} staged_bin/m35_flare2_paired.py
    cp ${flareCommonPy} staged_bin/m34_run_flare.py
    cp ${modelWrapperPy} staged_bin/m35b_create_model_wrapper.py
    python3 staged_bin/m35c_cluster_screen.py \
      --contract ${contract} --arm ${arm} \
      --selection-seed ${selectionSeed} --gmm-seed ${gmmSeed} --granularity ${granularity} \
      --reference-vcf ${referenceVcf} --reference-tbi ${referenceTbi} \
      --target-vcf ${targetVcf} --target-tbi ${targetTbi} \
      --sample-map ${sampleMap} --panel-macro-map ${panelMacroMap} \
      --prepare-receipt ${prepareReceipt} --genetic-map ${geneticMap} \
      --flare-jar /opt/flare/flare.jar \
      --model-wrapper staged_bin/m35b_create_model_wrapper.py \
      --upstream-builder /opt/flare/create_model_file.py \
      --outdir m35c_screen_${arm.toLowerCase()}_s${selectionSeed}_${granularity}_g${gmmSeed}
    """
}

process M35C_AGGREGATE_SOURCE_GATE {
    tag { "m35c_gate_${params.m35c_run_id}" }
    publishDir { "${params.m35c_results_dir}/${params.m35c_run_id}/gate" },
        mode: 'copy', overwrite: false
    container params.m35c_scoring_image
    containerOptions { "--network none --user ${params.m35c_container_user}" }
    cpus 1
    memory '2 GB'
    time '30m'

    input:
    path screenDirs
    path contract
    path aggregatorPy

    output:
    path 'm35c.source_cluster_gate.json', emit: gate_receipt
    path 'm35c.go_post_gate.token.json', optional: true, emit: go_token

    script:
    def screenArgs = screenDirs.collect { "--screen-dir ${it}" }.join(' ')
    """
    set -euo pipefail
    python3 ${aggregatorPy} --contract ${contract} ${screenArgs} \
      --output m35c.source_cluster_gate.json --go-token m35c.go_post_gate.token.json
    """
}
