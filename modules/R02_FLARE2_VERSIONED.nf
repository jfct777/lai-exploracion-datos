nextflow.enable.dsl=2

/*
 * FLARE2 is a three-stage procedure using the FLARE 0.6.0 JAR.  Nextflow
 * caches each completed stage independently.  The model stage stops before
 * inference if latent labels cannot be identified from fixed reference panels.
 */
process R02_FLARE2_PANEL_PROBABILITIES {
    tag { "flare2_panel_probabilities_chr${params.r02_flare2_chromosome}" }
    publishDir { "${params.r02_flare2_results}/panel_probabilities" }, mode: 'copy', overwrite: false
    container params.r02_flare2_container
    cpus { params.r02_flare2_cpus }
    memory { params.r02_flare2_memory }
    time { params.r02_flare2_panel_time }
    maxForks params.r02_flare2_max_forks

    input:
    path config
    path referenceVcf
    path referenceIndex
    path targetVcf
    path targetIndex
    path directPanelMap
    path finePanelMap
    path panelToAncestry
    path geneticMap
    path excludeMarkers
    path flareJar
    path officialBuilder
    path seededBuilder
    path roles
    path launcher
    path modelIo
    path m35Runner
    path m34Runner

    output:
    path 'flare2.plan.json', emit: plan
    path 'flare2.panels', emit: panels
    path 'flare2.panel.receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p runtime/bin
    cp -L ${launcher} runtime/bin/r02_flare_versioned.py
    cp -L ${modelIo} runtime/bin/r02_flare2_model_io.py
    cp -L ${m35Runner} runtime/bin/m35_flare2_paired.py
    cp -L ${m34Runner} runtime/bin/m34_run_flare.py
    cp -L ${seededBuilder} runtime/bin/r02_flare2_seeded_create_model.py
    PYTHONPATH=runtime/bin python3 runtime/bin/r02_flare_versioned.py prepare --config ${config} \
      --reference-vcf ${referenceVcf} --reference-index ${referenceIndex} \
      --target-vcf ${targetVcf} --target-index ${targetIndex} \
      --direct-panel-map ${directPanelMap} --fine-panel-map ${finePanelMap} \
      --panel-to-ancestry ${panelToAncestry} --genetic-map ${geneticMap} \
      --excludemarkers ${excludeMarkers} --flare-jar ${flareJar} \
      --official-builder ${officialBuilder} --seeded-builder runtime/bin/r02_flare2_seeded_create_model.py \
      --roles ${roles} --output flare2.plan.json
    PYTHONPATH=runtime/bin python3 runtime/bin/r02_flare_versioned.py panel-probabilities --plan flare2.plan.json \
      --flare-jar ${flareJar} --reference-vcf ${referenceVcf} --reference-index ${referenceIndex} \
      --target-vcf ${targetVcf} --target-index ${targetIndex} \
      --fine-panel-map ${finePanelMap} --genetic-map ${geneticMap} \
      --excludemarkers ${excludeMarkers} --output-prefix flare2 \
      --receipt flare2.panel.receipt.json
    """
}

process R02_FLARE2_BUILD_MODEL {
    tag { "flare2_build_model_chr${params.r02_flare2_chromosome}" }
    publishDir { "${params.r02_flare2_results}/model" }, mode: 'copy', overwrite: false
    container params.r02_flare2_container
    cpus 1
    memory { params.r02_flare2_model_memory }
    time { params.r02_flare2_model_time }

    input:
    path plan
    path panels
    path panelReceipt
    path officialBuilder
    path seededBuilder
    path launcher
    path modelIo
    path m35Runner
    path m34Runner

    output:
    path 'flare2.model', emit: model
    path 'flare2.log', emit: log
    path 'flare2.mapping.json', emit: mapping
    path 'flare2.model.receipt.json', emit: receipt
    path 'flare2.seeded_builder.receipt.json', emit: builder_receipt

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    mkdir -p runtime/bin
    cp -L ${launcher} runtime/bin/r02_flare_versioned.py
    cp -L ${modelIo} runtime/bin/r02_flare2_model_io.py
    cp -L ${m35Runner} runtime/bin/m35_flare2_paired.py
    cp -L ${m34Runner} runtime/bin/m34_run_flare.py
    cp -L ${seededBuilder} runtime/bin/r02_flare2_seeded_create_model.py
    PYTHONPATH=runtime/bin python3 runtime/bin/r02_flare_versioned.py build-model --plan ${plan} --panels ${panels} \
      --panel-receipt ${panelReceipt} --official-builder ${officialBuilder} \
      --seeded-builder runtime/bin/r02_flare2_seeded_create_model.py --output-prefix flare2 \
      --mapping flare2.mapping.json --receipt flare2.model.receipt.json
    """
}

process R02_FLARE2_CLUSTERED_INFERENCE {
    tag { "flare2_clustered_inference_chr${params.r02_flare2_chromosome}" }
    publishDir { "${params.r02_flare2_results}/inference" }, mode: 'copy', overwrite: false
    container params.r02_flare2_container
    cpus { params.r02_flare2_cpus }
    memory { params.r02_flare2_memory }
    time { params.r02_flare2_inference_time }
    maxForks params.r02_flare2_max_forks

    input:
    path plan
    path model
    path modelReceipt
    path mapping
    path referenceVcf
    path referenceIndex
    path targetVcf
    path targetIndex
    path finePanelMap
    path geneticMap
    path excludeMarkers
    path flareJar
    path launcher
    path modelIo
    path m35Runner
    path m34Runner

    output:
    path 'flare2.raw.anc.vcf.gz', emit: raw_prediction
    path 'flare2.anc.vcf.gz', emit: prediction
    path 'flare2.final.model', emit: final_model
    path 'flare2.final.log', emit: final_log
    path 'flare2.final.mapping.json', emit: final_mapping
    path 'flare2.inference.receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p runtime/bin
    cp -L ${launcher} runtime/bin/r02_flare_versioned.py
    cp -L ${modelIo} runtime/bin/r02_flare2_model_io.py
    cp -L ${m35Runner} runtime/bin/m35_flare2_paired.py
    cp -L ${m34Runner} runtime/bin/m34_run_flare.py
    PYTHONPATH=runtime/bin python3 runtime/bin/r02_flare_versioned.py infer --plan ${plan} --model ${model} \
      --model-receipt ${modelReceipt} --mapping ${mapping} --flare-jar ${flareJar} \
      --reference-vcf ${referenceVcf} --reference-index ${referenceIndex} \
      --target-vcf ${targetVcf} --target-index ${targetIndex} \
      --fine-panel-map ${finePanelMap} --genetic-map ${geneticMap} \
      --excludemarkers ${excludeMarkers} --output-prefix flare2.raw \
      --canonical-output flare2.anc.vcf.gz --final-model flare2.final.model \
      --final-log flare2.final.log --final-mapping flare2.final.mapping.json \
      --receipt flare2.inference.receipt.json
    """
}

/* Independent direct reference; it uses the same plan, VCF, marker exclusions,
 * map, seed, threads and JAR.  It is outside the three-stage FLARE2 chain. */
process R02_FLARE_0_6_0_DIRECT_REFERENCE {
    tag { "flare060_direct_chr${params.r02_flare2_chromosome}" }
    publishDir { "${params.r02_flare2_results}/direct" }, mode: 'copy', overwrite: false
    container params.r02_flare2_container
    cpus { params.r02_flare2_cpus }
    memory { params.r02_flare2_memory }
    time { params.r02_flare2_inference_time }

    input:
    path plan
    path referenceVcf
    path referenceIndex
    path targetVcf
    path targetIndex
    path directPanelMap
    path geneticMap
    path excludeMarkers
    path flareJar
    path launcher
    path modelIo
    path m35Runner
    path m34Runner

    output:
    path 'flare060.anc.vcf.gz', emit: prediction
    path 'flare060.receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p runtime/bin
    cp -L ${launcher} runtime/bin/r02_flare_versioned.py
    cp -L ${modelIo} runtime/bin/r02_flare2_model_io.py
    cp -L ${m35Runner} runtime/bin/m35_flare2_paired.py
    cp -L ${m34Runner} runtime/bin/m34_run_flare.py
    PYTHONPATH=runtime/bin python3 runtime/bin/r02_flare_versioned.py direct --plan ${plan} --flare-jar ${flareJar} \
      --reference-vcf ${referenceVcf} --reference-index ${referenceIndex} \
      --target-vcf ${targetVcf} --target-index ${targetIndex} \
      --direct-panel-map ${directPanelMap} --genetic-map ${geneticMap} \
      --excludemarkers ${excludeMarkers} --output-prefix flare060 \
      --receipt flare060.receipt.json
    """
}
