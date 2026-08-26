nextflow.enable.dsl=2

process M34_NAM_BUILD_FACTORIZED_MANIFEST {
    tag 'm34_factorized_manifest_R0'
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/manifest"
    }, mode: 'copy', overwrite: false,
       saveAs: { filename -> filename == 'factor_bundle' ? null : filename }
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_manifest_cpus }
    memory { params.m34_inputs_manifest_memory }
    time { params.m34_inputs_manifest_time }

    input:
    tuple val(fitSplit),
          path(fitSelected, stageAs: 'factor_bundle/FIT/selected_variant.npz'),
          path(fitTarget, stageAs: 'factor_bundle/FIT/target.npz'),
          path(fitReference, stageAs: 'factor_bundle/FIT/reference.npz'),
          path(fitMosaicReceipt, stageAs: 'factor_bundle/FIT/mosaic.receipt.json'),
          path(fitBridgeReceipt, stageAs: 'factor_bundle/FIT/bridge.receipt.json'),
          path(fitFlareDir, stageAs: 'factor_bundle/FIT/flare'),
          path(fitF0Dir, stageAs: 'factor_bundle/FIT/f0'),
          path(fitTruthDir, stageAs: 'factor_bundle/FIT/truth')
    tuple val(validSplit),
          path(validSelected, stageAs: 'factor_bundle/VALID/selected_variant.npz'),
          path(validTarget, stageAs: 'factor_bundle/VALID/target.npz'),
          path(validReference, stageAs: 'factor_bundle/VALID/reference.npz'),
          path(validMosaicReceipt, stageAs: 'factor_bundle/VALID/mosaic.receipt.json'),
          path(validBridgeReceipt, stageAs: 'factor_bundle/VALID/bridge.receipt.json'),
          path(validFlareDir, stageAs: 'factor_bundle/VALID/flare'),
          path(validF0Dir, stageAs: 'factor_bundle/VALID/f0'),
          path(validTruthDir, stageAs: 'factor_bundle/VALID/truth')
    path manifestBuilderPy

    output:
    tuple path('factor_bundle'),
          path('m34_factorized_manifest.json'),
          path('m34_factorized_manifest.receipt.json'),
          emit: bundle

    script:
    """
    set -euo pipefail
    test '${fitSplit}' = FIT
    test '${validSplit}' = VALID
    python3 ${manifestBuilderPy} \
      --fit-selected-variant factor_bundle/FIT/selected_variant.npz \
      --fit-target factor_bundle/FIT/target.npz \
      --fit-reference factor_bundle/FIT/reference.npz \
      --fit-f0 factor_bundle/FIT/f0/m34_f0.npz \
      --fit-marker-cm factor_bundle/FIT/f0/marker_cM.npz \
      --fit-truth factor_bundle/FIT/truth/truth.npz \
      --fit-mosaic-receipt factor_bundle/FIT/mosaic.receipt.json \
      --fit-bridge-receipt factor_bundle/FIT/bridge.receipt.json \
      --fit-flare-receipt factor_bundle/FIT/flare/m34_flare.receipt.json \
      --valid-selected-variant factor_bundle/VALID/selected_variant.npz \
      --valid-target factor_bundle/VALID/target.npz \
      --valid-reference factor_bundle/VALID/reference.npz \
      --valid-f0 factor_bundle/VALID/f0/m34_f0.npz \
      --valid-marker-cm factor_bundle/VALID/f0/marker_cM.npz \
      --valid-truth factor_bundle/VALID/truth/truth.npz \
      --valid-mosaic-receipt factor_bundle/VALID/mosaic.receipt.json \
      --valid-bridge-receipt factor_bundle/VALID/bridge.receipt.json \
      --valid-flare-receipt factor_bundle/VALID/flare/m34_flare.receipt.json \
      --manifest factor_bundle/factorized.manifest.json \
      --receipt factor_bundle/factorized.manifest.receipt.json
    cp factor_bundle/factorized.manifest.json m34_factorized_manifest.json
    cp factor_bundle/factorized.manifest.receipt.json m34_factorized_manifest.receipt.json
    """

    stub:
    """
    set -euo pipefail
    touch m34_factorized_manifest.json m34_factorized_manifest.receipt.json
    """
}
