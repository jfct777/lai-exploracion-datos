nextflow.enable.dsl=2

process M34_NAM_ALIGN_TRUTH {
    tag { "m34_truth_${split}" }
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/${split.toLowerCase()}/truth"
    }, mode: 'copy', overwrite: false,
       saveAs: { filename -> filename == "m34_${split.toLowerCase()}_truth" ? filename : null }
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_truth_cpus }
    memory { params.m34_inputs_truth_memory }
    time { params.m34_inputs_truth_time }
    maxForks params.m34_inputs_truth_max_forks

    input:
    tuple val(split),
          path(selectedLoci), path(targetRare), path(referenceRare),
          path(mosaicTruth), path(mosaicReceipt), path(bridgeReceipt),
          path(flareDir), path(f0Dir)
    path parserPy
    path mosaicPy
    path bridgeCorePy

    output:
    tuple val(split),
          path(selectedLoci), path(targetRare), path(referenceRare),
          path(mosaicReceipt), path(bridgeReceipt),
          path(flareDir), path(f0Dir), path("m34_${split.toLowerCase()}_truth"),
          emit: factors

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${parserPy} staged/bin/m34_parse_flare_truth.py
    cp ${mosaicPy} staged/bin/m34_generate_mosaics.py
    cp ${bridgeCorePy} staged/bin/m33_safe_bridge_core.py
    PYTHONPATH=staged/bin python3 staged/bin/m34_parse_flare_truth.py truth \
      --truth-segments ${mosaicTruth} \
      --f0 ${f0Dir}/m34_f0.npz \
      --marker-cm ${f0Dir}/marker_cM.npz \
      --ancestry-order AFR,EUR,NAM \
      --role ${split} \
      --outdir m34_${split.toLowerCase()}_truth
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m34_${split.toLowerCase()}_truth
    touch \
      m34_${split.toLowerCase()}_truth/truth.npz \
      m34_${split.toLowerCase()}_truth/truth.receipt.json
    """
}
