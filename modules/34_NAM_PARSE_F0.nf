nextflow.enable.dsl=2

process M34_NAM_PARSE_F0 {
    tag { "m34_parse_f0_${split}" }
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/${split.toLowerCase()}/f0"
    }, mode: 'copy', overwrite: false,
       saveAs: { filename -> filename == "m34_${split.toLowerCase()}_f0" ? filename : null }
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_parse_cpus }
    memory { params.m34_inputs_parse_memory }
    time { params.m34_inputs_parse_time }
    maxForks params.m34_inputs_parse_max_forks

    input:
    tuple val(split),
          path(selectedLoci), path(targetRare), path(referenceRare),
          path(mosaicTruth), path(mosaicReceipt), path(bridgeReceipt),
          path(flareDir)
    path geneticMap
    path parserPy
    path mosaicPy
    path bridgeCorePy

    output:
    tuple val(split),
          path(selectedLoci), path(targetRare), path(referenceRare),
          path(mosaicTruth), path(mosaicReceipt), path(bridgeReceipt),
          path(flareDir), path("m34_${split.toLowerCase()}_f0"),
          emit: parsed

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${parserPy} staged/bin/m34_parse_flare_truth.py
    cp ${mosaicPy} staged/bin/m34_generate_mosaics.py
    cp ${bridgeCorePy} staged/bin/m33_safe_bridge_core.py
    PYTHONPATH=staged/bin python3 staged/bin/m34_parse_flare_truth.py f0 \
      --flare-anc ${flareDir}/m34.anc.vcf.gz \
      --genetic-map ${geneticMap} \
      --ancestry-order AFR,EUR,NAM \
      --flare-id-map 0=AFR,1=EUR,2=NAM \
      --outdir m34_${split.toLowerCase()}_f0
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m34_${split.toLowerCase()}_f0
    touch \
      m34_${split.toLowerCase()}_f0/m34_f0.npz \
      m34_${split.toLowerCase()}_f0/marker_cM.npz \
      m34_${split.toLowerCase()}_f0/m34_f0.receipt.json
    """
}
