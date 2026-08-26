nextflow.enable.dsl=2

process M34_NAM_RUN_FLARE {
    tag { "m34_flare_${split}" }
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/${split.toLowerCase()}/flare"
    }, mode: 'copy', overwrite: false,
       saveAs: { filename -> filename == "m34_${split.toLowerCase()}_flare" ? filename : null }
    container params.m34_inputs_flare_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_flare_cpus }
    memory { params.m34_inputs_flare_memory }
    time { params.m34_inputs_flare_time }
    maxForks params.m34_inputs_flare_max_forks

    input:
    tuple val(split),
          path(referenceVcf), path(referenceTbi),
          path(targetVcf), path(targetTbi), path(sampleMap),
          path(selectedLoci), path(targetRare), path(referenceRare),
          path(mosaicTruth), path(mosaicReceipt), path(bridgeReceipt),
          path(flareContract)
    path geneticMap
    path flareJar
    path runFlarePy

    output:
    tuple val(split),
          path(selectedLoci), path(targetRare), path(referenceRare),
          path(mosaicTruth), path(mosaicReceipt), path(bridgeReceipt),
          path("m34_${split.toLowerCase()}_flare"),
          emit: baseline

    script:
    """
    set -euo pipefail
    python3 ${runFlarePy} \
      --contract ${flareContract} \
      --reference-vcf ${referenceVcf} \
      --reference-tbi ${referenceTbi} \
      --target-vcf ${targetVcf} \
      --target-tbi ${targetTbi} \
      --sample-map ${sampleMap} \
      --genetic-map ${geneticMap} \
      --flare-jar ${flareJar} \
      --outdir m34_${split.toLowerCase()}_flare
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m34_${split.toLowerCase()}_flare
    touch \
      m34_${split.toLowerCase()}_flare/m34.anc.vcf.gz \
      m34_${split.toLowerCase()}_flare/m34_flare.receipt.json
    """
}
