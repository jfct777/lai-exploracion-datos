nextflow.enable.dsl=2

process M34_NAM_BUILD_FLARE_CONTRACT {
    tag { "m34_flare_contract_${split}" }
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/${split.toLowerCase()}/flare_contract"
    }, mode: 'copy', overwrite: false,
       saveAs: { filename -> filename.endsWith('.flare.contract.json') ? filename : null }
    container params.m34_inputs_flare_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_flare_contract_cpus }
    memory { params.m34_inputs_flare_contract_memory }
    time { params.m34_inputs_flare_contract_time }

    input:
    tuple val(split),
          path(referenceVcf), path(referenceTbi),
          path(targetVcf), path(targetTbi), path(sampleMap),
          path(selectedLoci), path(targetRare), path(referenceRare),
          path(mosaicTruth), path(mosaicReceipt), path(bridgeReceipt)
    path experimentContract
    path geneticMap
    path flareJar
    path buildContractPy

    output:
    tuple val(split),
          path(referenceVcf), path(referenceTbi),
          path(targetVcf), path(targetTbi), path(sampleMap),
          path(selectedLoci), path(targetRare), path(referenceRare),
          path(mosaicTruth), path(mosaicReceipt), path(bridgeReceipt),
          path("m34_${split.toLowerCase()}.flare.contract.json"),
          emit: contracted

    script:
    """
    set -euo pipefail
    python3 ${buildContractPy} \
      --experiment ${experimentContract} \
      --reference-vcf ${referenceVcf} \
      --reference-tbi ${referenceTbi} \
      --target-vcf ${targetVcf} \
      --target-tbi ${targetTbi} \
      --sample-map ${sampleMap} \
      --genetic-map ${geneticMap} \
      --flare-jar ${flareJar} \
      --output m34_${split.toLowerCase()}.flare.contract.json
    """

    stub:
    """
    set -euo pipefail
    touch m34_${split.toLowerCase()}.flare.contract.json
    """
}
