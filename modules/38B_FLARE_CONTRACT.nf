nextflow.enable.dsl=2

process M38B_BUILD_FLARE_CONTRACT {
    tag { 'm38b_flare_contract_fit' }
    publishDir {
        "${params.m38b_prepare_results_dir}/${params.m38b_prepare_run_id}/fit/flare_contract"
    }, mode: 'copy', overwrite: false
    container params.m38b_prepare_flare_image
    containerOptions { "--network none --user ${params.m38b_prepare_container_user}" }
    cpus { params.m38b_prepare_contract_cpus }
    memory { params.m38b_prepare_contract_memory }
    time { params.m38b_prepare_contract_time }

    input:
    path referenceVcf
    path referenceTbi
    path targetVcf
    path targetTbi
    path sampleMap
    path geneticMap
    path experimentContract
    val flareJar
    path buildContractPy

    output:
    tuple path(referenceVcf), path(referenceTbi),
          path(targetVcf), path(targetTbi), path(sampleMap),
          path('m38b_fit_f_minus_s660.flare.contract.json'),
          emit: contracted

    script:
    """
    set -euo pipefail
    python3 '${buildContractPy}' \
      --experiment '${experimentContract}' \
      --reference-vcf '${referenceVcf}' \
      --reference-vcf-sha256 '${params.m38b_prepare_reference_vcf_sha256}' \
      --reference-tbi '${referenceTbi}' \
      --reference-tbi-sha256 '${params.m38b_prepare_reference_tbi_sha256}' \
      --target-vcf '${targetVcf}' \
      --target-vcf-sha256 '${params.m38b_prepare_target_vcf_sha256}' \
      --target-tbi '${targetTbi}' \
      --target-tbi-sha256 '${params.m38b_prepare_target_tbi_sha256}' \
      --sample-map '${sampleMap}' \
      --sample-map-sha256 '${params.m38b_prepare_sample_map_sha256}' \
      --genetic-map '${geneticMap}' \
      --genetic-map-sha256 '${params.m38b_prepare_genetic_map_sha256}' \
      --flare-jar '${flareJar}' \
      --flare-jar-sha256 '${params.m38b_prepare_flare_jar_sha256}' \
      --output m38b_fit_f_minus_s660.flare.contract.json
    """

    stub:
    """
    set -euo pipefail
    touch m38b_fit_f_minus_s660.flare.contract.json
    """
}
