nextflow.enable.dsl=2

process M38B_RUN_FLARE_F_MINUS_S660 {
    tag { 'm38b_flare_f_minus_s660_fit' }
    publishDir {
        "${params.m38b_prepare_results_dir}/${params.m38b_prepare_run_id}/fit/flare"
    }, mode: 'copy', overwrite: false,
       saveAs: { name -> name == 'm38b_fit_f_minus_s660_flare' ? name : null }
    container params.m38b_prepare_flare_image
    containerOptions { "--network none --user ${params.m38b_prepare_container_user}" }
    cpus { params.m38b_prepare_flare_cpus }
    memory { params.m38b_prepare_flare_memory }
    time { params.m38b_prepare_flare_time }
    maxForks 1

    input:
    tuple path(referenceVcf), path(referenceTbi),
          path(targetVcf), path(targetTbi), path(sampleMap),
          path(flareContract)
    path geneticMap
    val flareJar
    path sourceFiles

    output:
    path 'm38b_fit_f_minus_s660_flare', emit: baseline

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_run_flare.py \
      --contract '${flareContract}' \
      --reference-vcf '${referenceVcf}' \
      --reference-tbi '${referenceTbi}' \
      --target-vcf '${targetVcf}' \
      --target-tbi '${targetTbi}' \
      --sample-map '${sampleMap}' \
      --genetic-map '${geneticMap}' \
      --flare-jar '${flareJar}' \
      --outdir m38b_fit_f_minus_s660_flare
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m38b_fit_f_minus_s660_flare
    touch \
      m38b_fit_f_minus_s660_flare/m38b_f_minus_s660.anc.vcf.gz \
      m38b_fit_f_minus_s660_flare/m38b_flare.receipt.json
    """
}
