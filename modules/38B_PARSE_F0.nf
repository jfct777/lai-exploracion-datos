nextflow.enable.dsl=2

process M38B_PARSE_F_MINUS_S660_F0 {
    tag { 'm38b_parse_f_minus_s660_f0_fit' }
    publishDir {
        "${params.m38b_prepare_results_dir}/${params.m38b_prepare_run_id}/fit/f0"
    }, mode: 'copy', overwrite: false,
       saveAs: { name -> name == 'm38b_fit_f_minus_s660_f0' ? name : null }
    container params.m38b_prepare_python_image
    containerOptions { "--network none --user ${params.m38b_prepare_container_user}" }
    cpus { params.m38b_prepare_parse_cpus }
    memory { params.m38b_prepare_parse_memory }
    time { params.m38b_prepare_parse_time }
    maxForks 1

    input:
    path flareDir
    path geneticMap
    path sourceFiles

    output:
    path 'm38b_fit_f_minus_s660_f0', emit: parsed

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_parse_flare.py \
      --flare-anc '${flareDir}/m38b_f_minus_s660.anc.vcf.gz' \
      --flare-receipt '${flareDir}/m38b_flare.receipt.json' \
      --genetic-map '${geneticMap}' \
      --genetic-map-sha256 '${params.m38b_prepare_genetic_map_sha256}' \
      --expected-samples '${params.m38b_prepare_expected_fit_samples}' \
      --expected-markers '${params.m38b_prepare_expected_fminus_loci}' \
      --outdir m38b_fit_f_minus_s660_f0
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m38b_fit_f_minus_s660_f0
    touch \
      m38b_fit_f_minus_s660_f0/m38b_f_minus_s660_f0.npz \
      m38b_fit_f_minus_s660_f0/m38b_f_minus_s660_marker_cM.npz \
      m38b_fit_f_minus_s660_f0/m38b_f_minus_s660_f0.receipt.json
    """
}
