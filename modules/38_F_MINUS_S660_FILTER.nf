nextflow.enable.dsl=2

process M38_F_MINUS_S660_FILTER {
    tag { "m38_f_minus_s660_${split}" }
    publishDir {
        "${params.m38_fminus_results_dir}/${params.m38_fminus_run_id}/${split.toLowerCase()}/audit"
    }, mode: 'copy', overwrite: false, saveAs: { name ->
        name.endsWith('.receipt.json') ? name : null
    }
    container params.m38_fminus_python_image
    containerOptions { "--network none --user ${params.m38_fminus_container_user}" }
    cpus { params.m38_fminus_filter_cpus }
    memory { params.m38_fminus_filter_memory }
    time { params.m38_fminus_filter_time }
    maxForks params.m38_fminus_filter_max_forks

    input:
    tuple val(split), path(referenceVcf), path(targetVcf), path(selectedLoci),
          val(referenceSha256), val(targetSha256), val(selectedSha256),
          val(expectedTargetSamples)
    path sourceFiles

    output:
    tuple val(split),
          path("m38_${split.toLowerCase()}_f_minus_s660/m38_f_minus_s660_reference.chr22.vcf"),
          path("m38_${split.toLowerCase()}_f_minus_s660/m38_f_minus_s660_target.chr22.vcf"),
          path("m38_${split.toLowerCase()}_f_minus_s660/m38_f_minus_s660_filter.receipt.json"),
          emit: filtered

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38_build_f_minus_s660.py \
      --split '${split}' \
      --reference-vcf '${referenceVcf}' \
      --target-vcf '${targetVcf}' \
      --selected-loci '${selectedLoci}' \
      --reference-sha256 '${referenceSha256}' \
      --target-sha256 '${targetSha256}' \
      --selected-sha256 '${selectedSha256}' \
      --chromosome '${params.m38_fminus_chromosome}' \
      --expected-full-loci '${params.m38_fminus_expected_full_loci}' \
      --expected-selected-loci '${params.m38_fminus_expected_selected_loci}' \
      --expected-reference-samples '${params.m38_fminus_expected_reference_samples}' \
      --expected-target-samples '${expectedTargetSamples}' \
      --outdir m38_${split.toLowerCase()}_f_minus_s660
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m38_${split.toLowerCase()}_f_minus_s660
    touch \
      m38_${split.toLowerCase()}_f_minus_s660/m38_f_minus_s660_reference.chr22.vcf \
      m38_${split.toLowerCase()}_f_minus_s660/m38_f_minus_s660_target.chr22.vcf \
      m38_${split.toLowerCase()}_f_minus_s660/m38_f_minus_s660_filter.receipt.json
    """
}
