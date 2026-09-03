nextflow.enable.dsl=2

process M38B_PROJECT_FULL_AND_TRUTH {
    tag { 'm38b_project_full_and_truth_fit' }
    publishDir {
        "${params.m38b_prepare_results_dir}/${params.m38b_prepare_run_id}/fit/aligned"
    }, mode: 'copy', overwrite: false,
       saveAs: { name -> name == 'm38b_fit_aligned_baselines' ? name : null }
    container params.m38b_prepare_python_image
    containerOptions { "--network none --user ${params.m38b_prepare_container_user}" }
    cpus { params.m38b_prepare_project_cpus }
    memory { params.m38b_prepare_project_memory }
    time { params.m38b_prepare_project_time }
    maxForks 1

    input:
    path fminusDir
    path fullF0
    path fullMarkerCm
    path fullTruth
    path selectedLoci
    path experimentContract
    path sourceFiles

    output:
    path 'm38b_fit_aligned_baselines', emit: aligned

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_project_baselines.py \
      --experiment '${experimentContract}' \
      --full-f0 '${fullF0}' \
      --full-f0-sha256 '${params.m38b_prepare_full_f0_sha256}' \
      --full-marker-cm '${fullMarkerCm}' \
      --full-marker-cm-sha256 '${params.m38b_prepare_full_marker_cm_sha256}' \
      --full-truth '${fullTruth}' \
      --full-truth-sha256 '${params.m38b_prepare_full_truth_sha256}' \
      --selected-loci '${selectedLoci}' \
      --selected-loci-sha256 '${params.m38b_prepare_selected_loci_sha256}' \
      --fminus-f0 '${fminusDir}/m38b_f_minus_s660_f0.npz' \
      --fminus-marker-cm '${fminusDir}/m38b_f_minus_s660_marker_cM.npz' \
      --fminus-receipt '${fminusDir}/m38b_f_minus_s660_f0.receipt.json' \
      --expected-samples '${params.m38b_prepare_expected_fit_samples}' \
      --expected-full-markers '${params.m38b_prepare_expected_full_loci}' \
      --expected-selected-markers '${params.m38b_prepare_expected_selected_loci}' \
      --outdir m38b_fit_aligned_baselines
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m38b_fit_aligned_baselines
    touch \
      m38b_fit_aligned_baselines/m38b_f_full_projected_to_f_minus_s660.npz \
      m38b_fit_aligned_baselines/m38b_fit_truth_projected_to_f_minus_s660.npz \
      m38b_fit_aligned_baselines/m38b_common_marker_cM.npz \
      m38b_fit_aligned_baselines/m38b_baseline_alignment.receipt.json
    """
}
