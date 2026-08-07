nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// M25C — multiscale sensitivity of fold-specific rare-window PCA
// ---------------------------------------------------------------------------

process WRITE_RARE_WINDOW_SCALE_RUN_PROVENANCE {
    tag "rare_window_scale_run_provenance"
    publishDir "${params.rare_window_scale_results_dir}", mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '10m'

    input:
    val run_prov_b64

    output:
    path "run_provenance.json"

    script:
    """
    set -euo pipefail
    printf '%s' '${run_prov_b64}' | base64 -d > run_provenance.json
    """
}


process EVALUATE_RARE_WINDOW_SCALE_SENSITIVITY {
    tag "chr${chrom}_rare_window_scale_sensitivity"
    publishDir "${params.rare_window_scale_results_dir}/scale_sensitivity", mode: 'copy', overwrite: false
    cpus params.rare_window_scale_cpus
    memory params.rare_window_scale_memory
    time params.rare_window_scale_time

    input:
    val chrom
    path fold_features
    path fold_windows
    path fold_qc
    path fold_manifest
    path split_manifest
    path diagnostic_covariates
    path diagnostic_covariates_audit
    path diagnostic_covariates_manifest
    path preregistration
    path evaluate_scale_py
    path evaluate_pca_py
    path base_features_py
    path manifest_py
    val prov_b64

    output:
    path "chr${chrom}.multiscale_fold_metrics.tsv", emit: fold_metrics
    path "chr${chrom}.multiscale_sample_errors.tsv", emit: sample_errors
    path "chr${chrom}.multiscale_bootstrap.tsv", emit: bootstrap
    path "chr${chrom}.multiscale_diagnostics.tsv", emit: diagnostics
    path "chr${chrom}.multiscale_score_concordance.tsv", emit: concordance
    path "chr${chrom}.multiscale_site_bias.tsv", emit: site_bias
    path "chr${chrom}.multiscale_window_map.tsv", emit: window_map
    path "chr${chrom}.multiscale_window_contributions.tsv.gz", emit: contributions
    path "chr${chrom}.multiscale_oof_scores.tsv.gz", emit: scores
    path "chr${chrom}.multiscale_skill.{png,pdf}", emit: skill_figure
    path "chr${chrom}.multiscale_rank_curve.{png,pdf}", emit: rank_figure
    path "chr${chrom}.multiscale_site_bias.{png,pdf}", emit: bias_figure
    path "chr${chrom}.multiscale_score_concordance.{png,pdf}", emit: concordance_figure
    path "chr${chrom}.multiscale_summary.json", emit: summary
    path "chr${chrom}.multiscale.manifest.json", emit: manifest

    script:
    def scientificOutputs = [
        "chr${chrom}.multiscale_fold_metrics.tsv",
        "chr${chrom}.multiscale_sample_errors.tsv",
        "chr${chrom}.multiscale_bootstrap.tsv",
        "chr${chrom}.multiscale_diagnostics.tsv",
        "chr${chrom}.multiscale_score_concordance.tsv",
        "chr${chrom}.multiscale_site_bias.tsv",
        "chr${chrom}.multiscale_window_map.tsv",
        "chr${chrom}.multiscale_window_contributions.tsv.gz",
        "chr${chrom}.multiscale_oof_scores.tsv.gz",
        "chr${chrom}.multiscale_skill.png",
        "chr${chrom}.multiscale_skill.pdf",
        "chr${chrom}.multiscale_rank_curve.png",
        "chr${chrom}.multiscale_rank_curve.pdf",
        "chr${chrom}.multiscale_site_bias.png",
        "chr${chrom}.multiscale_site_bias.pdf",
        "chr${chrom}.multiscale_score_concordance.png",
        "chr${chrom}.multiscale_score_concordance.pdf",
        "chr${chrom}.multiscale_summary.json",
    ]
    def outputArgs = scientificOutputs.collect { "--output ${it}" }.join(' ')
    def stagedInputArgs = (fold_features + fold_windows).collect { "--input ${it}" }.join(' ')

    """
    set -euo pipefail

    python3 ${evaluate_scale_py} \
      --features ${fold_features} \
      --windows ${fold_windows} \
      --fold-qc ${fold_qc} \
      --fold-manifest ${fold_manifest} \
      --split-manifest ${split_manifest} \
      --diagnostic-covariates ${diagnostic_covariates} \
      --diagnostic-covariates-audit ${diagnostic_covariates_audit} \
      --diagnostic-covariates-manifest ${diagnostic_covariates_manifest} \
      --preregistration ${preregistration} \
      --chrom ${chrom} \
      --outer-folds '${params.rare_window_scale_outer_folds}' \
      --ranks '${params.rare_window_scale_ranks}' \
      --primary-rank ${params.rare_window_scale_primary_rank} \
      --bootstrap-replicates ${params.rare_window_scale_bootstrap_replicates} \
      --seed ${params.rare_window_scale_seed} \
      --out-prefix chr${chrom}

    python3 ${manifest_py} \
      --stage M25C_RARE_WINDOW_SCALE_SENSITIVITY \
      ${stagedInputArgs} \
      --input ${fold_qc} --input ${fold_manifest} --input ${split_manifest} \
      --input ${diagnostic_covariates} --input ${diagnostic_covariates_audit} \
      --input ${diagnostic_covariates_manifest} --input ${preregistration} \
      --input ${evaluate_scale_py} --input ${evaluate_pca_py} --input ${base_features_py} \
      ${outputArgs} \
      --provenance-b64 ${prov_b64} \
      --params-json '{"chrom":"${chrom}","scope":"post_hoc_scale_sensitivity_train_only","outer_folds":"${params.rare_window_scale_outer_folds}","ranks":"${params.rare_window_scale_ranks}","primary_rank":${params.rare_window_scale_primary_rank},"bootstrap_replicates":${params.rare_window_scale_bootstrap_replicates},"overlap_policy":"separate_phases"}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out chr${chrom}.multiscale.manifest.json
    """
}
