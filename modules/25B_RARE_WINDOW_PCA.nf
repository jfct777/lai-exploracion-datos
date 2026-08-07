nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// M25B — fold-specific rare-window PCA benchmark (canonical TRAIN only)
// ---------------------------------------------------------------------------

process WRITE_RARE_WINDOW_PCA_RUN_PROVENANCE {
    tag "rare_window_pca_run_provenance"
    publishDir "${params.rare_window_pca_results_dir}", mode: 'copy', overwrite: false
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


process BUILD_TRAIN_DIAGNOSTIC_COVARIATES {
    tag "m25b_train_diagnostic_covariates"
    publishDir "${params.rare_window_pca_results_dir}/diagnostic_covariates", mode: 'copy', overwrite: false
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    path split_manifest
    path feature_store
    path covariates_py
    path base_features_py
    path manifest_py
    val prov_b64

    output:
    path "train_diagnostic_covariates.tsv", emit: covariates
    path "train_diagnostic_covariates.audit.json", emit: audit
    path "train_diagnostic_covariates.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail

    python3 ${covariates_py} \
      --split-manifest ${split_manifest} \
      --feature-store ${feature_store} \
      --expected-train ${params.rare_window_pca_expected_train_samples} \
      --output train_diagnostic_covariates.tsv \
      --audit train_diagnostic_covariates.audit.json

    python3 ${manifest_py} \
      --stage BUILD_TRAIN_DIAGNOSTIC_COVARIATES \
      --input ${split_manifest} --input ${feature_store} \
      --input ${covariates_py} --input ${base_features_py} \
      --output train_diagnostic_covariates.tsv \
      --output train_diagnostic_covariates.audit.json \
      --provenance-b64 ${prov_b64} \
      --params-json '{"scope":"canonical_train_diagnostic_only","expected_train":${params.rare_window_pca_expected_train_samples},"fields":["Q_NAM","Q_EUR","Q_EAS","Q_AFR","cohort"]}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out train_diagnostic_covariates.manifest.json
    """
}


process BUILD_FOLD_RARE_WINDOW_FEATURES {
    tag "chr${chrom}_fold_specific_features"
    publishDir "${params.rare_window_pca_results_dir}/fold_features", mode: 'copy', overwrite: false
    cpus params.rare_window_pca_extract_cpus
    memory params.rare_window_pca_extract_memory
    time params.rare_window_pca_extract_time

    input:
    tuple val(chrom), path(vcf), path(vcf_tbi), path(reference_fai),
          path(split_manifest), path(upstream_qc), path(upstream_manifest),
          path(fold_features_py), path(base_features_py), path(manifest_py)
    val prov_b64

    output:
    path "chr${chrom}.fold*.rare_sites.tsv.gz", emit: sites
    path "chr${chrom}.fold*.windows.tsv", emit: windows
    path "chr${chrom}.fold*.sample_window_features.tsv.gz", emit: features
    path "chr${chrom}.fold_features_qc.json", emit: qc
    path "chr${chrom}.fold_features.manifest.json", emit: manifest

    script:
    def folds = params.rare_window_pca_outer_folds.toString().tokenize(',')
    def scientificOutputs = folds.collectMany { fold -> [
        "chr${chrom}.fold${fold}.rare_sites.tsv.gz",
        "chr${chrom}.fold${fold}.windows.tsv",
        "chr${chrom}.fold${fold}.sample_window_features.tsv.gz",
    ] } + ["chr${chrom}.fold_features_qc.json"]
    def outputArgs = scientificOutputs.collect { "--output ${it}" }.join(' ')

    """
    set -euo pipefail

    python3 ${fold_features_py} \
      --vcf ${vcf} \
      --vcf-index ${vcf_tbi} \
      --reference-fai ${reference_fai} \
      --split-manifest ${split_manifest} \
      --upstream-qc ${upstream_qc} \
      --upstream-manifest ${upstream_manifest} \
      --chrom ${chrom} \
      --expected-train-samples ${params.rare_window_pca_expected_train_samples} \
      --expected-test-samples ${params.rare_window_pca_expected_test_samples} \
      --expected-input-sites ${params.rare_window_pca_expected_input_sites} \
      --outer-folds '${params.rare_window_pca_outer_folds}' \
      --min-mac ${params.rare_window_pca_min_mac} \
      --max-maf ${params.rare_window_pca_max_maf} \
      --window-size-bp ${params.rare_window_pca_window_size_bp} \
      --outdir .

    python3 ${manifest_py} \
      --stage BUILD_FOLD_RARE_WINDOW_FEATURES \
      --input ${vcf} --input ${vcf_tbi} --input ${reference_fai} \
      --input ${split_manifest} --input ${upstream_qc} --input ${upstream_manifest} \
      --input ${fold_features_py} --input ${base_features_py} \
      ${outputArgs} \
      --provenance-b64 ${prov_b64} \
      --params-json '{"chrom":"${chrom}","scope":"canonical_train_outer_fold","outer_folds":"${params.rare_window_pca_outer_folds}","min_mac":${params.rare_window_pca_min_mac},"max_maf":${params.rare_window_pca_max_maf},"window_size_bp":${params.rare_window_pca_window_size_bp}}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out chr${chrom}.fold_features.manifest.json
    """
}


process EVALUATE_FOLD_RARE_WINDOW_PCA {
    tag "chr${chrom}_fold_pca"
    publishDir "${params.rare_window_pca_results_dir}/pca", mode: 'copy', overwrite: false
    cpus params.rare_window_pca_cpus
    memory params.rare_window_pca_memory
    time params.rare_window_pca_time

    input:
    val chrom
    path fold_features
    path fold_windows
    path fold_qc
    path fold_manifest
    path split_manifest
    path train_covariates
    path train_covariates_audit
    path train_covariates_manifest
    path evaluate_py
    path base_features_py
    path manifest_py
    val prov_b64

    output:
    path "chr${chrom}.pca_fold_metrics.tsv", emit: metrics
    path "chr${chrom}.pca_sample_errors.tsv", emit: sample_errors
    path "chr${chrom}.pca_bootstrap.tsv", emit: bootstrap
    path "chr${chrom}.pca_subspace_stability.tsv", emit: stability
    path "chr${chrom}.pca_diagnostics.tsv", emit: diagnostics
    path "chr${chrom}.pca_scores.tsv.gz", emit: scores
    path "chr${chrom}.pca_loadings.tsv.gz", emit: loadings
    path "chr${chrom}.pca_oof_baselines.tsv.gz", emit: baselines
    path "chr${chrom}.pca_oof_reconstructions.tsv.gz", emit: reconstructions
    path "chr${chrom}.pca_benchmark.png", emit: png
    path "chr${chrom}.pca_benchmark.pdf", emit: pdf
    path "chr${chrom}.pca_summary.json", emit: summary
    path "chr${chrom}.pca.manifest.json", emit: manifest

    script:
    def scientificOutputs = [
        "chr${chrom}.pca_fold_metrics.tsv",
        "chr${chrom}.pca_sample_errors.tsv",
        "chr${chrom}.pca_bootstrap.tsv",
        "chr${chrom}.pca_subspace_stability.tsv",
        "chr${chrom}.pca_diagnostics.tsv",
        "chr${chrom}.pca_scores.tsv.gz",
        "chr${chrom}.pca_loadings.tsv.gz",
        "chr${chrom}.pca_oof_baselines.tsv.gz",
        "chr${chrom}.pca_oof_reconstructions.tsv.gz",
        "chr${chrom}.pca_benchmark.png",
        "chr${chrom}.pca_benchmark.pdf",
        "chr${chrom}.pca_summary.json",
    ]
    def outputArgs = scientificOutputs.collect { "--output ${it}" }.join(' ')
    def foldInputArgs = (fold_features + fold_windows).collect { "--input ${it}" }.join(' ')

    """
    set -euo pipefail

    python3 ${evaluate_py} \
      --features ${fold_features} \
      --windows ${fold_windows} \
      --fold-qc ${fold_qc} \
      --fold-manifest ${fold_manifest} \
      --split-manifest ${split_manifest} \
      --feature-store ${train_covariates} \
      --train-covariates-audit ${train_covariates_audit} \
      --train-covariates-manifest ${train_covariates_manifest} \
      --chrom ${chrom} \
      --outer-folds '${params.rare_window_pca_outer_folds}' \
      --ranks '${params.rare_window_pca_ranks}' \
      --primary-rank ${params.rare_window_pca_primary_rank} \
      --bootstrap-replicates ${params.rare_window_pca_bootstrap_replicates} \
      --seed ${params.rare_window_pca_seed} \
      --out-prefix chr${chrom}

    python3 ${manifest_py} \
      --stage EVALUATE_FOLD_RARE_WINDOW_PCA \
      --input ${fold_qc} --input ${fold_manifest} --input ${split_manifest} \
      --input ${train_covariates} --input ${train_covariates_audit} \
      --input ${train_covariates_manifest} --input ${evaluate_py} --input ${base_features_py} \
      ${foldInputArgs} \
      ${outputArgs} \
      --provenance-b64 ${prov_b64} \
      --params-json '{"chrom":"${chrom}","ranks":"${params.rare_window_pca_ranks}","rank_selection":"none_capacity_curve","primary_rank":${params.rare_window_pca_primary_rank},"bootstrap_replicates":${params.rare_window_pca_bootstrap_replicates},"seed":${params.rare_window_pca_seed}}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out chr${chrom}.pca.manifest.json
    """
}
