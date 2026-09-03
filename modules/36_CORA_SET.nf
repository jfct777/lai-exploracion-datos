nextflow.enable.dsl=2

process M36_CORA_SET_PLAN {
    tag "m36_cora_set_plan_${params.m36_cora_feature_chrom}"
    cpus params.m36_cora_cpus
    memory params.m36_cora_memory
    time params.m36_cora_time

    input:
    path events
    path covariates
    path components
    path targets
    path contract
    path cora_py
    path models_py

    output:
    path "m36_cora_event_tokens.tsv", emit: event_tokens
    path "m36_cora_trial_plan.tsv", emit: trial_plan
    path "m36_cora_component_folds.tsv", emit: component_folds
    path "m36_cora_smoke_summary.json", emit: summary

    script:
    """
    set -euo pipefail
    python3 ${cora_py} \\
      --events ${events} \\
      --covariates ${covariates} \\
      --components ${components} \\
      --targets ${targets} \\
      --contract ${contract} \\
      --feature-chrom ${params.m36_cora_feature_chrom} \\
      --model-families ${params.m36_cora_model_families} \\
      --halving-budgets ${params.m36_cora_halving_budgets} \\
      --halving-eta ${params.m36_cora_halving_eta} \\
      --n-folds ${params.m36_cora_n_folds} \\
      --smoke-only \\
      --outdir .
    """
}

process M36_CORA_SET_TRAIN {
    tag "m36_cora_set_${mode}_${params.m36_cora_feature_chrom}"
    publishDir { "gs://teams-usp/frank/lai-exploracion-datos/runs/${params.m36_cora_run_id}/m36_cora_set/" }, mode: 'copy', overwrite: false
    container params.m36_cora_pytorch_image
    cpus params.m36_cora_train_cpus
    memory params.m36_cora_train_memory
    time params.m36_cora_train_time

    input:
    val mode
    path loci
    path carriers
    path missing
    path covariates
    path components
    path targets
    path materialization_receipt
    path train_py
    path cora_py
    path models_py
    path train_receipt_py

    output:
    path "m36_cora_train_summary.json", emit: summary
    path "m36_cora_*_halving.tsv", emit: halving
    path "m36_cora_*_predictions.tsv", emit: predictions
    path "m36_cora_*_component_metrics.tsv", emit: component_metrics
    path "m36_cora_train_publication_receipt.json", emit: publication_receipt

    script:
    def syntheticFlag = mode == 'smoke' ? '--synthetic-smoke' : ''
    """
    set -euo pipefail
    python3 ${train_py} \\
      --mode ${mode} ${syntheticFlag} \\
      --loci ${loci} \\
      --carriers ${carriers} \\
      --missing ${missing} \\
      --covariates ${covariates} \\
      --components ${components} \\
      --targets ${targets} \\
      --materialization-receipt ${materialization_receipt} \\
      --feature-chrom ${params.m36_cora_feature_chrom} \\
      --model-families ${params.m36_cora_model_families} \\
      --halving-budgets ${params.m36_cora_halving_budgets} \\
      --halving-eta ${params.m36_cora_halving_eta} \\
      --n-folds ${params.m36_cora_n_folds} \\
      --seed ${params.m36_cora_seed} \\
      --train-seeds ${params.m36_cora_train_seeds} \\
      --bootstrap-reps ${params.m36_cora_bootstrap_reps} \\
      --positive-min-relative-mse-reduction ${params.m36_cora_positive_min_relative_mse_reduction} \\
      --outdir .
    python3 ${train_receipt_py} \\
      --materialization-receipt ${materialization_receipt} \\
      --train-summary m36_cora_train_summary.json \\
      --out m36_cora_train_publication_receipt.json
    """
}

process M36_CORA_MATERIALIZE {
    tag "m36_cora_materialize_${params.m36_cora_feature_chrom}"
    // Resolve the namespace at task creation, after command-line params are
    // applied.  A config-time interpolation would freeze it at UNSET.
    publishDir { "gs://teams-usp/frank/lai-exploracion-datos/runs/${params.m36_cora_run_id}/m36_cora_set/" }, mode: 'copy', overwrite: false
    container params.m36_cora_pytorch_image
    cpus params.m36_cora_materialize_cpus
    memory params.m36_cora_materialize_memory
    time params.m36_cora_materialize_time

    input:
    path rare_vcf
    path locus_metadata
    path genetic_map
    path sample_metadata
    path pcrelate_components
    path asibd_manifest
    path asibd_segments
    path materialize_py

    output:
    path "m36_cora_loci.tsv", emit: loci
    path "m36_cora_carriers.tsv", emit: carriers
    path "m36_cora_missing.tsv", emit: missing
    path "m36_cora_covariates.tsv", emit: covariates
    path "m36_cora_components.tsv", emit: components
    path "m36_cora_external_targets.tsv", emit: targets
    path "m36_cora_external_targets_zero*.tsv", emit: sensitivity_targets, optional: true
    path "m36_cora_materialization_receipt.json", emit: receipt

    script:
    def asibdArgs = asibd_segments.collect { segment -> segment.getName() }.join(' ')
    """
    set -euo pipefail
    python3 ${materialize_py} \\
      --rare-vcf ${rare_vcf} \\
      --locus-metadata ${locus_metadata} \\
      --genetic-map ${genetic_map} \\
      --sample-metadata ${sample_metadata} \\
      --pcrelate-components ${pcrelate_components} \\
      --asibd-manifest ${asibd_manifest} \\
      --asibd-segments ${asibdArgs} \\
      --feature-chrom ${params.m36_cora_feature_chrom} \\
      --zero-negative-ratios ${params.m36_cora_zero_negative_ratios} \\
      --max-positives-per-stratum ${params.m36_cora_max_positives_per_stratum} \\
      --seed ${params.m36_cora_seed} \\
      --outdir .
    """
}

process M36_CORA_CANONICAL_ADAPTER {
    tag 'm36_cora_canonical_adapter'
    container params.m36_cora_pytorch_image
    cpus 1
    memory '1 GB'
    time '15m'

    input:
    path metadata
    path m20_feature_store
    path modeling_master
    path adapter_py

    output:
    path 'm36_cora_sample_metadata.tsv', emit: sample_metadata
    path 'm36_cora_pcrelate_components.tsv', emit: pcrelate_components
    path 'm36_cora_canonical_adapter_receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    python3 ${adapter_py} --metadata ${metadata} --m20-feature-store ${m20_feature_store} \\
      --modeling-master ${modeling_master} --outdir .
    """
}

// A single Batch task for the explicitly requested technical end-to-end run.
// It keeps the three authenticated stages in one failure domain so an
// interrupted submitter cannot leave a real adapter job without scheduling
// its dependent materializer/trainer. Standalone materialize/train modes
// remain available above for normal production orchestration.
process M36_CORA_MATERIALIZE_TRAIN {
    tag "m36_cora_materialize_train_${params.m36_cora_feature_chrom}"
    publishDir { "gs://teams-usp/frank/lai-exploracion-datos/runs/${params.m36_cora_run_id}/m36_cora_set/" }, mode: 'copy', overwrite: false
    container params.m36_cora_pytorch_image
    cpus params.m36_cora_train_cpus
    memory params.m36_cora_train_memory
    time params.m36_cora_train_time

    input:
    path rare_vcf
    path locus_metadata
    path genetic_map
    path canonical_metadata
    path m20_feature_store
    path modeling_master
    path asibd_manifest
    path asibd_segments
    path adapter_py
    path materialize_py
    path train_py
    path cora_py
    path models_py
    path train_receipt_py

    output:
    path "m36_cora_loci.tsv", emit: loci
    path "m36_cora_carriers.tsv", emit: carriers
    path "m36_cora_missing.tsv", emit: missing
    path "m36_cora_covariates.tsv", emit: covariates
    path "m36_cora_components.tsv", emit: components
    path "m36_cora_external_targets.tsv", emit: targets
    path "m36_cora_external_targets_zero*.tsv", emit: sensitivity_targets, optional: true
    path "m36_cora_materialization_receipt.json", emit: receipt
    path "m36_cora_train_summary.json", emit: summary
    path "m36_cora_train_publication_receipt.json", emit: publication_receipt
    path "m36_cora_*_halving.tsv", emit: halving
    path "m36_cora_*_predictions.tsv", emit: predictions
    path "m36_cora_*_component_metrics.tsv", emit: component_metrics

    script:
    def asibdArgs = asibd_segments.collect { segment -> segment.getName() }.join(' ')
    """
    set -euo pipefail
    python3 ${adapter_py} --metadata ${canonical_metadata} --m20-feature-store ${m20_feature_store} \\
      --modeling-master ${modeling_master} --outdir .
    python3 ${materialize_py} \\
      --rare-vcf ${rare_vcf} \\
      --locus-metadata ${locus_metadata} \\
      --genetic-map ${genetic_map} \\
      --sample-metadata m36_cora_sample_metadata.tsv \\
      --pcrelate-components m36_cora_pcrelate_components.tsv \\
      --asibd-manifest ${asibd_manifest} \\
      --asibd-segments ${asibdArgs} \\
      --feature-chrom ${params.m36_cora_feature_chrom} \\
      --zero-negative-ratios ${params.m36_cora_zero_negative_ratios} \\
      --max-positives-per-stratum ${params.m36_cora_max_positives_per_stratum} \\
      --seed ${params.m36_cora_seed} \\
      --outdir .
    python3 ${train_py} --mode train \\
      --loci m36_cora_loci.tsv --carriers m36_cora_carriers.tsv --missing m36_cora_missing.tsv \\
      --covariates m36_cora_covariates.tsv --components m36_cora_components.tsv \\
      --targets m36_cora_external_targets.tsv \\
      --materialization-receipt m36_cora_materialization_receipt.json \\
      --feature-chrom ${params.m36_cora_feature_chrom} \\
      --model-families ${params.m36_cora_model_families} \\
      --halving-budgets ${params.m36_cora_halving_budgets} \\
      --halving-eta ${params.m36_cora_halving_eta} --n-folds ${params.m36_cora_n_folds} \\
      --seed ${params.m36_cora_seed} --train-seeds ${params.m36_cora_train_seeds} \\
      --bootstrap-reps ${params.m36_cora_bootstrap_reps} \\
      --positive-min-relative-mse-reduction ${params.m36_cora_positive_min_relative_mse_reduction} \\
      --outdir .
    python3 ${train_receipt_py} --materialization-receipt m36_cora_materialization_receipt.json \\
      --train-summary m36_cora_train_summary.json --out m36_cora_train_publication_receipt.json
    """
}
