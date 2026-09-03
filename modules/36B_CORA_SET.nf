nextflow.enable.dsl=2

process M36B_CORA_SET_TRAIN {
    tag "m36b_cora_set_${params.m36b_feature_chrom}"
    publishDir { "${params.m36b_output_prefix}" }, mode: 'copy', overwrite: false
    container params.m36b_pytorch_image
    cpus params.m36b_cpus
    memory params.m36b_memory
    time params.m36b_time

    input:
    path loci
    path carriers
    path missing
    path covariates
    path components
    path targets
    path materialization_receipt
    path train_py
    path cached_models_py
    path upstream_train_py
    path upstream_models_py
    path cora_py
    path receipt_py
    path run_config
    path design_contract
    path module_nf
    path workflow_nf

    output:
    path 'm36b_train_summary.json', emit: summary
    path 'm36b_architecture_screen.tsv', emit: architecture_screen
    path 'm36b_predictions.tsv', emit: predictions
    path 'm36b_paired_effects.tsv', emit: paired_effects
    path 'm36b_control_diagnostics.json', emit: control_diagnostics
    path 'm36b_provenance_receipt.json', emit: provenance_receipt

    script:
    def cli = [
        'feature-chrom': params.m36b_feature_chrom,
        'model-families': params.m36b_model_families,
        'halving-budgets': params.m36b_halving_budgets,
        'halving-eta': params.m36b_halving_eta,
        'outer-folds': params.m36b_outer_folds,
        'inner-folds': params.m36b_inner_folds,
        'train-seeds': params.m36b_train_seeds,
        'bootstrap-reps': params.m36b_bootstrap_reps,
        'pair-batch-size': params.m36b_pair_batch_size,
        'permutation-swap-multiplier': params.m36b_permutation_swap_multiplier,
        'minimum-moved-carrier-fraction': params.m36b_minimum_moved_carrier_fraction,
        'positive-control-budgets': params.m36b_positive_control_budgets,
        'positive-control-seed': params.m36b_positive_control_seed,
        'learning-rate': params.m36b_learning_rate,
        'weight-decay': params.m36b_weight_decay,
        'huber-delta': params.m36b_huber_delta,
        'minimum-relative-mse-reduction': params.m36b_minimum_relative_mse_reduction,
        'minimum-positive-folds': params.m36b_minimum_positive_folds,
    ].collect { key, value -> "--${key} ${value}" }.join(' ')
    """
    set -euo pipefail
    python3 ${train_py} \\
      --loci ${loci} --carriers ${carriers} --missing ${missing} \\
      --covariates ${covariates} --components ${components} --targets ${targets} \\
      --materialization-receipt ${materialization_receipt} \\
      ${cli} \\
      --outdir .

    python3 ${receipt_py} \\
      --materialization-receipt ${materialization_receipt} \\
      --train-summary m36b_train_summary.json \\
      --run-config ${run_config} \\
      --design-contract ${design_contract} \\
      --code ${train_py} --code ${cached_models_py} --code ${upstream_train_py} \\
      --code ${upstream_models_py} --code ${cora_py} --code ${receipt_py} \\
      --code ${module_nf} --code ${workflow_nf} \\
      --input loci=${loci} --input carriers=${carriers} --input missing=${missing} \\
      --input covariates=${covariates} --input components=${components} --input targets=${targets} \\
      --output m36b_train_summary.json --output m36b_architecture_screen.tsv \\
      --output m36b_predictions.tsv --output m36b_paired_effects.tsv \\
      --output m36b_control_diagnostics.json \\
      --output-prefix '${params.m36b_output_prefix}' \\
      --out m36b_provenance_receipt.json
    """
}
