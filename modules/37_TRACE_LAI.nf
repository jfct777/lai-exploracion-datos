nextflow.enable.dsl=2

process M37_TRACE_BIND_MARKER_AXIS {
    tag { "${split}_marker_axis" }
    input:
    tuple val(split), path(f0), path(marker_cm), path(source_receipt)
    path source_files
    output:
    tuple val(split), path("${split}.marker_axis.npz"), path("${split}.marker_axis.receipt.json"), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_bind_marker_axis.py \\
      --f0 ${f0} --marker-cm ${marker_cm} --source-receipt ${source_receipt} \\
      --output '${split}.marker_axis.npz'
    """
}

process M37_TRACE_SHAM_REFERENCE {
    tag { "${split}_sham" }
    input:
    tuple val(split), path(reference)
    path source_files
    output:
    tuple val(split), path("${split}.sham.reference.npz"), path("${split}.sham.reference.receipt.json"), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_sham.py --reference ${reference} \\
      --seed '${params.m37_sham_seed}' --output '${split}.sham.reference.npz'
    """
}

process M37_TRACE_MATERIALIZE {
    tag { "${split}_${arm}" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/features" }, mode: 'copy', overwrite: false
    cpus params.m37_materialize_cpus
    memory params.m37_materialize_memory
    time params.m37_materialize_time

    input:
    tuple val(split), val(arm), path(selected), path(target), path(reference_bundle), path(f0), path(marker_cm), path(marker_axis_receipt)
    path source_files

    output:
    tuple val(split), val(arm), path("${split}.${arm}.trace.npz"), path("${split}.${arm}.trace.receipt.json"), emit: bundle

    script:
    def reference = reference_bundle.find { path -> path.name.endsWith('.npz') }
    def referenceReceipt = reference_bundle.find { path -> path.name.endsWith('.receipt.json') }
    def referenceReceiptFlag = referenceReceipt ? "--reference-receipt '${referenceReceipt}'" : ''
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_materialize.py \\
      --selected ${selected} --target ${target} --reference-fit-folds ${reference} ${referenceReceiptFlag} \\
      --f0 ${f0} --marker-cm ${marker_cm} --marker-axis-receipt ${marker_axis_receipt} --arm '${arm}' \\
      --beta-prior-strength '${params.m37_beta_prior_strength}' \\
      --output '${split}.${arm}.trace.npz'
    """
}

process M37_TRACE_TRAIN {
    tag { "${candidate_id}_${family}_${arm}" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/predictions" }, mode: 'copy', overwrite: false
    cpus params.m37_train_cpus
    memory params.m37_train_memory
    time params.m37_train_time
    maxForks params.m37_train_max_forks

    input:
    tuple val(candidate_id), val(family), val(arm), val(hazard), val(evidence_scale), val(hidden), val(depth), val(kernel), val(dropout), val(seed), val(learning_rate), val(dilations), path(fit_features, stageAs: 'fit/*'), path(fit_features_receipt, stageAs: 'fit/*'), path(valid_features, stageAs: 'predict/*'), path(valid_features_receipt, stageAs: 'predict/*')
    path fit_truth
    path source_files

    output:
    tuple val(candidate_id), val(family), val(arm), path("${candidate_id}.${family}.${arm}.prediction.npz"), path("${candidate_id}.${family}.${arm}.prediction.receipt.json"), emit: bundle
    tuple val(candidate_id), val(family), val(arm), path("${candidate_id}.${family}.${arm}.checkpoint.pt"), optional: true, emit: checkpoint

    script:
    def checkpointFlag = family == 'tcn' ? "--checkpoint '${candidate_id}.${family}.${arm}.checkpoint.pt'" : ''
    """
    set -euo pipefail
    export USER=m37-runner
    export LOGNAME=m37-runner
    export TORCHINDUCTOR_CACHE_DIR="\$PWD/.torch-cache"
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_train.py \\
      --features ${fit_features} --features-receipt ${fit_features_receipt} \\
      --predict-features ${valid_features} --predict-features-receipt ${valid_features_receipt} \\
      --truth ${fit_truth} --candidate-id '${candidate_id}' --family '${family}' --arm '${arm}' \\
      --hazard-per-morgan '${hazard}' --evidence-scale '${evidence_scale}' \\
      --hidden-dim '${hidden}' --depth '${depth}' --kernel-size '${kernel}' --dropout '${dropout}' \\
      --dilations '${dilations}' --seed '${seed}' --updates '${params.m37_updates}' --learning-rate '${learning_rate}' \\
      --batch-people '${params.m37_batch_people}' --marker-shard '${params.m37_marker_shard}' \\
      --validation-every '${params.m37_validation_every}' --early-stopping-patience '${params.m37_early_stopping_patience}' \\
      --tune-fraction '${params.m37_tune_fraction}' \\
      --split-seed '${params.m37_split_seed}' \\
      --event-radius-cm '${params.m37_event_radius_cm}' \\
      ${checkpointFlag} --output '${candidate_id}.${family}.${arm}.prediction.npz'
    """
}

process M37_TRACE_SCORE {
    tag { "${root}_${candidate_id}_${family}_${arm}" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/metrics" }, mode: 'copy', overwrite: false
    cpus 1
    memory '4 GB'
    time '20m'

    input:
    tuple val(root), val(candidate_id), val(family), val(arm), path(prediction), path(prediction_receipt), path(valid_features), path(valid_features_receipt)
    path valid_truth
    path source_files

    output:
    tuple val(root), val(candidate_id), val(family), val(arm), path("${candidate_id}.${family}.${arm}.metrics.json"), path("${candidate_id}.${family}.${arm}.metrics.receipt.json"), emit: bundle

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_score.py \\
      --prediction ${prediction} --prediction-receipt ${prediction_receipt} \\
      --features ${valid_features} --features-receipt ${valid_features_receipt} --truth ${valid_truth} \\
      --root '${root}' --candidate-id '${candidate_id}' --family '${family}' --arm '${arm}' \\
      --output '${candidate_id}.${family}.${arm}.metrics.json'
    """
}

process M37_TRACE_COLLECT_METRICS {
    tag { "${root}_paired_metrics" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/promotion" }, mode: 'copy', overwrite: false
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    tuple val(root), path(metric_files), path(receipt_files)
    path source_files

    output:
    tuple val(root), path("m37.${root}.paired_metrics.json"), path("m37.${root}.paired_metrics.receipt.json"), emit: bundle

    script:
    def metricFlags = metric_files.collect { path -> "--metric '${path}'" }.join(' ')
    def receiptFlags = receipt_files.collect { path -> "--receipt '${path}'" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_collect_metrics.py \\
      --root '${root}' --expected-evaluation-split FIT_TUNE \\
      ${metricFlags} ${receiptFlags} --output 'm37.${root}.paired_metrics.json'
    """
}

process M37_TRACE_SUCCESSIVE_HALVING {
    tag { "${root}_paired-candidate-promotion" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/promotion" }, mode: 'copy', overwrite: false
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    tuple val(root), path(metrics_json), path(metrics_receipt)
    path source_files

    output:
    tuple val(root), path('m37.successive_halving.json'), path('m37.successive_halving.receipt.json'), emit: plan

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_trace_successive_halving.py \\
      --metrics-json ${metrics_json} --metrics-receipt ${metrics_receipt} \\
      --keep-fraction '${params.m37_keep_fraction}' \\
      --boundary-tolerance-cm '${params.m37_promotion_boundary_tolerance_cm}' \\
      --minimum-f1-gain '${params.m37_promotion_minimum_f1_gain}' \\
      --maximum-log-loss-increase '${params.m37_promotion_maximum_log_loss_increase}' \\
      --bootstrap-seed '${params.m37_promotion_bootstrap_seed}' \\
      --bootstrap-draws '${params.m37_promotion_bootstrap_draws}' \\
      --minimum-replication-roots '${params.m37_promotion_minimum_replication_roots}' \\
      --output m37.successive_halving.json
    """
}

process M37_TRACE_READY {
    tag { "${root}_${candidate_id}_${family}_${arm}" }
    publishDir { "${params.m37_results_dir}/${params.m37_run_id}/provenance" }, mode: 'copy', overwrite: false

    input:
    tuple val(root), val(candidate_id), val(family), val(arm), path(metrics), path(receipt)
    path run_overlay
    val run_overlay_uri
    path auth_files

    output:
    tuple val(root), val(candidate_id), val(family), val(arm), path("${candidate_id}.${family}.${arm}.manifest.json"), path("${candidate_id}.${family}.${arm}.READY.json"), emit: ready

    script:
    def authFlags = auth_files.collect { path -> "--auth-file '${path}'" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${auth_files} staged/bin/
    python3 staged/bin/m37_trace_provenance.py \\
      --artifact ${metrics} --receipt ${receipt} --run-id '${params.m37_run_id}' \\
      --container-digest '${params.m37_container_digest}' \\
      --root '${root}' --candidate-id '${candidate_id}' --family '${family}' --arm '${arm}' \\
      --run-overlay ${run_overlay} --run-overlay-uri '${run_overlay_uri}' \\
      ${authFlags} \\
      --output-prefix '${candidate_id}.${family}.${arm}'
    """
}

process M37_ADAPT_M34_TRACE_TRUTH {
    tag { 'm34-phase-to-trace-diploid' }
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    path m34_truth
    path m34_f0
    path source_files

    output:
    path 'm37.trace_truth.npz', emit: truth
    path 'm37.trace_truth.receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m37_m34_adapter.py \\
      --m34-truth ${m34_truth} --m34-f0 ${m34_f0} --output m37.trace_truth.npz
    """
}
