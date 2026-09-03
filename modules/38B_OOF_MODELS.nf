nextflow.enable.dsl=2

process M38B_AUTHENTICATE_MODEL_CONTRACT {
    tag 'M38B_contract_chain'
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/contract" }, mode: 'copy', overwrite: false
    cpus 1; memory '2 GB'; time '10m'; maxForks 1
    input:
    path contract
    path amendment1
    path amendment2
    path sourceFiles
    path provenanceFiles
    output:
    path 'm38b.model_contract.receipt.json', emit: receipt
    script:
    def sourceFlags = (sourceFiles + provenanceFiles + [contract, amendment1, amendment2])
        .collect { "--source '${it}'" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_validate_model_contract.py \
      --contract '${contract}' --amendment '${amendment1}' --amendment-2 '${amendment2}' \
      --expected-contract-sha256 '${params.m38b_oof_contract_sha256}' \
      --expected-amendment-sha256 '${params.m38b_oof_amendment_1_sha256}' \
      --expected-amendment-2-sha256 '${params.m38b_oof_amendment_2_sha256}' \
      ${sourceFlags} \
      --output m38b.model_contract.receipt.json
    """
    stub: "touch m38b.model_contract.receipt.json"
}

process M38B_APPLY_PRIMARY_SUBSET {
    tag 'chr22_R0_FIT_Sstar'
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/factors" }, mode: 'copy', overwrite: false
    cpus 1; memory '4 GB'; time '20m'; maxForks 1
    input:
    path looSubset
    path looReceipt
    path selected
    path target
    path reference
    path sourceFiles
    output:
    path 'm38b_primary_factors', emit: directory
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_subset_factors.py \
      --loo-subset '${looSubset}' --loo-receipt '${looReceipt}' \
      --selected '${selected}' --target '${target}' --reference '${reference}' \
      --expected-loo-sha256 '${params.m38b_oof_loo_sha256}' \
      --expected-loo-receipt-sha256 '${params.m38b_oof_loo_receipt_sha256}' \
      --expected-selected-sha256 '${params.m38b_oof_selected_sha256}' \
      --expected-target-sha256 '${params.m38b_oof_target_sha256}' \
      --expected-reference-sha256 '${params.m38b_oof_reference_sha256}' \
      --expected-loci 660 --outdir m38b_primary_factors
    """
    stub: "mkdir -p m38b_primary_factors && touch m38b_primary_factors/m38b_primary_selected_loci.npz m38b_primary_factors/m38b_primary_target_rare_diploid.npz m38b_primary_factors/m38b_primary_reference_rare_summary.npz m38b_primary_factors/m38b_primary_factor_subset.receipt.json"
}

process M38B_BIND_MARKER_AXIS {
    tag 'chr22_R0_FIT_Fminus_axis'
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/axes" }, mode: 'copy', overwrite: false
    cpus 1; memory '4 GB'; time '20m'; maxForks 1
    input:
    path minusF0
    path markerCm
    path alignmentReceipt
    path sourceFiles
    output:
    tuple path('m38b.marker_axis.npz'), path('m38b.marker_axis.receipt.json'), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_bind_marker_axis.py \
      --f0 '${minusF0}' --marker-cm '${markerCm}' --alignment-receipt '${alignmentReceipt}' \
      --expected-f0-sha256 '${params.m38b_oof_minus_f0_sha256}' \
      --expected-marker-cm-sha256 '${params.m38b_oof_marker_cm_sha256}' \
      --expected-alignment-receipt-sha256 '${params.m38b_oof_alignment_receipt_sha256}' \
      --output m38b.marker_axis.npz --adapter-receipt m38b.marker_axis.adapter.receipt.json
    """
    stub: "touch m38b.marker_axis.npz m38b.marker_axis.receipt.json"
}

process M38B_STRICT_SHAM {
    tag 'chr22_R0_FIT_SHAM_seed3401103'
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/controls/sham" }, mode: 'copy', overwrite: false
    cpus 1; memory '4 GB'; time '20m'; maxForks 1
    input:
    path factorDir
    path sourceFiles
    output:
    tuple path('m38b.strict_sham.reference.npz'), path('m38b.strict_sham.reference.receipt.json'), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_strict_sham.py \
      --reference '${factorDir}/m38b_primary_reference_rare_summary.npz' \
      --source-receipt '${factorDir}/m38b_primary_factor_subset.receipt.json' \
      --seed '${params.m38b_oof_sham_seed}' --output m38b.strict_sham.reference.npz \
      --receipt m38b.strict_sham.reference.receipt.json
    """
    stub: "touch m38b.strict_sham.reference.npz m38b.strict_sham.reference.receipt.json"
}

process M38B_MATERIALIZE_ARM {
    tag { "chr22_R0_FIT_${arm}_Sstar" }
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/features" }, mode: 'copy', overwrite: false
    cpus 4; memory '12 GB'; time '60m'; maxForks params.m38b_oof_materialize_max_forks
    input:
    tuple val(arm), path(selected), path(target), path(reference), path(factorsReceipt),
          path(referenceReceipt), path(minusF0), path(markerAxis), path(markerAxisReceipt)
    path sourceFiles
    output:
    tuple val(arm), path("m38b.${arm}.trace.npz"), path("m38b.${arm}.trace.receipt.json"), emit: bundle
    script:
    def shamFlag = arm == 'SHAM' ? "--reference-receipt '${referenceReceipt}'" : ''
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_materialize_arm.py \
      --selected '${selected}' --target '${target}' --reference '${reference}' \
      --factors-receipt '${factorsReceipt}' ${shamFlag} --f0 '${minusF0}' \
      --marker-axis '${markerAxis}' --marker-axis-receipt '${markerAxisReceipt}' \
      --arm '${arm}' --beta-prior-strength '${params.m38b_oof_beta_prior_strength}' \
      --output 'm38b.${arm}.trace.npz'
    """
    stub: "touch m38b.${arm}.trace.npz m38b.${arm}.trace.receipt.json"
}

process M38B_FREEZE_FOLDS {
    tag 'chr22_R0_FIT_3fold'
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/folds" }, mode: 'copy', overwrite: false
    cpus 1; memory '4 GB'; time '20m'; maxForks 1
    input:
    tuple path(sourceFeatures), path(sourceReceipt)
    path sourceFiles
    output:
    tuple path('m38b.folds.npz'), path('m38b.folds.receipt.json'), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_make_folds.py \
      --source '${sourceFeatures}' --source-receipt '${sourceReceipt}' \
      --outer-seed '${params.m38b_oof_outer_seed}' --inner-seed-start '${params.m38b_oof_inner_seed_start}' \
      --output m38b.folds.npz
    """
    stub: "touch m38b.folds.npz m38b.folds.receipt.json"
}

process M38B_POSITIVE_CONTROL {
    tag { "POS_fold${fold}_${logicalId}" }
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/controls/positive/features" }, mode: 'copy', overwrite: false
    cpus 1; memory '8 GB'; time '30m'; maxForks params.m38b_oof_positive_max_forks
    input:
    tuple val(fold), val(logicalId), val(delta), path(reFeatures), path(reReceipt),
          path(rdFeatures), path(rdReceipt), path(truth), path(truthReceipt),
          path(folds), path(foldsReceipt)
    path sourceFiles
    output:
    tuple val(fold), val(logicalId), val(delta), path("m38b.${logicalId}.fold${fold}.npz"),
          path("m38b.${logicalId}.fold${fold}.receipt.json"), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_positive_control.py \
      --real-features '${reFeatures}' --real-receipt '${reReceipt}' \
      --rd-features '${rdFeatures}' --rd-receipt '${rdReceipt}' \
      --truth '${truth}' --truth-receipt '${truthReceipt}' \
      --folds '${folds}' --folds-receipt '${foldsReceipt}' --fold '${fold}' --delta '${delta}' \
      --output 'm38b.${logicalId}.fold${fold}.npz' --receipt 'm38b.${logicalId}.fold${fold}.receipt.json'
    """
    stub: "touch m38b.${logicalId}.fold${fold}.npz m38b.${logicalId}.fold${fold}.receipt.json"
}

process M38B_PARTITION_FEATURES {
    tag { "${identity}_fold${fold}" }
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/partitions/features" }, mode: 'copy', overwrite: false
    cpus 1; memory '8 GB'; time '45m'; maxForks params.m38b_oof_partition_max_forks
    input:
    tuple val(fold), val(arm), val(identity), val(delta), path(sourceFeatures), path(sourceReceipt), path(folds), path(foldsReceipt)
    path sourceFiles
    output:
    tuple val(fold), val(arm), val(identity), val(delta), path("m38b.${identity}.fold${fold}.fit.features.npz"),
          path("m38b.${identity}.fold${fold}.score.features.npz"),
          path("m38b.${identity}.fold${fold}.features.receipt.json"), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_partition_fold.py features \
      --source '${sourceFeatures}' --source-receipt '${sourceReceipt}' --arm '${arm}' \
      --folds '${folds}' --folds-receipt '${foldsReceipt}' --fold '${fold}' \
      --fit-output 'm38b.${identity}.fold${fold}.fit.features.npz' \
      --score-output 'm38b.${identity}.fold${fold}.score.features.npz' \
      --receipt 'm38b.${identity}.fold${fold}.features.receipt.json'
    """
    stub: "touch m38b.${identity}.fold${fold}.fit.features.npz m38b.${identity}.fold${fold}.score.features.npz m38b.${identity}.fold${fold}.features.receipt.json"
}

process M38B_PARTITION_TRUTH {
    tag { "truth_fold${fold}" }
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/partitions/truth" }, mode: 'copy', overwrite: false
    cpus 1; memory '8 GB'; time '45m'; maxForks 3
    input:
    tuple val(fold), path(sourceTruth), path(sourceReceipt), path(folds), path(foldsReceipt)
    path sourceFiles
    output:
    tuple val(fold), path("m38b.fold${fold}.fit.truth.npz"), path("m38b.fold${fold}.score.truth.npz"),
          path("m38b.fold${fold}.truth.receipt.json"), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_partition_fold.py truth \
      --source '${sourceTruth}' --source-receipt '${sourceReceipt}' \
      --expected-source-sha256 '${params.m38b_oof_truth_sha256}' \
      --expected-source-receipt-sha256 '${params.m38b_oof_alignment_receipt_sha256}' \
      --folds '${folds}' --folds-receipt '${foldsReceipt}' --fold '${fold}' \
      --fit-output 'm38b.fold${fold}.fit.truth.npz' --score-output 'm38b.fold${fold}.score.truth.npz' \
      --receipt 'm38b.fold${fold}.truth.receipt.json'
    """
    stub: "touch m38b.fold${fold}.fit.truth.npz m38b.fold${fold}.score.truth.npz m38b.fold${fold}.truth.receipt.json"
}

process M38B_TRAIN_FOLD {
    tag { "${family}_${identity}_fold${fold}_seed${seed}" }
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/predictions/folds" }, mode: 'copy', overwrite: false
    cpus 4; memory '16 GB'; time '2h'; maxForks params.m38b_oof_train_max_forks
    input:
    tuple val(family), val(arm), val(identity), val(delta), val(fold), val(seed), path(fitFeatures),
          path(scoreFeatures), path(featureReceipt), path(fitTruth), path(truthReceipt)
    path modelContractReceipt
    path sourceFiles
    output:
    tuple val(family), val(arm), val(identity), val(delta), val(fold), val(seed),
          path("m38b.${family}.${identity}.fold${fold}.seed${seed}.prediction.npz"),
          path("m38b.${family}.${identity}.fold${fold}.seed${seed}.prediction.receipt.json"), emit: bundle
    tuple val(family), val(arm), val(identity), val(delta), val(fold), val(seed),
          path("m38b.${family}.${identity}.fold${fold}.seed${seed}.checkpoint.pt"),
          optional: true, emit: checkpoints
    script:
    def checkpointFlag = family == 'tcn' ? "--checkpoint 'm38b.${family}.${identity}.fold${fold}.seed${seed}.checkpoint.pt'" : ''
    """
    set -euo pipefail
    export USER=m38b-runner LOGNAME=m38b-runner TORCHINDUCTOR_CACHE_DIR=\"\$PWD/.torch-cache\"
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_train_fold.py \
      --family '${family}' --arm '${arm}' --fold '${fold}' --seed '${seed}' \
      --fit-features '${fitFeatures}' --score-features '${scoreFeatures}' --feature-receipt '${featureReceipt}' \
      --fit-truth '${fitTruth}' --truth-receipt '${truthReceipt}' --model-contract-receipt '${modelContractReceipt}' \
      --lambda-grid '${params.m38b_oof_lambda_grid}' --event-radius-cm '${params.m38b_oof_event_radius_cm}' \
      --evidence-scale '${params.m38b_oof_evidence_scale}' --hidden-dim '${params.m38b_oof_hidden_dim}' \
      --depth '${params.m38b_oof_depth}' --kernel-size '${params.m38b_oof_kernel_size}' \
      --dropout '${params.m38b_oof_dropout}' --dilations '${params.m38b_oof_dilations}' \
      --updates '${params.m38b_oof_updates}' --learning-rate '${params.m38b_oof_learning_rate}' \
      --batch-people '${params.m38b_oof_batch_people}' --marker-shard '${params.m38b_oof_marker_shard}' \
      --validation-every '${params.m38b_oof_validation_every}' --patience '${params.m38b_oof_patience}' \
      ${checkpointFlag} --output 'm38b.${family}.${identity}.fold${fold}.seed${seed}.prediction.npz' \
      --receipt 'm38b.${family}.${identity}.fold${fold}.seed${seed}.prediction.receipt.json'
    """
    stub: "touch m38b.${family}.${identity}.fold${fold}.seed${seed}.prediction.npz m38b.${family}.${identity}.fold${fold}.seed${seed}.prediction.receipt.json"
}

process M38B_COLLECT_OOF {
    tag { "${family}_${identity}_OOF" }
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/predictions/oof" }, mode: 'copy', overwrite: false
    cpus 1; memory '8 GB'; time '30m'; maxForks params.m38b_oof_collect_max_forks
    input:
    tuple val(family), val(arm), val(identity), val(delta), path(predictions), path(predictionReceipts), path(folds), path(foldsReceipt)
    path sourceFiles
    output:
    tuple val(family), val(arm), val(identity), val(delta), path("m38b.${family}.${identity}.oof.npz"),
          path("m38b.${family}.${identity}.oof.receipt.json"), emit: bundle
    script:
    def predictionFlags = predictions.collect { "--prediction '${it}'" }.join(' ')
    def receiptFlags = predictionReceipts.collect { "--prediction-receipt '${it}'" }.join(' ')
    def deltaFlag = arm == 'POSITIVE' ? "--positive-delta '${delta}'" : ''
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_collect_oof.py \
      --folds '${folds}' --folds-receipt '${foldsReceipt}' ${predictionFlags} ${receiptFlags} \
      --family '${family}' --arm '${arm}' ${deltaFlag} --output 'm38b.${family}.${identity}.oof.npz' \
      --receipt 'm38b.${family}.${identity}.oof.receipt.json'
    """
    stub: "touch m38b.${family}.${identity}.oof.npz m38b.${family}.${identity}.oof.receipt.json"
}

process M38B_PACK_BASELINES {
    tag 'chr22_R0_FIT_baselines'
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/predictions/baselines" }, mode: 'copy', overwrite: false
    cpus 1; memory '8 GB'; time '30m'; maxForks 1
    input:
    tuple path(fullF0), path(minusF0), path(markerCm), path(alignmentReceipt), path(folds), path(foldsReceipt)
    path modelContractReceipt
    path sourceFiles
    output:
    path 'm38b_packed_baselines', emit: directory
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_pack_scoring.py baselines \
      --full-f0 '${fullF0}' --minus-f0 '${minusF0}' --marker-cm '${markerCm}' \
      --alignment-receipt '${alignmentReceipt}' --folds '${folds}' --folds-receipt '${foldsReceipt}' \
      --expected-full-f0-sha256 '${params.m38b_oof_full_f0_sha256}' \
      --expected-minus-f0-sha256 '${params.m38b_oof_minus_f0_sha256}' \
      --expected-marker-cm-sha256 '${params.m38b_oof_marker_cm_sha256}' \
      --expected-alignment-receipt-sha256 '${params.m38b_oof_alignment_receipt_sha256}' \
      --model-contract-receipt '${modelContractReceipt}' --outdir m38b_packed_baselines
    """
    stub: "mkdir -p m38b_packed_baselines && touch m38b_packed_baselines/m38b_full.oof.npz m38b_packed_baselines/m38b_minus.oof.npz m38b_packed_baselines/m38b_RD.oof.npz m38b_packed_baselines/m38b_baselines.receipt.json"
}

process M38B_PACK_TRUTH {
    tag 'chr22_R0_FIT_score_truth'
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/score" }, mode: 'copy', overwrite: false
    cpus 1; memory '8 GB'; time '30m'; maxForks 1
    input:
    tuple path(truth), path(markerCm), path(alignmentReceipt), path(folds), path(foldsReceipt)
    path modelContractReceipt
    path sourceFiles
    output:
    tuple path('m38b.score.truth.npz'), path('m38b.score.truth.receipt.json'), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_pack_scoring.py truth \
      --truth '${truth}' --marker-cm '${markerCm}' --alignment-receipt '${alignmentReceipt}' \
      --expected-truth-sha256 '${params.m38b_oof_truth_sha256}' \
      --expected-marker-cm-sha256 '${params.m38b_oof_marker_cm_sha256}' \
      --expected-alignment-receipt-sha256 '${params.m38b_oof_alignment_receipt_sha256}' \
      --folds '${folds}' --folds-receipt '${foldsReceipt}' --model-contract-receipt '${modelContractReceipt}' \
      --output m38b.score.truth.npz
    """
    stub: "touch m38b.score.truth.npz m38b.score.truth.receipt.json"
}

process M38B_SCORE_FAMILY {
    tag { "${family}_OOF_score" }
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/score" }, mode: 'copy', overwrite: false
    cpus 2; memory '12 GB'; time '60m'; maxForks 2
    input:
    tuple val(family), val(arms), path(oofPredictions), path(oofReceipts), path(baselineDir), path(truth), path(truthReceipt)
    path sourceFiles
    output:
    tuple val(family), path("m38b.${family}.metrics.json"), path("m38b.${family}.metrics.per_person.npz"),
          path("m38b.${family}.metrics.receipt.json"), emit: bundle
    script:
    def byArm = [:]; def receiptByArm = [:]
    arms.eachWithIndex { arm, i -> byArm[arm as String] = oofPredictions[i]; receiptByArm[arm as String] = oofReceipts[i] }
    if (!byArm.keySet().containsAll(['RE','SHAM'])) throw new IllegalArgumentException("${family} lacks RE/SHAM")
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_score_oof.py --family '${family}' \
      --prediction 'full=${baselineDir}/m38b_full.oof.npz' --prediction 'minus=${baselineDir}/m38b_minus.oof.npz' \
      --prediction 'RD=${baselineDir}/m38b_RD.oof.npz' --prediction 'RE=${byArm['RE']}' \
      --prediction 'SHAM=${byArm['SHAM']}' \
      --prediction-receipt 'full=${baselineDir}/m38b_baselines.receipt.json' \
      --prediction-receipt 'minus=${baselineDir}/m38b_baselines.receipt.json' \
      --prediction-receipt 'RD=${baselineDir}/m38b_baselines.receipt.json' \
      --prediction-receipt 'RE=${receiptByArm['RE']}' --prediction-receipt 'SHAM=${receiptByArm['SHAM']}' \
      --truth '${truth}' --truth-receipt '${truthReceipt}' \
      --contrast 'full-minus=full,minus' --contrast 'RE-RD=RE,RD' --contrast 'RE-SHAM=RE,SHAM' \
      --contrast 'RE-full=RE,full' --bootstrap-replicates '${params.m38b_oof_bootstrap_replicates}' \
      --bootstrap-seed '${params.m38b_oof_bootstrap_seed}' --output 'm38b.${family}.metrics.json' \
      --per-person-output 'm38b.${family}.metrics.per_person.npz'
    """
    stub: "touch m38b.${family}.metrics.json m38b.${family}.metrics.per_person.npz m38b.${family}.metrics.receipt.json"
}

process M38B_SCORE_POSITIVE {
    tag 'TCN_positive_grid_score'
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/controls/positive/score" }, mode: 'copy', overwrite: false
    cpus 2; memory '12 GB'; time '60m'; maxForks 1
    input:
    tuple val(ids), path(predictions), path(receipts), path(truth), path(truthReceipt)
    path sourceFiles
    output:
    tuple path('m38b.positive.metrics.json'), path('m38b.positive.metrics.receipt.json'), emit: bundle
    script:
    def predictionFlags = ids.indices.collect { i -> "--prediction '${ids[i]}=${predictions[i]}'" }.join(' ')
    def receiptFlags = ids.indices.collect { i -> "--prediction-receipt '${ids[i]}=${receipts[i]}'" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_score_positive.py ${predictionFlags} ${receiptFlags} \
      --truth '${truth}' --truth-receipt '${truthReceipt}' \
      --bootstrap-replicates '${params.m38b_oof_bootstrap_replicates}' \
      --bootstrap-seed '${params.m38b_oof_bootstrap_seed}' --output m38b.positive.metrics.json
    """
    stub: "touch m38b.positive.metrics.json m38b.positive.metrics.receipt.json"
}

process M38B_FINAL_DECISION {
    tag 'M38B_prespecified_gates'
    publishDir { "${params.m38b_oof_results_dir}/${params.m38b_oof_run_id}/decision" }, mode: 'copy', overwrite: false
    cpus 1; memory '2 GB'; time '15m'; maxForks 1
    input:
    tuple path(analytic), path(analyticReceipt), path(tcn), path(tcnReceipt),
          path(positive), path(positiveReceipt)
    path sourceFiles
    output:
    tuple path('m38b.final_decision.json'), path('m38b.final_decision.receipt.json'), emit: bundle
    script:
    """
    set -euo pipefail
    mkdir -p staged/bin && cp ${sourceFiles} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_decide.py \
      --analytic '${analytic}' --analytic-receipt '${analyticReceipt}' \
      --tcn '${tcn}' --tcn-receipt '${tcnReceipt}' \
      --positive '${positive}' --positive-receipt '${positiveReceipt}' \
      --output m38b.final_decision.json
    """
    stub: "touch m38b.final_decision.json m38b.final_decision.receipt.json"
}
