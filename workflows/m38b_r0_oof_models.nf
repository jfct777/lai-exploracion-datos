nextflow.enable.dsl=2

include {
  M38B_AUTHENTICATE_MODEL_CONTRACT;
  M38B_APPLY_PRIMARY_SUBSET;
  M38B_BIND_MARKER_AXIS;
  M38B_STRICT_SHAM;
  M38B_MATERIALIZE_ARM;
  M38B_FREEZE_FOLDS;
  M38B_POSITIVE_CONTROL;
  M38B_PARTITION_FEATURES;
  M38B_PARTITION_TRUTH;
  M38B_TRAIN_FOLD;
  M38B_COLLECT_OOF;
  M38B_PACK_BASELINES;
  M38B_PACK_TRUTH;
  M38B_SCORE_FAMILY;
  M38B_SCORE_POSITIVE;
  M38B_FINAL_DECISION
} from '../modules/38B_OOF_MODELS'

workflow {
    def required = [
        'm38b_oof_run_id', 'm38b_oof_results_dir', 'm38b_oof_loo_subset',
        'm38b_oof_loo_receipt', 'm38b_oof_loo_sha256',
        'm38b_oof_loo_receipt_sha256', 'm38b_oof_selected',
        'm38b_oof_target', 'm38b_oof_reference', 'm38b_oof_minus_f0',
        'm38b_oof_full_f0', 'm38b_oof_truth', 'm38b_oof_marker_cm',
        'm38b_oof_alignment_receipt',
    ]
    required.each { key -> if (!params[key]) error "--${key} is required" }
    if (params.m38b_oof_results_dir != 'gs://teams-usp/frank/lai-exploracion-datos/runs')
        error 'M38B derived outputs must remain in the personal project bucket'
    if (params.m38b_oof_root != 'R0' || params.m38b_oof_partition != 'FIT')
        error 'M38B model workflow is restricted to chr22 R0 FIT'
    if (!(params.m38b_oof_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error 'M38B run ID is unsafe'
    def forbidden = [params.m38b_oof_loo_subset, params.m38b_oof_selected,
                     params.m38b_oof_target, params.m38b_oof_reference,
                     params.m38b_oof_minus_f0, params.m38b_oof_full_f0,
                     params.m38b_oof_truth, params.m38b_oof_marker_cm]*.toString()
    if (forbidden.any { value -> value.toLowerCase().contains('/valid/') ||
                                  value.toLowerCase().contains('/test/') })
        error 'VALID and TEST inputs are forbidden'
    def hashKeys = [
        'm38b_oof_loo_sha256', 'm38b_oof_loo_receipt_sha256',
        'm38b_oof_selected_sha256',
        'm38b_oof_target_sha256', 'm38b_oof_reference_sha256',
        'm38b_oof_minus_f0_sha256', 'm38b_oof_full_f0_sha256',
        'm38b_oof_truth_sha256', 'm38b_oof_marker_cm_sha256',
        'm38b_oof_alignment_receipt_sha256',
        'm38b_oof_contract_sha256', 'm38b_oof_amendment_1_sha256',
        'm38b_oof_amendment_2_sha256',
    ]
    if (!hashKeys.every { key -> params[key] instanceof String &&
                                params[key] ==~ /[0-9a-f]{64}/ })
        error 'M38B canonical SHA-256 pin is missing or malformed'
    def expectedImage = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99'
    if (params.m38b_oof_python_image != expectedImage || params.m38b_oof_container_user != '0:0')
        error 'M38B runtime image or user differs from the pinned runtime'
    if (params.m38b_oof_lambda_grid != '0,0.25,0.5,1,2' ||
        params.m38b_oof_sham_seed != 3401103 ||
        params.m38b_oof_outer_seed != 38032026 ||
        params.m38b_oof_inner_seed_start != 38100000 ||
        params.m38b_oof_bootstrap_seed != 38200103 ||
        params.m38b_oof_bootstrap_replicates != 10000 ||
        params.m38b_oof_beta_prior_strength != 0.5 ||
        params.m38b_oof_event_radius_cm != 0.2)
        error 'M38B frozen analytical, SHAM, or bootstrap parameters differ'

    def repoDir = projectDir.resolve('..')
    def contract = file("${repoDir}/conf/m38b_r0_oof_contract.json", checkIfExists: true)
    def amendment1 = file("${repoDir}/conf/m38b_r0_oof_amendment_1.json", checkIfExists: true)
    def amendment2 = file("${repoDir}/conf/m38b_r0_oof_amendment_2.json", checkIfExists: true)
    def sourceNames = [
        'm33_safe_bridge_core.py', 'm34_generate_mosaics.py',
        'm34_parse_flare_truth.py',
        'm37_trace_core.py', 'm37_bind_marker_axis.py',
        'm37_trace_materialize.py', 'm37_trace_train.py', 'm38b_oof_core.py',
        'm38b_validate_model_contract.py', 'm38b_subset_factors.py',
        'm38b_bind_marker_axis.py', 'm38b_strict_sham.py', 'm38b_materialize_arm.py',
        'm38b_make_folds.py', 'm38b_positive_control.py', 'm38b_partition_fold.py',
        'm38b_train_fold.py', 'm38b_collect_oof.py', 'm38b_pack_scoring.py',
        'm38b_score_oof.py', 'm38b_score_positive.py', 'm38b_decide.py',
    ]
    sources = Channel.value(sourceNames.collect { name ->
        file("${repoDir}/bin/${name}", checkIfExists: true)
    })
    provenanceSources = Channel.value([
        file("${repoDir}/conf/m38b_r0_oof_models.config", checkIfExists: true),
        file("${repoDir}/modules/38B_OOF_MODELS.nf", checkIfExists: true),
        file("${repoDir}/workflows/m38b_r0_oof_models.nf", checkIfExists: true),
    ])

    M38B_AUTHENTICATE_MODEL_CONTRACT(
        contract, amendment1, amendment2, sources, provenanceSources,
    )
    contractReceipt = M38B_AUTHENTICATE_MODEL_CONTRACT.out.receipt.first()

    M38B_APPLY_PRIMARY_SUBSET(
        file(params.m38b_oof_loo_subset, checkIfExists: true),
        file(params.m38b_oof_loo_receipt, checkIfExists: true),
        file(params.m38b_oof_selected, checkIfExists: true),
        file(params.m38b_oof_target, checkIfExists: true),
        file(params.m38b_oof_reference, checkIfExists: true), sources,
    )
    factorDir = M38B_APPLY_PRIMARY_SUBSET.out.directory.first()
    M38B_BIND_MARKER_AXIS(
        file(params.m38b_oof_minus_f0, checkIfExists: true),
        file(params.m38b_oof_marker_cm, checkIfExists: true),
        file(params.m38b_oof_alignment_receipt, checkIfExists: true), sources,
    )
    markerBundle = M38B_BIND_MARKER_AXIS.out.bundle.first()
    M38B_STRICT_SHAM(factorDir, sources)
    shamBundle = M38B_STRICT_SHAM.out.bundle.first()

    materialRows = factorDir.combine(markerBundle).combine(shamBundle).flatMap {
        dir, markerAxis, markerReceipt, shamReference, shamReceipt ->
        def selected = dir.resolve('m38b_primary_selected_loci.npz')
        def target = dir.resolve('m38b_primary_target_rare_diploid.npz')
        def reference = dir.resolve('m38b_primary_reference_rare_summary.npz')
        def factorsReceipt = dir.resolve('m38b_primary_factor_subset.receipt.json')
        def fminus = file(params.m38b_oof_minus_f0, checkIfExists: true)
        [
          tuple('RE', selected, target, reference, factorsReceipt, factorsReceipt,
                fminus, markerAxis, markerReceipt),
          tuple('RD', selected, target, reference, factorsReceipt, factorsReceipt,
                fminus, markerAxis, markerReceipt),
          tuple('SHAM', selected, target, shamReference, factorsReceipt, shamReceipt,
                fminus, markerAxis, markerReceipt),
        ]
    }
    M38B_MATERIALIZE_ARM(materialRows, sources)
    materialBundles = M38B_MATERIALIZE_ARM.out.bundle
    reBundle = materialBundles.filter { arm, feature, receipt -> arm == 'RE' }.first()
    rdBundle = materialBundles.filter { arm, feature, receipt -> arm == 'RD' }.first()

    M38B_FREEZE_FOLDS(
        reBundle.map { arm, feature, receipt -> tuple(feature, receipt) }, sources,
    )
    foldBundle = M38B_FREEZE_FOLDS.out.bundle.first()

    truthPath = file(params.m38b_oof_truth, checkIfExists: true)
    alignmentReceipt = file(params.m38b_oof_alignment_receipt, checkIfExists: true)
    positiveRows = reBundle.combine(rdBundle).combine(foldBundle).flatMap {
        reArm, reFeature, reReceipt, rdArm, rdFeature, rdReceipt, folds, foldsReceipt ->
        [
          tuple(0, 'POS_d0', 0.0), tuple(0, 'POS_d0p25', 0.25),
          tuple(0, 'POS_d0p5', 0.5), tuple(0, 'POS_d1', 1.0), tuple(0, 'POS_d2', 2.0),
          tuple(1, 'POS_d0', 0.0), tuple(1, 'POS_d0p25', 0.25),
          tuple(1, 'POS_d0p5', 0.5), tuple(1, 'POS_d1', 1.0), tuple(1, 'POS_d2', 2.0),
          tuple(2, 'POS_d0', 0.0), tuple(2, 'POS_d0p25', 0.25),
          tuple(2, 'POS_d0p5', 0.5), tuple(2, 'POS_d1', 1.0), tuple(2, 'POS_d2', 2.0),
        ].collect { row -> tuple(row[0], row[1], row[2], reFeature, reReceipt,
                                 rdFeature, rdReceipt, truthPath, alignmentReceipt,
                                 folds, foldsReceipt) }
    }
    M38B_POSITIVE_CONTROL(positiveRows, sources)

    realPartitionRows = materialBundles.filter { arm, feature, receipt -> arm != 'RD' }
        .combine(foldBundle).flatMap { arm, feature, receipt, folds, foldsReceipt ->
            (0..<3).collect { fold -> tuple(fold, arm, arm, 'NA', feature, receipt,
                                            folds, foldsReceipt) }
        }
    positivePartitionRows = M38B_POSITIVE_CONTROL.out.bundle.combine(foldBundle).map {
        fold, identity, delta, feature, receipt, folds, foldsReceipt ->
        tuple(fold, 'POSITIVE', identity, delta, feature, receipt, folds, foldsReceipt)
    }
    M38B_PARTITION_FEATURES(realPartitionRows.mix(positivePartitionRows), sources)

    truthPartitionRows = foldBundle.flatMap { folds, foldsReceipt ->
        (0..<3).collect { fold -> tuple(fold, truthPath, alignmentReceipt, folds, foldsReceipt) }
    }
    M38B_PARTITION_TRUTH(truthPartitionRows, sources)

    // Seven feature rows share each fold; combine deliberately broadcasts the
    // single authenticated truth row to all seven without lossy duplicate-key joins.
    joined = M38B_PARTITION_FEATURES.out.bundle.combine(M38B_PARTITION_TRUTH.out.bundle, by: 0)
    trainRows = joined.flatMap { fold, arm, identity, delta, fitFeatures, scoreFeatures,
                                featureReceipt, fitTruth, scoreTruth, truthReceipt ->
        if (arm == 'POSITIVE') {
            return [1103, 2207, 3301].collect { seed ->
                tuple('tcn', arm, identity, delta, fold, seed, fitFeatures,
                      scoreFeatures, featureReceipt, fitTruth, truthReceipt)
            }
        }
        def rows = [tuple('analytic', arm, identity, delta, fold, 1103, fitFeatures,
                          scoreFeatures, featureReceipt, fitTruth, truthReceipt)]
        rows.addAll([1103, 2207, 3301].collect { seed ->
            tuple('tcn', arm, identity, delta, fold, seed, fitFeatures,
                  scoreFeatures, featureReceipt, fitTruth, truthReceipt)
        })
        return rows
    }
    M38B_TRAIN_FOLD(trainRows, contractReceipt, sources)

    groupedPredictions = M38B_TRAIN_FOLD.out.bundle.map {
        family, arm, identity, delta, fold, seed, prediction, receipt ->
        tuple([family, arm, identity, delta], prediction, receipt)
    }.groupTuple(by: 0).combine(foldBundle).map {
        key, predictions, receipts, folds, foldsReceipt ->
        tuple(key[0], key[1], key[2], key[3], predictions, receipts, folds, foldsReceipt)
    }
    M38B_COLLECT_OOF(groupedPredictions, sources)

    baselineRows = foldBundle.map { folds, foldsReceipt ->
        tuple(file(params.m38b_oof_full_f0, checkIfExists: true),
              file(params.m38b_oof_minus_f0, checkIfExists: true),
              file(params.m38b_oof_marker_cm, checkIfExists: true), alignmentReceipt,
              folds, foldsReceipt)
    }
    M38B_PACK_BASELINES(baselineRows, contractReceipt, sources)
    truthRows = foldBundle.map { folds, foldsReceipt ->
        tuple(truthPath, file(params.m38b_oof_marker_cm, checkIfExists: true),
              alignmentReceipt, folds, foldsReceipt)
    }
    M38B_PACK_TRUTH(truthRows, contractReceipt, sources)

    baselineDir = M38B_PACK_BASELINES.out.directory.first()
    scoreTruth = M38B_PACK_TRUTH.out.bundle.first()
    realScores = M38B_COLLECT_OOF.out.bundle
        .filter { family, arm, identity, delta, prediction, receipt -> arm != 'POSITIVE' }
        .map { family, arm, identity, delta, prediction, receipt ->
            tuple(family, arm, prediction, receipt)
        }.groupTuple(by: 0).combine(baselineDir).combine(scoreTruth).map {
            family, arms, predictions, receipts, baseDir, truth, truthReceipt ->
            tuple(family, arms, predictions, receipts, baseDir, truth, truthReceipt)
        }
    M38B_SCORE_FAMILY(realScores, sources)

    positiveScores = M38B_COLLECT_OOF.out.bundle
        .filter { family, arm, identity, delta, prediction, receipt -> arm == 'POSITIVE' }
        .map { family, arm, identity, delta, prediction, receipt ->
            tuple('positive', identity, prediction, receipt)
        }.groupTuple(by: 0).combine(scoreTruth).map {
            key, identities, predictions, receipts, truth, truthReceipt ->
            tuple(identities, predictions, receipts, truth, truthReceipt)
        }
    M38B_SCORE_POSITIVE(positiveScores, sources)

    familyScores = M38B_SCORE_FAMILY.out.bundle.collect()
        .combine(M38B_SCORE_POSITIVE.out.bundle.first()).map {
            rows, positive, positiveReceipt ->
            def byFamily = rows.collectEntries { row -> [(row[0]): row] }
            if (byFamily.keySet() != ['analytic', 'tcn'] as Set)
                error 'M38B final decision requires analytic and TCN scores'
            tuple(byFamily.analytic[1], byFamily.analytic[3],
                  byFamily.tcn[1], byFamily.tcn[3], positive, positiveReceipt)
        }
    M38B_FINAL_DECISION(familyScores, sources)
}
