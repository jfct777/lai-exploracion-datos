nextflow.enable.dsl=2

include { m37_compact_capacity_families; m37_compact_decision_parts; M37_TRACE_COMPACT_CAPACITY_SCREEN; M37_TRACE_COMPACT_CAPACITY_REPLICATION; M37_TRACE_COMPACT_SWEEP; M37_TRACE_COMPACT_DECISION } from '../modules/37_TRACE_COMPACT_SWEEP'
include { M37_TRACE_COLLECT_METRICS } from '../modules/37_TRACE_LAI'

workflow {
    ['m37_run_id', 'm37_root', 'm37_results_dir', 'm37_run_overlay_config',
     'm37_run_overlay_uri', 'm37_fit_truth', 'm37_compact_candidate_manifest',
     'm37_compact_parent_contract', 'm37_compact_contract_amendment',
     'm37_compact_canonical_metrics', 'm37_compact_canonical_metrics_receipt',
     'm37_fit_f0_receipt'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (!(params.m37_run_id ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/)) {
        error '--m37_run_id contains unsupported characters'
    }
    if (params.m37_root != 'R0') error 'the compact sweep is restricted to R0 FIT/TUNE'
    if (!(params.m37_compact_feature_files instanceof List) ||
        !(params.m37_compact_feature_receipts instanceof List) ||
        params.m37_compact_feature_files.size() != 5 ||
        params.m37_compact_feature_receipts.size() != 5) {
        error 'the compact sweep needs five materialized features and five receipts'
    }
    ['m37_valid_selected', 'm37_valid_target', 'm37_valid_reference_folds',
     'm37_valid_f0', 'm37_valid_marker_cm', 'm37_valid_f0_receipt',
     'm37_valid_truth'].each { key ->
        if (params[key]) error "--${key} is forbidden in the compact FIT/TUNE workflow"
    }

    def repoDir = projectDir.resolve('..')
    def overlayValue = params.m37_run_overlay_config as String
    def manifestValue = params.m37_compact_candidate_manifest as String
    def parentValue = params.m37_compact_parent_contract as String
    def amendmentValue = params.m37_compact_contract_amendment as String
    def runOverlay = overlayValue.startsWith('/') || overlayValue.startsWith('gs://') ?
        file(overlayValue, checkIfExists: true) : file("${repoDir}/${overlayValue}", checkIfExists: true)
    def candidateManifest = manifestValue.startsWith('/') || manifestValue.startsWith('gs://') ?
        file(manifestValue, checkIfExists: true) : file("${repoDir}/${manifestValue}", checkIfExists: true)
    def parentContract = parentValue.startsWith('/') || parentValue.startsWith('gs://') ?
        file(parentValue, checkIfExists: true) : file("${repoDir}/${parentValue}", checkIfExists: true)
    def contractAmendment = amendmentValue.startsWith('/') || amendmentValue.startsWith('gs://') ?
        file(amendmentValue, checkIfExists: true) : file("${repoDir}/${amendmentValue}", checkIfExists: true)
    def sourceFiles = [
        'm33_safe_bridge_core.py', 'm37_trace_core.py', 'm37_trace_train.py',
        'm37_trace_score.py', 'm37_trace_collect_metrics.py',
        'm37_trace_successive_halving.py', 'm37_trace_compact_sweep.py',
    ].collect { name -> file("${repoDir}/bin/${name}", checkIfExists: true) }
    sourceFiles += [
        file("${repoDir}/modules/37_TRACE_COMPACT_SWEEP.nf", checkIfExists: true),
        file("${repoDir}/workflows/m37_trace_compact_sweep.nf", checkIfExists: true),
        file("${repoDir}/conf/m37_trace_compact_sweep.config", checkIfExists: true),
        file("${repoDir}/conf/m37_trace_gcp.config", checkIfExists: true),
    ]

    def capacitySources = [
        'm33_safe_bridge_core.py', 'm37_trace_core.py', 'm37_trace_train.py',
        'm37_trace_compact_positive_control.py',
    ].collect { name -> file("${repoDir}/bin/${name}", checkIfExists: true) }

    M37_TRACE_COMPACT_CAPACITY_SCREEN(
        channel.of(tuple(candidateManifest, parentContract, contractAmendment)),
        capacitySources,
    )
    def replicationInput = M37_TRACE_COMPACT_CAPACITY_SCREEN.out.evidence.map {
        screen, screenReceipt, selection, selectionReceipt ->
        tuple(candidateManifest, parentContract, contractAmendment,
              screen, screenReceipt, selection, selectionReceipt)
    }
    M37_TRACE_COMPACT_CAPACITY_REPLICATION(replicationInput, capacitySources)

    // HMM remains an independent exploratory lane.  The synthetic artifact is
    // inspected before constructing any real-data TCN task: a failed TCN gate
    // removes only TCN, while the 12 HMM pairs remain runnable and reportable.
    def familyInput = M37_TRACE_COMPACT_CAPACITY_REPLICATION.out.evidence.flatMap { positive, positiveReceipt ->
        def families = m37_compact_capacity_families(positive)
        def featureFiles = params.m37_compact_feature_files.collect {
            value -> file(value, checkIfExists: true)
        }
        def featureReceipts = params.m37_compact_feature_receipts.collect {
            value -> file(value, checkIfExists: true)
        }
        families.collect { family ->
            tuple(family, candidateManifest, parentContract, contractAmendment,
                  file(params.m37_compact_canonical_metrics, checkIfExists: true),
                  file(params.m37_compact_canonical_metrics_receipt, checkIfExists: true),
                  file(params.m37_fit_truth, checkIfExists: true),
                  file(params.m37_fit_f0_receipt, checkIfExists: true),
                  featureFiles, featureReceipts, runOverlay, positive, positiveReceipt)
        }
    }
    M37_TRACE_COMPACT_SWEEP(familyInput, sourceFiles)

    def collectionInput = M37_TRACE_COMPACT_SWEEP.out.bundle
        .map { _family, metrics, receipts, _equivalence, _equivalenceReceipt, _audit, _auditReceipt ->
            tuple(metrics, receipts)
        }
        .collect(flat: false)
        .map { rows ->
            tuple(params.m37_root as String,
                  rows.collectMany { row -> row[0] as List },
                  rows.collectMany { row -> row[1] as List })
        }
    def collectionSources = [
        file("${repoDir}/bin/m37_trace_core.py", checkIfExists: true),
        file("${repoDir}/bin/m37_trace_collect_metrics.py", checkIfExists: true),
    ]
    M37_TRACE_COLLECT_METRICS(collectionInput, collectionSources)

    def familyAudits = M37_TRACE_COMPACT_SWEEP.out.bundle
        .map { _family, _metrics, _receipts, _equivalence, _equivalenceReceipt, audit, auditReceipt ->
            tuple(audit, auditReceipt)
        }
        .collect(flat: false)
    def decisionInput = M37_TRACE_COLLECT_METRICS.out.bundle
        .combine(familyAudits)
        .map { combined ->
            def parts = m37_compact_decision_parts(combined)
            tuple(parts[0], parts[1], parts[2], parts[3], parts[4])
        }
    M37_TRACE_COMPACT_DECISION(
        decisionInput,
        [file("${repoDir}/bin/m37_trace_core.py", checkIfExists: true),
         file("${repoDir}/bin/m37_trace_collect_metrics.py", checkIfExists: true),
         file("${repoDir}/bin/m37_trace_successive_halving.py", checkIfExists: true),
         file("${repoDir}/bin/m37_trace_compact_decision.py", checkIfExists: true),
         file("${repoDir}/modules/37_TRACE_COMPACT_SWEEP.nf", checkIfExists: true),
         file("${repoDir}/workflows/m37_trace_compact_sweep.nf", checkIfExists: true),
         file("${repoDir}/conf/m37_trace_compact_sweep.config", checkIfExists: true),
         file("${repoDir}/conf/m37_trace_gcp.config", checkIfExists: true)],
    )
}
