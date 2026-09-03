nextflow.enable.dsl=2

include { M37_TRACE_BIND_MARKER_AXIS; M37_TRACE_SHAM_REFERENCE; M37_TRACE_MATERIALIZE; M37_TRACE_TRAIN; M37_TRACE_SCORE; M37_TRACE_COLLECT_METRICS; M37_TRACE_SUCCESSIVE_HALVING; M37_TRACE_READY } from '../modules/37_TRACE_LAI'

workflow {
    ['m37_run_id', 'm37_root', 'm37_results_dir', 'm37_run_overlay_config', 'm37_run_overlay_uri',
     'm37_fit_selected', 'm37_fit_target',
     'm37_fit_reference_folds', 'm37_fit_f0', 'm37_fit_marker_cm', 'm37_fit_f0_receipt',
     'm37_fit_truth'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (!(params.m37_run_id ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/)) error '--m37_run_id contains unsupported characters'
    if (!(params.m37_root ==~ /R[0-9]+/)) error '--m37_root must be an explicit mosaic root such as R0'
    if (!(params.m37_candidates instanceof List) || params.m37_candidates.isEmpty()) error '--m37_candidates must be nonempty'

    def requiredArms = ['RE', 'RD', 'POOLED', 'SHAM', 'GEOMETRY'] as Set
    def parameterKeys = ['family', 'hazard_per_morgan', 'evidence_scale', 'hidden_dim', 'depth',
                         'kernel_size', 'dropout', 'seed', 'learning_rate', 'dilations']
    params.m37_candidates.each { row ->
        if (!row.containsKey('candidate_id') || !row.containsKey('arm')) error 'every M37 row needs candidate_id and arm'
    }
    params.m37_candidates.groupBy { it.candidate_id as String }.each { candidateId, rows ->
        if ((rows.collect { it.arm as String } as Set) != requiredArms || rows.size() != requiredArms.size()) {
            error "candidate ${candidateId} must declare each of RE/RD/POOLED/SHAM/GEOMETRY exactly once"
        }
        if (rows.collect { row -> parameterKeys.collect { key -> row[key] } }.unique().size() != 1) {
            error "candidate ${candidateId} changes model parameters between arms"
        }
    }

    def repoDir = projectDir.resolve('..')
    def overlayValue = params.m37_run_overlay_config as String
    def runOverlay = overlayValue.startsWith('/') ? file(overlayValue, checkIfExists: true) :
                     file("${repoDir}/${overlayValue}", checkIfExists: true)
    def axisSources = ['m33_safe_bridge_core.py', 'm37_trace_core.py', 'm37_bind_marker_axis.py']
        .collect { name -> file("${repoDir}/bin/${name}", checkIfExists: true) }
    def sources = ['m33_safe_bridge_core.py', 'm37_trace_core.py', 'm37_trace_materialize.py',
                   'm37_trace_train.py', 'm37_trace_score.py', 'm37_trace_provenance.py', 'm37_trace_sham.py']
        .collect { name -> file("${repoDir}/bin/${name}", checkIfExists: true) }
    def readyAuth = (axisSources + sources + [
        file("${repoDir}/conf/m37_trace_lai.config", checkIfExists: true),
        file("${repoDir}/conf/m37_trace_sweep_contract.json", checkIfExists: true),
        file("${repoDir}/bin/m37_trace_collect_metrics.py", checkIfExists: true),
        file("${repoDir}/bin/m37_trace_successive_halving.py", checkIfExists: true),
        file("${repoDir}/modules/37_TRACE_LAI.nf", checkIfExists: true),
        file("${repoDir}/workflows/m37_trace_lai.nf", checkIfExists: true),
    ]).unique { path -> path.name }

    M37_TRACE_BIND_MARKER_AXIS(
        Channel.fromList([tuple('FIT', file(params.m37_fit_f0), file(params.m37_fit_marker_cm),
                                file(params.m37_fit_f0_receipt))]),
        axisSources,
    )
    def featureRows = [
        tuple('FIT', 'RE', file(params.m37_fit_selected), file(params.m37_fit_target), [file(params.m37_fit_reference_folds)], file(params.m37_fit_f0)),
        tuple('FIT', 'RD', file(params.m37_fit_selected), file(params.m37_fit_target), [file(params.m37_fit_reference_folds)], file(params.m37_fit_f0)),
        tuple('FIT', 'POOLED', file(params.m37_fit_selected), file(params.m37_fit_target), [file(params.m37_fit_reference_folds)], file(params.m37_fit_f0)),
        tuple('FIT', 'GEOMETRY', file(params.m37_fit_selected), file(params.m37_fit_target), [file(params.m37_fit_reference_folds)], file(params.m37_fit_f0)),
    ]
    M37_TRACE_SHAM_REFERENCE(
        Channel.fromList([tuple('FIT', file(params.m37_fit_reference_folds))]),
        [file("${repoDir}/bin/m37_trace_sham.py", checkIfExists: true),
         file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists: true),
         file("${repoDir}/bin/m37_trace_core.py", checkIfExists: true)],
    )
    def shamFeatures = M37_TRACE_SHAM_REFERENCE.out.bundle.map { split, sham, shamReceipt ->
        tuple(split, 'SHAM', file(params.m37_fit_selected), file(params.m37_fit_target), [sham, shamReceipt],
              file(params.m37_fit_f0))
    }
    // One authenticated FIT marker axis is deliberately reused by every arm.
    // ``join`` is one-to-one and would silently retain only the first arm;
    // keyed ``combine`` broadcasts the axis to the complete paired family.
    def materializeRows = Channel.fromList(featureRows).mix(shamFeatures)
        .combine(M37_TRACE_BIND_MARKER_AXIS.out.bundle, by: 0)
        .map { split, arm, selected, target, reference, f0, markerAxis, markerAxisReceipt ->
            tuple(split, arm, selected, target, reference, f0, markerAxis, markerAxisReceipt)
        }
    M37_TRACE_MATERIALIZE(materializeRows, sources)

    def fitByArm = M37_TRACE_MATERIALIZE.out.bundle
        .filter { split, arm, path, receipt -> split == 'FIT' }
        .map { split, arm, path, receipt -> tuple(arm, path, receipt) }
    def candidates = Channel.fromList(params.m37_candidates.collect { row ->
        tuple(row.candidate_id as String, row.family as String, row.arm as String,
              row.hazard_per_morgan as Double, row.evidence_scale as Double,
              row.hidden_dim as Integer, row.depth as Integer, row.kernel_size as Integer,
              row.dropout as Double, row.seed as Integer, row.learning_rate as Double,
              row.dilations as String)
    })
    def withFit = candidates.combine(fitByArm)
        .filter { candidateId, family, arm, hazard, scale, hidden, depth, kernel, dropout, seed, lr, dilations, featureArm, fit, fitReceipt -> arm == featureArm }
        .map { candidateId, family, arm, hazard, scale, hidden, depth, kernel, dropout, seed, lr, dilations, featureArm, fit, fitReceipt ->
            tuple(candidateId, family, arm, hazard, scale, hidden, depth, kernel, dropout, seed, lr,
                  dilations, fit, fitReceipt)
        }
    // Triage sees FIT only. The trainer creates a stable TRAIN/TUNE split and
    // derives its marker calendar solely from TRAIN people.
    def tuning = withFit.map { candidateId, family, arm, hazard, scale, hidden, depth, kernel, dropout, seed, lr, dilations, fit, fitReceipt ->
        tuple(candidateId, family, arm, hazard, scale, hidden, depth, kernel, dropout, seed, lr,
              dilations, fit, fitReceipt, fit, fitReceipt)
    }
    M37_TRACE_TRAIN(tuning, file(params.m37_fit_truth), sources)
    def scored = M37_TRACE_TRAIN.out.bundle.combine(fitByArm)
        .filter { candidateId, family, arm, prediction, predictionReceipt, featureArm, fit, fitReceipt -> arm == featureArm }
        .map { candidateId, family, arm, prediction, predictionReceipt, featureArm, fit, fitReceipt ->
            tuple(params.m37_root as String, candidateId, family, arm, prediction, predictionReceipt, fit, fitReceipt)
        }
    M37_TRACE_SCORE(scored, file(params.m37_fit_truth), sources)
    def collectionInput = M37_TRACE_SCORE.out.bundle
        .map { root, candidateId, family, arm, metrics, receipt -> tuple(metrics, receipt) }
        .collect(flat: false)
        .map { rows -> tuple(params.m37_root as String,
                             rows.collect { row -> row[0] }, rows.collect { row -> row[1] }) }
    def collectionSources = [file("${repoDir}/bin/m37_trace_core.py", checkIfExists: true),
                             file("${repoDir}/bin/m37_trace_collect_metrics.py", checkIfExists: true)]
    M37_TRACE_COLLECT_METRICS(collectionInput, collectionSources)
    M37_TRACE_SUCCESSIVE_HALVING(
        M37_TRACE_COLLECT_METRICS.out.bundle,
        [file("${repoDir}/bin/m37_trace_core.py", checkIfExists: true),
         file("${repoDir}/bin/m37_trace_successive_halving.py", checkIfExists: true)],
    )
    M37_TRACE_READY(M37_TRACE_SCORE.out.bundle, runOverlay, params.m37_run_overlay_uri as String, readyAuth)
}
