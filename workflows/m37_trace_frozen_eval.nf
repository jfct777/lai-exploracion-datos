nextflow.enable.dsl=2

include { M37_TRACE_BIND_MARKER_AXIS; M37_TRACE_SHAM_REFERENCE; M37_TRACE_MATERIALIZE; M37_TRACE_TRAIN; M37_TRACE_SCORE; M37_TRACE_READY } from '../modules/37_TRACE_LAI'

/* Evaluate exactly one preselected candidate as a five-arm family. FIT truth
 * is available to training; disjoint VALID truth is opened only by scoring. */
workflow {
    ['m37_run_id', 'm37_root', 'm37_results_dir', 'm37_run_overlay_config', 'm37_run_overlay_uri',
     'm37_frozen_candidate', 'm37_fit_selected', 'm37_fit_target',
     'm37_fit_reference_folds', 'm37_fit_f0', 'm37_fit_marker_cm', 'm37_fit_f0_receipt', 'm37_fit_truth',
     'm37_valid_selected', 'm37_valid_target', 'm37_valid_reference_folds',
     'm37_valid_f0', 'm37_valid_marker_cm', 'm37_valid_f0_receipt', 'm37_valid_truth'].each { key ->
        if (!params[key]) error "--${key} is required for frozen evaluation"
    }
    if (!(params.m37_frozen_candidate instanceof Map)) error '--m37_frozen_candidate must be exactly one sealed candidate row'
    if (!(params.m37_root ==~ /R[0-9]+/)) error '--m37_root must be an explicit mosaic root such as R0'
    def row = params.m37_frozen_candidate
    ['candidate_id','family','hazard_per_morgan','evidence_scale','hidden_dim','depth','kernel_size',
     'dropout','seed','learning_rate','dilations'].each { key ->
        if (!row.containsKey(key)) error "frozen candidate lacks ${key}"
    }
    if (row.containsKey('arm')) error 'the frozen candidate is a decoder specification, not a single arm'

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
        file("${repoDir}/modules/37_TRACE_LAI.nf", checkIfExists: true),
        file("${repoDir}/workflows/m37_trace_frozen_eval.nf", checkIfExists: true),
    ]).unique { path -> path.name }

    M37_TRACE_BIND_MARKER_AXIS(Channel.fromList([
        tuple('FIT', file(params.m37_fit_f0), file(params.m37_fit_marker_cm), file(params.m37_fit_f0_receipt)),
        tuple('VALID', file(params.m37_valid_f0), file(params.m37_valid_marker_cm), file(params.m37_valid_f0_receipt)),
    ]), axisSources)
    M37_TRACE_SHAM_REFERENCE(Channel.fromList([
        tuple('FIT', file(params.m37_fit_reference_folds)),
        tuple('VALID', file(params.m37_valid_reference_folds)),
    ]), [file("${repoDir}/bin/m37_trace_sham.py", checkIfExists: true),
         file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists: true),
         file("${repoDir}/bin/m37_trace_core.py", checkIfExists: true)])
    def arms = ['RE', 'RD', 'POOLED', 'GEOMETRY']
    def featureRows = Channel.fromList(arms.collectMany { currentArm ->
        [tuple('FIT', currentArm, file(params.m37_fit_selected), file(params.m37_fit_target), [file(params.m37_fit_reference_folds)], file(params.m37_fit_f0)),
         tuple('VALID', currentArm, file(params.m37_valid_selected), file(params.m37_valid_target), [file(params.m37_valid_reference_folds)], file(params.m37_valid_f0))]
    })
    def shamRows = M37_TRACE_SHAM_REFERENCE.out.bundle.map { split, sham, shamReceipt ->
        split == 'FIT' ?
            tuple(split, 'SHAM', file(params.m37_fit_selected), file(params.m37_fit_target), [sham, shamReceipt], file(params.m37_fit_f0)) :
            tuple(split, 'SHAM', file(params.m37_valid_selected), file(params.m37_valid_target), [sham, shamReceipt], file(params.m37_valid_f0))
    }
    def materializeRows = featureRows.mix(shamRows).combine(M37_TRACE_BIND_MARKER_AXIS.out.bundle, by: 0)
        .map { split, arm, selected, target, reference, f0, markerAxis, markerAxisReceipt ->
            tuple(split, arm, selected, target, reference, f0, markerAxis, markerAxisReceipt)
        }
    M37_TRACE_MATERIALIZE(materializeRows, sources)

    def fit = M37_TRACE_MATERIALIZE.out.bundle
        .filter { split, arm, path, receipt -> split == 'FIT' }
        .map { split, arm, path, receipt -> tuple(arm, path, receipt) }
    def valid = M37_TRACE_MATERIALIZE.out.bundle
        .filter { split, arm, path, receipt -> split == 'VALID' }
        .map { split, arm, path, receipt -> tuple(arm, path, receipt) }
    def candidates = Channel.fromList((arms + ['SHAM']).collect { currentArm ->
        tuple(row.candidate_id as String, row.family as String, currentArm,
              row.hazard_per_morgan as Double, row.evidence_scale as Double,
              row.hidden_dim as Integer, row.depth as Integer, row.kernel_size as Integer,
              row.dropout as Double, row.seed as Integer, row.learning_rate as Double,
              row.dilations as String)
    })
    def withFit = candidates.combine(fit)
        .filter { candidateId, family, arm, hazard, scale, hidden, depth, kernel, dropout, seed, lr, dilations, fitArm, fitFeatures, fitReceipt -> arm == fitArm }
    def candidate = withFit.combine(valid)
        .filter { candidateId, family, arm, hazard, scale, hidden, depth, kernel, dropout, seed, lr, dilations, fitArm, fitFeatures, fitReceipt, validArm, validFeatures, validReceipt -> arm == validArm }
        .map { candidateId, family, arm, hazard, scale, hidden, depth, kernel, dropout, seed, lr, dilations, fitArm, fitFeatures, fitReceipt, validArm, validFeatures, validReceipt ->
            tuple(candidateId, family, arm, hazard, scale, hidden, depth, kernel, dropout, seed, lr,
                  dilations, fitFeatures, fitReceipt, validFeatures, validReceipt)
        }
    M37_TRACE_TRAIN(candidate, file(params.m37_fit_truth), sources)
    def scored = M37_TRACE_TRAIN.out.bundle.combine(valid)
        .filter { candidateId, family, arm, prediction, predictionReceipt, validArm, validFeatures, validReceipt -> arm == validArm }
        .map { candidateId, family, arm, prediction, predictionReceipt, validArm, validFeatures, validReceipt ->
            tuple(params.m37_root as String, candidateId, family, arm, prediction, predictionReceipt, validFeatures, validReceipt)
        }
    M37_TRACE_SCORE(scored, file(params.m37_valid_truth), sources)
    M37_TRACE_READY(M37_TRACE_SCORE.out.bundle, runOverlay, params.m37_run_overlay_uri as String, readyAuth)
}
