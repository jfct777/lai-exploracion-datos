nextflow.enable.dsl=2

String m34Sha256(target) {
    java.security.MessageDigest.getInstance('SHA-256')
        .digest(new File(target.toString()).bytes).encodeHex().toString()
}

include { M34_NAM_TRAIN_FACTORIZED } from '../modules/34_NAM_TRAIN_FACTORIZED'
include { M34_NAM_TRAIN_TRANSFORMER_FACTORIZED } from '../modules/34_NAM_TRAIN_TRANSFORMER_FACTORIZED'
include { M34_NAM_SCORE_VALID } from '../modules/34_NAM_SCORE'

workflow {
    if (!params.m34_inputs_run_id ||
        !(params.m34_inputs_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error 'm34_inputs_run_id must be a valid explicit run identifier'
    if (params.m34_batch_region &&
        !(params.m34_batch_region ==~ /[a-z]+-[a-z]+[0-9]/))
        error 'm34_batch_region must be a valid Google Cloud region'
    if (!params.m34_inputs_results_dir || !params.m34_pending_factor_bundle ||
        !params.m34_inputs_adaptive_contract || !params.m34_pending_plan)
        error 'results, factor bundle, adaptive contract and pending plan are required'

    def runResults = file(
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}",
        checkIfExists: false,
    )
    if (runResults.exists())
        error 'the run-specific results directory already exists; outputs are append-safe'

    def expectedPytorch = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99'
    if (params.m34_inputs_pytorch_image != expectedPytorch)
        error 'the M34 pending workflow requires the pinned PyTorch runtime'

    def bundleFile = file(params.m34_pending_factor_bundle, checkIfExists: true)
    def contractFile = file(params.m34_inputs_adaptive_contract, checkIfExists: true)
    def pendingFile = file(params.m34_pending_plan, checkIfExists: true)
    def pending = new groovy.json.JsonSlurper().parse(pendingFile)
    if (pending.stage != 'M34_PENDING_TASK_SELECTION' ||
        pending.status != 'PASS_EXACT_COMPLEMENT' || pending.test_opened != false ||
        !(['M34_TRIAGE_PLAN', 'M34_LOCAL_EXPANSION_PLAN',
           'M34_RADIUS_SENSITIVITY_PLAN', 'M34_FINALIST_PLAN']
          .contains(pending.source_plan_stage)) ||
        pending.task_count < 1 || pending.pending_count < 1 ||
        pending.completed_count + pending.pending_count != pending.task_count ||
        pending.completed.size() != pending.completed_count ||
        pending.pending_tasks.size() != pending.pending_count)
        error 'pending task selection does not close the frozen triage grid'

    factorBundle = Channel.value(bundleFile)
    adaptiveContract = Channel.value(contractFile)
    if (pending.input_sha256.contract != m34Sha256(contractFile) ||
        pending.input_sha256.factorized_manifest !=
            m34Sha256(bundleFile.resolve('factorized.manifest.json')))
        error 'pending task selection is bound to different contract or factors'
    taskInputs = Channel.fromList(pending.pending_tasks.collect { task ->
        def encoded = groovy.json.JsonOutput.toJson(task)
            .getBytes('UTF-8').encodeBase64().toString()
        tuple(task.family as String, task.config_id as String,
              task.arm as String, encoded)
    })
    standardTaskInputs = taskInputs.filter { family, configId, arm, taskBase64 ->
        family != 'transformer_small'
    }
    transformerTaskInputs = taskInputs.filter { family, configId, arm, taskBase64 ->
        family == 'transformer_small'
    }

    trainerPy = Channel.value(file("${baseDir}/../bin/m34_train_factorized.py", checkIfExists: true))
    transformerTrainerPy = Channel.value(file("${baseDir}/../bin/m34_train_transformer_factorized.py", checkIfExists: true))
    adaptiveSweepPy = Channel.value(file("${baseDir}/../bin/m34_adaptive_sweep.py", checkIfExists: true))
    materializePy = Channel.value(file("${baseDir}/../bin/m34_materialize.py", checkIfExists: true))
    modelsPy = Channel.value(file("${baseDir}/../bin/m34_models.py", checkIfExists: true))
    packedTrainPy = Channel.value(file("${baseDir}/../bin/m34_train_packed.py", checkIfExists: true))
    bridgeCorePy = Channel.value(file("${baseDir}/../bin/m33_safe_bridge_core.py", checkIfExists: true))
    m33MaterializePy = Channel.value(file("${baseDir}/../bin/m33_materialize.py", checkIfExists: true))
    m33ContractPy = Channel.value(file("${baseDir}/../bin/m33_m0_contract.py", checkIfExists: true))
    scorerPy = Channel.value(file("${baseDir}/../bin/m34_score_predictions.py", checkIfExists: true))

    M34_NAM_TRAIN_FACTORIZED(
        standardTaskInputs, factorBundle, adaptiveContract,
        trainerPy, adaptiveSweepPy, materializePy, modelsPy,
        packedTrainPy, bridgeCorePy, m33MaterializePy, m33ContractPy,
    )
    M34_NAM_TRAIN_TRANSFORMER_FACTORIZED(
        transformerTaskInputs, factorBundle, adaptiveContract,
        transformerTrainerPy, trainerPy, adaptiveSweepPy, materializePy,
        modelsPy, packedTrainPy, bridgeCorePy, m33MaterializePy, m33ContractPy,
    )

    completedPredictions = Channel.fromList(pending.completed.collect { row ->
        def task = row.task
        tuple(task.family as String, task.config_id as String, task.arm as String,
              file(row.prediction as String, checkIfExists: true))
    })
    standardPredictions = M34_NAM_TRAIN_FACTORIZED.out.trained.map {
        family, configId, arm, prediction, receipt, model ->
        tuple(family, configId, arm, prediction)
    }
    transformerPredictions = M34_NAM_TRAIN_TRANSFORMER_FACTORIZED.out.trained.map {
        family, configId, arm, prediction, receipt, model, batchingReceipt ->
        tuple(family, configId, arm, prediction)
    }
    allPredictions = completedPredictions
        .mix(standardPredictions)
        .mix(transformerPredictions)
    validTruth = Channel.value(file(
        bundleFile.resolve('VALID/truth/truth.npz'), checkIfExists: true,
    ))
    M34_NAM_SCORE_VALID(allPredictions, validTruth, scorerPy)
}
