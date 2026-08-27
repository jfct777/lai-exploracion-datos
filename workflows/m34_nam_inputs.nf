nextflow.enable.dsl=2

String m34Sha256(target) {
    java.security.MessageDigest.getInstance('SHA-256')
        .digest(new File(target.toString()).bytes).encodeHex().toString()
}

String m34RadiusToken(value) {
    def radius = new BigDecimal(value.toString()).stripTrailingZeros()
    if (radius <= 0)
        error 'radius_cM must be positive'
    def plain = radius.toPlainString()
    if (!(plain ==~ /[0-9]+(?:\.[0-9]+)?/))
        error 'radius_cM cannot be represented as a safe path token'
    return "r${plain.replace('.', 'p')}cM"
}

String m34TaskToken(task, String radiusToken) {
    def components = [
        task.sweep_stage as String,
        task.rotation as String,
        "seed${task.seed}",
        "u${task.maximum_updates}",
        radiusToken,
    ]
    if (!components.every { it ==~ /[A-Za-z0-9._-]+/ })
        error 'task identity cannot be represented as a safe path token'
    return components.join('_')
}

include { M34_NAM_VALIDATE_EXPERIMENT_CONTRACT } from '../modules/34_NAM_EXPERIMENT_CONTRACT'
include { M34_NAM_GENERATE_MOSAICS } from '../modules/34_NAM_MOSAICS'
include { M34_NAM_PREPARE_PANEL_FACTORS } from '../modules/34_NAM_PANEL_FACTORS'
include { M34_NAM_TABIX_INDEX } from '../modules/34_NAM_TABIX'
include { M34_NAM_BUILD_FLARE_CONTRACT } from '../modules/34_NAM_FLARE_CONTRACT'
include { M34_NAM_RUN_FLARE } from '../modules/34_NAM_FLARE_RUN'
include { M34_NAM_PARSE_F0 } from '../modules/34_NAM_PARSE_F0'
include { M34_NAM_ALIGN_TRUTH } from '../modules/34_NAM_ALIGN_TRUTH'
include { M34_NAM_BUILD_FACTORIZED_MANIFEST } from '../modules/34_NAM_FACTORIZED_MANIFEST'
include { M34_NAM_BUILD_TRIAGE_PLAN } from '../modules/34_NAM_TRIAGE_PLAN'
include { M34_NAM_TRAIN_FACTORIZED } from '../modules/34_NAM_TRAIN_FACTORIZED'
include { M34_NAM_TRAIN_TRANSFORMER_FACTORIZED } from '../modules/34_NAM_TRAIN_TRANSFORMER_FACTORIZED'
include { M34_NAM_PACK_BASELINE } from '../modules/34_NAM_PACK_BASELINE'
include { M34_NAM_SCORE_VALID } from '../modules/34_NAM_SCORE'

workflow {
    if (!params.m34_inputs_run_id ||
        !(params.m34_inputs_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error 'm34_inputs_run_id must be a valid explicit run identifier'
    if (!params.m34_inputs_results_dir)
        error 'm34_inputs_results_dir is required'
    if (!(['R0', 'R1', 'R2'].contains(params.m34_inputs_root)))
        error 'm34_inputs_root must be R0, R1 or R2'
    if (!(['small', 'pilot_128', 'medium'].contains(params.m34_inputs_target_size)))
        error 'm34_inputs_target_size must be small, pilot_128 or medium'
    if (!params.m34_inputs_phased_vcf ||
        !params.m34_inputs_split_tsv ||
        !params.m34_inputs_genetic_map ||
        !params.m34_inputs_flare_jar ||
        !params.m34_inputs_experiment_contract ||
        !params.m34_inputs_adaptive_contract)
        error 'panel, split, map, FLARE jar and both M34 contracts are required'

    def localInputs = [
        params.m34_inputs_phased_vcf,
        params.m34_inputs_split_tsv,
        params.m34_inputs_genetic_map,
        params.m34_inputs_flare_jar,
        params.m34_inputs_experiment_contract,
        params.m34_inputs_adaptive_contract,
    ]
    if (params.m34_inputs_task_plan)
        localInputs.add(params.m34_inputs_task_plan)
    if (localInputs.any { it.toString().contains('://') })
        error 'm34_nam_inputs accepts local paths only'

    def runResults = file(
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}",
        checkIfExists: false,
    )
    if (runResults.exists() && !workflow.resume)
        error 'the run-specific results directory already exists; outputs are append-safe'

    def expectedPytorch = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99'
    def expectedTabix = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-tabix@sha256:e730c35759e3851a92d7f3a6619333105331d97f1ae44a50dfa8d59745c43e54'
    def expectedFlare = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m30-flare-runtime@sha256:86bf36c5d23407ed187d546f2420a0d2c44fbb6eed12ba81ddfc0f75df6b3a84'
    if (params.m34_inputs_pytorch_image != expectedPytorch)
        error 'the M34 model and bridge image differs from the pinned PyTorch runtime'
    if (params.m34_inputs_tabix_image != expectedTabix)
        error 'the M34 Tabix image differs from the pinned m33-tabix runtime'
    if (params.m34_inputs_flare_image != expectedFlare)
        error 'the M34 FLARE image differs from the pinned runtime'

    def experimentFile = file(params.m34_inputs_experiment_contract, checkIfExists: true)
    def experimentSha256 = java.security.MessageDigest.getInstance('SHA-256')
        .digest(experimentFile.bytes).encodeHex().toString()
    if (experimentSha256 != params.m34_inputs_experiment_contract_sha256)
        error 'm34_nam_experiment_contract.json SHA-256 differs'
    def experiment = new groovy.json.JsonSlurper().parse(experimentFile)
    def mixture = experiment.mosaics.primary_mixture_proportions
    def generations = experiment.mosaics.primary_admixture_generations
    def seeds = experiment.mosaics.seeds
    def rootId = params.m34_inputs_root as String
    def targetSizeId = params.m34_inputs_target_size as String
    def targetSize = experiment.mosaics.target_sizes[targetSizeId]
    def roles = experiment.roles
    if (experiment.experiment_id != 'M34_NAM_EXPLORATORY_CHR22' ||
        experiment.status != 'CONTRACT_ONLY_NO_REAL_RESULTS' ||
        experiment.chromosome.toString().replaceFirst('^chr', '') != '22' ||
        experiment.ancestry_order != ['AFR', 'EUR', 'NAM'] ||
        mixture != [AFR: 0.25, EUR: 0.60, NAM: 0.15] ||
        generations != 12 ||
        seeds[rootId + '_FIT'] == null || seeds[rootId + '_VALID'] == null ||
        targetSize == null || targetSize.people != targetSize.fit + targetSize.valid ||
        targetSize.fit % 8 != 0 || targetSize.valid % 8 != 0 ||
        roles.mosaic_fit_donors != 'SOURCE_VALID' ||
        roles.mosaic_valid_donors != 'SOURCE_TEST' ||
        experiment.rare_definition.minimum_mac != 2 ||
        experiment.rare_definition.maximum_maf_exclusive != 0.01)
        error 'M34 selected-root scientific inputs differ from the authenticated experiment contract'
    if ((params.m34_inputs_fit_people as Integer) != (targetSize.fit as Integer) ||
        (params.m34_inputs_valid_people as Integer) != (targetSize.valid as Integer))
        error 'M34 manifest split sizes differ from the selected target size'
    def mixtureArgument = "AFR=${mixture.AFR},EUR=${mixture.EUR},NAM=${mixture.NAM}"

    def splitCases = [
        tuple(
            'FIT', roles.mosaic_fit_donors as String,
            roles.mosaic_valid_donors as String, 'all',
            seeds[rootId + '_FIT'] as Integer, targetSize.fit as Integer,
            "M34_${rootId}_FIT" as String,
            mixtureArgument, generations as Double,
        ),
        tuple(
            'VALID', roles.mosaic_valid_donors as String,
            roles.mosaic_fit_donors as String, 'all',
            seeds[rootId + '_VALID'] as Integer, targetSize.valid as Integer,
            "M34_${rootId}_VALID" as String,
            mixtureArgument, generations as Double,
        ),
    ]
    if (splitCases*.get(0) != ['FIT', 'VALID'])
        error 'the M34 input channel must contain exactly FIT then VALID'

    phasedVcf = Channel.value(file(params.m34_inputs_phased_vcf, checkIfExists: true))
    splitTsv = Channel.value(file(params.m34_inputs_split_tsv, checkIfExists: true))
    geneticMap = Channel.value(file(params.m34_inputs_genetic_map, checkIfExists: true))
    flareJar = Channel.value(file(params.m34_inputs_flare_jar, checkIfExists: true))
    experimentContract = Channel.value(experimentFile)
    adaptiveContract = Channel.value(file(params.m34_inputs_adaptive_contract, checkIfExists: true))

    validatorPy = Channel.value(file("${baseDir}/../bin/m34_validate_experiment_contract.py", checkIfExists: true))
    mosaicPy = Channel.value(file("${baseDir}/../bin/m34_generate_mosaics.py", checkIfExists: true))
    bridgePy = Channel.value(file("${baseDir}/../bin/m34_prepare_panel_factors.py", checkIfExists: true))
    bridgeCorePy = Channel.value(file("${baseDir}/../bin/m33_safe_bridge_core.py", checkIfExists: true))
    buildFlareContractPy = Channel.value(file("${baseDir}/../bin/m34_build_flare_contract.py", checkIfExists: true))
    runFlarePy = Channel.value(file("${baseDir}/../bin/m34_run_flare.py", checkIfExists: true))
    parseFlareTruthPy = Channel.value(file("${baseDir}/../bin/m34_parse_flare_truth.py", checkIfExists: true))
    manifestBuilderPy = Channel.value(file("${baseDir}/../bin/m34_build_factorized_manifest.py", checkIfExists: true))
    adaptiveSweepPy = Channel.value(file("${baseDir}/../bin/m34_adaptive_sweep.py", checkIfExists: true))
    trainerPy = Channel.value(file("${baseDir}/../bin/m34_train_factorized.py", checkIfExists: true))
    transformerTrainerPy = Channel.value(file("${baseDir}/../bin/m34_train_transformer_factorized.py", checkIfExists: true))
    materializePy = Channel.value(file("${baseDir}/../bin/m34_materialize.py", checkIfExists: true))
    modelsPy = Channel.value(file("${baseDir}/../bin/m34_models.py", checkIfExists: true))
    packedTrainPy = Channel.value(file("${baseDir}/../bin/m34_train_packed.py", checkIfExists: true))
    m33MaterializePy = Channel.value(file("${baseDir}/../bin/m33_materialize.py", checkIfExists: true))
    m33ContractPy = Channel.value(file("${baseDir}/../bin/m33_m0_contract.py", checkIfExists: true))
    packBaselinePy = Channel.value(file("${baseDir}/../bin/m34_pack_f0_prediction.py", checkIfExists: true))
    scorerPy = Channel.value(file("${baseDir}/../bin/m34_score_predictions.py", checkIfExists: true))

    M34_NAM_VALIDATE_EXPERIMENT_CONTRACT(
        experimentContract,
        validatorPy,
        params.m34_inputs_experiment_contract_sha256,
        rootId,
        targetSizeId,
    )
    validatedExperiment = M34_NAM_VALIDATE_EXPERIMENT_CONTRACT.out.validated.map {
        contract, receipt, sha256 -> contract
    }.first()

    M34_NAM_GENERATE_MOSAICS(
        Channel.fromList(splitCases), phasedVcf, splitTsv, geneticMap, mosaicPy,
    )
    mosaicAux = M34_NAM_GENERATE_MOSAICS.out.mosaics.map {
        split, donorRole, mosaicVcf, truth, donorAudit, mosaicReceipt ->
        tuple(split, truth, mosaicReceipt)
    }
    bridgeInputs = M34_NAM_GENERATE_MOSAICS.out.mosaics.map {
        split, donorRole, mosaicVcf, truth, donorAudit, mosaicReceipt ->
        tuple(split, donorRole, mosaicVcf)
    }
    M34_NAM_PREPARE_PANEL_FACTORS(
        bridgeInputs, phasedVcf, splitTsv, geneticMap,
        bridgePy, mosaicPy, bridgeCorePy,
    )

    indexInputs = M34_NAM_PREPARE_PANEL_FACTORS.out.factors.flatMap {
        split, referenceVcf, targetVcf, sampleMap, selectedLoci,
        targetRare, referenceRare, bridgeReceipt ->
        [tuple(split, 'REFERENCE', referenceVcf), tuple(split, 'TARGET', targetVcf)]
    }
    M34_NAM_TABIX_INDEX(indexInputs)
    indexBundles = M34_NAM_TABIX_INDEX.out.indexed
        .groupTuple(by: 0, size: 2)
        .map { split, rolesList, vcfs, indexes ->
            def referenceIndex = rolesList.indexOf('REFERENCE')
            def targetIndex = rolesList.indexOf('TARGET')
            if (referenceIndex < 0 || targetIndex < 0)
                error "${split} lacks one typed Tabix index"
            tuple(split, indexes[referenceIndex], indexes[targetIndex])
        }
    bridgeIndexed = M34_NAM_PREPARE_PANEL_FACTORS.out.factors.join(indexBundles)
    flareInputs = bridgeIndexed.join(mosaicAux).map {
        split, referenceVcf, targetVcf, sampleMap, selectedLoci,
        targetRare, referenceRare, bridgeReceipt, referenceTbi, targetTbi,
        mosaicTruth, mosaicReceipt ->
        tuple(
            split, referenceVcf, referenceTbi, targetVcf, targetTbi, sampleMap,
            selectedLoci, targetRare, referenceRare,
            mosaicTruth, mosaicReceipt, bridgeReceipt,
        )
    }
    M34_NAM_BUILD_FLARE_CONTRACT(
        flareInputs, validatedExperiment, geneticMap, flareJar, buildFlareContractPy,
    )
    M34_NAM_RUN_FLARE(
        M34_NAM_BUILD_FLARE_CONTRACT.out.contracted,
        geneticMap, flareJar, runFlarePy,
    )
    M34_NAM_PARSE_F0(
        M34_NAM_RUN_FLARE.out.baseline,
        geneticMap, parseFlareTruthPy, mosaicPy, bridgeCorePy,
    )
    M34_NAM_ALIGN_TRUTH(
        M34_NAM_PARSE_F0.out.parsed,
        parseFlareTruthPy, mosaicPy, bridgeCorePy,
    )

    fitFactors = M34_NAM_ALIGN_TRUTH.out.factors.filter {
        split, selected, target, reference, mosaicReceipt,
        bridgeReceipt, flareDir, f0Dir, truthDir -> split == 'FIT'
    }
    validFactors = M34_NAM_ALIGN_TRUTH.out.factors.filter {
        split, selected, target, reference, mosaicReceipt,
        bridgeReceipt, flareDir, f0Dir, truthDir -> split == 'VALID'
    }
    M34_NAM_BUILD_FACTORIZED_MANIFEST(fitFactors, validFactors, manifestBuilderPy)

    if (params.m34_inputs_task_plan) {
        def planFile = file(params.m34_inputs_task_plan, checkIfExists: true)
        if (!params.m34_inputs_task_plan_sha256 ||
            m34Sha256(planFile) != params.m34_inputs_task_plan_sha256)
            error 'M34 replication task-plan SHA-256 differs'
        def plan = new groovy.json.JsonSlurper().parse(planFile)
        if (plan.stage != 'M34_EXPLORATORY_128_REPLICATION_PLAN' ||
            plan.status != 'PLAN_ONLY_NO_EXECUTION_TEST_CLOSED' ||
            plan.claim_level != 'exploratory' || plan.test_opened != false ||
            plan.target_size != [people: 128, fit: 96, valid: 32] ||
            plan.task_count != plan.tasks.size() || plan.task_count != 12)
            error 'M34 128-mosaic replication plan identity differs'
        def rootTasks = plan.tasks.findAll { task -> task.rotation == rootId }
        if (rootTasks.size() != 4 ||
            rootTasks.collect { it.config_id }.toSet() != ['bilstm_r1', 'unet_r1'].toSet() ||
            rootTasks.collect { it.arm }.toSet() != ['RD', 'RE'].toSet() ||
            !rootTasks.every { task ->
                task.seed == 1103 && task.radius_cM == 0.2 &&
                task.sweep_stage == 'replication_128' && task.maximum_updates == 3200
            } || targetSizeId != 'pilot_128')
            error 'M34 selected-root replication tasks differ from the PRE plan'
        taskInputs = Channel.fromList(rootTasks.collect { task ->
            def taskBase64 = groovy.json.JsonOutput.toJson(task)
                .getBytes('UTF-8').encodeBase64().toString()
            def radiusCm = new BigDecimal(task.radius_cM.toString())
            def radiusToken = m34RadiusToken(radiusCm)
            def taskToken = m34TaskToken(task, radiusToken)
            tuple(task.family as String, task.config_id as String,
                  task.arm as String, radiusCm, radiusToken, taskToken, taskBase64)
        })
    } else {
        M34_NAM_BUILD_TRIAGE_PLAN(adaptiveContract, adaptiveSweepPy)
        taskInputs = M34_NAM_BUILD_TRIAGE_PLAN.out.plan.flatMap { planPath ->
            def plan = new groovy.json.JsonSlurper().parse(planPath.toFile())
            if (plan.stage != 'M34_TRIAGE_PLAN' ||
                plan.status != 'PLAN_ONLY_NO_EXECUTION' ||
                plan.task_count != plan.tasks.size())
                error 'M34 triage plan identity or task count differs'
            plan.tasks.collect { task ->
                def taskBase64 = groovy.json.JsonOutput.toJson(task)
                    .getBytes('UTF-8').encodeBase64().toString()
                def radiusCm = new BigDecimal(task.radius_cM.toString())
                def radiusToken = m34RadiusToken(radiusCm)
                def taskToken = m34TaskToken(task, radiusToken)
                tuple(task.family as String, task.config_id as String,
                      task.arm as String, radiusCm, radiusToken, taskToken, taskBase64)
            }
        }
    }
    factorBundle = M34_NAM_BUILD_FACTORIZED_MANIFEST.out.bundle.map {
        bundle, manifest, receipt -> bundle
    }.first()
    standardTaskInputs = taskInputs.filter {
        family, configId, arm, radiusCm, radiusToken, taskToken, taskBase64 ->
        family != 'transformer_small'
    }
    transformerTaskInputs = taskInputs.filter {
        family, configId, arm, radiusCm, radiusToken, taskToken, taskBase64 ->
        family == 'transformer_small'
    }
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

    validF0 = M34_NAM_ALIGN_TRUTH.out.factors
        .filter { split, selected, target, reference, mosaicReceipt,
                  bridgeReceipt, flareDir, f0Dir, truthDir -> split == 'VALID' }
        .map { split, selected, target, reference, mosaicReceipt,
               bridgeReceipt, flareDir, f0Dir, truthDir -> tuple(split, f0Dir) }
        .first()
    validTruth = M34_NAM_ALIGN_TRUTH.out.factors
        .filter { split, selected, target, reference, mosaicReceipt,
                  bridgeReceipt, flareDir, f0Dir, truthDir -> split == 'VALID' }
        .map { split, selected, target, reference, mosaicReceipt,
               bridgeReceipt, flareDir, f0Dir, truthDir -> truthDir.resolve('truth.npz') }
        .first()
    M34_NAM_PACK_BASELINE(validF0, packBaselinePy, bridgeCorePy)
    standardPredictions = M34_NAM_TRAIN_FACTORIZED.out.trained.map {
        family, configId, arm, radiusCm, radiusToken, taskToken, taskBase64,
        prediction, receipt, model ->
        tuple(family, configId, arm, radiusCm, radiusToken, taskToken,
              taskBase64, prediction)
    }
    transformerPredictions = M34_NAM_TRAIN_TRANSFORMER_FACTORIZED.out.trained.map {
        family, configId, arm, radiusCm, radiusToken, taskToken, taskBase64,
        prediction, receipt, model, batchingReceipt ->
        tuple(family, configId, arm, radiusCm, radiusToken, taskToken,
              taskBase64, prediction)
    }
    trainedPredictions = standardPredictions.mix(transformerPredictions)
    baselinePredictions = M34_NAM_PACK_BASELINE.out.prediction.map {
        family, configId, arm, prediction ->
        def task = [
            family: family, config_id: configId, arm: arm,
            seed: 0, rotation: rootId, radius_cM: 0.2,
            sweep_stage: 'baseline', maximum_updates: 0,
            learning_rate: 0.0, weight_decay: 0.0,
        ]
        def taskBase64 = groovy.json.JsonOutput.toJson(task)
            .getBytes('UTF-8').encodeBase64().toString()
        def radiusCm = new BigDecimal('0.2')
        def radiusToken = m34RadiusToken(radiusCm)
        def taskToken = "baseline_${rootId}_${radiusToken}"
        tuple(family, configId, arm, radiusCm, radiusToken, taskToken,
              taskBase64, prediction)
    }
    scoringInputs = trainedPredictions.mix(baselinePredictions)
    M34_NAM_SCORE_VALID(scoringInputs, validTruth, scorerPy)
}
