nextflow.enable.dsl=2

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
        params.m34_inputs_results_dir,
    ]
    if (localInputs.any { it.toString().contains('://') })
        error 'm34_nam_inputs accepts local paths only'

    def runResults = new File(
        params.m34_inputs_results_dir.toString(),
        params.m34_inputs_run_id.toString(),
    )
    if (runResults.exists())
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
    def small = experiment.mosaics.target_sizes.small
    def roles = experiment.roles
    if (experiment.experiment_id != 'M34_NAM_EXPLORATORY_CHR22' ||
        experiment.status != 'CONTRACT_ONLY_NO_REAL_RESULTS' ||
        experiment.chromosome.toString().replaceFirst('^chr', '') != '22' ||
        experiment.ancestry_order != ['AFR', 'EUR', 'NAM'] ||
        mixture != [AFR: 0.25, EUR: 0.60, NAM: 0.15] ||
        generations != 12 ||
        seeds.R0_FIT != 1439610605 || seeds.R0_VALID != 1702577247 ||
        small.fit != 24 || small.valid != 8 || small.people != 32 ||
        roles.mosaic_fit_donors != 'SOURCE_VALID' ||
        roles.mosaic_valid_donors != 'SOURCE_TEST' ||
        experiment.rare_definition.minimum_mac != 2 ||
        experiment.rare_definition.maximum_maf_exclusive != 0.01)
        error 'M34 R0 scientific inputs differ from the authenticated experiment contract'
    def mixtureArgument = "AFR=${mixture.AFR},EUR=${mixture.EUR},NAM=${mixture.NAM}"

    def splitCases = [
        tuple(
            'FIT', roles.mosaic_fit_donors as String,
            roles.mosaic_valid_donors as String, 'all',
            seeds.R0_FIT as Integer, small.fit as Integer, 'M34_R0_FIT',
            mixtureArgument, generations as Double,
        ),
        tuple(
            'VALID', roles.mosaic_valid_donors as String,
            roles.mosaic_fit_donors as String, 'all',
            seeds.R0_VALID as Integer, small.valid as Integer, 'M34_R0_VALID',
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

    M34_NAM_BUILD_TRIAGE_PLAN(adaptiveContract, adaptiveSweepPy)
    taskInputs = M34_NAM_BUILD_TRIAGE_PLAN.out.plan.flatMap { planPath ->
        def plan = new groovy.json.JsonSlurper().parse(planPath.toFile())
        if (plan.stage != 'M34_TRIAGE_PLAN' ||
            plan.status != 'PLAN_ONLY_NO_EXECUTION' ||
            plan.task_count != plan.tasks.size())
            error 'M34 triage plan identity or task count differs'
        plan.tasks.collect { task ->
            def taskJson = groovy.json.JsonOutput.toJson(task)
            def taskBase64 = taskJson.getBytes('UTF-8').encodeBase64().toString()
            tuple(task.family as String, task.config_id as String,
                  task.arm as String, taskBase64)
        }
    }
    factorBundle = M34_NAM_BUILD_FACTORIZED_MANIFEST.out.bundle.map {
        bundle, manifest, receipt -> bundle
    }.first()
    standardTaskInputs = taskInputs.filter { family, configId, arm, taskBase64 ->
        family != 'transformer_small'
    }
    transformerTaskInputs = taskInputs.filter { family, configId, arm, taskBase64 ->
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
        family, configId, arm, prediction, receipt, model ->
        tuple(family, configId, arm, prediction)
    }
    transformerPredictions = M34_NAM_TRAIN_TRANSFORMER_FACTORIZED.out.trained.map {
        family, configId, arm, prediction, receipt, model, batchingReceipt ->
        tuple(family, configId, arm, prediction)
    }
    trainedPredictions = standardPredictions.mix(transformerPredictions)
    scoringInputs = trainedPredictions.mix(M34_NAM_PACK_BASELINE.out.prediction)
    M34_NAM_SCORE_VALID(scoringInputs, validTruth, scorerPy)
}
