nextflow.enable.dsl=2

include {
    M35B_PREPARE_BALANCED_REFERENCE
    M35B_CLUSTER_SCREEN
    M35B_AGGREGATE_CLUSTER_GATE
    M35B_RUN_PREASSIGNED_FINAL
    M35B_PACK_PREDICTION
    M35B_SCORE_PAIRED
} from '../modules/35B_FLARE2_BALANCED'

workflow {
    def required = [
        params.m35b_run_id, params.m35b_results_dir, params.m35b_contract,
        params.m35b_roles, params.m35b_reference_vcf, params.m35b_reference_tbi,
        params.m35b_target_vcf, params.m35b_target_tbi, params.m35b_genetic_map,
        params.m35b_flare2_image, params.m35b_tabix_image, params.m35b_scoring_image,
    ]
    if (required.any { value -> !value })
        error 'M35B requires an explicit run, frozen inputs and pinned runtimes'
    if (!(params.m35b_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error 'm35b_run_id must be a safe explicit identifier'
    if (!params.m35b_results_dir.startsWith('gs://teams-usp/frank/lai-exploracion-datos/'))
        error 'M35B persistent outputs are restricted to the frank GCS prefix'
    if (![params.m35b_tabix_image, params.m35b_scoring_image].every { it.contains('@sha256:') })
        error 'M35B shared runtime images must use immutable registry digests'
    if (!(params.m35b_flare2_image.contains('@sha256:') || params.m35b_flare2_image.startsWith('sha256:')))
        error 'M35B FLARE2 runtime must use an immutable registry digest or local image ID'
    if (params.m35b_executor == 'google-batch' && !params.m35b_flare2_image.contains('@sha256:'))
        error 'M35B Google Batch requires a registry image pinned by digest'

    def contractFile = file(params.m35b_contract, checkIfExists: true)
    def contract = new groovy.json.JsonSlurper().parse(contractFile)
    if (contract.experiment_id != 'M35B_FLARE2_BALANCED_SENSITIVITY_CHR22' ||
        contract.status != 'PREREGISTERED_EXPLORATORY_SCREEN' ||
        contract.cluster_screen.primary_gate != 'all_9_coarse_selection_by_gmm_combinations_must_pass')
        error 'M35B contract identity or prospective gate differs'

    def repoDir = baseDir.resolve('..')
    def selectionSeeds = contract.reference_balance.selection_seeds.collect { it as Integer }
    def gmmSeeds = contract.cluster_screen.gmm_seeds.collect { it as Integer }
    def primary = contract.primary_final_pair
    def runFinal = params.m35b_run_final instanceof Boolean \
        ? params.m35b_run_final : params.m35b_run_final.toString().toBoolean()

    M35B_PREPARE_BALANCED_REFERENCE(
        Channel.fromList(selectionSeeds),
        Channel.value(file(params.m35b_roles, checkIfExists: true)),
        Channel.value(file(params.m35b_reference_vcf, checkIfExists: true)),
        Channel.value(file(params.m35b_reference_tbi, checkIfExists: true)),
        Channel.value(file(params.m35b_target_vcf, checkIfExists: true)),
        Channel.value(file(params.m35b_target_tbi, checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_prepare_balanced_reference.py", checkIfExists: true)),
    )

    def prepared = M35B_PREPARE_BALANCED_REFERENCE.out.balanced_reference
    def screenCases = prepared.flatMap {
        selectionSeed, referenceVcf, referenceTbi, coarseSample, coarseMacro,
        fineSample, fineMacro, prepareReceipt ->
        def cases = []
        gmmSeeds.each { gmmSeed ->
            cases << tuple(selectionSeed, 'coarse', gmmSeed, referenceVcf, referenceTbi,
                           coarseSample, coarseMacro, prepareReceipt)
            cases << tuple(selectionSeed, 'fine', gmmSeed, referenceVcf, referenceTbi,
                           fineSample, fineMacro, prepareReceipt)
        }
        cases
    }
    M35B_CLUSTER_SCREEN(
        screenCases,
        Channel.value(contractFile),
        Channel.value(file(params.m35b_target_vcf, checkIfExists: true)),
        Channel.value(file(params.m35b_target_tbi, checkIfExists: true)),
        Channel.value(file(params.m35b_genetic_map, checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_cluster_screen.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35_flare2_paired.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_run_flare.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_create_model_wrapper.py", checkIfExists: true)),
    )

    def screened = M35B_CLUSTER_SCREEN.out.screen
    M35B_AGGREGATE_CLUSTER_GATE(
        screened.map { selectionSeed, granularity, gmmSeed, screenDir -> screenDir }.collect(),
        Channel.value(contractFile),
        Channel.value(file("${repoDir}/bin/m35b_aggregate_cluster_gate.py", checkIfExists: true)),
    )

    if (runFinal) {
        if (!params.m35b_truth_npz || !params.m35b_canonical_f0_metrics)
            error 'M35B final scoring requires the frozen truth and contextual M34 F0 metrics'
        def primaryScreen = screened.filter {
            selectionSeed, granularity, gmmSeed, screenDir ->
            selectionSeed == primary.selection_seed && granularity == primary.granularity &&
                gmmSeed == primary.gmm_seed
        }
        def primaryReference = prepared.filter {
            selectionSeed, referenceVcf, referenceTbi, coarseSample, coarseMacro,
            fineSample, fineMacro, prepareReceipt -> selectionSeed == primary.selection_seed
        }.map {
            selectionSeed, referenceVcf, referenceTbi, coarseSample, coarseMacro,
            fineSample, fineMacro, prepareReceipt ->
            tuple(selectionSeed, referenceVcf, referenceTbi, prepareReceipt)
        }
        def finalInput = primaryScreen.join(primaryReference, by: 0).map {
            selectionSeed, granularity, gmmSeed, screenDir,
            referenceVcf, referenceTbi, prepareReceipt ->
            tuple(selectionSeed, granularity, gmmSeed, screenDir,
                  referenceVcf, referenceTbi, prepareReceipt)
        }
        M35B_RUN_PREASSIGNED_FINAL(
            finalInput,
            M35B_AGGREGATE_CLUSTER_GATE.out.go_token,
            M35B_AGGREGATE_CLUSTER_GATE.out.gate_receipt,
            Channel.value(contractFile),
            Channel.value(file(params.m35b_target_vcf, checkIfExists: true)),
            Channel.value(file(params.m35b_target_tbi, checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m35b_run_final_pair.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m35b_cluster_screen.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m35_flare2_paired.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m34_run_flare.py", checkIfExists: true)),
        )
        def rawPredictions = M35B_RUN_PREASSIGNED_FINAL.out.direct_prediction
            .map { anc -> tuple('FLARE_0_6_BALANCED', anc) }
            .mix(M35B_RUN_PREASSIGNED_FINAL.out.flare2_prediction
                 .map { anc -> tuple('FLARE2_BALANCED', anc) })
        M35B_PACK_PREDICTION(
            rawPredictions,
            Channel.value(file(params.m35b_genetic_map, checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m34_parse_flare_truth.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m34_generate_mosaics.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m34_pack_f0_prediction.py", checkIfExists: true)),
        )
        M35B_SCORE_PAIRED(
            M35B_PACK_PREDICTION.out.prediction.filter { row -> row[0] == 'FLARE_0_6_BALANCED' },
            M35B_PACK_PREDICTION.out.prediction.filter { row -> row[0] == 'FLARE2_BALANCED' },
            Channel.value(file(params.m35b_truth_npz, checkIfExists: true)),
            Channel.value(file(params.m35b_canonical_f0_metrics, checkIfExists: true)),
            M35B_RUN_PREASSIGNED_FINAL.out.inference_receipt,
            Channel.value(file("${repoDir}/bin/m34_score_predictions.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m35b_summarize_paired.py", checkIfExists: true)),
        )
    }
}
