nextflow.enable.dsl=2

include { M35_FLARE2_PAIRED_BASELINE; M35_PACK_FLARE_PREDICTION; M35_SCORE_PAIRED } from '../modules/35_FLARE2_PAIRED'

workflow {
    def required = [
        params.m35_run_id, params.m35_results_dir, params.m35_contract,
        params.m35_reference_vcf, params.m35_reference_tbi,
        params.m35_target_vcf, params.m35_target_tbi, params.m35_sample_map, params.m35_panel_macro_map,
        params.m35_genetic_map, params.m35_container_image, params.m35_scoring_image,
    ]
    if (required.any { value -> !value })
        error 'M35 requires an explicit run, approved input locations, and pinned runtimes'
    if (!(params.m35_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error 'm35_run_id must be a safe explicit identifier'
    if (!params.m35_results_dir.startsWith('gs://teams-usp/frank/lai-exploracion-datos/'))
        error 'M35 outputs are restricted to the frank GCS prefix'
    if (![params.m35_container_image, params.m35_scoring_image].every { it.contains('@sha256:') })
        error 'M35 runtime images must use immutable @sha256 digests'
    if (!params.m35_preflight_only && (!params.m35_truth_npz || !params.m35_canonical_f0_metrics))
        error 'M35 scoring requires sealed M34 truth and canonical F0 metrics after truth-blind inference'

    def contractFile = file(params.m35_contract, checkIfExists: true)
    def contractPayload = new groovy.json.JsonSlurper().parse(contractFile)
    if (contractPayload.experiment_id != 'M35_FLARE2_PAIRED_CHR22' ||
        contractPayload.status != 'PLAN_ONLY_PRECHECK_DEFAULT' ||
        contractPayload.chromosome.toString().replaceFirst('^chr', '') != '22' ||
        contractPayload.methods.flare_0_6.parameters.seed != 3401103)
        error 'M35 paired contract identity or canonical M34 seed differs'

    def repoDir = baseDir.resolve('..')
    def inference = M35_FLARE2_PAIRED_BASELINE(
        Channel.value(contractFile),
        Channel.value(file(params.m35_reference_vcf, checkIfExists: true)),
        Channel.value(file(params.m35_reference_tbi, checkIfExists: true)),
        Channel.value(file(params.m35_target_vcf, checkIfExists: true)),
        Channel.value(file(params.m35_target_tbi, checkIfExists: true)),
        Channel.value(file(params.m35_sample_map, checkIfExists: true)),
        Channel.value(file(params.m35_panel_macro_map, checkIfExists: true)),
        Channel.value(file(params.m35_genetic_map, checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35_flare2_paired.py", checkIfExists: true)),
    )

    if (!params.m35_preflight_only) {
        def rawPredictions = inference.out.flare060_prediction.map { anc -> tuple('FLARE_0_6', anc) }
            .mix(inference.out.flare2_prediction.map { anc -> tuple('FLARE2', anc) })
        def packed = M35_PACK_FLARE_PREDICTION(
            rawPredictions,
            Channel.value(file(params.m35_genetic_map, checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m34_parse_flare_truth.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m34_generate_mosaics.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m34_pack_f0_prediction.py", checkIfExists: true)),
        )
        M35_SCORE_PAIRED(
            packed.out.prediction.filter { row -> row[0] == 'FLARE_0_6' },
            packed.out.prediction.filter { row -> row[0] == 'FLARE2' },
            Channel.value(file(params.m35_truth_npz, checkIfExists: true)),
            Channel.value(file(params.m35_canonical_f0_metrics, checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m34_score_predictions.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m35_verify_direct_f0.py", checkIfExists: true)),
            Channel.value(file("${repoDir}/bin/m35_summarize_paired.py", checkIfExists: true)),
        )
    }
}
