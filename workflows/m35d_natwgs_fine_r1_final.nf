nextflow.enable.dsl=2

include {
    M35D_RUN_R1_FINAL_PAIR
    M35D_PACK_R1_PREDICTION
    M35D_SCORE_R1_PAIR
} from '../modules/35D_NATWGS_FINE_R1_FINAL'

workflow {
    def required = [
        params.m35d_run_id, params.m35d_results_dir, params.m35d_contract,
        params.m35d_gate, params.m35d_token, params.m35d_screen_dir,
        params.m35d_reference_vcf, params.m35d_target_vcf,
        params.m35d_coarse_sample_map, params.m35d_coarse_panel_macro_map,
        params.m35d_genetic_map, params.m35d_truth_npz,
        params.m35d_canonical_f0_metrics, params.m35d_flare2_image,
        params.m35d_scoring_image,
    ]
    if (required.any { value -> !value })
        error 'M35D final requires the frozen gate, final pair and R1 scoring inputs'
    if (required.any { value -> value.toString().toLowerCase().contains('m34-nam-128-r2') })
        error 'M35D final forbids every R2 input'
    if (!params.m35d_results_dir.startsWith('gs://teams-usp/frank/lai-exploracion-datos/'))
        error 'M35D outputs are restricted to the frank GCS prefix'
    def contractFile = file(params.m35d_contract, checkIfExists: true)
    def contract = new groovy.json.JsonSlurper().parse(contractFile)
    if (contract.experiment_id != 'M35D_NATWGS_FINE_R1_EXPLORATORY_CHR22' ||
        contract.preassigned_final_pair.granularity != 'fine')
        error 'M35D final contract differs'
    def repoDir = baseDir.resolve('..')
    M35D_RUN_R1_FINAL_PAIR(
        Channel.value(contractFile),
        Channel.value(file(params.m35d_gate, checkIfExists: true)),
        Channel.value(file(params.m35d_token, checkIfExists: true)),
        Channel.value(file(params.m35d_screen_dir, checkIfExists: true)),
        Channel.value(file(params.m35d_reference_vcf, checkIfExists: true)),
        Channel.value(file(params.m35d_target_vcf, checkIfExists: true)),
        Channel.value(file(params.m35d_coarse_sample_map, checkIfExists: true)),
        Channel.value(file(params.m35d_coarse_panel_macro_map, checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35d_natwgs_fine_r1.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35c_prepare_source_comparison.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_prepare_balanced_reference.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35_flare2_paired.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_run_flare.py", checkIfExists: true)),
    )
    def predictions = M35D_RUN_R1_FINAL_PAIR.out.direct
        .map { path -> tuple('FLARE_F0_SAME_69', path) }
        .mix(M35D_RUN_R1_FINAL_PAIR.out.flare2
             .map { path -> tuple('FLARE2_NATWGS_FINE_SAME_69', path) })
    M35D_PACK_R1_PREDICTION(
        predictions,
        Channel.value(file(params.m35d_genetic_map, checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_parse_flare_truth.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_generate_mosaics.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_pack_f0_prediction.py", checkIfExists: true)),
    )
    M35D_SCORE_R1_PAIR(
        M35D_PACK_R1_PREDICTION.out.prediction.filter { it[0] == 'FLARE_F0_SAME_69' },
        M35D_PACK_R1_PREDICTION.out.prediction.filter { it[0] == 'FLARE2_NATWGS_FINE_SAME_69' },
        Channel.value(file(params.m35d_truth_npz, checkIfExists: true)),
        Channel.value(file(params.m35d_canonical_f0_metrics, checkIfExists: true)),
        M35D_RUN_R1_FINAL_PAIR.out.receipt, Channel.value(contractFile),
        Channel.value(file("${repoDir}/bin/m34_score_predictions.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35d_natwgs_fine_r1.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35d_subset_truth.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_parse_flare_truth.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_generate_mosaics.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35c_prepare_source_comparison.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_prepare_balanced_reference.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35_flare2_paired.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_run_flare.py", checkIfExists: true)),
    )
}
