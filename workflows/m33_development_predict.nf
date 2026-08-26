nextflow.enable.dsl=2

include { M33_DEVELOPMENT_SCREEN_PREDICT } from '../modules/33_DEVELOPMENT_SCREEN'

workflow {
    def required = [
        'm33_dev_predict_run_id', 'm33_dev_predict_results_dir',
        'm33_dev_predict_runtime_dir', 'm33_dev_predict_training_dir',
        'm33_dev_predict_oci_image'
    ]
    required.each { key -> if (!params[key]) error "--${key} is required" }
    def repoDir = projectDir.resolve('..')
    def sourceNames = [
        'm33_predict_development.py', 'm33_train_development.py', 'm33_materialize.py',
        'm33_m0_contract.py', 'm33_t0a_models.py', 'm33_t0a_forward.py',
        'm33_safe_bridge_core.py', 'm33_score_development.py',
        'm30_flare_scorer.py', 'm28d_b0_scorer.py'
    ]
    def sources = sourceNames.collect { name -> file("${repoDir}/bin/${name}", checkIfExists:true) }
    def candidates = params.m33_dev_predict_candidates
    if (!(candidates instanceof List) || candidates.isEmpty()) {
        error '--m33_dev_predict_candidates must be a nonempty list'
    }
    Channel.fromList(candidates.collect { row ->
        def checkpoint = file("${params.m33_dev_predict_training_dir}/${row.bundle}/model.pt",
                              checkIfExists:true)
        tuple(row.rotation as String, row.family as String, row.radius as Double,
              row.share as String, row.arm as String, checkpoint)
    }).set { candidate_ch }
    M33_DEVELOPMENT_SCREEN_PREDICT(
        candidate_ch, sources,
        file("${repoDir}/conf/m33_pre4_preregistration.json", checkIfExists:true),
    )
}
