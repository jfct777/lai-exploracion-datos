nextflow.enable.dsl=2

include { M33_DEVELOPMENT_SCREEN_TRAIN } from '../modules/33_DEVELOPMENT_SCREEN'

workflow {
    def required = [
        'm33_dev_screen_run_id', 'm33_dev_screen_results_dir',
        'm33_dev_screen_runtime_dir', 'm33_dev_screen_oci_image'
    ]
    required.each { key -> if (!params[key]) error "--${key} is required" }
    if (!(params.m33_dev_screen_run_id ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/)) {
        error '--m33_dev_screen_run_id contains unsupported characters'
    }
    def repoDir = projectDir.resolve('..')
    def sourceNames = [
        'm33_train_development.py', 'm33_materialize.py', 'm33_m0_contract.py',
        'm33_t0a_models.py', 'm33_t0a_forward.py', 'm33_safe_bridge_core.py',
        'm33_score_development.py', 'm30_flare_scorer.py', 'm28d_b0_scorer.py'
    ]
    def sources = sourceNames.collect { name -> file("${repoDir}/bin/${name}", checkIfExists:true) }
    def candidates = params.m33_dev_screen_candidates
    if (!(candidates instanceof List) || candidates.isEmpty()) {
        error '--m33_dev_screen_candidates must be a nonempty list'
    }
    Channel.fromList(candidates.collect { row ->
        tuple(row.rotation as String, row.family as String, row.radius as Double,
              row.share as String, row.beta as Double, row.seed as Integer, row.arm as String)
    }).set { candidate_ch }
    M33_DEVELOPMENT_SCREEN_TRAIN(
        candidate_ch, sources,
        file("${repoDir}/conf/m33_pre4_preregistration.json", checkIfExists:true),
        file("${repoDir}/conf/m33_pre4a_boundary_weight_amendment.json", checkIfExists:true),
    )
}
