nextflow.enable.dsl=2

include { M33_DEVELOPMENT_SCREEN_SCORE } from '../modules/33_DEVELOPMENT_SCREEN'

workflow {
    def required = [
        'm33_dev_score_run_id', 'm33_dev_score_results_dir',
        'm33_dev_score_prediction_dir', 'm33_dev_score_runtime_dir',
        'm33_dev_score_oci_image'
    ]
    required.each { key -> if (!params[key]) error "--${key} is required" }
    def repoDir = projectDir.resolve('..')
    def sources = [
        'm33_score_development.py', 'm33_safe_bridge_core.py',
        'm30_flare_scorer.py', 'm28d_b0_scorer.py'
    ].collect { name -> file("${repoDir}/bin/${name}", checkIfExists:true) }
    def candidates = params.m33_dev_score_candidates
    if (!(candidates instanceof List) || candidates.isEmpty()) {
        error '--m33_dev_score_candidates must be a nonempty list'
    }
    Channel.fromList(candidates.collect { row ->
        def prediction = file(
            "${params.m33_dev_score_prediction_dir}/${row.bundle}/score.prediction.npz",
            checkIfExists:true,
        )
        tuple(row.rotation as String, row.family as String, row.radius as Double,
              row.share as String, row.arm as String, prediction)
    }).set { candidate_ch }
    M33_DEVELOPMENT_SCREEN_SCORE(
        candidate_ch,
        file("${params.m33_dev_score_runtime_dir}/indexed/root-386357765/target.vcf.gz", checkIfExists:true),
        file("${params.m33_dev_score_runtime_dir}/generation/root-386357765/m28_lai_truth.private.tsv.gz", checkIfExists:true),
        file("${params.m33_dev_score_runtime_dir}/flare/root-386357765/flare.map", checkIfExists:true),
        sources,
    )
}
