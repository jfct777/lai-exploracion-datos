nextflow.enable.dsl=2

import groovy.json.JsonOutput

include { IBD_COMMUNITY_ENHANCED } from '../modules/16_5_IBD_COMMUNITY_ENHANCED'
include {
    WRITE_M16_5_RUN_PROVENANCE;
    WRITE_M16_5_MANIFEST;
    COMPARE_M16_5_ORIENTATION
} from '../modules/16_5_ORIENTATION_CLOSURE'


workflow {
    def required = [
        ibd_enhanced_min_edge_bp: 5000000,
        ibd_enhanced_min_max_segment_bp: 500000,
        ibd_enhanced_edge_weight_transform: 'log1p',
        ibd_enhanced_seed: 42,
        ibd_enhanced_leiden_resolutions: '0.5,0.8,1.0,1.2,1.5,2.0,3.0',
        ibd_enhanced_leiden_n_seeds: 25,
        ibd_enhanced_leiden_min_community_size: 3,
        ibd_enhanced_leiden_consensus_resolution: 1.0,
        ibd_enhanced_nmf_k_values: '2,3,4,5,6,8,10,12,15,20',
        ibd_enhanced_nmf_inits: 30,
        ibd_enhanced_nmf_init_mode: 'random-cophenetic',
        ibd_enhanced_nmf_max_iter: 500,
        ibd_enhanced_nmf_tol: 1e-5,
        ibd_enhanced_laplacian_normalize: true,
        ibd_enhanced_nmf_operational_k: 8,
        ibd_enhanced_kinship_segment_mb: 3.0,
        ibd_enhanced_founder_intra_inter_ratio: 3.0,
        ibd_enhanced_founder_min_silhouette: 0.0,
        ibd_enhanced_validation_resolution: 1.0,
    ]
    required.each { name, expected ->
        def observed = params[name]
        if( observed.toString() != expected.toString() ) {
            throw new IllegalStateException(
                "m16_5_minor.nf freezes corrida_C: params.${name}=${observed}, expected ${expected}"
            )
        }
    }
    if( !params.ibd_enhanced_input_dir ) {
        throw new IllegalStateException('Set --ibd_enhanced_input_dir to the immutable M14-minor aggregate')
    }
    if( !params.ibd_enhanced_historical_dir ) {
        throw new IllegalStateException('Set --ibd_enhanced_historical_dir to canonical M16.5-ALT')
    }
    if( !params.ibd_enhanced_cohort_summary ) {
        throw new IllegalStateException('Set --ibd_enhanced_cohort_summary to a M14-minor per-chromosome summary')
    }

    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: 'unknown'
    def provenance = [
        git_commit       : gitCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : params.m16_5_analysis_container_image,
        container_sha256 : params.m16_5_analysis_container_digest,
        run_id           : params.cloud_run_id,
    ]
    def provenanceB64 = JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : 'M16.5 internal stability under M14 ALT-to-minor orientation only; no downstream or TEST',
        input_m14_minor  : params.ibd_enhanced_input_dir,
        historical_m16_5: params.ibd_enhanced_historical_dir,
        frozen_parameters: required,
    ]
    def runProvenanceB64 = JsonOutput.prettyPrint(JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()
    WRITE_M16_5_RUN_PROVENANCE(channel.value(runProvenanceB64))

    def repoDir = projectDir.resolve('..')
    def ibdScript = file("${repoDir}/bin/ibd_community_enhanced.py")
    def comparePy = file("${repoDir}/bin/compare_m16_5_orientation.py")
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py")
    def inputDir = params.ibd_enhanced_input_dir.toString()
    def historicalDir = params.ibd_enhanced_historical_dir.toString()

    def allSegments = file("${inputDir}/all_pairwise_segments.tsv.gz", checkIfExists: true)
    def pairSummary = file("${inputDir}/pair_sharing_summary.tsv", checkIfExists: true)
    def individualSummary = file("${inputDir}/individual_sharing_summary.tsv", checkIfExists: true)
    def cohortSummary = file(params.ibd_enhanced_cohort_summary, checkIfExists: true)
    def metadata = params.ibd_enhanced_metadata_file \
        ? file(params.ibd_enhanced_metadata_file, checkIfExists: true) \
        : file("${repoDir}/conf/empty.txt")

    def m16Input = channel.value(tuple(allSegments, pairSummary, individualSummary, ibdScript))
    IBD_COMMUNITY_ENHANCED(m16Input, channel.value(metadata))

    WRITE_M16_5_MANIFEST(
        channel.value(allSegments),
        channel.value(pairSummary),
        channel.value(individualSummary),
        IBD_COMMUNITY_ENHANCED.out.graph_edges,
        IBD_COMMUNITY_ENHANCED.out.graph_summary,
        IBD_COMMUNITY_ENHANCED.out.leiden_assignments,
        IBD_COMMUNITY_ENHANCED.out.global_summary,
        channel.value(ibdScript),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )

    COMPARE_M16_5_ORIENTATION(
        channel.value(file("${historicalDir}/leiden_assignments.tsv", checkIfExists: true)),
        channel.value(file("${historicalDir}/graph_edges.tsv.gz", checkIfExists: true)),
        channel.value(file("${historicalDir}/graph_summary.json", checkIfExists: true)),
        IBD_COMMUNITY_ENHANCED.out.leiden_assignments,
        IBD_COMMUNITY_ENHANCED.out.graph_edges,
        IBD_COMMUNITY_ENHANCED.out.graph_summary,
        channel.value(cohortSummary),
        channel.value(comparePy),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )
}
