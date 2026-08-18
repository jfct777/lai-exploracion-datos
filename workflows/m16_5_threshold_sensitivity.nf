nextflow.enable.dsl=2

include {
    WRITE_M16_5_THRESHOLD_SENSITIVITY_PROVENANCE;
    RUN_M16_5_THRESHOLD_SENSITIVITY
} from '../modules/16_5_THRESHOLD_SENSITIVITY'

workflow {
    def repoDir = projectDir.resolve('..')
    if( !params.m16_5_sensitivity_results_dir )
        throw new IllegalStateException('Set --m16_5_sensitivity_results_dir to a new, empty run directory')
    if( params.m16_5_sensitivity_results_dir.toString().contains('m16-5-minor-20260806d') )
        throw new IllegalStateException('The canonical M16.5 run directory is immutable')

    def m14Root = params.m16_5_sensitivity_m14_input_dir
    def pairSummary = file("${m14Root}/pair_sharing_summary.tsv", checkIfExists: true)
    def individualSummary = file("${m14Root}/individual_sharing_summary.tsv", checkIfExists: true)
    def globalSummary = file("${m14Root}/global_sharing_summary.json", checkIfExists: true)
    def metadata = file(params.m16_5_sensitivity_metadata_file, checkIfExists: true)
    def burdenTable = file(params.m16_5_sensitivity_burden_file, checkIfExists: true)
    def pcrelate = file(params.m16_5_sensitivity_pcrelate_file, checkIfExists: true)
    def preregistration = file("${repoDir}/conf/m16_5_threshold_sensitivity_preregistration.json", checkIfExists: true)
    def evaluatorPy = file("${repoDir}/bin/evaluate_m16_5_threshold_sensitivity.py", checkIfExists: true)
    def corePy = file("${repoDir}/bin/ibd_community_enhanced.py", checkIfExists: true)

    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: workflow.commitId
    if( !gitCommit )
        throw new IllegalStateException('Set DNABR_GIT_COMMIT to the exact source commit')
    def provenance = [
        git_commit       : gitCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        container_image  : params.m16_5_sensitivity_container_image,
        scientific_scope : 'Separate descriptive threshold sensitivity for M16.5; no NMF and no canonical replacement',
        selection_policy : 'Report all preregistered configurations; fineSTRUCTURE is descriptive only',
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(provenance)
    ).bytes.encodeBase64().toString()

    WRITE_M16_5_THRESHOLD_SENSITIVITY_PROVENANCE(channel.value(provenanceB64))
    RUN_M16_5_THRESHOLD_SENSITIVITY(
        pairSummary,
        individualSummary,
        globalSummary,
        metadata,
        burdenTable,
        pcrelate,
        preregistration,
        evaluatorPy,
        corePy,
        WRITE_M16_5_THRESHOLD_SENSITIVITY_PROVENANCE.out.provenance,
    )
}
