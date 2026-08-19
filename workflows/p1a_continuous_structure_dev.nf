nextflow.enable.dsl=2

include { P1A_CONTINUOUS_STRUCTURE_DEV } from '../modules/P1A_CONTINUOUS_STRUCTURE_DEV'

workflow {
    def repoDir = projectDir.resolve('..')
    def runId = params.p1a_run_id
    if( !runId || !(runId ==~ /[a-z0-9][a-z0-9._-]{2,63}/) )
        throw new IllegalStateException('Set DNABR_RUN_ID to a 3-64 character lowercase run identifier')
    if( !params.p1a_preflight_report )
        throw new IllegalStateException('Set --p1a_preflight_report to the immutable PASS preflight report')
    if( !params.p1a_results_dir )
        throw new IllegalStateException('Set --p1a_results_dir to a new output prefix')

    def codeCommit = System.getenv('DNABR_GIT_COMMIT')
    if( !codeCommit || !(codeCommit ==~ /[0-9a-f]{40}/) )
        throw new IllegalStateException('Set DNABR_GIT_COMMIT to the exact 40-character source commit')

    def pairs = file(params.p1a_pairs, checkIfExists: true)
    def globalSummary = file(params.p1a_global_summary, checkIfExists: true)
    def metadata = file(params.p1a_metadata, checkIfExists: true)
    def burden = file(params.p1a_burden, checkIfExists: true)
    def featureStore = file(params.p1a_feature_store, checkIfExists: true)
    def modelingMaster = file(params.p1a_modeling_master, checkIfExists: true)
    def splitManifest = file(params.p1a_split_manifest, checkIfExists: true)
    def preflightReport = file(params.p1a_preflight_report, checkIfExists: true)
    def preregistration = file(
        "${repoDir}/conf/p1a_continuous_structure_preregistration.json",
        checkIfExists: true,
    )
    def runnerPy = file("${repoDir}/bin/run_p1a_continuous_structure_dev.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    def provenance = [
        git_commit       : codeCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        container_path   : params.p1a_container_image,
        container_sha256 : params.p1a_container_digest,
        scientific_scope : 'DEV-only transductive regional association; no fold3, graph nulls, novel-Brazil audit or NAM',
        primary_graph    : 'binary M14-minor using all 54522 observed pairs',
        sensitivity_graph: 'same all-DNABR TRAIN anchors, log1p(total_shared_bp/1Mb) weights only; cannot rescue binary failure',
        result_uri       : params.p1a_results_dir,
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(provenance))
        .bytes.encodeBase64().toString()

    P1A_CONTINUOUS_STRUCTURE_DEV(
        pairs,
        globalSummary,
        metadata,
        burden,
        featureStore,
        modelingMaster,
        splitManifest,
        preflightReport,
        preregistration,
        runnerPy,
        manifestPy,
        codeCommit,
        provenanceB64,
    )
}
