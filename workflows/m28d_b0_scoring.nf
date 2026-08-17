nextflow.enable.dsl=2

include {
    WRITE_M28D_B0_SCORING_PROVENANCE;
    VALIDATE_M28D_B0_SCORER;
    AUTHENTICATE_M28D_B0_PAIR;
    SCORE_M28D_B0_PAIR
} from '../modules/28D_B0_SCORING'

workflow {
    def repoDir = projectDir.resolve('..')
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        run_id: workflow.runName,
        container_path: params.m28d_container_image,
        container_sha256: params.m28d_container_digest,
        scientific_scope: 'descriptive B0 scoring on protected M28 validation seed; no inference, SESOI selection or BR/BS authorization',
        nextflow_command: workflow.commandLine,
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(provenance)
    ).bytes.encodeBase64().toString()

    def truth = file(params.m28d_truth, checkIfExists: true)
    def b0Markers = file(params.m28d_b0_markers, checkIfExists: true)
    def geneticMap = file(params.m28d_genetic_map, checkIfExists: true)
    def fbA = file(params.m28d_fb_A, checkIfExists: true)
    def mspA = file(params.m28d_msp_A, checkIfExists: true)
    def fbB = file(params.m28d_fb_B, checkIfExists: true)
    def mspB = file(params.m28d_msp_B, checkIfExists: true)
    def comparison = file(params.m28d_m28c_comparison, checkIfExists: true)
    def simulationManifest = file(params.m28d_simulation_manifest, checkIfExists: true)
    def b0PreflightManifest = file(params.m28d_b0_preflight_manifest, checkIfExists: true)
    def ingestReport = file(params.m28d_ingest_report, checkIfExists: true)
    def inferenceManifestA = file(params.m28d_inference_manifest_A, checkIfExists: true)
    def inferenceManifestB = file(params.m28d_inference_manifest_B, checkIfExists: true)
    def preregistration = file(params.m28d_preregistration, checkIfExists: true)
    def scorerPy = file("${repoDir}/bin/m28d_b0_scorer.py", checkIfExists: true)
    def knownAnswersPy = file("${repoDir}/bin/m28d_b0_known_answers.py", checkIfExists: true)
    def unitTestPy = file("${repoDir}/tests/test_m28d_b0_scorer.py", checkIfExists: true)

    WRITE_M28D_B0_SCORING_PROVENANCE(channel.value(provenanceB64))
    VALIDATE_M28D_B0_SCORER(
        channel.value(scorerPy),
        channel.value(knownAnswersPy),
        channel.value(unitTestPy),
        channel.value(preregistration),
        WRITE_M28D_B0_SCORING_PROVENANCE.out.provenance,
    )
    AUTHENTICATE_M28D_B0_PAIR(
        VALIDATE_M28D_B0_SCORER.out.receipt,
        channel.value(truth),
        channel.value(b0Markers),
        channel.value(geneticMap),
        channel.value(fbA),
        channel.value(mspA),
        channel.value(fbB),
        channel.value(mspB),
        channel.value(comparison),
        channel.value(simulationManifest),
        channel.value(b0PreflightManifest),
        channel.value(ingestReport),
        channel.value(inferenceManifestA),
        channel.value(inferenceManifestB),
        channel.value(preregistration),
        channel.value(scorerPy),
    )
    SCORE_M28D_B0_PAIR(
        VALIDATE_M28D_B0_SCORER.out.receipt,
        AUTHENTICATE_M28D_B0_PAIR.out.receipt,
        channel.value(truth),
        channel.value(b0Markers),
        channel.value(geneticMap),
        channel.value(fbA),
        channel.value(mspA),
        channel.value(fbB),
        channel.value(mspB),
        channel.value(comparison),
        channel.value(simulationManifest),
        channel.value(b0PreflightManifest),
        channel.value(ingestReport),
        channel.value(inferenceManifestA),
        channel.value(inferenceManifestB),
        channel.value(preregistration),
        channel.value(scorerPy),
        WRITE_M28D_B0_SCORING_PROVENANCE.out.provenance,
    )
}
