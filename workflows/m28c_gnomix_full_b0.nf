nextflow.enable.dsl=2

include {
    WRITE_M28C_GNOMIX_FULL_B0_PROVENANCE;
    VALIDATE_M28C_GNOMIX_FULL_B0;
    TRAIN_M28C_GNOMIX_FULL_B0;
    INFER_M28C_GNOMIX_FULL_B0
} from '../modules/28C_GNOMIX_FULL_B0'

workflow {
    if (!(params.m28c_full_replicate in ['A', 'B'])) {
        error '--m28c_full_replicate must be A or B'
    }
    def repoDir = projectDir.resolve('..')
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        run_id: workflow.runName,
        replicate: params.m28c_full_replicate,
        container_path: params.m28c_full_container_image,
        container_sha256: params.m28c_full_container_digest,
        scientific_scope: 'protected full-B0 Gnomix resource benchmark; no TARGET truth/accuracy, screen, BR or BS claim',
        nextflow_command: workflow.commandLine,
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(provenance)
    ).bytes.encodeBase64().toString()

    def referenceVcf = file(params.m28c_full_reference_vcf, checkIfExists: true)
    def referenceTbi = file(params.m28c_full_reference_tbi, checkIfExists: true)
    def targetVcf = file(params.m28c_full_target_vcf, checkIfExists: true)
    def targetTbi = file(params.m28c_full_target_tbi, checkIfExists: true)
    def sampleMap = file(params.m28c_full_sample_map, checkIfExists: true)
    def b0Markers = file(params.m28c_full_b0_markers, checkIfExists: true)
    def geneticMap = file(params.m28c_full_genetic_map, checkIfExists: true)
    def gnomixConfig = file(params.m28c_full_gnomix_config, checkIfExists: true)
    def preregistration = file(params.m28c_full_preregistration, checkIfExists: true)
    def runnerPy = file("${repoDir}/bin/m28c_gnomix_training_smoke.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    WRITE_M28C_GNOMIX_FULL_B0_PROVENANCE(channel.value(provenanceB64))
    VALIDATE_M28C_GNOMIX_FULL_B0(
        channel.value(referenceVcf), channel.value(referenceTbi),
        channel.value(targetVcf), channel.value(targetTbi),
        channel.value(sampleMap), channel.value(b0Markers),
        channel.value(geneticMap), channel.value(gnomixConfig),
        channel.value(preregistration), channel.value(runnerPy),
        channel.value(manifestPy), WRITE_M28C_GNOMIX_FULL_B0_PROVENANCE.out.provenance,
        channel.value(provenanceB64),
    )
    TRAIN_M28C_GNOMIX_FULL_B0(
        channel.value(params.m28c_full_replicate), channel.value(referenceVcf),
        channel.value(sampleMap), channel.value(geneticMap), channel.value(gnomixConfig),
        channel.value(preregistration), VALIDATE_M28C_GNOMIX_FULL_B0.out.report,
        channel.value(runnerPy), channel.value(manifestPy),
        WRITE_M28C_GNOMIX_FULL_B0_PROVENANCE.out.provenance,
        channel.value(provenanceB64),
    )
    INFER_M28C_GNOMIX_FULL_B0(
        TRAIN_M28C_GNOMIX_FULL_B0.out.bundle, channel.value(targetVcf),
        VALIDATE_M28C_GNOMIX_FULL_B0.out.report, channel.value(gnomixConfig),
        channel.value(preregistration), channel.value(runnerPy), channel.value(manifestPy),
        WRITE_M28C_GNOMIX_FULL_B0_PROVENANCE.out.provenance,
        channel.value(provenanceB64),
    )
}
