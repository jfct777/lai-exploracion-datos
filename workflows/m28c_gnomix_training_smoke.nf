nextflow.enable.dsl=2

include {
    WRITE_M28C_GNOMIX_SMOKE_PROVENANCE;
    PREPARE_M28C_GNOMIX_SMOKE;
    TRAIN_M28C_GNOMIX_SMOKE;
    INFER_M28C_GNOMIX_SMOKE;
    COMPARE_M28C_GNOMIX_SMOKE
} from '../modules/28C_GNOMIX_TRAINING_SMOKE'

workflow {
    def required = [
        m28c_smoke_reference_vcf: params.m28c_smoke_reference_vcf,
        m28c_smoke_reference_tbi: params.m28c_smoke_reference_tbi,
        m28c_smoke_target_vcf: params.m28c_smoke_target_vcf,
        m28c_smoke_target_tbi: params.m28c_smoke_target_tbi,
        m28c_smoke_sample_map: params.m28c_smoke_sample_map,
        m28c_smoke_b0_markers: params.m28c_smoke_b0_markers,
        m28c_smoke_genetic_map: params.m28c_smoke_genetic_map,
        m28c_smoke_gnomix_config: params.m28c_smoke_gnomix_config,
        m28c_smoke_preregistration: params.m28c_smoke_preregistration,
        m28c_smoke_container_image: params.m28c_smoke_container_image,
    ]
    required.each { name, value ->
        if (!value) {
            error "--${name} is required"
        }
    }

    def repoDir = projectDir.resolve('..')
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        run_id: workflow.runName,
        container_path: params.m28c_smoke_container_image,
        container_sha256: params.m28c_smoke_container_digest,
        scientific_scope: 'technical Gnomix training smoke; no TARGET truth/accuracy, ceiling, BR or BS claim',
        nextflow_command: workflow.commandLine,
        gnomix_seed: 42,
        replicates: ['A', 'B'],
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(provenance)
    ).bytes.encodeBase64().toString()

    def referenceVcf = file(params.m28c_smoke_reference_vcf, checkIfExists: true)
    def referenceTbi = file(params.m28c_smoke_reference_tbi, checkIfExists: true)
    def targetVcf = file(params.m28c_smoke_target_vcf, checkIfExists: true)
    def targetTbi = file(params.m28c_smoke_target_tbi, checkIfExists: true)
    def sampleMap = file(params.m28c_smoke_sample_map, checkIfExists: true)
    def b0Markers = file(params.m28c_smoke_b0_markers, checkIfExists: true)
    def geneticMap = file(params.m28c_smoke_genetic_map, checkIfExists: true)
    def gnomixConfig = file(params.m28c_smoke_gnomix_config, checkIfExists: true)
    def preregistration = file(params.m28c_smoke_preregistration, checkIfExists: true)
    def smokePy = file("${repoDir}/bin/m28c_gnomix_training_smoke.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)
    def replicates = Channel.of('A', 'B')

    WRITE_M28C_GNOMIX_SMOKE_PROVENANCE(channel.value(provenanceB64))
    PREPARE_M28C_GNOMIX_SMOKE(
        channel.value(referenceVcf),
        channel.value(referenceTbi),
        channel.value(targetVcf),
        channel.value(targetTbi),
        channel.value(sampleMap),
        channel.value(b0Markers),
        channel.value(geneticMap),
        channel.value(gnomixConfig),
        channel.value(preregistration),
        channel.value(smokePy),
        channel.value(manifestPy),
        WRITE_M28C_GNOMIX_SMOKE_PROVENANCE.out.provenance,
        channel.value(provenanceB64),
    )
    TRAIN_M28C_GNOMIX_SMOKE(
        replicates,
        PREPARE_M28C_GNOMIX_SMOKE.out.reference,
        channel.value(sampleMap),
        channel.value(geneticMap),
        channel.value(gnomixConfig),
        channel.value(preregistration),
        PREPARE_M28C_GNOMIX_SMOKE.out.report,
        channel.value(smokePy),
        channel.value(manifestPy),
        WRITE_M28C_GNOMIX_SMOKE_PROVENANCE.out.provenance,
        channel.value(provenanceB64),
    )
    INFER_M28C_GNOMIX_SMOKE(
        TRAIN_M28C_GNOMIX_SMOKE.out.bundle,
        PREPARE_M28C_GNOMIX_SMOKE.out.target,
        PREPARE_M28C_GNOMIX_SMOKE.out.report,
        channel.value(gnomixConfig),
        channel.value(preregistration),
        channel.value(smokePy),
        channel.value(manifestPy),
        WRITE_M28C_GNOMIX_SMOKE_PROVENANCE.out.provenance,
        channel.value(provenanceB64),
    )

    def inferenceA = INFER_M28C_GNOMIX_SMOKE.out.bundle
        .filter { replicate, directory -> replicate == 'A' }
        .map { replicate, directory -> directory }
    def inferenceB = INFER_M28C_GNOMIX_SMOKE.out.bundle
        .filter { replicate, directory -> replicate == 'B' }
        .map { replicate, directory -> directory }
    COMPARE_M28C_GNOMIX_SMOKE(
        inferenceA,
        inferenceB,
        channel.value(preregistration),
        channel.value(smokePy),
        channel.value(manifestPy),
        WRITE_M28C_GNOMIX_SMOKE_PROVENANCE.out.provenance,
        channel.value(provenanceB64),
    )
}
