nextflow.enable.dsl=2

include {
    WRITE_M28C_B0_RUN_PROVENANCE;
    RUN_M28C_B0_INPUT_PREFLIGHT;
    AUDIT_M28C_B0_GNOMIX_INGEST
} from '../modules/28C_B0_INPUT_PREFLIGHT'

workflow {
    def required = [
        m28c_b0_tree_sequence: params.m28c_b0_tree_sequence,
        m28c_b0_pool_manifest: params.m28c_b0_pool_manifest,
        m28c_b0_mosaic_events: params.m28c_b0_mosaic_events,
        m28c_b0_markers: params.m28c_b0_markers,
        m28c_b0_preregistration: params.m28c_b0_preregistration,
        m28c_b0_container_image: params.m28c_b0_container_image,
        m28c_gnomix_ingest_preregistration: params.m28c_gnomix_ingest_preregistration,
        m28c_gnomix_container_image: params.m28c_gnomix_container_image,
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
        container_path: params.m28c_b0_container_image,
        container_sha256: params.m28c_b0_container_digest,
        gnomix_container_path: params.m28c_gnomix_container_image,
        gnomix_container_sha256: params.m28c_gnomix_container_digest,
        scientific_scope: 'technical B0 input preflight; no LAI and no effect estimation',
        nextflow_command: workflow.commandLine,
        root_seed: params.m28c_b0_root_seed,
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(provenance)
    ).bytes.encodeBase64().toString()
    def ingestProvenance = provenance + [
        container_path: params.m28c_gnomix_container_image,
        container_sha256: params.m28c_gnomix_container_digest,
        scientific_scope: 'technical Gnomix ingest audit; no training, truth, or effect estimation',
    ]
    def ingestProvenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(ingestProvenance)
    ).bytes.encodeBase64().toString()

    def treeSequence = file(params.m28c_b0_tree_sequence, checkIfExists: true)
    def poolManifest = file(params.m28c_b0_pool_manifest, checkIfExists: true)
    def mosaicEvents = file(params.m28c_b0_mosaic_events, checkIfExists: true)
    def b0Markers = file(params.m28c_b0_markers, checkIfExists: true)
    def preregistration = file(params.m28c_b0_preregistration, checkIfExists: true)
    def materializePy = file("${repoDir}/bin/materialize_m28c_b0_inputs.py", checkIfExists: true)
    def ingestPreregistration = file(params.m28c_gnomix_ingest_preregistration, checkIfExists: true)
    def ingestPy = file("${repoDir}/bin/audit_m28c_gnomix_ingest.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    WRITE_M28C_B0_RUN_PROVENANCE(channel.value(provenanceB64))
    RUN_M28C_B0_INPUT_PREFLIGHT(
        channel.value(treeSequence),
        channel.value(poolManifest),
        channel.value(mosaicEvents),
        channel.value(b0Markers),
        channel.value(preregistration),
        channel.value(materializePy),
        channel.value(manifestPy),
        WRITE_M28C_B0_RUN_PROVENANCE.out,
        channel.value(provenanceB64),
    )
    AUDIT_M28C_B0_GNOMIX_INGEST(
        RUN_M28C_B0_INPUT_PREFLIGHT.out.reference_vcf,
        RUN_M28C_B0_INPUT_PREFLIGHT.out.target_vcf,
        RUN_M28C_B0_INPUT_PREFLIGHT.out.report,
        channel.value(ingestPreregistration),
        channel.value(ingestPy),
        channel.value(manifestPy),
        WRITE_M28C_B0_RUN_PROVENANCE.out,
        channel.value(ingestProvenanceB64),
    )
}
