nextflow.enable.dsl=2

include {
    WRITE_M28_RUN_PROVENANCE;
    RUN_M28_SIMULATION_PREFLIGHT
} from '../modules/28_LAI_SIMULATION_PREFLIGHT'

workflow {
    if (!params.m28_genetic_map) {
        error "--m28_genetic_map is required"
    }
    if (!params.m28_container_image) {
        error "--m28_container_image is required"
    }
    def repoDir = projectDir.resolve('..')
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        run_id: workflow.runName,
        container_path: params.m28_container_image,
        container_sha256: params.m28_container_digest,
        scientific_scope: 'technical preflight only; no LAI and no effect estimation',
        nextflow_command: workflow.commandLine,
        map_uri: params.m28_genetic_map,
        preregistration: params.m28_preregistration,
        root_seed: params.m28_root_seed,
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(provenance)
    ).bytes.encodeBase64().toString()

    def geneticMap = file(params.m28_genetic_map, checkIfExists: true)
    def preregistration = file(params.m28_preregistration, checkIfExists: true)
    def preflightPy = file("${repoDir}/bin/m28_simulation_preflight.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    WRITE_M28_RUN_PROVENANCE(channel.value(provenanceB64))
    RUN_M28_SIMULATION_PREFLIGHT(
        channel.value(geneticMap),
        channel.value(preregistration),
        channel.value(preflightPy),
        channel.value(manifestPy),
        WRITE_M28_RUN_PROVENANCE.out,
        channel.value(params.m28_root_seed as Integer),
        channel.value(provenanceB64),
    )
}
