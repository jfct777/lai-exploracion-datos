nextflow.enable.dsl=2

include {
    WRITE_LAI_PILOT_PREFLIGHT_RUN_PROVENANCE;
    AUDIT_LAI_PILOT_PREFLIGHT
} from '../modules/27_LAI_PILOT_PREFLIGHT'

workflow {
    def repoDir = projectDir.resolve('..')
    def gnomixReferenceVcf = file(params.lai_pilot_preflight_gnomix_reference_vcf, checkIfExists: true)
    def externalPanelVcf = file(params.lai_pilot_preflight_external_panel_vcf, checkIfExists: true)
    def gnomixModel = file(params.lai_pilot_preflight_gnomix_model, checkIfExists: true)
    def gnomixConfig = file(params.lai_pilot_preflight_gnomix_config, checkIfExists: true)
    def geneticMap = file(params.lai_pilot_preflight_genetic_map, checkIfExists: true)
    def metadata = file(params.lai_pilot_preflight_metadata, checkIfExists: true)
    def top95Nam = file(params.lai_pilot_preflight_top95_nam, checkIfExists: true)
    def namUnrelatedKeep = file(params.lai_pilot_preflight_nam_unrelated_keep, checkIfExists: true)
    def preregistration = file(
        "${repoDir}/conf/m27_lai_pilot_preflight_preregistration.json",
        checkIfExists: true,
    )
    def auditPy = file("${repoDir}/bin/audit_lai_pilot_preflight.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: 'unknown'
    def provenance = [
        git_commit       : gitCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : params.m16_5_analysis_container_image,
        container_sha256 : params.m16_5_analysis_container_digest,
        run_id           : params.cloud_run_id,
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : 'Read-only chr22 identifiability preflight; no simulation, model inference, training or TEST access',
        gnomix_reference : params.lai_pilot_preflight_gnomix_reference_vcf,
        external_panel   : params.lai_pilot_preflight_external_panel_vcf,
        gnomix_model     : params.lai_pilot_preflight_gnomix_model,
        genetic_map      : params.lai_pilot_preflight_genetic_map,
    ]
    def runProvenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()

    WRITE_LAI_PILOT_PREFLIGHT_RUN_PROVENANCE(channel.value(runProvenanceB64))
    AUDIT_LAI_PILOT_PREFLIGHT(
        channel.value(gnomixReferenceVcf),
        channel.value(externalPanelVcf),
        channel.value(gnomixModel),
        channel.value(gnomixConfig),
        channel.value(geneticMap),
        channel.value(metadata),
        channel.value(top95Nam),
        channel.value(namUnrelatedKeep),
        channel.value(preregistration),
        channel.value(auditPy),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )
}
