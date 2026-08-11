nextflow.enable.dsl=2

import groovy.json.JsonOutput

include {
    WRITE_M16_TARGET_AUDIT_RUN_PROVENANCE;
    AUDIT_M16_TARGET_VIABILITY
} from '../modules/26_M16_TARGET_VIABILITY_AUDIT'

workflow {
    def repoDir = projectDir.resolve('..')
    def minorAssignments = file(params.m16_target_audit_minor_assignments, checkIfExists: true)
    def modelingMaster = file(params.m16_target_audit_modeling_master, checkIfExists: true)
    def splitManifest = file(params.m16_target_audit_split_manifest, checkIfExists: true)
    def preregistration = file(
        "${repoDir}/conf/m26_m16_target_audit_preregistration.json",
        checkIfExists: true,
    )
    def auditPy = file("${repoDir}/bin/audit_m16_target_viability.py")
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py")

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
        scientific_scope : 'Single-pass M16.5-minor target audit; TRAIN/VALIDATION only; no model, TEST, reclustering or genotype access',
        minor_assignments: params.m16_target_audit_minor_assignments,
        modeling_master : params.m16_target_audit_modeling_master,
        split_manifest  : params.m16_target_audit_split_manifest,
    ]
    def runProvenanceB64 = JsonOutput.prettyPrint(JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()

    WRITE_M16_TARGET_AUDIT_RUN_PROVENANCE(channel.value(runProvenanceB64))
    AUDIT_M16_TARGET_VIABILITY(
        channel.value(minorAssignments),
        channel.value(modelingMaster),
        channel.value(splitManifest),
        channel.value(preregistration),
        channel.value(auditPy),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )
}
