nextflow.enable.dsl=2

include {
    WRITE_RARE_SCAFFOLD_BRIDGE_RUN_PROVENANCE;
    AUDIT_RARE_SCAFFOLD_BRIDGE
} from '../modules/27B_RARE_SCAFFOLD_BRIDGE'

workflow {
    def repoDir = projectDir.resolve('..')
    def rawWgsVcf = file(params.rare_scaffold_bridge_raw_wgs_vcf, checkIfExists: true)
    def phasedScaffoldVcf = file(params.rare_scaffold_bridge_phased_scaffold_vcf, checkIfExists: true)
    def gnomixReferenceVcf = file(params.rare_scaffold_bridge_gnomix_reference_vcf, checkIfExists: true)
    def metadata = file(params.rare_scaffold_bridge_metadata, checkIfExists: true)
    def preregistration = file(
        "${repoDir}/conf/m27b_rare_scaffold_bridge_preregistration.json",
        checkIfExists: true,
    )
    def auditPy = file("${repoDir}/bin/audit_rare_scaffold_bridge.py", checkIfExists: true)
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
        scientific_scope : 'Read-only chr22 raw-WGS to phased-scaffold bridge audit; no PC-Relate, simulation, LAI, training or TEST access',
        raw_wgs_vcf      : params.rare_scaffold_bridge_raw_wgs_vcf,
        phased_scaffold  : params.rare_scaffold_bridge_phased_scaffold_vcf,
        gnomix_reference : params.rare_scaffold_bridge_gnomix_reference_vcf,
        sample_metadata  : params.rare_scaffold_bridge_metadata,
    ]
    def runProvenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()

    WRITE_RARE_SCAFFOLD_BRIDGE_RUN_PROVENANCE(channel.value(runProvenanceB64))
    AUDIT_RARE_SCAFFOLD_BRIDGE(
        channel.value(rawWgsVcf),
        channel.value(phasedScaffoldVcf),
        channel.value(gnomixReferenceVcf),
        channel.value(metadata),
        channel.value(preregistration),
        channel.value(auditPy),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )
}
