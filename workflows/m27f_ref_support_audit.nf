nextflow.enable.dsl=2

include {
    WRITE_M27F_REF_RUN_PROVENANCE;
    PROJECT_M27F_REF_PANEL;
    AUDIT_M27F_REF_SUPPORT
} from '../modules/27F_REF_SUPPORT_AUDIT'

workflow {
    def repoDir = projectDir.resolve('..')
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        syntax_parser: System.getenv('NXF_SYNTAX_PARSER') ?: 'default',
        run_id: params.cloud_run_id,
        container_path: params.m27f_ref_container_image,
        container_sha256: params.m27f_ref_container_digest,
        scientific_scope: 'Mechanical DISCOVERY_CORE and REF projections followed by REF-only support audit; VALID and TEST remain sealed',
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command: workflow.commandLine,
        launch_dir: workflow.launchDir.toString(),
        project_dir: projectDir.toString(),
        raw_wgs_vcf: params.m27f_ref_raw_wgs_vcf,
        source_panel_vcf: params.m27f_ref_source_panel_vcf,
        split_private: params.m27f_ref_split_private,
        split_manifest: params.m27f_ref_split_manifest,
        source_valid_opened: false,
        source_test_opened: false,
    ]
    def runProvenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()

    def sourcePanel = file(params.m27f_ref_source_panel_vcf, checkIfExists: true)
    def rawWgs = file(params.m27f_ref_raw_wgs_vcf, checkIfExists: true)
    def splitPrivate = file(params.m27f_ref_split_private, checkIfExists: true)
    def splitPublic = file(params.m27f_ref_split_public, checkIfExists: true)
    def splitManifest = file(params.m27f_ref_split_manifest, checkIfExists: true)
    def m27eManifest = file(params.m27f_ref_m27e_manifest, checkIfExists: true)
    def m27eSupport = file(params.m27f_ref_m27e_support, checkIfExists: true)
    def preregistration = file("${repoDir}/conf/m27f_ref_support_preregistration.json", checkIfExists: true)
    def m27ePreregistration = file("${repoDir}/conf/m27e_ibd_rare_transfer_preregistration.json", checkIfExists: true)

    WRITE_M27F_REF_RUN_PROVENANCE(channel.value(runProvenanceB64))
    PROJECT_M27F_REF_PANEL(
        channel.value(sourcePanel),
        channel.value(splitPrivate),
        channel.value(splitPublic),
        channel.value(splitManifest),
        channel.value(preregistration),
        channel.value(file("${repoDir}/bin/project_m27f_ref_panel.py", checkIfExists: true)),
    )
    AUDIT_M27F_REF_SUPPORT(
        channel.value(rawWgs),
        PROJECT_M27F_REF_PANEL.out.discovery_projection,
        PROJECT_M27F_REF_PANEL.out.ref_projection,
        channel.value(splitPrivate),
        channel.value(splitManifest),
        PROJECT_M27F_REF_PANEL.out.public_receipt,
        channel.value(m27eManifest),
        channel.value(m27eSupport),
        channel.value(m27ePreregistration),
        channel.value(preregistration),
        channel.value(file("${repoDir}/bin/audit_m27f_ref_support.py", checkIfExists: true)),
        channel.value(file("${repoDir}/bin/audit_m27e_ibd_rare_transfer.py", checkIfExists: true)),
        channel.value(file("${repoDir}/bin/audit_rare_scaffold_bridge.py", checkIfExists: true)),
        channel.value(file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)),
        channel.value(provenanceB64),
    )
}
