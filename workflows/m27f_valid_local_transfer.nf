nextflow.enable.dsl=2

include {
    WRITE_M27F_VALID_RUN_PROVENANCE;
    PROJECT_M27F_VALID_PANEL;
    AUDIT_M27F_VALID_TRANSFER
} from '../modules/27F_VALID_LOCAL_TRANSFER'

workflow {
    def repoDir = projectDir.resolve('..')
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        syntax_parser: System.getenv('NXF_SYNTAX_PARSER') ?: 'default',
        run_id: params.cloud_run_id,
        container_path: params.m27f_valid_container_image,
        container_sha256: params.m27f_valid_container_digest,
        container_options: params.m27f_valid_container_options,
        scientific_scope: 'One-shot SOURCE_VALID structural transfer audit; SOURCE_TEST remains sealed',
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command: workflow.commandLine,
        launch_dir: workflow.launchDir.toString(),
        project_dir: projectDir.toString(),
        source_panel_vcf: params.m27f_valid_source_panel_vcf,
        split_private: params.m27f_valid_split_private,
        ref_eligible_catalog: params.m27f_valid_ref_eligible_catalog,
        historical_baseline_vcf: params.m27f_valid_historical_baseline_vcf,
        source_valid_opened_once: true,
        source_test_opened: false,
    ]
    def runProvenanceB64 = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(runProvenance))
        .bytes.encodeBase64().toString()

    def sourcePanel = file(params.m27f_valid_source_panel_vcf, checkIfExists: true)
    def splitPrivate = file(params.m27f_valid_split_private, checkIfExists: true)
    def splitPublic = file(params.m27f_valid_split_public, checkIfExists: true)
    def splitManifest = file(params.m27f_valid_split_manifest, checkIfExists: true)
    def refEligible = file(params.m27f_valid_ref_eligible_catalog, checkIfExists: true)
    def refSupportPublic = file(params.m27f_valid_ref_support_public, checkIfExists: true)
    def refSupportManifest = file(params.m27f_valid_ref_support_manifest, checkIfExists: true)
    def baseline = file(params.m27f_valid_historical_baseline_vcf, checkIfExists: true)
    def geneticMap = file(params.m27f_valid_genetic_map, checkIfExists: true)
    def preregistration = file("${repoDir}/conf/m27f_valid_transfer_preregistration.json", checkIfExists: true)

    WRITE_M27F_VALID_RUN_PROVENANCE(channel.value(runProvenanceB64))
    PROJECT_M27F_VALID_PANEL(
        channel.value(sourcePanel),
        channel.value(splitPrivate),
        channel.value(splitPublic),
        channel.value(splitManifest),
        channel.value(preregistration),
        channel.value(file("${repoDir}/bin/project_m27f_valid_panel.py", checkIfExists: true)),
        channel.value(file("${repoDir}/bin/project_m27f_ref_panel.py", checkIfExists: true)),
    )
    AUDIT_M27F_VALID_TRANSFER(
        PROJECT_M27F_VALID_PANEL.out.valid_projection,
        channel.value(splitPrivate),
        channel.value(splitManifest),
        PROJECT_M27F_VALID_PANEL.out.public_receipt,
        channel.value(refEligible),
        channel.value(refSupportPublic),
        channel.value(refSupportManifest),
        channel.value(baseline),
        channel.value(geneticMap),
        channel.value(preregistration),
        channel.value(file("${repoDir}/bin/audit_m27f_valid_transfer.py", checkIfExists: true)),
        channel.value(file("${repoDir}/bin/audit_m27f_ref_support.py", checkIfExists: true)),
        channel.value(file("${repoDir}/bin/audit_rare_scaffold_bridge.py", checkIfExists: true)),
        channel.value(file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)),
        channel.value(provenanceB64),
    )
}
