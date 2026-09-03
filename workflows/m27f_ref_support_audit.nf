nextflow.enable.dsl=2

include {
    WRITE_M27F_SUPPORT_RUN_PROVENANCE;
    PROJECT_M27F_SUPPORT_PANEL;
    AUDIT_M27F_REF_SUPPORT;
    CLAIM_M27F_VALIDATION_OPENING;
    AUDIT_M27F_VALID_SUPPORT
} from '../modules/27F_REF_SUPPORT_AUDIT'

workflow {
    def repoDir = projectDir.resolve('..')
    if (!params.m27f_support_validation_claim_registry) {
        error "--m27f_support_validation_claim_registry is required"
    }
    if (!params.m27f_support_validation_claim_uri) {
        error "--m27f_support_validation_claim_uri is required"
    }
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        syntax_parser: System.getenv('NXF_SYNTAX_PARSER') ?: 'default',
        run_id: params.cloud_run_id,
        container_path: params.m27f_support_container_image,
        container_sha256: params.m27f_support_container_digest,
        container_options: params.m27f_support_container_options,
        scientific_scope: 'Mechanical DISCOVERY/REF/VALID projection, REF-only selection and one frozen VALID evaluation; SOURCE_TEST absent',
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance)
        .bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command: workflow.commandLine,
        launch_dir: workflow.launchDir.toString(),
        project_dir: projectDir.toString(),
        raw_wgs_vcf: params.m27f_support_raw_wgs_vcf,
        source_panel_vcf: params.m27f_support_source_panel_vcf,
        baseline_vcf: params.m27f_support_baseline_vcf,
        split_private: params.m27f_support_split_private,
        split_manifest: params.m27f_support_split_manifest,
        validation_claim_registry: params.m27f_support_validation_claim_registry,
        validation_claim_key: params.m27f_support_validation_claim_key,
        validation_claim_uri: params.m27f_support_validation_claim_uri,
        support_threshold_source: 'conf/m27f_ref_support_preregistration.json',
        source_test_projection_created: false,
        source_test_genotypes_opened: false,
        simulation_performed: false,
        lai_performed: false,
        model_training_performed: false,
    ]
    def runProvenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(runProvenance)
    ).bytes.encodeBase64().toString()

    def sourcePanel = file(params.m27f_support_source_panel_vcf, checkIfExists: true)
    def rawWgs = file(params.m27f_support_raw_wgs_vcf, checkIfExists: true)
    def baseline = file(params.m27f_support_baseline_vcf, checkIfExists: true)
    def splitPrivate = file(params.m27f_support_split_private, checkIfExists: true)
    def splitPublic = file(params.m27f_support_split_public, checkIfExists: true)
    def splitManifest = file(params.m27f_support_split_manifest, checkIfExists: true)
    def m27eManifest = file(params.m27f_support_m27e_manifest, checkIfExists: true)
    def m27eSupport = file(params.m27f_support_m27e_support, checkIfExists: true)
    def preregistration = file(
        "${repoDir}/conf/m27f_ref_support_preregistration.json",
        checkIfExists: true
    )
    def m27ePreregistration = file(
        "${repoDir}/conf/m27e_ibd_rare_transfer_preregistration.json",
        checkIfExists: true
    )
    def projectionPy = file(
        "${repoDir}/bin/project_m27f_ref_panel.py", checkIfExists: true
    )
    def refAuditPy = file(
        "${repoDir}/bin/audit_m27f_ref_support.py", checkIfExists: true
    )
    def validAuditPy = file(
        "${repoDir}/bin/audit_m27f_valid_support.py", checkIfExists: true
    )
    def claimPy = file(
        "${repoDir}/bin/claim_m27f_validation_opening.py", checkIfExists: true
    )
    def validationContractPy = file(
        "${repoDir}/bin/m27f_validation_contract.py", checkIfExists: true
    )
    def m27ePy = file(
        "${repoDir}/bin/audit_m27e_ibd_rare_transfer.py", checkIfExists: true
    )
    def bridgePy = file(
        "${repoDir}/bin/audit_rare_scaffold_bridge.py", checkIfExists: true
    )
    def manifestPy = file(
        "${repoDir}/bin/write_stage_manifest.py", checkIfExists: true
    )

    WRITE_M27F_SUPPORT_RUN_PROVENANCE(channel.value(runProvenanceB64))

    PROJECT_M27F_SUPPORT_PANEL(
        channel.value(sourcePanel),
        channel.value(splitPrivate),
        channel.value(splitPublic),
        channel.value(splitManifest),
        channel.value(preregistration),
        channel.value(projectionPy),
        channel.value(manifestPy),
        WRITE_M27F_SUPPORT_RUN_PROVENANCE.out,
        channel.value(provenanceB64),
    )

    AUDIT_M27F_REF_SUPPORT(
        channel.value(rawWgs),
        channel.value(baseline),
        PROJECT_M27F_SUPPORT_PANEL.out.discovery_projection,
        PROJECT_M27F_SUPPORT_PANEL.out.ref_projection,
        channel.value(splitPrivate),
        channel.value(splitManifest),
        PROJECT_M27F_SUPPORT_PANEL.out.public_receipt,
        PROJECT_M27F_SUPPORT_PANEL.out.manifest,
        channel.value(m27eManifest),
        channel.value(m27eSupport),
        channel.value(m27ePreregistration),
        channel.value(preregistration),
        channel.value(refAuditPy),
        channel.value(m27ePy),
        channel.value(bridgePy),
        channel.value(manifestPy),
        WRITE_M27F_SUPPORT_RUN_PROVENANCE.out,
        channel.value(provenanceB64),
    )

    CLAIM_M27F_VALIDATION_OPENING(
        PROJECT_M27F_SUPPORT_PANEL.out.public_receipt,
        PROJECT_M27F_SUPPORT_PANEL.out.manifest,
        AUDIT_M27F_REF_SUPPORT.out.private_support,
        AUDIT_M27F_REF_SUPPORT.out.private_primary_catalog,
        AUDIT_M27F_REF_SUPPORT.out.public_support,
        AUDIT_M27F_REF_SUPPORT.out.manifest,
        channel.value(preregistration),
        channel.value(claimPy),
        channel.value(validationContractPy),
        channel.value(validAuditPy),
        channel.value(refAuditPy),
        channel.value(m27ePy),
        channel.value(bridgePy),
        channel.value(params.m27f_support_container_image),
        channel.value(params.m27f_support_container_digest),
        channel.value(params.cloud_run_id),
        channel.value(params.m27f_support_validation_claim_registry),
        channel.value(params.m27f_support_validation_claim_key),
    )

    def authorizedOpening = CLAIM_M27F_VALIDATION_OPENING.out.receipt.filter {
        receipt ->
            new groovy.json.JsonSlurper().parseText(receipt.text).decision ==
                'VALIDATION_OPENING_FROZEN'
    }

    AUDIT_M27F_VALID_SUPPORT(
        PROJECT_M27F_SUPPORT_PANEL.out.valid_projection,
        channel.value(splitPrivate),
        channel.value(splitManifest),
        PROJECT_M27F_SUPPORT_PANEL.out.public_receipt,
        PROJECT_M27F_SUPPORT_PANEL.out.manifest,
        AUDIT_M27F_REF_SUPPORT.out.private_support,
        AUDIT_M27F_REF_SUPPORT.out.private_primary_catalog,
        AUDIT_M27F_REF_SUPPORT.out.public_support,
        AUDIT_M27F_REF_SUPPORT.out.manifest,
        authorizedOpening,
        channel.value(preregistration),
        channel.value(validAuditPy),
        channel.value(refAuditPy),
        channel.value(m27ePy),
        channel.value(bridgePy),
        channel.value(claimPy),
        channel.value(validationContractPy),
        channel.value(params.m27f_support_container_image),
        channel.value(params.m27f_support_container_digest),
        channel.value(manifestPy),
        WRITE_M27F_SUPPORT_RUN_PROVENANCE.out,
        channel.value(provenanceB64),
    )
}
