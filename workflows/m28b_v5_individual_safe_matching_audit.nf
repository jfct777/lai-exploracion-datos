nextflow.enable.dsl=2

include {
    WRITE_M28B_V5_RUN_PROVENANCE;
    RUN_M28B_V5_DEVELOPMENT;
    RUN_M28B_V5_VALIDATION
} from '../modules/28B_V5_INDIVIDUAL_SAFE_MATCHING_AUDIT'

workflow {
    def required = [
        m28b_v5_dev_tree: params.m28b_v5_dev_tree,
        m28b_v5_dev_pools: params.m28b_v5_dev_pools,
        m28b_v5_dev_preflight_report: params.m28b_v5_dev_preflight_report,
        m28b_v5_dev_preflight_manifest: params.m28b_v5_dev_preflight_manifest,
        m28b_v5_validation_tree: params.m28b_v5_validation_tree,
        m28b_v5_validation_pools: params.m28b_v5_validation_pools,
        m28b_v5_validation_preflight_report: params.m28b_v5_validation_preflight_report,
        m28b_v5_validation_preflight_manifest: params.m28b_v5_validation_preflight_manifest,
        m28b_v5_preflight_reproducibility: params.m28b_v5_preflight_reproducibility,
        m28b_v5_genetic_map: params.m28b_v5_genetic_map,
        m28b_v5_baseline_template: params.m28b_v5_baseline_template,
        m28b_v5_container_image: params.m28b_v5_container_image,
    ]
    required.each { key, value -> if (!value) error "--${key} is required" }
    def repoDir = projectDir.resolve('..')
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        run_id: workflow.runName,
        container_path: params.m28b_v5_container_image,
        container_sha256: params.m28b_v5_container_digest,
        container_options: params.m28b_v5_container_options,
        scientific_scope: 'Individual-safe DEV plus one technical validation; no LAI, TARGET or truth',
        nextflow_command: workflow.commandLine,
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(provenance)
    ).bytes.encodeBase64().toString()

    def devTree = file(params.m28b_v5_dev_tree, checkIfExists: true)
    def devPools = file(params.m28b_v5_dev_pools, checkIfExists: true)
    def devReport = file(params.m28b_v5_dev_preflight_report, checkIfExists: true)
    def devManifest = file(params.m28b_v5_dev_preflight_manifest, checkIfExists: true)
    def validationTree = file(params.m28b_v5_validation_tree, checkIfExists: true)
    def validationPools = file(params.m28b_v5_validation_pools, checkIfExists: true)
    def validationReport = file(params.m28b_v5_validation_preflight_report, checkIfExists: true)
    def validationManifest = file(params.m28b_v5_validation_preflight_manifest, checkIfExists: true)
    def reproducibility = file(params.m28b_v5_preflight_reproducibility, checkIfExists: true)
    def geneticMap = file(params.m28b_v5_genetic_map, checkIfExists: true)
    def baselineTemplate = file(params.m28b_v5_baseline_template, checkIfExists: true)
    def m28Preregistration = file(params.m28b_v5_m28_preregistration, checkIfExists: true)
    def preregistration = file(params.m28b_v5_preregistration, checkIfExists: true)
    def auditV5Py = file("${repoDir}/bin/m28b_optimal_matching_audit.py", checkIfExists: true)
    def auditV3Py = file("${repoDir}/bin/m28b_generic_capacity_audit.py", checkIfExists: true)
    def auditV2Py = file("${repoDir}/bin/m28b_joint_capacity_audit.py", checkIfExists: true)
    def auditV1Py = file("${repoDir}/bin/m28b_marker_capacity_audit.py", checkIfExists: true)
    def m28Py = file("${repoDir}/bin/m28_simulation_preflight.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    WRITE_M28B_V5_RUN_PROVENANCE(channel.value(provenanceB64))
    RUN_M28B_V5_DEVELOPMENT(
        channel.value(devTree), channel.value(devPools), channel.value(devReport),
        channel.value(devManifest), channel.value(reproducibility), channel.value(geneticMap),
        channel.value(baselineTemplate), channel.value(m28Preregistration),
        channel.value(preregistration), channel.value(auditV5Py), channel.value(auditV3Py),
        channel.value(auditV2Py), channel.value(auditV1Py), channel.value(m28Py),
        channel.value(manifestPy), WRITE_M28B_V5_RUN_PROVENANCE.out,
        channel.value(provenanceB64),
    )
    def frozenForValidation = RUN_M28B_V5_DEVELOPMENT.out.frozen.filter { frozen ->
        def selection = new groovy.json.JsonSlurper().parse(frozen.toFile())
        selection.frozen_K != null
    }
    RUN_M28B_V5_VALIDATION(
        channel.value(validationTree), channel.value(validationPools),
        channel.value(validationReport), channel.value(validationManifest),
        channel.value(reproducibility), channel.value(geneticMap),
        channel.value(baselineTemplate), channel.value(m28Preregistration),
        channel.value(preregistration), frozenForValidation,
        channel.value(auditV5Py), channel.value(auditV3Py), channel.value(auditV2Py),
        channel.value(auditV1Py), channel.value(m28Py), channel.value(manifestPy),
        WRITE_M28B_V5_RUN_PROVENANCE.out, channel.value(provenanceB64),
    )
}
