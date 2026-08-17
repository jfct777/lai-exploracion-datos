nextflow.enable.dsl=2

include {
    WRITE_M28B_V2_RUN_PROVENANCE;
    RUN_M28B_V2_JOINT_CAPACITY_AUDIT
} from '../modules/28B_V2_LAI_JOINT_CAPACITY_AUDIT'

workflow {
    def required = [
        m28b_v2_tree_sequence: params.m28b_v2_tree_sequence,
        m28b_v2_pool_manifest: params.m28b_v2_pool_manifest,
        m28b_v2_genetic_map: params.m28b_v2_genetic_map,
        m28b_v2_baseline_template: params.m28b_v2_baseline_template,
        m28b_v2_container_image: params.m28b_v2_container_image,
    ]
    required.each { key, value -> if (!value) error "--${key} is required" }
    def repoDir = projectDir.resolve('..')
    def provenance = [
        git_commit: System.getenv('DNABR_GIT_COMMIT') ?: 'unknown',
        nextflow_version: workflow.nextflow.version.toString(),
        run_id: workflow.runName,
        container_path: params.m28b_v2_container_image,
        container_sha256: params.m28b_v2_container_digest,
        container_options: params.m28b_v2_container_options,
        scientific_scope: 'joint marker capacity only; no LAI, TARGET, truth or donor selection',
        nextflow_command: workflow.commandLine,
        baseline_template_uri: 'gs://projects-usp/dna-do-brasil/dnabr-lai-gnomix/vcf_fixed/dnabr.refpop.fixed.chr22.vcf.gz',
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(provenance)
    ).bytes.encodeBase64().toString()

    def treeSequence = file(params.m28b_v2_tree_sequence, checkIfExists: true)
    def poolManifest = file(params.m28b_v2_pool_manifest, checkIfExists: true)
    def geneticMap = file(params.m28b_v2_genetic_map, checkIfExists: true)
    def baselineTemplate = file(params.m28b_v2_baseline_template, checkIfExists: true)
    def m28Preregistration = file(params.m28b_v2_m28_preregistration, checkIfExists: true)
    def preregistration = file(params.m28b_v2_preregistration, checkIfExists: true)
    def auditV2Py = file("${repoDir}/bin/m28b_joint_capacity_audit.py", checkIfExists: true)
    def auditV1Py = file("${repoDir}/bin/m28b_marker_capacity_audit.py", checkIfExists: true)
    def m28Py = file("${repoDir}/bin/m28_simulation_preflight.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)

    WRITE_M28B_V2_RUN_PROVENANCE(channel.value(provenanceB64))
    RUN_M28B_V2_JOINT_CAPACITY_AUDIT(
        channel.value(treeSequence), channel.value(poolManifest), channel.value(geneticMap),
        channel.value(baselineTemplate), channel.value(m28Preregistration),
        channel.value(preregistration), channel.value(auditV2Py), channel.value(auditV1Py),
        channel.value(m28Py), channel.value(manifestPy),
        WRITE_M28B_V2_RUN_PROVENANCE.out, channel.value(provenanceB64),
    )
}
