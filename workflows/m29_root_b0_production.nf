nextflow.enable.dsl=2

include {
    WRITE_M29_ROOT_B0_PROVENANCE;
    SELECT_M29_ROOT_B0;
    MATERIALIZE_M29_ROOT_B0;
    INGEST_M29_ROOT_B0
} from '../modules/29_ROOT_B0_PRODUCTION'

workflow {
    def required = [
        params.m29_b0_root17_tree, params.m29_b0_root17_pools, params.m29_b0_root17_report,
        params.m29_b0_root17_manifest, params.m29_b0_root17_mosaic,
        params.m29_b0_root18_tree, params.m29_b0_root18_pools, params.m29_b0_root18_report,
        params.m29_b0_root18_manifest, params.m29_b0_root18_mosaic,
        params.m29_b0_reproducibility, params.m29_b0_genetic_map, params.m29_b0_baseline_template,
        params.m29_b0_m28_contract, params.m29_b0_m28b_contract, params.m29_b0_production_contract
    ]
    if (required.any { value -> !value }) error 'All M29 root-B0 inputs are required; no historical B0 may be substituted'
    def repoDir = projectDir.resolve('..')
    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: workflow.commitId
    if (!gitCommit) error 'Set DNABR_GIT_COMMIT to the exact source commit'
    def provenance = [
        git_commit       : gitCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        nextflow_command : workflow.commandLine,
        run_name         : workflow.runName,
        simulation_image : params.m29_b0_sim_container,
        gnomix_image     : params.m29_b0_gnomix_container,
        results_dir      : params.m29_b0_results_dir,
        scientific_scope : 'Root-specific B0 selection, VCF materialization and Gnomix ingest only; no truth, training or inference',
    ]
    def provenanceB64 = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(provenance)
    ).bytes.encodeBase64().toString()
    WRITE_M29_ROOT_B0_PROVENANCE(channel.value(provenanceB64))
    roots = Channel.of(
        tuple('root17', 20260817, file(params.m29_b0_root17_tree, checkIfExists: true), file(params.m29_b0_root17_pools, checkIfExists: true), file(params.m29_b0_root17_report, checkIfExists: true), file(params.m29_b0_root17_manifest, checkIfExists: true), file(params.m29_b0_root17_mosaic, checkIfExists: true)),
        tuple('root18', 20260818, file(params.m29_b0_root18_tree, checkIfExists: true), file(params.m29_b0_root18_pools, checkIfExists: true), file(params.m29_b0_root18_report, checkIfExists: true), file(params.m29_b0_root18_manifest, checkIfExists: true), file(params.m29_b0_root18_mosaic, checkIfExists: true))
    )
    productionContract = file(params.m29_b0_production_contract, checkIfExists: true)
    manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)
    SELECT_M29_ROOT_B0(
        roots,
        file(params.m29_b0_reproducibility, checkIfExists: true),
        file(params.m29_b0_genetic_map, checkIfExists: true),
        file(params.m29_b0_baseline_template, checkIfExists: true),
        file(params.m29_b0_m28_contract, checkIfExists: true),
        file(params.m29_b0_m28b_contract, checkIfExists: true),
        productionContract,
        file("${repoDir}/bin/prepare_m29_root_b0.py", checkIfExists: true),
        file("${repoDir}/bin/m28b_optimal_matching_audit.py", checkIfExists: true),
        file("${repoDir}/bin/m28b_generic_capacity_audit.py", checkIfExists: true),
        file("${repoDir}/bin/m28b_joint_capacity_audit.py", checkIfExists: true),
        file("${repoDir}/bin/m28b_marker_capacity_audit.py", checkIfExists: true),
        file("${repoDir}/bin/m28_simulation_preflight.py", checkIfExists: true),
        manifestPy,
        WRITE_M29_ROOT_B0_PROVENANCE.out.provenance
    )
    MATERIALIZE_M29_ROOT_B0(
        SELECT_M29_ROOT_B0.out.selected,
        productionContract,
        file("${repoDir}/bin/materialize_m29_root_b0.py", checkIfExists: true),
        file("${repoDir}/bin/materialize_m28c_b0_inputs.py", checkIfExists: true),
        manifestPy,
        WRITE_M29_ROOT_B0_PROVENANCE.out.provenance
    )
    INGEST_M29_ROOT_B0(
        MATERIALIZE_M29_ROOT_B0.out.materialized,
        productionContract,
        file("${repoDir}/bin/ingest_m29_root_b0.py", checkIfExists: true),
        file("${repoDir}/bin/audit_m28c_gnomix_ingest.py", checkIfExists: true),
        manifestPy,
        WRITE_M29_ROOT_B0_PROVENANCE.out.provenance
    )
}
