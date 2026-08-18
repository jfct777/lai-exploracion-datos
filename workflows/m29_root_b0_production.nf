nextflow.enable.dsl=2

include { SELECT_M29_ROOT_B0; MATERIALIZE_M29_ROOT_B0; INGEST_M29_ROOT_B0 } from '../modules/29_ROOT_B0_PRODUCTION'

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
        manifestPy
    )
    MATERIALIZE_M29_ROOT_B0(
        SELECT_M29_ROOT_B0.out.selected,
        productionContract,
        file("${repoDir}/bin/materialize_m29_root_b0.py", checkIfExists: true),
        file("${repoDir}/bin/materialize_m28c_b0_inputs.py", checkIfExists: true),
        manifestPy
    )
    INGEST_M29_ROOT_B0(
        MATERIALIZE_M29_ROOT_B0.out.materialized,
        productionContract,
        file("${repoDir}/bin/ingest_m29_root_b0.py", checkIfExists: true),
        file("${repoDir}/bin/audit_m28c_gnomix_ingest.py", checkIfExists: true),
        manifestPy
    )
}
