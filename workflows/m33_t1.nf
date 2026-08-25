nextflow.enable.dsl=2

include { M33_T1_PREFLIGHT; M33_T1_BACKWARD; M33_T1_COMPARE } from '../modules/33_T1_BACKWARD_DRY_RUN'

workflow {
    if (!params.m33_t1_run_id) error 'm33_t1_run_id is required'
    if (!params.m33_t1_results_dir) error 'm33_t1_results_dir is required'
    if (!(params.m33_t1_implementation_commit ==~ /[0-9a-f]{40}/))
        error 'exact T1 implementation commit is required'
    if (!(params.m33_t1_oci_image ==~ /us-central1-docker\.pkg\.dev\/uspbr-242713\/dnabr-lai\/m33-t0a@sha256:[0-9a-f]{64}/))
        error 'T1 OCI image must be fixed by project digest'

    def repoDir = projectDir.resolve('..')
    def t0bChildren = files(params.m33_t1_t0b_child_receipts, checkIfExists:true)
    if (t0bChildren.size() != 5) error 'T1 requires exactly five authenticated T0b child receipts'

    M33_T1_PREFLIGHT(
        file("${repoDir}/bin/m33_t1_preflight.py", checkIfExists:true),
        file("${repoDir}/conf/m33_t1_contract.json", checkIfExists:true),
        file(params.m33_t1_source_auth, checkIfExists:true),
        params.m33_t1_implementation_commit,
        params.m33_t1_oci_image,
        file(params.m33_t1_t0b_aggregate, checkIfExists:true),
        t0bChildren,
    )
    preflightReceipt = M33_T1_PREFLIGHT.out.receipt

    cases = Channel.of(
        tuple('local_linear', 0),
        tuple('local_linear', 1),
        tuple('small_residual_cnn_1d', 0),
        tuple('small_residual_cnn_1d', 1),
    )
    M33_T1_BACKWARD(
        cases, preflightReceipt,
        file("${repoDir}/bin/m33_t1_backward.py", checkIfExists:true),
        file("${repoDir}/bin/m33_t1_preflight.py", checkIfExists:true),
        file("${repoDir}/bin/m33_t0a_models.py", checkIfExists:true),
        file(params.m33_t1_source_auth, checkIfExists:true),
        file("${repoDir}/conf/m33_pre4_preregistration.json", checkIfExists:true),
        file("${repoDir}/conf/m33_t1_contract.json", checkIfExists:true),
        params.m33_t1_implementation_commit,
        params.m33_t1_oci_image,
    )
    M33_T1_COMPARE(
        M33_T1_BACKWARD.out.receipt.collect(), preflightReceipt,
        file("${repoDir}/bin/m33_t1_compare.py", checkIfExists:true),
        file("${repoDir}/bin/m33_t1_preflight.py", checkIfExists:true),
        file(params.m33_t1_source_auth, checkIfExists:true),
        file("${repoDir}/conf/m33_t1_contract.json", checkIfExists:true),
        params.m33_t1_implementation_commit,
        params.m33_t1_oci_image,
    )
}
