nextflow.enable.dsl=2

include { M33_T0B_PREFLIGHT; M33_T0B_FORWARD; M33_T0B_COMPARE } from '../modules/33_T0B_FULL_CHR22'

workflow {
    if (!params.m33_t0b_run_id) error 'm33_t0b_run_id is required'
    if (!params.m33_t0b_results_dir) error 'm33_t0b_results_dir is required'
    if (!(params.m33_t0b_implementation_commit ==~ /[0-9a-f]{40}/))
        error 'exact T0b implementation commit is required'
    if (!(params.m33_t0b_oci_image ==~ /us-central1-docker\.pkg\.dev\/uspbr-242713\/dnabr-lai\/m33-t0a@sha256:[0-9a-f]{64}/))
        error 'T0b OCI image must be fixed by project digest'
    if (params.m33_t0b_marker_count != 79791)
        error 'T0b full chromosome 22 requires exactly 79791 markers'

    def repoDir = projectDir.resolve('..')
    def root17 = [file(params.m33_t0b_root17_dir, checkIfExists:true),
                  file(params.m33_t0b_root17_verify, checkIfExists:true),
                  file(params.m33_t0b_root17_map, checkIfExists:true)]
    def root18 = [file(params.m33_t0b_root18_dir, checkIfExists:true),
                  file(params.m33_t0b_root18_verify, checkIfExists:true),
                  file(params.m33_t0b_root18_map, checkIfExists:true)]
    def t0aChildren = file(params.m33_t0b_t0a_child_receipts, checkIfExists:true)
    if (t0aChildren.size() != 12) error 'T0b requires exactly 12 local T0a child receipts'

    M33_T0B_PREFLIGHT(
        file("${repoDir}/bin/m33_t0b_preflight.py", checkIfExists:true),
        file("${repoDir}/conf/m33_t0b_contract.json", checkIfExists:true),
        file(params.m33_t0b_source_auth, checkIfExists:true),
        params.m33_t0b_implementation_commit,
        file(params.m33_t0b_t0a_aggregate, checkIfExists:true),
        file(params.m33_t0b_t0a_source_auth, checkIfExists:true),
        t0aChildren,
        root17[0], root18[0], root17[1], root18[1], root17[2], root18[2],
    )
    preflightReceipt = M33_T0B_PREFLIGHT.out.receipt.first()

    cases = Channel.of(
        tuple('root17', 20260817, 'small_residual_cnn_1d', 0, root17[0], root17[1], root17[2]),
        tuple('root17', 20260817, 'small_residual_cnn_1d', 1, root17[0], root17[1], root17[2]),
        tuple('root18', 20260818, 'small_residual_cnn_1d', 0, root18[0], root18[1], root18[2]),
        tuple('root17', 20260817, 'local_linear', 0, root17[0], root17[1], root17[2]),
        tuple('root18', 20260818, 'local_linear', 0, root18[0], root18[1], root18[2])
    )
    M33_T0B_FORWARD(
        cases, preflightReceipt,
        file("${repoDir}/bin/m33_t0b_forward.py", checkIfExists:true),
        file("${repoDir}/bin/m33_t0a_forward.py", checkIfExists:true),
        file("${repoDir}/bin/m33_t0a_models.py", checkIfExists:true),
        file("${repoDir}/bin/m33_materialize.py", checkIfExists:true),
        file("${repoDir}/bin/m33_m0_factorized_lazy_technical_kat.py", checkIfExists:true),
        file("${repoDir}/bin/m33_m0_contract.py", checkIfExists:true),
        file("${repoDir}/bin/m31_ordered_linear.py", checkIfExists:true),
        file(params.m33_t0b_source_auth, checkIfExists:true),
        file("${repoDir}/conf/m33_pre4_preregistration.json", checkIfExists:true),
        file("${repoDir}/conf/m33_t0b_contract.json", checkIfExists:true),
        params.m33_t0b_implementation_commit,
        params.m33_t0b_oci_image,
    )
    M33_T0B_COMPARE(
        M33_T0B_FORWARD.out.receipt.collect(), preflightReceipt,
        file("${repoDir}/bin/m33_t0b_compare.py", checkIfExists:true),
        file(params.m33_t0b_source_auth, checkIfExists:true),
        file("${repoDir}/conf/m33_t0b_contract.json", checkIfExists:true),
        params.m33_t0b_implementation_commit,
        params.m33_t0b_oci_image,
    )
}
