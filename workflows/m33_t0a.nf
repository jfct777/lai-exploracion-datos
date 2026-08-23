nextflow.enable.dsl=2

include { M33_T0A_FORWARD; M33_T0A_STRESS; M33_T0A_COMPARE } from '../modules/33_T0A_FORWARD'

workflow {
    if (!params.m33_t0a_run_id) error 'm33_t0a_run_id is required'
    if (!params.m33_t0a_results_dir) error 'm33_t0a_results_dir is required'
    if (!(params.m33_t0a_implementation_commit ==~ /[0-9a-f]{40}/))
        error 'exact T0a implementation commit is required'
    if (!(params.m33_t0a_oci_image ==~ /us-central1-docker\.pkg\.dev\/uspbr-242713\/dnabr-lai\/m33-t0a@sha256:[0-9a-f]{64}/))
        error 'T0a OCI image must be fixed by project digest'
    def repoDir = projectDir.resolve('..')
    def root17 = [file(params.m33_t0a_root17_dir, checkIfExists:true),
                  file(params.m33_t0a_root17_verify, checkIfExists:true),
                  file(params.m33_t0a_root17_map, checkIfExists:true)]
    def root18 = [file(params.m33_t0a_root18_dir, checkIfExists:true),
                  file(params.m33_t0a_root18_verify, checkIfExists:true),
                  file(params.m33_t0a_root18_map, checkIfExists:true)]
    cases = Channel.of(
        tuple('root17', 20260817, 'local_linear', 0, *root17),
        tuple('root17', 20260817, 'local_linear', 1, *root17),
        tuple('root17', 20260817, 'small_residual_cnn_1d', 0, *root17),
        tuple('root17', 20260817, 'small_residual_cnn_1d', 1, *root17),
        tuple('root18', 20260818, 'local_linear', 0, *root18),
        tuple('root18', 20260818, 'local_linear', 1, *root18),
        tuple('root18', 20260818, 'small_residual_cnn_1d', 0, *root18),
        tuple('root18', 20260818, 'small_residual_cnn_1d', 1, *root18)
    )
    M33_T0A_FORWARD(
        cases,
        file("${repoDir}/bin/m33_t0a_forward.py", checkIfExists:true),
        file("${repoDir}/bin/m33_t0a_models.py", checkIfExists:true),
        file("${repoDir}/bin/m33_materialize.py", checkIfExists:true),
        file("${repoDir}/bin/m33_m0_factorized_lazy_technical_kat.py", checkIfExists:true),
        file("${repoDir}/bin/m33_m0_contract.py", checkIfExists:true),
        file("${repoDir}/bin/m31_ordered_linear.py", checkIfExists:true),
        file(params.m33_t0a_source_auth, checkIfExists:true),
        file("${repoDir}/conf/m33_pre4_preregistration.json", checkIfExists:true),
        params.m33_t0a_implementation_commit,
        params.m33_t0a_oci_image,
    )
    stressCases = Channel.of(
        tuple('local_linear', 0), tuple('local_linear', 1),
        tuple('small_residual_cnn_1d', 0), tuple('small_residual_cnn_1d', 1)
    )
    M33_T0A_STRESS(
        stressCases,
        file("${repoDir}/bin/m33_t0a_stress.py", checkIfExists:true),
        file("${repoDir}/bin/m33_t0a_forward.py", checkIfExists:true),
        file("${repoDir}/bin/m33_t0a_models.py", checkIfExists:true),
        file("${repoDir}/bin/m33_materialize.py", checkIfExists:true),
        file("${repoDir}/bin/m33_m0_factorized_lazy_technical_kat.py", checkIfExists:true),
        file("${repoDir}/bin/m33_m0_contract.py", checkIfExists:true),
        file("${repoDir}/bin/m31_ordered_linear.py", checkIfExists:true),
        file(params.m33_t0a_source_auth, checkIfExists:true),
        file("${repoDir}/conf/m33_pre4_preregistration.json", checkIfExists:true),
        params.m33_t0a_implementation_commit,
        params.m33_t0a_oci_image,
    )
    M33_T0A_COMPARE(
        M33_T0A_FORWARD.out.receipt.collect(),
        M33_T0A_STRESS.out.receipt.collect(),
        file("${repoDir}/bin/m33_t0a_compare.py", checkIfExists:true),
        file(params.m33_t0a_source_auth, checkIfExists:true),
        params.m33_t0a_implementation_commit,
        params.m33_t0a_oci_image,
    )
}
