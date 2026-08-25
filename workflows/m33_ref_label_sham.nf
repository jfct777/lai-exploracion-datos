nextflow.enable.dsl=2

include { M33_REF_LABEL_SHAM_KAT } from '../modules/33_REF_LABEL_SHAM_KAT'

workflow {
    if (!params.m33_ref_label_sham_run_id)
        error 'm33_ref_label_sham_run_id is required'
    if (!params.m33_ref_label_sham_results_dir)
        error 'm33_ref_label_sham_results_dir is required'
    if (!params.m33_ref_label_sham_source_auth)
        error 'm33_ref_label_sham_source_auth is required'
    if (!(params.m33_ref_label_sham_implementation_commit ==~ /[0-9a-f]{40}/))
        error 'exact REF-label sham implementation commit is required'
    if (!(params.m33_ref_label_sham_oci_image ==~ /us-central1-docker\.pkg\.dev\/uspbr-242713\/dnabr-lai\/m33-t0a@sha256:[0-9a-f]{64}/))
        error 'REF-label sham OCI image must be fixed by project digest'

    def repoDir = projectDir.resolve('..')
    M33_REF_LABEL_SHAM_KAT(
        file("${repoDir}/bin/m33_ref_label_sham_kat.py", checkIfExists:true),
        file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists:true),
        file("${repoDir}/bin/m33_ref_label_sham_source_auth.py", checkIfExists:true),
        file("${repoDir}/conf/m33_ref_label_sham_contract.json", checkIfExists:true),
        file("${repoDir}/conf/m33_pre4_preregistration.json", checkIfExists:true),
        file("${repoDir}/conf/m33_m0_materializer_contract.json", checkIfExists:true),
        file("${repoDir}/conf/m33_ref_label_sham.config", checkIfExists:true),
        file("${repoDir}/modules/33_REF_LABEL_SHAM_KAT.nf", checkIfExists:true),
        file("${repoDir}/workflows/m33_ref_label_sham.nf", checkIfExists:true),
        file("${repoDir}/tests/test_m33_ref_label_sham.py", checkIfExists:true),
        file(params.m33_ref_label_sham_source_auth, checkIfExists:true),
        params.m33_ref_label_sham_implementation_commit,
        params.m33_ref_label_sham_oci_image,
    )
}
