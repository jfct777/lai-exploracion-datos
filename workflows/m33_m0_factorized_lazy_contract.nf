nextflow.enable.dsl=2

include { M33_VALIDATE_FACTORIZED_LAZY_CONTRACT } from '../modules/33_M0_FACTORIZED_LAZY_CONTRACT'

workflow {
    if (!params.m33_factorized_lazy_run_id) error 'm33_factorized_lazy_run_id is required'
    def repoDir = projectDir.resolve('..')
    M33_VALIDATE_FACTORIZED_LAZY_CONTRACT(
        file("${repoDir}/bin/m33_m0_factorized_lazy_amendment.py", checkIfExists:true),
        file("${repoDir}/bin/m33_materialize.py", checkIfExists:true),
        file("${repoDir}/bin/m33_m0_contract.py", checkIfExists:true),
        file(params.m33_factorized_lazy_contract, checkIfExists:true),
        file(params.m33_factorized_lazy_materializer_contract, checkIfExists:true),
        file(params.m33_factorized_lazy_sanitized_f0_contract, checkIfExists:true),
        file(params.m33_factorized_lazy_pre4_preregistration, checkIfExists:true),
        file("${repoDir}/tests/test_m33_m0_factorized_lazy_amendment.py", checkIfExists:true),
        file("${repoDir}/tests/test_m33_materialize.py", checkIfExists:true)
    )
}
