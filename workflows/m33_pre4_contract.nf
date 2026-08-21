nextflow.enable.dsl=2

include { M33_AUTHENTICATE_CONTRACT_SOURCES; M33_VALIDATE_PRE4_CONTRACT } from '../modules/33_PRE4_CONTRACT'

workflow {
    if (!params.m33_pre4_run_id) error 'm33_pre4_run_id is required'
    if (!(params.m33_pre4_git_commit ==~ /[0-9a-f]{40}/)) error 'm33_pre4_git_commit must be exact'
    def repoDir = projectDir.resolve('..')
    def sourceAuthPy = file("${repoDir}/bin/m33_pre4_source_auth.py", checkIfExists:true)
    def contractPy = file("${repoDir}/bin/m33_pre4_contract.py", checkIfExists:true)
    def preregistration = file(params.m33_pre4_preregistration, checkIfExists:true)
    def configNf = file("${repoDir}/conf/m33_pre4_contract.config", checkIfExists:true)
    def moduleNf = file("${repoDir}/modules/33_PRE4_CONTRACT.nf", checkIfExists:true)
    def workflowNf = file("${repoDir}/workflows/m33_pre4_contract.nf", checkIfExists:true)
    def contractTests = file("${repoDir}/tests/test_m33_pre4_contract.py", checkIfExists:true)
    def nextflowTests = file("${repoDir}/tests/test_m33_pre4_nextflow.py", checkIfExists:true)
    M33_AUTHENTICATE_CONTRACT_SOURCES(sourceAuthPy, contractPy, preregistration, configNf,
        moduleNf, workflowNf, contractTests, nextflowTests, params.m33_pre4_git_commit, repoDir.toString())
    M33_VALIDATE_PRE4_CONTRACT(sourceAuthPy, contractPy, preregistration, configNf,
        moduleNf, workflowNf, contractTests, nextflowTests,
        M33_AUTHENTICATE_CONTRACT_SOURCES.out.auth, params.m33_pre4_git_commit,
        repoDir.toString(), params.m33_pre4_container_image, nextflow.version.toString())
}
