nextflow.enable.dsl=2

include { M33_AUTHENTICATE_ASSET_CONTRACT_SOURCES; M33_VALIDATE_ASSET_MANIFEST_CONTRACT } from '../modules/33_ASSET_MANIFEST_CONTRACT'

workflow {
    if (!params.m33_asset_contract_run_id) error 'm33_asset_contract_run_id is required'
    if (!(params.m33_asset_contract_git_commit ==~ /[0-9a-f]{40}/)) error 'm33_asset_contract_git_commit must be exact'
    def repoDir = projectDir.resolve('..')
    def sourceAuthPy = file("${repoDir}/bin/m33_asset_manifest_source_auth.py", checkIfExists:true)
    def contractPy = file("${repoDir}/bin/m33_asset_manifest_contract.py", checkIfExists:true)
    def contractJson = file(params.m33_asset_contract_json, checkIfExists:true)
    def configNf = file("${repoDir}/conf/m33_asset_manifest_contract.config", checkIfExists:true)
    def moduleNf = file("${repoDir}/modules/33_ASSET_MANIFEST_CONTRACT.nf", checkIfExists:true)
    def workflowNf = file("${repoDir}/workflows/m33_asset_manifest_contract.nf", checkIfExists:true)
    def contractTests = file("${repoDir}/tests/test_m33_asset_manifest_contract.py", checkIfExists:true)
    def nextflowTests = file("${repoDir}/tests/test_m33_asset_manifest_nextflow.py", checkIfExists:true)
    M33_AUTHENTICATE_ASSET_CONTRACT_SOURCES(
        sourceAuthPy, contractPy, contractJson, configNf, moduleNf, workflowNf,
        contractTests, nextflowTests, params.m33_asset_contract_git_commit, repoDir.toString()
    )
    M33_VALIDATE_ASSET_MANIFEST_CONTRACT(
        sourceAuthPy, contractPy, contractJson, configNf, moduleNf, workflowNf,
        contractTests, nextflowTests, M33_AUTHENTICATE_ASSET_CONTRACT_SOURCES.out.auth,
        params.m33_asset_contract_git_commit, repoDir.toString(), nextflow.version.toString()
    )
}
