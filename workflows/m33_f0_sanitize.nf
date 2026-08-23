nextflow.enable.dsl=2

include { M33_SANITIZE_FLARE_F0 } from '../modules/33_SANITIZE_FLARE_F0'

workflow {
    if (!params.m33_f0_sanitize_roots_manifest) error 'm33_f0_sanitize_roots_manifest is required'
    if (!params.m33_f0_sanitize_source_auth) error 'm33_f0_sanitize_source_auth is required'
    if (!(params.m33_f0_sanitize_git_commit ==~ /[0-9a-f]{40}/)) {
        error 'm33_f0_sanitize_git_commit must be an exact commit'
    }
    def repo = projectDir.resolve('..')
    def runner = file("${repo}/bin/m33_f0_sanitize.py", checkIfExists:true)
    def core = file("${repo}/bin/m33_safe_bridge_core.py", checkIfExists:true)
    def contract = file("${repo}/conf/m33_m0_f0_sanitized_amendment_contract.json", checkIfExists:true)
    def sourceAuth = file(params.m33_f0_sanitize_source_auth, checkIfExists:true)
    def configNf = file("${repo}/conf/m33_f0_sanitize.config", checkIfExists:true)
    def moduleNf = file("${repo}/modules/33_SANITIZE_FLARE_F0.nf", checkIfExists:true)
    def workflowNf = file("${repo}/workflows/m33_f0_sanitize.nf", checkIfExists:true)
    def runnerTest = file("${repo}/tests/test_m33_f0_sanitize.py", checkIfExists:true)
    def nextflowTest = file("${repo}/tests/test_m33_f0_sanitize_nextflow.py", checkIfExists:true)
    def roots = Channel
        .fromPath(params.m33_f0_sanitize_roots_manifest, checkIfExists:true)
        .splitCsv(header:true, sep:'\t')
        .map { row ->
            tuple(
                row.root_seed as Integer,
                file(row.flare_anc, checkIfExists:true),
                file(row.target_rare_diploid, checkIfExists:true),
            )
        }
    M33_SANITIZE_FLARE_F0(
        roots, runner, core, contract, sourceAuth, configNf, moduleNf, workflowNf,
        runnerTest, nextflowTest, params.m33_f0_sanitize_git_commit,
    )
}
