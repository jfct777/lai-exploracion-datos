nextflow.enable.dsl=2

include { M33_SAFE_BRIDGE_TECHNICAL_KAT_ROOT } from '../modules/33_SAFE_BRIDGE_TECHNICAL_KAT'

workflow {
    def repo = projectDir.resolve('..')
    def required = ['run_id', 'git_commit', 'root17_snapshot', 'root18_snapshot',
                    'root17_flare_tbi', 'root18_flare_tbi', 'source_auth']
    def missing = required.findAll { !params["m33_safe_bridge_technical_${it}"] }
    if (missing) error "M33 SAFE_BRIDGE technical KAT requires: ${missing.join(', ')}"
    if (!(params.m33_safe_bridge_technical_git_commit ==~ /[0-9a-f]{40}/)) {
        error 'M33 SAFE_BRIDGE technical KAT requires an exact Git commit'
    }

    def contract = file("${repo}/conf/m33_safe_bridge_technical_kat_contract.json", checkIfExists:true)
    def authorization = file("${repo}/conf/m33_safe_bridge_technical_kat_authorization.json", checkIfExists:true)
    def sourceAuth = file(params.m33_safe_bridge_technical_source_auth, checkIfExists:true)
    def runner = file("${repo}/bin/m33_safe_bridge_technical_kat.py", checkIfExists:true)
    def core = file("${repo}/bin/m33_safe_bridge_core.py", checkIfExists:true)
    def a0 = file("${repo}/bin/m33_a0_real_adapter.py", checkIfExists:true)
    def linear = file("${repo}/bin/m31_ordered_linear.py", checkIfExists:true)
    def rare = file("${repo}/bin/m31_ordered_rare_preflight.py", checkIfExists:true)
    def configNf = file("${repo}/conf/m33_safe_bridge_technical_kat.config", checkIfExists:true)
    def moduleNf = file("${repo}/modules/33_SAFE_BRIDGE_TECHNICAL_KAT.nf", checkIfExists:true)
    def workflowNf = file("${repo}/workflows/m33_safe_bridge_technical_kat.nf", checkIfExists:true)
    def runnerTest = file("${repo}/tests/test_m33_safe_bridge_technical_kat.py", checkIfExists:true)
    def nextflowTest = file("${repo}/tests/test_m33_safe_bridge_technical_kat_nextflow.py", checkIfExists:true)

    def rootTuple = { String label, int seed, String snapshot, String flareTbi ->
        def base = file(snapshot, checkIfExists:true)
        tuple(label, seed,
              file("${base}/m28_sources.trees", checkIfExists:true),
              file("${base}/m28_pools.private.tsv", checkIfExists:true),
              file("${base}/m28_rare_catalog.tsv.gz", checkIfExists:true),
              file("${base}/m28_rare_haplotypes.tsv.gz", checkIfExists:true),
              file("${base}/m31_ordered_rare.sites.tsv.gz", checkIfExists:true),
              file("${base}/m31_ordered_rare.target.tsv.gz", checkIfExists:true),
              file("${base}/m28c_b0_reference.vcf.gz", checkIfExists:true),
              file("${base}/m28c_b0_reference.vcf.gz.tbi", checkIfExists:true),
              file("${base}/m28c_b0_reference_pairs.private.tsv", checkIfExists:true),
              file("${base}/m28c_b0_reference.sample_map.tsv", checkIfExists:true),
              file("${base}/genetic.map.chr22", checkIfExists:true),
              file("${base}/${label}.flare.anc.vcf.gz", checkIfExists:true),
              file(flareTbi, checkIfExists:true))
    }
    def roots = Channel.of(
        rootTuple('root17', 20260817, params.m33_safe_bridge_technical_root17_snapshot,
                  params.m33_safe_bridge_technical_root17_flare_tbi),
        rootTuple('root18', 20260818, params.m33_safe_bridge_technical_root18_snapshot,
                  params.m33_safe_bridge_technical_root18_flare_tbi),
    )
    M33_SAFE_BRIDGE_TECHNICAL_KAT_ROOT(
        roots, contract, authorization, sourceAuth, runner, core, a0, linear, rare,
        configNf, moduleNf, workflowNf, runnerTest, nextflowTest,
        params.m33_safe_bridge_technical_git_commit,
    )
}
