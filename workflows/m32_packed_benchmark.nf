nextflow.enable.dsl=2

include {
    M32_AUTHENTICATE_PACKED_SOURCES
    M32_MATERIALIZE_PACKED_TENSOR
    M32_BENCHMARK_PACKED_TENSOR
} from '../modules/32_PACKED_BENCHMARK'

workflow {
    def names = ['run_id','git_commit','genetic_map'] +
        ['root17','root18'].collectMany { root ->
            ['grid','rare','sites','target','tree','pools','flare_vcf'].collect { "${root}_${it}" }
        }
    def missing = names.findAll { !params["m32_pack_${it}"] }
    if (missing) error "M32 packed benchmark requires explicit parameters: ${missing.join(', ')}"
    if (!(params.m32_pack_git_commit ==~ /[0-9a-f]{40}/)) error 'M32 packed benchmark requires an exact Git commit'

    def repoDir = projectDir.resolve('..')
    def sourceAuthPy = file("${repoDir}/bin/m32_source_auth.py", checkIfExists:true)
    def benchmarkPy = file("${repoDir}/bin/m32_packed_benchmark.py", checkIfExists:true)
    def orderedLinearPy = file("${repoDir}/bin/m31_ordered_linear.py", checkIfExists:true)
    def contractPy = file("${repoDir}/bin/m32_locus_contract.py", checkIfExists:true)
    def smokePy = file("${repoDir}/bin/m32_locus_smoke.py", checkIfExists:true)
    def preregistration = file(params.m32_pack_preregistration, checkIfExists:true)
    def configNf = file("${repoDir}/conf/m32_packed_benchmark.config", checkIfExists:true)
    def moduleNf = file("${repoDir}/modules/32_PACKED_BENCHMARK.nf", checkIfExists:true)
    def workflowNf = file("${repoDir}/workflows/m32_packed_benchmark.nf", checkIfExists:true)
    M32_AUTHENTICATE_PACKED_SOURCES(
        sourceAuthPy, benchmarkPy, orderedLinearPy, contractPy, smokePy,
        preregistration, configNf, moduleNf, workflowNf,
        params.m32_pack_git_commit, repoDir.toString()
    )
    def sourceAuth = M32_AUTHENTICATE_PACKED_SOURCES.out.auth
    def roots = Channel.of(
        tuple('root17', 20260817,
            file(params.m32_pack_root17_grid, checkIfExists:true),
            file(params.m32_pack_root17_rare, checkIfExists:true),
            file(params.m32_pack_root17_sites, checkIfExists:true),
            file(params.m32_pack_root17_target, checkIfExists:true),
            file(params.m32_pack_root17_tree, checkIfExists:true),
            file(params.m32_pack_root17_pools, checkIfExists:true),
            file(params.m32_pack_root17_flare_vcf, checkIfExists:true)),
        tuple('root18', 20260818,
            file(params.m32_pack_root18_grid, checkIfExists:true),
            file(params.m32_pack_root18_rare, checkIfExists:true),
            file(params.m32_pack_root18_sites, checkIfExists:true),
            file(params.m32_pack_root18_target, checkIfExists:true),
            file(params.m32_pack_root18_tree, checkIfExists:true),
            file(params.m32_pack_root18_pools, checkIfExists:true),
            file(params.m32_pack_root18_flare_vcf, checkIfExists:true))
    )
    M32_MATERIALIZE_PACKED_TENSOR(
        roots,
        file(params.m32_pack_genetic_map, checkIfExists:true),
        preregistration, benchmarkPy, orderedLinearPy, contractPy, smokePy,
        configNf, moduleNf, workflowNf, sourceAuthPy, sourceAuth,
        params.m32_pack_git_commit,
        workflow.nextflow.version.toString(),
        params.m32_pack_container_image,
        params.m32_pack_expected_memory_bytes
    )
    M32_BENCHMARK_PACKED_TENSOR(
        M32_MATERIALIZE_PACKED_TENSOR.out.tensor,
        preregistration, benchmarkPy, orderedLinearPy, contractPy, smokePy,
        configNf, moduleNf, workflowNf, sourceAuthPy,
        params.m32_pack_git_commit,
        workflow.nextflow.version.toString(),
        params.m32_pack_container_image,
        params.m32_pack_expected_memory_bytes
    )
}
