nextflow.enable.dsl=2

include { M33_M0_FACTORIZED_LAZY_TECHNICAL_KAT } from '../modules/33_M0_FACTORIZED_LAZY_TECHNICAL_KAT'

workflow {
    if (!params.m33_lazy_kat_run_id) error 'm33_lazy_kat_run_id is required'
    if (!(params.m33_lazy_kat_implementation_commit ==~ /[0-9a-f]{40}/)) error 'exact implementation commit is required'
    def repoDir = projectDir.resolve('..')
    roots = Channel.of(
        tuple('root17', 20260817, file(params.m33_lazy_kat_root17_dir, checkIfExists:true),
              file(params.m33_lazy_kat_root17_verify, checkIfExists:true),
              file(params.m33_lazy_kat_root17_map, checkIfExists:true)),
        tuple('root18', 20260818, file(params.m33_lazy_kat_root18_dir, checkIfExists:true),
              file(params.m33_lazy_kat_root18_verify, checkIfExists:true),
              file(params.m33_lazy_kat_root18_map, checkIfExists:true))
    )
    M33_M0_FACTORIZED_LAZY_TECHNICAL_KAT(
        roots,
        file("${repoDir}/bin/m33_m0_factorized_lazy_technical_kat.py", checkIfExists:true),
        file("${repoDir}/bin/m33_materialize.py", checkIfExists:true),
        file("${repoDir}/bin/m33_m0_contract.py", checkIfExists:true),
        file("${repoDir}/bin/m31_ordered_linear.py", checkIfExists:true),
        file(params.m33_lazy_kat_source_auth, checkIfExists:true),
        params.m33_lazy_kat_implementation_commit
    )
}
