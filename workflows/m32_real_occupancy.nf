nextflow.enable.dsl=2

include { M32_MATERIALIZE_COORDINATES; M32_REAL_OCCUPANCY_SCREEN } from '../modules/32_REAL_OCCUPANCY'

workflow {
    def required = [params.m32_occ_run_id, params.m32_occ_git_commit, params.m32_occ_genetic_map,
        params.m32_occ_root17_sites, params.m32_occ_root17_flare_vcf,
        params.m32_occ_root18_sites, params.m32_occ_root18_flare_vcf]
    if (required.any { !it }) error 'M32 real occupancy requires explicit run, commit, map, sites and FLARE grid inputs'
    if (!(params.m32_occ_git_commit ==~ /[0-9a-f]{40}/)) error 'M32 occupancy requires an exact Git commit'
    def repoDir = projectDir.resolve('..')
    def preregistration = file(params.m32_occ_preregistration, checkIfExists: true)
    def preparePy = file("${repoDir}/bin/m32_prepare_coordinates.py", checkIfExists: true)
    def realPy = file("${repoDir}/bin/m32_real_occupancy.py", checkIfExists: true)
    def contractPy = file("${repoDir}/bin/m32_locus_contract.py", checkIfExists: true)
    def occupancyPy = file("${repoDir}/bin/m32_locus_occupancy.py", checkIfExists: true)
    def smokePy = file("${repoDir}/bin/m32_locus_smoke.py", checkIfExists: true)
    def tensorPy = file("${repoDir}/bin/m32_locus_tensor.py", checkIfExists: true)
    roots = Channel.of(
        tuple('root17', 20260817, file(params.m32_occ_genetic_map, checkIfExists: true), file(params.m32_occ_root17_sites, checkIfExists: true), file(params.m32_occ_root17_flare_vcf, checkIfExists: true)),
        tuple('root18', 20260818, file(params.m32_occ_genetic_map, checkIfExists: true), file(params.m32_occ_root18_sites, checkIfExists: true), file(params.m32_occ_root18_flare_vcf, checkIfExists: true))
    )
    M32_MATERIALIZE_COORDINATES(roots, preregistration, preparePy, contractPy, smokePy, tensorPy, occupancyPy)
    M32_REAL_OCCUPANCY_SCREEN(
        M32_MATERIALIZE_COORDINATES.out.coordinates,
        preregistration, realPy, preparePy, contractPy, occupancyPy, smokePy, tensorPy,
        file("${repoDir}/conf/m32_real_occupancy.config", checkIfExists: true),
        file("${repoDir}/modules/32_REAL_OCCUPANCY.nf", checkIfExists: true),
        file("${repoDir}/workflows/m32_real_occupancy.nf", checkIfExists: true),
        params.m32_occ_git_commit, repoDir.toString(), workflow.nextflow.version.toString()
    )
}
