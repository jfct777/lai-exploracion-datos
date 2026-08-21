nextflow.enable.dsl=2

include { M32_LOCUS_SEQUENCE_SMOKE } from '../modules/32_LOCUS_SEQUENCE_SMOKE'

workflow {
    if (!params.m32_smoke_run_id) {
        error 'M32 synthetic smoke requires an explicit --m32_smoke_run_id'
    }
    if (!(params.m32_smoke_git_commit ==~ /[0-9a-f]{40}/)) {
        error 'M32 synthetic smoke requires an explicit exact --m32_smoke_git_commit'
    }
    def repoDir = projectDir.resolve('..')

    M32_LOCUS_SEQUENCE_SMOKE(
        params.m32_smoke_run_id,
        params.m32_smoke_seed as Integer,
        file(params.m32_smoke_preregistration, checkIfExists: true),
        file("${repoDir}/bin/m32_locus_contract.py", checkIfExists: true),
        file("${repoDir}/bin/m32_locus_tensor.py", checkIfExists: true),
        file("${repoDir}/bin/m32_locus_occupancy.py", checkIfExists: true),
        file("${repoDir}/bin/m32_locus_smoke.py", checkIfExists: true),
        file("${repoDir}/conf/m32_locus_sequence_smoke.config", checkIfExists: true),
        file("${repoDir}/modules/32_LOCUS_SEQUENCE_SMOKE.nf", checkIfExists: true),
        file("${repoDir}/workflows/m32_locus_sequence_smoke.nf", checkIfExists: true),
        params.m32_smoke_git_commit,
        repoDir.toString()
    )
}
