nextflow.enable.dsl=2

include { RUN_M29_SAME_LOCUS_DEV } from '../modules/29_SAME_LOCUS_DEV'

workflow {
    def required = [
        'm29_genetic_map',
        'm29_root_a_tree', 'm29_root_a_pools', 'm29_root_a_report', 'm29_root_a_manifest',
        'm29_root_a_catalog', 'm29_root_a_haplotypes', 'm29_root_a_truth', 'm29_root_a_fb',
        'm29_root_a_msp', 'm29_root_a_binding',
        'm29_root_b_tree', 'm29_root_b_pools', 'm29_root_b_report', 'm29_root_b_manifest',
        'm29_root_b_catalog', 'm29_root_b_haplotypes', 'm29_root_b_truth', 'm29_root_b_fb',
        'm29_root_b_msp', 'm29_root_b_binding'
    ]
    required.each { key ->
        if (!params[key]) error "--${key} is required; historical M28C predictions are not valid substitutes"
    }

    def repoDir = projectDir.resolve('..')
    def dotGit = new File(repoDir.toFile(), '.git')
    def gitDir = dotGit
    if (dotGit.isFile()) {
        def gitDirPath = dotGit.text.trim().replaceFirst(/^gitdir:\s*/, '')
        gitDir = new File(gitDirPath)
        if (!gitDir.isAbsolute()) gitDir = new File(dotGit.parentFile, gitDirPath)
    }
    def headValue = new File(gitDir, 'HEAD').text.trim()
    def repositoryHead = headValue
    if (headValue.startsWith('ref:')) {
        def headRef = headValue.substring(4).trim()
        def looseRef = new File(gitDir, headRef)
        repositoryHead = looseRef.exists()
            ? looseRef.text.trim()
            : new File(gitDir, 'packed-refs').readLines()
                .find { line -> line.endsWith(" ${headRef}") }
                ?.split(' ')[0]
    }
    if (!(repositoryHead ==~ /[0-9a-f]{40}/)) {
        error 'M29/M29R could not resolve an exact 40-character commit from the repository'
    }

    root_a = Channel.value(tuple(
        20260817,
        file(params.m29_root_a_tree, checkIfExists: true), file(params.m29_root_a_pools, checkIfExists: true),
        file(params.m29_root_a_report, checkIfExists: true), file(params.m29_root_a_manifest, checkIfExists: true),
        file(params.m29_root_a_catalog, checkIfExists: true), file(params.m29_root_a_haplotypes, checkIfExists: true),
        file(params.m29_root_a_truth, checkIfExists: true), file(params.m29_root_a_fb, checkIfExists: true),
        file(params.m29_root_a_msp, checkIfExists: true), file(params.m29_root_a_binding, checkIfExists: true)
    ))
    root_b = Channel.value(tuple(
        20260818,
        file(params.m29_root_b_tree, checkIfExists: true), file(params.m29_root_b_pools, checkIfExists: true),
        file(params.m29_root_b_report, checkIfExists: true), file(params.m29_root_b_manifest, checkIfExists: true),
        file(params.m29_root_b_catalog, checkIfExists: true), file(params.m29_root_b_haplotypes, checkIfExists: true),
        file(params.m29_root_b_truth, checkIfExists: true), file(params.m29_root_b_fb, checkIfExists: true),
        file(params.m29_root_b_msp, checkIfExists: true), file(params.m29_root_b_binding, checkIfExists: true)
    ))

    RUN_M29_SAME_LOCUS_DEV(
        file(params.m29_preregistration, checkIfExists: true),
        file(params.m29_genetic_map, checkIfExists: true),
        root_a, root_b,
        file("${repoDir}/bin/m29_same_locus_dev.py", checkIfExists: true),
        file("${repoDir}/bin/m28d_b0_scorer.py", checkIfExists: true),
        repositoryHead
    )
}
