nextflow.enable.dsl=2

include { M31_ORDERED_RARE_PREFLIGHT } from '../modules/31_ORDERED_RARE_PREFLIGHT'

workflow {
    def required = [
        params.m31_preflight_preregistration,
        params.m31_preflight_root17_tree, params.m31_preflight_root17_pools,
        params.m31_preflight_root17_catalog, params.m31_preflight_root17_haplotypes,
        params.m31_preflight_root18_tree, params.m31_preflight_root18_pools,
        params.m31_preflight_root18_catalog, params.m31_preflight_root18_haplotypes
    ]
    if (required.any { value -> !value }) {
        error 'M31 ordered-rare smoke requires authenticated tree/pools/catalog/haplotypes for root17 and root18'
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
        error 'M31 could not resolve an exact 40-character commit from the repository'
    }
    roots = Channel.of(
        tuple(
            'root17', 20260817,
            file(params.m31_preflight_root17_tree, checkIfExists: true),
            file(params.m31_preflight_root17_pools, checkIfExists: true),
            file(params.m31_preflight_root17_catalog, checkIfExists: true),
            file(params.m31_preflight_root17_haplotypes, checkIfExists: true)
        ),
        tuple(
            'root18', 20260818,
            file(params.m31_preflight_root18_tree, checkIfExists: true),
            file(params.m31_preflight_root18_pools, checkIfExists: true),
            file(params.m31_preflight_root18_catalog, checkIfExists: true),
            file(params.m31_preflight_root18_haplotypes, checkIfExists: true)
        )
    )

    M31_ORDERED_RARE_PREFLIGHT(
        roots,
        file(params.m31_preflight_preregistration, checkIfExists: true),
        file("${repoDir}/bin/m31_ordered_rare_preflight.py", checkIfExists: true),
        repositoryHead
    )
}
