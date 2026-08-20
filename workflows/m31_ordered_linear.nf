nextflow.enable.dsl=2

include { M31_ORDERED_LINEAR_DEV } from '../modules/31_ORDERED_LINEAR'

workflow {
    def required = [
        'm31_ordered_linear_run_id',
        'm31_ordered_linear_preregistration',
        'm31_ordered_linear_genetic_map',
        'm31_ordered_linear_container_image',
        'm31_ordered_linear_container_digest',
        'm31_ordered_linear_root17_sites', 'm31_ordered_linear_root17_target',
        'm31_ordered_linear_root17_tree', 'm31_ordered_linear_root17_pools',
        'm31_ordered_linear_root17_truth', 'm31_ordered_linear_root17_flare_vcf',
        'm31_ordered_linear_root17_flare_audit',
        'm31_ordered_linear_root18_sites', 'm31_ordered_linear_root18_target',
        'm31_ordered_linear_root18_tree', 'm31_ordered_linear_root18_pools',
        'm31_ordered_linear_root18_truth', 'm31_ordered_linear_root18_flare_vcf',
        'm31_ordered_linear_root18_flare_audit'
    ]
    required.each { key ->
        if (!params[key]) error "--${key} is required for the two-root M31 ordered-linear DEV run"
    }
    if (!(params.m31_ordered_linear_run_id ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/)) {
        error '--m31_ordered_linear_run_id contains unsupported characters'
    }
    if (!params.m31_ordered_linear_container_image.contains('@sha256:')) {
        error '--m31_ordered_linear_container_image must use an immutable @sha256 digest'
    }
    if (!params.m31_ordered_linear_container_image.endsWith(params.m31_ordered_linear_container_digest)) {
        error '--m31_ordered_linear_container_digest does not match the container image'
    }
    if (!params.m31_ordered_linear_results_dir.contains(params.m31_ordered_linear_run_id)) {
        error '--m31_ordered_linear_results_dir must be namespaced by the run ID'
    }
    def resultsDir = new File(params.m31_ordered_linear_results_dir)
    if (resultsDir.exists()) {
        error "M31 ordered-linear results already exist and will not be reused or overwritten: ${resultsDir}"
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

    def runnerFile = new File(repoDir.toFile(), 'bin/m31_ordered_linear.py')
    if (!runnerFile.isFile()) {
        error 'M31 ordered-linear runner is missing: bin/m31_ordered_linear.py'
    }
    def codeSha256 = java.security.MessageDigest.getInstance('SHA-256')
        .digest(runnerFile.bytes).encodeHex().toString()
    if (!(codeSha256 ==~ /[0-9a-f]{64}/)) {
        error 'M31 could not compute the runner SHA-256'
    }
    def preregistrationFile = new File(params.m31_ordered_linear_preregistration)
    if (!preregistrationFile.isFile()) {
        error 'M31 ordered-linear preregistration is missing'
    }
    def contractSha256 = java.security.MessageDigest.getInstance('SHA-256')
        .digest(preregistrationFile.bytes).encodeHex().toString()
    def provenance = [
        experiment_id: 'M31_ORDERED_LINEAR_DEV',
        status: 'SCAFFOLD_SELFTEST_ONLY',
        run_id: params.m31_ordered_linear_run_id,
        git_commit: repositoryHead,
        code_sha256: codeSha256,
        preregistration_sha256: contractSha256,
        container_image: params.m31_ordered_linear_container_image,
        container_digest: params.m31_ordered_linear_container_digest,
        nextflow_version: workflow.nextflow.version.toString(),
        roots: [root17: 20260817, root18: 20260818],
        fitted_model_executed: false
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()

    root17 = channel.value(tuple(
        'root17', 20260817,
        file(params.m31_ordered_linear_root17_sites, checkIfExists: true),
        file(params.m31_ordered_linear_root17_target, checkIfExists: true),
        file(params.m31_ordered_linear_root17_tree, checkIfExists: true),
        file(params.m31_ordered_linear_root17_pools, checkIfExists: true),
        file(params.m31_ordered_linear_root17_truth, checkIfExists: true),
        file(params.m31_ordered_linear_root17_flare_vcf, checkIfExists: true),
        file(params.m31_ordered_linear_root17_flare_audit, checkIfExists: true)
    ))
    root18 = channel.value(tuple(
        'root18', 20260818,
        file(params.m31_ordered_linear_root18_sites, checkIfExists: true),
        file(params.m31_ordered_linear_root18_target, checkIfExists: true),
        file(params.m31_ordered_linear_root18_tree, checkIfExists: true),
        file(params.m31_ordered_linear_root18_pools, checkIfExists: true),
        file(params.m31_ordered_linear_root18_truth, checkIfExists: true),
        file(params.m31_ordered_linear_root18_flare_vcf, checkIfExists: true),
        file(params.m31_ordered_linear_root18_flare_audit, checkIfExists: true)
    ))

    M31_ORDERED_LINEAR_DEV(
        file(params.m31_ordered_linear_preregistration, checkIfExists: true),
        file(params.m31_ordered_linear_genetic_map, checkIfExists: true),
        root17,
        root18,
        file(runnerFile, checkIfExists: true),
        provenanceB64
    )
}
