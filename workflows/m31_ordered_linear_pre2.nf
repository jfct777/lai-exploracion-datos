nextflow.enable.dsl=2

include {
    M31_PRE2_VERIFY_AUTHORIZATION
    M31_PRE2_VERIFY_TECHNICAL
    M31_PRE2_FIT_PREDICT
    M31_PRE2_VERIFY_WORKERS
    M31_PRE2_ROOT17_GATE
    M31_PRE2_SCORE_ROOT18
} from '../modules/31_ORDERED_LINEAR_PRE2'

workflow {
    def required = [
        'm31_pre2_run_id', 'm31_pre2_preregistration', 'm31_pre2_genetic_map',
        'm31_pre2_container_image', 'm31_pre2_container_digest',
        'm31_pre2_execution_authorization', 'm31_pre2_cost_cap_usd',
        'm31_pre2_pre1_c_checkpoint', 'm31_pre2_pre1_c_prediction',
        'm31_pre2_root18_truth_source',
        'm31_pre2_root17_sites', 'm31_pre2_root17_target', 'm31_pre2_root17_tree',
        'm31_pre2_root17_pools', 'm31_pre2_root17_truth',
        'm31_pre2_root17_flare_vcf', 'm31_pre2_root17_flare_audit',
        'm31_pre2_root18_sites', 'm31_pre2_root18_target', 'm31_pre2_root18_tree',
        'm31_pre2_root18_pools', 'm31_pre2_root18_flare_vcf',
        'm31_pre2_root18_flare_audit'
    ]
    required.each { key ->
        if (!params[key]) error "--${key} is required for M31 PRE2"
    }
    if (!(params.m31_pre2_run_id ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/)) {
        error '--m31_pre2_run_id contains unsupported characters'
    }
    if (!(params.m31_pre2_container_image == params.m31_pre2_container_digest
          || params.m31_pre2_container_image.endsWith("@${params.m31_pre2_container_digest}"))) {
        error '--m31_pre2_container_digest does not match the image'
    }
    if (!params.m31_pre2_results_dir.contains(params.m31_pre2_run_id)) {
        error '--m31_pre2_results_dir must be namespaced by run ID'
    }
    if (!params.m31_pre2_work_dir.contains(params.m31_pre2_run_id)) {
        error '--m31_pre2_work_dir must be namespaced by run ID'
    }
    def repoDir = projectDir.resolve('..')
    def usableLocalDiskGiB = repoDir.toFile().usableSpace / (1024.0 * 1024.0 * 1024.0)
    if (usableLocalDiskGiB < params.m31_pre2_min_local_disk_gib) {
        error "M31 PRE2 requires at least ${params.m31_pre2_min_local_disk_gib} GiB local free space; observed ${String.format('%.2f', usableLocalDiskGiB)} GiB"
    }
    def resultsDir = file(params.m31_pre2_results_dir)
    def durableStageDirs = ['authorization', 'technical', 'workers', 'gate', 'score']
        .collect { name -> file("${resultsDir}/${name}") }
    if (durableStageDirs.any { item -> item.exists() }) {
        error "M31 PRE2 durable stage output already exists and will not be reused: ${resultsDir}"
    }

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
        repositoryHead = looseRef.exists() ? looseRef.text.trim() : new File(gitDir, 'packed-refs')
            .readLines().find { line -> line.endsWith(" ${headRef}") }?.split(' ')[0]
    }
    if (!(repositoryHead ==~ /[0-9a-f]{40}/)) {
        error 'M31 PRE2 could not resolve the exact repository commit'
    }

    def contractFile = new File(params.m31_pre2_preregistration)
    def pipelineFile = new File(repoDir.toFile(), 'bin/m31_pre2_pipeline.py')
    def runnerFile = new File(repoDir.toFile(), 'bin/run_m31_ordered_linear.py')
    def coreFile = new File(repoDir.toFile(), 'bin/m31_ordered_linear.py')
    def receiptFile = new File(repoDir.toFile(), 'bin/m31_pre2_receipt.py')
    def validatorFile = new File(repoDir.toFile(), 'bin/m31_pre2_contract.py')
    def moduleFile = new File(repoDir.toFile(), 'modules/31_ORDERED_LINEAR_PRE2.nf')
    def workflowFile = new File(repoDir.toFile(), 'workflows/m31_ordered_linear_pre2.nf')
    def configFile = new File(repoDir.toFile(), 'conf/m31_ordered_linear_pre2.config')
    [contractFile, pipelineFile, runnerFile, coreFile, receiptFile, validatorFile,
     moduleFile, workflowFile, configFile].each { item ->
        if (!item.isFile()) error "M31 PRE2 source is absent: ${item}"
    }
    def fileSha256 = { File item ->
        java.security.MessageDigest.getInstance('SHA-256').digest(item.bytes).encodeHex().toString()
    }
    def contractSha256 = fileSha256.call(contractFile)
    if (contractSha256 != 'fd2d7b6d287913636be6e83ad542a40ffbc26c961d769e9c181955d89efa76bf') {
        error 'M31 PRE2 immutable contract SHA-256 differs'
    }
    def executionSourceSha256 = [
        pipelineFile, validatorFile, receiptFile, moduleFile, workflowFile, configFile
    ]
        .collectEntries { item -> [(item.name): fileSha256.call(item)] }
    def executionSourceSha256Json = groovy.json.JsonOutput.toJson(executionSourceSha256)

    root17 = channel.value(tuple(
        'root17', 20260817,
        file(params.m31_pre2_root17_sites, checkIfExists: true),
        file(params.m31_pre2_root17_target, checkIfExists: true),
        file(params.m31_pre2_root17_tree, checkIfExists: true),
        file(params.m31_pre2_root17_pools, checkIfExists: true),
        file(params.m31_pre2_root17_truth, checkIfExists: true),
        file(params.m31_pre2_root17_flare_vcf, checkIfExists: true),
        file(params.m31_pre2_root17_flare_audit, checkIfExists: true)
    ))
    root18Features = channel.value(tuple(
        'root18', 20260818,
        file(params.m31_pre2_root18_sites, checkIfExists: true),
        file(params.m31_pre2_root18_target, checkIfExists: true),
        file(params.m31_pre2_root18_tree, checkIfExists: true),
        file(params.m31_pre2_root18_pools, checkIfExists: true),
        file(params.m31_pre2_root18_flare_vcf, checkIfExists: true),
        file(params.m31_pre2_root18_flare_audit, checkIfExists: true)
    ))

    M31_PRE2_VERIFY_AUTHORIZATION(
        file(params.m31_pre2_execution_authorization, checkIfExists: true),
        file(contractFile, checkIfExists: true), file(pipelineFile, checkIfExists: true),
        file(validatorFile, checkIfExists: true), file(runnerFile, checkIfExists: true),
        file(coreFile, checkIfExists: true), file(receiptFile, checkIfExists: true),
        params.m31_pre2_run_id, repositoryHead, params.m31_pre2_container_digest,
        executionSourceSha256Json,
        params.m31_pre2_cost_cap_usd
    )

    M31_PRE2_VERIFY_TECHNICAL(
        file(contractFile, checkIfExists: true), file(validatorFile, checkIfExists: true),
        file(pipelineFile, checkIfExists: true), file(runnerFile, checkIfExists: true),
        file(coreFile, checkIfExists: true), file(receiptFile, checkIfExists: true),
        file(params.m31_pre2_pre1_c_checkpoint, checkIfExists: true),
        file(params.m31_pre2_pre1_c_prediction, checkIfExists: true),
        params.m31_pre2_pre1_c_checkpoint_sha256,
        params.m31_pre2_pre1_c_prediction_sha256,
        groovy.json.JsonOutput.toJson(params.m31_pre2_pre1_c_metrics)
    )

    workers = channel.of(1, 4, 8)
    M31_PRE2_FIT_PREDICT(
        workers, file(contractFile, checkIfExists: true),
        file(params.m31_pre2_genetic_map, checkIfExists: true), root17, root18Features,
        file(pipelineFile, checkIfExists: true), file(validatorFile, checkIfExists: true),
        file(runnerFile, checkIfExists: true), file(coreFile, checkIfExists: true),
        file(receiptFile, checkIfExists: true),
        file(moduleFile, checkIfExists: true),
        file(workflowFile, checkIfExists: true), file(configFile, checkIfExists: true),
        M31_PRE2_VERIFY_AUTHORIZATION.out.report,
        params.m31_pre2_run_id,
        contractSha256, fileSha256.call(runnerFile), fileSha256.call(coreFile), repositoryHead,
        executionSourceSha256Json,
        params.m31_pre2_container_digest
    )
    workerDirs = M31_PRE2_FIT_PREDICT.out.worker_bundle.map { workersValue, directory -> directory }.collect()
    worker4 = M31_PRE2_FIT_PREDICT.out.worker_bundle
        .filter { workersValue, directory -> workersValue == 4 }
        .map { workersValue, directory -> directory }

    M31_PRE2_VERIFY_WORKERS(
        workerDirs, file(pipelineFile, checkIfExists: true),
        file(validatorFile, checkIfExists: true), file(runnerFile, checkIfExists: true),
        file(coreFile, checkIfExists: true), file(receiptFile, checkIfExists: true)
    )
    M31_PRE2_ROOT17_GATE(
        file(contractFile, checkIfExists: true), worker4,
        M31_PRE2_VERIFY_WORKERS.out.screen, M31_PRE2_VERIFY_TECHNICAL.out.evidence,
        file(pipelineFile, checkIfExists: true), file(validatorFile, checkIfExists: true),
        file(runnerFile, checkIfExists: true), file(coreFile, checkIfExists: true),
        file(receiptFile, checkIfExists: true),
        file(moduleFile, checkIfExists: true), file(workflowFile, checkIfExists: true),
        file(configFile, checkIfExists: true), M31_PRE2_VERIFY_AUTHORIZATION.out.report,
        params.m31_pre2_run_id
    )
    M31_PRE2_SCORE_ROOT18(
        M31_PRE2_ROOT17_GATE.out.open_token, M31_PRE2_ROOT17_GATE.out.receipt,
        M31_PRE2_ROOT17_GATE.out.metrics, M31_PRE2_VERIFY_TECHNICAL.out.evidence,
        M31_PRE2_VERIFY_WORKERS.out.screen, worker4,
        file(contractFile, checkIfExists: true),
        file(params.m31_pre2_genetic_map, checkIfExists: true), root18Features,
        file(pipelineFile, checkIfExists: true), file(validatorFile, checkIfExists: true),
        file(runnerFile, checkIfExists: true), file(coreFile, checkIfExists: true),
        file(receiptFile, checkIfExists: true),
        file(moduleFile, checkIfExists: true), file(workflowFile, checkIfExists: true),
        file(configFile, checkIfExists: true), M31_PRE2_VERIFY_AUTHORIZATION.out.report,
        params.m31_pre2_run_id, params.m31_pre2_root18_truth_source,
        params.m31_pre2_opening_ledger_uri
    )
}
