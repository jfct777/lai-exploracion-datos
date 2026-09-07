nextflow.enable.dsl = 2

include { M39_ORDERED_THROUGHPUT } from '../modules/39_ORDERED_THROUGHPUT'

workflow {
    ['m39_store_dir', 'm39_parent_receipt', 'm39_folds', 'm39_profile_config',
     'm39_output_dir', 'm39_source_commit', 'm39_profile_sha256', 'm39_run_token'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (!(params.m39_source_commit ==~ /[0-9a-f]{40}/)) error 'Invalid source commit'
    if (!(params.m39_profile_sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid profile seal'
    if (!(params.m39_run_token ==~ /[0-9a-f]{16}/)) error 'Invalid isolated run token'
    if (!(params.m39_container_user ==~ /[0-9]{1,10}:[0-9]{1,10}/)) error 'Invalid container uid:gid'
    def profileFile = file(params.m39_profile_config, checkIfExists: true)
    def digest = java.security.MessageDigest.getInstance('SHA-256')
        .digest(profileFile.bytes).encodeHex().toString()
    if (digest != params.m39_profile_sha256) error 'Profile differs from launch seal'
    def profile = new groovy.json.JsonSlurper().parseText(profileFile.text)
    if (!(profile.cases instanceof List) || profile.cases.size() != 6 ||
        profile.core_sites != 256 || profile.max_seconds != 900)
        error 'Expected six bounded throughput cases'
    def expected = ['cnn', 'attention'].collectMany { family ->
        ["${family}-small-grouped-b1-real".toString(), "${family}-small-grouped-b2-real".toString(),
         "${family}-small-random-b2-real".toString()]
    }
    def ids = profile.cases.collect { spec ->
        if (!(spec.id instanceof String) || !(spec.id in expected) ||
            !(spec.family in ['cnn', 'attention']) || !(spec.policy in ['grouped', 'random']) ||
            !(spec.batch_size in [1, 2]) || spec.arm != 'real' ||
            spec.id != "${spec.family}-small-${spec.policy}-b${spec.batch_size}-real".toString())
            error 'Unexpected or unsafe throughput case'
        spec.id
    }
    if (ids.toSet() != expected.toSet()) error 'Duplicate or missing throughput cases'
    def store = file(params.m39_store_dir, checkIfExists: true)
    if (!java.nio.file.Files.isDirectory(store)) error 'Store must be a directory'
    if (!(params.m39_store_dir.toString() ==~ /\/[A-Za-z0-9_.\/-]+/) ||
        store.toRealPath().toString() != params.m39_store_dir.toString())
        error 'Store bind source must be an absolute resolved shell-safe directory'
    def parentReceipt = file(params.m39_parent_receipt, checkIfExists: true)
    def folds = file(params.m39_folds, checkIfExists: true)
    if (!java.nio.file.Files.isRegularFile(parentReceipt) || !java.nio.file.Files.isRegularFile(folds))
        error 'Receipt and folds must be regular files'
    def output = file(params.m39_output_dir)
    if (java.nio.file.Files.exists(output, java.nio.file.LinkOption.NOFOLLOW_LINKS) || workflow.resume)
        error 'Use a new immutable output directory; resume is disabled'
    def repoDir = projectDir.resolve('..')
    def sources = ['m39_profile_ordered_throughput.py', 'm39_throughput_sampling.py',
                   'm39_profile_ordered_training.py', 'm39_ordered_models.py', 'm39_ordered_batches.py',
                   'm39_ordered_context.py', 'm39_carrier_context.py', 'm34_prepare_panel_factors.py',
                   'm34_generate_mosaics.py', 'm33_safe_bridge_core.py']
        .collect { file("${repoDir}/bin/${it}", checkIfExists: true) }
    M39_ORDERED_THROUGHPUT(channel.fromList(ids), channel.value(store.toString()),
        channel.value(parentReceipt), channel.value(folds), channel.value(profileFile),
        channel.value(sources), channel.value(params.m39_source_commit))
}
