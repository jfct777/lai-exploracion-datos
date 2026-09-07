nextflow.enable.dsl = 2

include { M39_ORDERED_TRAINING_PROFILE } from '../modules/39_ORDERED_TRAINING_PROFILE'

workflow {
    ['m39_store_dir', 'm39_parent_receipt', 'm39_folds', 'm39_profile_config',
     'm39_output_dir', 'm39_source_commit', 'm39_profile_sha256'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (!(params.m39_source_commit ==~ /[0-9a-f]{40}/)) error 'Invalid source commit'
    if (!(params.m39_profile_sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid profile seal'
    def profileFile = file(params.m39_profile_config, checkIfExists: true)
    def digest = java.security.MessageDigest.getInstance('SHA-256')
        .digest(profileFile.bytes).encodeHex().toString()
    if (digest != params.m39_profile_sha256) error 'Profile differs from launch seal'
    def profile = new groovy.json.JsonSlurper().parseText(profileFile.text)
    if (!(profile.cases instanceof List) || !profile.cases || profile.cases.size() > 32)
        error 'A bounded nonempty cases array is required'
    def ids = profile.cases.collect { spec ->
        if (!(spec.id instanceof String) || !(spec.id ==~ /[a-z0-9][a-z0-9_-]{0,79}/))
            error 'Unsafe profile case identifier'
        if (!(spec.family in ['cnn', 'attention']) || !(spec.size in ['small', 'medium']) ||
            !(spec.batch_size in [1, 2]) || !(spec.window in ['median', 'maximum']) ||
            !(spec.arm in ['common', 'real']) || spec.core_sites != 256)
            error "Case ${spec.id} is outside the authorized technical profile"
        spec.id
    }
    if (ids.unique(false).size() != ids.size()) error 'Duplicate case identifiers'
    def store = file(params.m39_store_dir, checkIfExists: true)
    if (!java.nio.file.Files.isDirectory(store)) error 'Store must be a directory'
    def parentReceipt = file(params.m39_parent_receipt, checkIfExists: true)
    def folds = file(params.m39_folds, checkIfExists: true)
    if (!java.nio.file.Files.isRegularFile(parentReceipt) || !java.nio.file.Files.isRegularFile(folds))
        error 'Receipt and folds must be regular files'
    def output = file(params.m39_output_dir)
    if (java.nio.file.Files.exists(output, java.nio.file.LinkOption.NOFOLLOW_LINKS) || workflow.resume)
        error 'Use a new immutable output directory; resume is disabled for this profile'
    def repoDir = projectDir.resolve('..')
    def sources = ['m39_profile_ordered_training.py', 'm39_ordered_models.py',
                   'm39_ordered_batches.py', 'm39_ordered_context.py', 'm39_carrier_context.py',
                   'm34_prepare_panel_factors.py', 'm34_generate_mosaics.py', 'm33_safe_bridge_core.py']
        .collect { file("${repoDir}/bin/${it}", checkIfExists: true) }
    // Only the case channel is a queue: all authenticated inputs are shared values.
    M39_ORDERED_TRAINING_PROFILE(channel.fromList(ids), channel.value(store),
        channel.value(parentReceipt), channel.value(folds), channel.value(profileFile),
        channel.value(sources), channel.value(params.m39_source_commit))
}
