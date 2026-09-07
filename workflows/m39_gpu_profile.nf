nextflow.enable.dsl = 2

include { M39_GPU_SERIAL_PROFILE } from '../modules/39_GPU_SERIAL_PROFILE'

workflow {
    if (!(params.m39_gpu_run_id ==~ /m39-gpu-[a-z0-9-]{3,45}/)) error 'Use a unique M39_GPU_RUN_ID'
    if (!(params.m39_gpu_image ==~ /us-central1-docker\.pkg\.dev\/uspbr-242713\/dnabr-lai\/[a-z0-9-]+@sha256:[0-9a-f]{64}/))
        error 'M39_GPU_IMAGE must be a project-owned digest'
    ['m39_store_dir', 'm39_parent_receipt', 'm39_folds', 'm39_profile_config',
     'm39_source_seal', 'm39_output_dir', 'm39_source_commit', 'm39_profile_sha256'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (!(params.m39_source_commit ==~ /[0-9a-f]{40}/)) error 'Full source commit required'
    if (!(params.m39_profile_sha256 ==~ /[0-9a-f]{64}/)) error 'Profile digest required'
    if (workflow.resume) error 'Use a new immutable GPU measurement; resume is disabled'
    def profile = file(params.m39_profile_config, checkIfExists: true)
    def digest = java.security.MessageDigest.getInstance('SHA-256').digest(profile.bytes).encodeHex().toString()
    if (digest != params.m39_profile_sha256) error 'Profile differs from launch seal'
    def seal = file(params.m39_source_seal, checkIfExists: true)
    def binding = new groovy.json.JsonSlurper().parseText(seal.text)
    if (binding.source_commit != params.m39_source_commit || binding.profile_sha256 != digest)
        error 'Source seal binding differs'
    def repoDir = projectDir.resolve('..')
    def sources = binding.source_sha256.keySet().sort().collect { name ->
        if (!(name ==~ /m[0-9]+_[a-z0-9_]+\.py/)) error 'Unsafe source member'
        file("${repoDir}/bin/${name}", checkIfExists: true)
    }
    M39_GPU_SERIAL_PROFILE(
        channel.value(file(params.m39_store_dir, checkIfExists: true)),
        channel.value(file(params.m39_parent_receipt, checkIfExists: true)),
        channel.value(file(params.m39_folds, checkIfExists: true)), channel.value(profile),
        channel.value(seal), channel.value(sources), channel.value(params.m39_source_commit))
}
