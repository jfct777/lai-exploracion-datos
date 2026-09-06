nextflow.enable.dsl=2

include { M39_ORDERED_CONTEXT_PROFILE } from '../modules/39_ORDERED_CONTEXT_PROFILE'

workflow {
    ['m39_input_dir', 'm39_output_dir', 'm39_contract', 'm39_ordered_profile', 'm39_folds'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    def bridgeFile = file(params.m39_contract, checkIfExists: true)
    def profileFile = file(params.m39_ordered_profile, checkIfExists: true)
    def bridge = new groovy.json.JsonSlurper().parseText(bridgeFile.text)
    def profile = new groovy.json.JsonSlurper().parseText(profileFile.text)
    if (profile.scope != 'technical_inner_TRAIN_only') error 'Only technical inner TRAIN is allowed'
    def inputs = bridge.inputs.values().collect { spec ->
        if (!(spec.path ==~ /[A-Za-z0-9_.-]+/) || !(spec.sha256 ==~ /[a-f0-9]{64}/))
            error 'Invalid staged name or hash'
        file("${params.m39_input_dir}/${spec.path}", checkIfExists: true)
    }
    def folds = file(params.m39_folds, checkIfExists: true)
    if (folds.name != profile.folds.path) error 'Fold filename differs from profile'
    // Path implements Iterable: list += path would append its path components.
    inputs.add(folds)
    if (inputs.size() != bridge.inputs.size() + 1 || inputs.any { !java.nio.file.Files.isRegularFile(it) })
        error 'Staged input inventory must contain seven bridge files and one folds file'
    if (inputs*.name.unique().size() != inputs.size()) error 'Duplicate inputs'
    if (file(params.m39_output_dir).exists() && !workflow.resume) error 'Choose a new output directory'
    def repoDir = projectDir.resolve('..')
    def sources = ['m39_profile_ordered.py', 'm39_ordered_context.py', 'm39_carrier_context.py',
                   'm34_prepare_panel_factors.py', 'm34_generate_mosaics.py', 'm33_safe_bridge_core.py']
                  .collect { file("${repoDir}/bin/${it}", checkIfExists: true) }
    M39_ORDERED_CONTEXT_PROFILE(channel.value(inputs), channel.value(bridgeFile),
                               channel.value(profileFile), channel.value(sources))
}
