nextflow.enable.dsl=2

include { M39_ORDERED_DEVELOPMENT_STORE } from '../modules/39_ORDERED_DEVELOPMENT_STORE'

workflow {
    ['m39_input_dir', 'm39_output_dir', 'm39_contract', 'm39_ordered_profile', 'm39_folds', 'm39_roles'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    def roles = params.m39_roles.toString().split(',').toList()
    if (roles.unique().size() != roles.size() || roles.any { !(it in ['TRAIN', 'SELECT']) })
        error 'Only explicit TRAIN/SELECT roles may be materialized'
    def bridgeFile = file(params.m39_contract, checkIfExists: true)
    def profileFile = file(params.m39_ordered_profile, checkIfExists: true)
    def bridge = new groovy.json.JsonSlurper().parseText(bridgeFile.text)
    def profile = new groovy.json.JsonSlurper().parseText(profileFile.text)
    def inputs = bridge.inputs.values().collect { spec ->
        if (!(spec.path ==~ /[A-Za-z0-9_.-]+/) || !(spec.sha256 ==~ /[a-f0-9]{64}/))
            error 'Invalid staged name or hash'
        file("${params.m39_input_dir}/${spec.path}", checkIfExists: true)
    }
    def folds = file(params.m39_folds, checkIfExists: true)
    if (folds.name != profile.folds.path) error 'Fold filename differs from profile'
    inputs.add(folds)
    if (inputs.size() != bridge.inputs.size() + 1 || inputs*.name.unique().size() != inputs.size())
        error 'Staged input inventory differs'
    if (file(params.m39_output_dir).exists() && !workflow.resume) error 'Choose a new output directory'
    def repoDir = projectDir.resolve('..')
    def sources = ['m39_ordered_training_store.py', 'm39_profile_ordered.py', 'm39_ordered_context.py',
                   'm39_carrier_context.py', 'm34_prepare_panel_factors.py',
                   'm34_generate_mosaics.py', 'm33_safe_bridge_core.py']
                  .collect { file("${repoDir}/bin/${it}", checkIfExists: true) }
    M39_ORDERED_DEVELOPMENT_STORE(channel.value(inputs), channel.value(bridgeFile),
                                channel.value(profileFile), channel.value(sources), channel.value(roles.join(' ')))
}
