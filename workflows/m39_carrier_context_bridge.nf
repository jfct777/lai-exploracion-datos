nextflow.enable.dsl=2

include { M39_CARRIER_CONTEXT_BRIDGE } from '../modules/39_CARRIER_CONTEXT_BRIDGE'

workflow {
    ['m39_input_dir', 'm39_output_dir', 'm39_contract'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    // This workflow materializes FIT inputs only; it has no truth or model input.
    def contractFile = file(params.m39_contract, checkIfExists: true)
    def contract = new groovy.json.JsonSlurper().parseText(contractFile.text)
    if (contract.scope != 'technical_only_chr22_R0_FIT')
        error 'The bridge requires the explicit technical FIT scope'
    def inputs = contract.inputs.values().collect { spec ->
        if (!(spec.path ==~ /[A-Za-z0-9_.-]+/) || !(spec.sha256 ==~ /[a-f0-9]{64}/))
            error 'Invalid staged filename or digest'
        file("${params.m39_input_dir}/${spec.path}", checkIfExists: true)
    }
    if (inputs*.name.unique().size() != inputs.size()) error 'Duplicate input names'
    def outputDir = file(params.m39_output_dir)
    if (outputDir.exists() && !workflow.resume)
        error 'Output directory already exists; use a new run identifier'
    def repoDir = projectDir.resolve('..')
    def sources = [
        'm39_carrier_context.py', 'm34_prepare_panel_factors.py',
        'm34_generate_mosaics.py', 'm33_safe_bridge_core.py',
    ].collect { file("${repoDir}/bin/${it}", checkIfExists: true) }
    M39_CARRIER_CONTEXT_BRIDGE(channel.value(inputs), channel.value(contractFile), channel.value(sources))
}
