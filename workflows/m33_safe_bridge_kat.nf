nextflow.enable.dsl=2

include { SAFE_BRIDGE_KAT } from '../modules/33_SAFE_BRIDGE_KAT'

workflow {
    fixture_ch = Channel.fromPath(params.fixture, checkIfExists: true)
    contract_ch = Channel.fromPath(params.contract, checkIfExists: true)
    base_contract_ch = Channel.fromPath(params.base_contract, checkIfExists: true)
    runner_ch = Channel.fromPath(params.runner, checkIfExists: true)
    core_ch = Channel.fromPath(params.core, checkIfExists: true)
    SAFE_BRIDGE_KAT(fixture_ch, contract_ch, base_contract_ch, runner_ch, core_ch)
}
