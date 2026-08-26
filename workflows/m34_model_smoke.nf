nextflow.enable.dsl=2

include { M34_MODEL_SMOKE } from '../modules/34_MODEL_SMOKE'

workflow {
    if (!params.m34_smoke_run_id)
        error 'm34_smoke_run_id is required'
    if (!params.m34_smoke_results_dir)
        error 'm34_smoke_results_dir is required'
    if (!(params.m34_smoke_oci_image ==~ /.+@sha256:[0-9a-f]{64}/))
        error 'm34_smoke_oci_image must be fixed by digest'
    if (params.m34_smoke_ancestries < 2)
        error 'm34_smoke_ancestries must be at least two'
    if (params.m34_smoke_channels <= 0)
        error 'm34_smoke_channels must be positive'

    def repoDir = projectDir.resolve('..')
    def contractFile = file(params.m34_smoke_contract, checkIfExists: true)
    def contract = new groovy.json.JsonSlurper().parse(contractFile)
    if (contract.scope.ancestries.size() != params.m34_smoke_ancestries)
        error 'contract and configured ancestry counts differ'

    def cases = []
    contract.families.each { family, familySpec ->
        familySpec.configs.each { config ->
            cases << tuple(family as String, config.id as String)
        }
    }
    if (cases.size() != 35)
        error 'M34 technical smoke requires exactly 35 declared configurations'

    M34_MODEL_SMOKE(
        Channel.fromList(cases),
        contractFile,
        file("${repoDir}/bin/m34_model_smoke.py", checkIfExists: true),
        file("${repoDir}/bin/m34_models.py", checkIfExists: true),
        file("${repoDir}/bin/m34_adaptive_sweep.py", checkIfExists: true),
    )
}
