nextflow.enable.dsl=2

include { M36B_CORA_SET_TRAIN } from '../modules/36B_CORA_SET'

workflow {
    if (!params.m36b_run_id || !(params.m36b_run_id ==~ /m36b-cora-[a-z0-9][a-z0-9-]{2,60}/)) {
        error 'M36B requires a unique m36b_run_id'
    }
    if (params.m36b_feature_chrom != 'chr22') {
        error 'The first M36B screen is preregistered for chr22 only'
    }
    def repoDir = projectDir.resolve('..')
    def required = [
        params.m36b_loci, params.m36b_carriers, params.m36b_missing,
        params.m36b_covariates, params.m36b_components, params.m36b_targets,
        params.m36b_materialization_receipt, params.m36b_run_config,
    ]
    if (required.any { value -> !value }) error 'M36B requires all six materialized tables and their receipt'

    def runConfigValue = params.m36b_run_config as String
    def runConfig = runConfigValue.startsWith('/') || runConfigValue.startsWith('gs://') ?
        file(runConfigValue, checkIfExists: true) : file("${repoDir}/${runConfigValue}", checkIfExists: true)

    M36B_CORA_SET_TRAIN(
        file(params.m36b_loci, checkIfExists: true),
        file(params.m36b_carriers, checkIfExists: true),
        file(params.m36b_missing, checkIfExists: true),
        file(params.m36b_covariates, checkIfExists: true),
        file(params.m36b_components, checkIfExists: true),
        file(params.m36b_targets, checkIfExists: true),
        file(params.m36b_materialization_receipt, checkIfExists: true),
        file("${repoDir}/bin/m36b_cora_train.py", checkIfExists: true),
        file("${repoDir}/bin/m36b_cora_models.py", checkIfExists: true),
        file("${repoDir}/bin/m36_cora_train.py", checkIfExists: true),
        file("${repoDir}/bin/m36_cora_models.py", checkIfExists: true),
        file("${repoDir}/bin/m36_cora_set.py", checkIfExists: true),
        file("${repoDir}/bin/m36b_cora_receipt.py", checkIfExists: true),
        runConfig,
        file("${repoDir}/conf/m36b_cora_preregistration.json", checkIfExists: true),
        file("${repoDir}/modules/36B_CORA_SET.nf", checkIfExists: true),
        file("${repoDir}/workflows/m36b_cora_set.nf", checkIfExists: true)
    )
}
