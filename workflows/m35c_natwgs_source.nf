nextflow.enable.dsl=2

include {
    M35C_PREPARE_SOURCE_REFERENCES
    M35C_CLUSTER_SCREEN
    M35C_AGGREGATE_SOURCE_GATE
} from '../modules/35C_NATWGS_SOURCE'

workflow {
    def required = [
        params.m35c_run_id, params.m35c_results_dir, params.m35c_contract,
        params.m35c_roles, params.m35c_phased_scaffold_vcf,
        params.m35c_target_vcf, params.m35c_target_tbi, params.m35c_genetic_map,
        params.m35c_m27d_manifest, params.m35c_m27d_strata,
        params.m35c_m27d_training_set, params.m35c_m27d_related_pairs,
        params.m35c_m34_donor_audit, params.m35c_m34_mosaic_receipt,
        params.m35c_flare2_image, params.m35c_tabix_image, params.m35c_scoring_image,
    ]
    if (required.any { value -> !value })
        error 'M35C requires an explicit run, frozen inputs and pinned runtimes'
    if (!(params.m35c_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error 'm35c_run_id must be a safe explicit identifier'
    if (!params.m35c_results_dir.startsWith('gs://teams-usp/frank/lai-exploracion-datos/'))
        error 'M35C persistent outputs are restricted to the frank GCS prefix'
    if (![params.m35c_tabix_image, params.m35c_scoring_image].every { it.contains('@sha256:') })
        error 'M35C shared runtime images must use immutable registry digests'
    if (!(params.m35c_flare2_image.contains('@sha256:') || params.m35c_flare2_image.startsWith('sha256:')))
        error 'M35C FLARE2 runtime must use an immutable registry digest or local image ID'
    if (params.m35c_executor == 'google-batch' && !params.m35c_flare2_image.contains('@sha256:'))
        error 'M35C Google Batch requires a registry image pinned by digest'

    def contractFile = file(params.m35c_contract, checkIfExists: true)
    def contract = new groovy.json.JsonSlurper().parse(contractFile)
    if (contract.experiment_id != 'M35C_NATWGS_SOURCE_SENSITIVITY_CHR22' ||
        contract.status != 'PREREGISTERED_EXPLORATORY_SOURCE_SCREEN' ||
        contract.cluster_screen.primary_gate != 'all_9_NATWGS_coarse_combinations_must_pass')
        error 'M35C contract identity or prospective gate differs'

    def repoDir = baseDir.resolve('..')
    def selectionSeeds = contract.reference_design.selection_seeds.collect { it as Integer }
    def gmmSeeds = contract.cluster_screen.gmm_seeds.collect { it as Integer }

    M35C_PREPARE_SOURCE_REFERENCES(
        Channel.fromList(selectionSeeds),
        Channel.value(contractFile),
        Channel.value(file(params.m35c_roles, checkIfExists: true)),
        Channel.value(file(params.m35c_phased_scaffold_vcf, checkIfExists: true)),
        Channel.value(file(params.m35c_target_vcf, checkIfExists: true)),
        Channel.value(file(params.m35c_target_tbi, checkIfExists: true)),
        Channel.value(file(params.m35c_m27d_manifest, checkIfExists: true)),
        Channel.value(file(params.m35c_m27d_strata, checkIfExists: true)),
        Channel.value(file(params.m35c_m27d_training_set, checkIfExists: true)),
        Channel.value(file(params.m35c_m27d_related_pairs, checkIfExists: true)),
        Channel.value(file(params.m35c_m34_donor_audit, checkIfExists: true)),
        Channel.value(file(params.m35c_m34_mosaic_receipt, checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35c_prepare_source_comparison.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_prepare_balanced_reference.py", checkIfExists: true)),
    )

    def prepared = M35C_PREPARE_SOURCE_REFERENCES.out.source_references
    def screenCases = prepared.flatMap {
        selectionSeed,
        externalVcf, externalTbi, externalCoarseSample, externalCoarseMacro,
        externalFineSample, externalFineMacro, externalSelected,
        natwgsVcf, natwgsTbi, natwgsCoarseSample, natwgsCoarseMacro,
        natwgsFineSample, natwgsFineMacro, natwgsSelected, prepareReceipt ->
        def cases = []
        gmmSeeds.each { gmmSeed ->
            cases << tuple('EXTERNAL_NAM', selectionSeed, 'coarse', gmmSeed,
                           externalVcf, externalTbi, externalCoarseSample, externalCoarseMacro,
                           prepareReceipt)
            cases << tuple('EXTERNAL_NAM', selectionSeed, 'fine', gmmSeed,
                           externalVcf, externalTbi, externalFineSample, externalFineMacro,
                           prepareReceipt)
            cases << tuple('NATWGS', selectionSeed, 'coarse', gmmSeed,
                           natwgsVcf, natwgsTbi, natwgsCoarseSample, natwgsCoarseMacro,
                           prepareReceipt)
            cases << tuple('NATWGS', selectionSeed, 'fine', gmmSeed,
                           natwgsVcf, natwgsTbi, natwgsFineSample, natwgsFineMacro,
                           prepareReceipt)
        }
        cases
    }
    M35C_CLUSTER_SCREEN(
        screenCases,
        Channel.value(contractFile),
        Channel.value(file(params.m35c_target_vcf, checkIfExists: true)),
        Channel.value(file(params.m35c_target_tbi, checkIfExists: true)),
        Channel.value(file(params.m35c_genetic_map, checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35c_cluster_screen.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35c_prepare_source_comparison.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_prepare_balanced_reference.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35_flare2_paired.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_run_flare.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_create_model_wrapper.py", checkIfExists: true)),
    )
    M35C_AGGREGATE_SOURCE_GATE(
        M35C_CLUSTER_SCREEN.out.screen.map {
            arm, selectionSeed, granularity, gmmSeed, screenDir -> screenDir
        }.collect(),
        Channel.value(contractFile),
        Channel.value(file("${repoDir}/bin/m35c_aggregate_source_gate.py", checkIfExists: true)),
    )
}
