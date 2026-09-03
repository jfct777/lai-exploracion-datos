nextflow.enable.dsl=2

include {
    M35D_PREPARE_R1_NATWGS_REFERENCE
    M35D_R1_CLUSTER_SCREEN
    M35D_AGGREGATE_R1_GATE
} from '../modules/35D_NATWGS_FINE_R1_SCREEN'

workflow {
    def required = [
        params.m35d_run_id, params.m35d_results_dir, params.m35d_contract,
        params.m35d_roles, params.m35d_phased_scaffold_vcf,
        params.m35d_target_vcf, params.m35d_target_tbi, params.m35d_genetic_map,
        params.m35d_m27d_manifest, params.m35d_m27d_strata,
        params.m35d_m27d_training_set, params.m35d_m27d_related_pairs,
        params.m35d_r1_donor_audit, params.m35d_r1_mosaic_receipt,
        params.m35d_flare2_image, params.m35d_tabix_image, params.m35d_scoring_image,
    ]
    if (required.any { value -> !value })
        error 'M35D screen requires explicit frozen inputs and pinned runtimes'
    if (required.any { value -> value.toString().toLowerCase().contains('m34-nam-128-r2') })
        error 'M35D screen forbids every R2 input'
    if (!(params.m35d_run_id ==~ /[a-z0-9][a-z0-9._-]{2,63}/))
        error 'm35d_run_id must be a safe explicit identifier'
    if (!params.m35d_results_dir.startsWith('gs://teams-usp/frank/lai-exploracion-datos/'))
        error 'M35D outputs are restricted to the frank GCS prefix'
    if (![params.m35d_tabix_image, params.m35d_scoring_image].every { it.contains('@sha256:') })
        error 'M35D shared images must use immutable registry digests'
    if (!(params.m35d_flare2_image.contains('@sha256:') ||
          params.m35d_flare2_image.startsWith('sha256:')))
        error 'M35D FLARE2 image must be immutable'

    def contractFile = file(params.m35d_contract, checkIfExists: true)
    def contract = new groovy.json.JsonSlurper().parse(contractFile)
    if (contract.experiment_id != 'M35D_NATWGS_FINE_R1_EXPLORATORY_CHR22' ||
        contract.status != 'PREREGISTERED_BEFORE_R1_TRUTH_OPENING' ||
        contract.cluster_screen.primary_gate != 'all_9_NATWGS_fine_combinations_must_pass')
        error 'M35D contract identity or gate differs'

    def repoDir = baseDir.resolve('..')
    def selections = contract.reference_design.selection_seeds.collect { it as Integer }
    def gmms = contract.cluster_screen.gmm_seeds.collect { it as Integer }
    M35D_PREPARE_R1_NATWGS_REFERENCE(
        Channel.fromList(selections), Channel.value(contractFile),
        Channel.value(file(params.m35d_roles, checkIfExists: true)),
        Channel.value(file(params.m35d_phased_scaffold_vcf, checkIfExists: true)),
        Channel.value(file(params.m35d_target_vcf, checkIfExists: true)),
        Channel.value(file(params.m35d_target_tbi, checkIfExists: true)),
        Channel.value(file(params.m35d_m27d_manifest, checkIfExists: true)),
        Channel.value(file(params.m35d_m27d_strata, checkIfExists: true)),
        Channel.value(file(params.m35d_m27d_training_set, checkIfExists: true)),
        Channel.value(file(params.m35d_m27d_related_pairs, checkIfExists: true)),
        Channel.value(file(params.m35d_r1_donor_audit, checkIfExists: true)),
        Channel.value(file(params.m35d_r1_mosaic_receipt, checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35d_natwgs_fine_r1.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35c_prepare_source_comparison.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_prepare_balanced_reference.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35_flare2_paired.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_run_flare.py", checkIfExists: true)),
    )
    def prepared = M35D_PREPARE_R1_NATWGS_REFERENCE.out.reference
    def cases = prepared.flatMap {
        selection, refVcf, refTbi, coarseSample, coarseMacro,
        fineSample, fineMacro, selected, receipt ->
        def rows = []
        gmms.each { gmm ->
            rows << tuple(selection, 'fine', gmm, refVcf, refTbi,
                          fineSample, fineMacro, receipt)
            rows << tuple(selection, 'coarse', gmm, refVcf, refTbi,
                          coarseSample, coarseMacro, receipt)
        }
        rows
    }
    M35D_R1_CLUSTER_SCREEN(
        cases, Channel.value(contractFile),
        Channel.value(file(params.m35d_target_vcf, checkIfExists: true)),
        Channel.value(file(params.m35d_target_tbi, checkIfExists: true)),
        Channel.value(file(params.m35d_genetic_map, checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35d_natwgs_fine_r1.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35c_prepare_source_comparison.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_prepare_balanced_reference.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35_flare2_paired.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_run_flare.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_create_model_wrapper.py", checkIfExists: true)),
    )
    M35D_AGGREGATE_R1_GATE(
        M35D_R1_CLUSTER_SCREEN.out.screen.map {
            selection, granularity, gmm, directory -> directory
        }.collect(), Channel.value(contractFile),
        Channel.value(file("${repoDir}/bin/m35d_natwgs_fine_r1.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35c_prepare_source_comparison.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35b_prepare_balanced_reference.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m35_flare2_paired.py", checkIfExists: true)),
        Channel.value(file("${repoDir}/bin/m34_run_flare.py", checkIfExists: true)),
    )
}
