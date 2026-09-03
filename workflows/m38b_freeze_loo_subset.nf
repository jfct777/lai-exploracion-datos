nextflow.enable.dsl=2

include { M38B_FREEZE_LOO_SUBSET } from '../modules/38B_LOO_SUBSET'

workflow {
    ['m38b_run_id', 'm38b_panel_vcf', 'm38b_split_tsv', 'm38b_selected_loci'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    def repoDir = projectDir.resolve('..')
    def sources = [
        'm33_safe_bridge_core.py',
        'm38b_build_loo_subset.py',
    ].collect { name -> file("${repoDir}/bin/${name}", checkIfExists: true) }
    M38B_FREEZE_LOO_SUBSET(
        Channel.of(tuple(
            file(params.m38b_panel_vcf, checkIfExists: true),
            file(params.m38b_split_tsv, checkIfExists: true),
            file(params.m38b_selected_loci, checkIfExists: true),
        )),
        sources,
    )
}
