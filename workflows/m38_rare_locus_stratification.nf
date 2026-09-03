nextflow.enable.dsl=2

include { M38_RARE_LOCUS_STRATIFICATION } from '../modules/38_RARE_LOCUS_STRATIFICATION'

workflow {
    [
        'm38_run_id', 'm38_results_dir', 'm38_selected_loci',
        'm38_reference_summary', 'm38_audit_tsv', 'm38_audit_summary',
        'm38_selected_sha256', 'm38_reference_sha256',
        'm38_audit_tsv_sha256', 'm38_audit_summary_sha256',
        'm38_f0_overlap_assertion_source',
    ].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (!(params.m38_run_id ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/)) {
        error '--m38_run_id contains unsupported characters'
    }
    if (params.m38_expected_loci != 660) error '--m38_expected_loci must remain 660'
    if (!params.m38_f0_contains_selected_rare_loci) {
        error 'the known overlap between F0 and the selected loci must be acknowledged'
    }

    def repoDir = projectDir.resolve('..')
    def sources = [
        file("${repoDir}/bin/m38_stratify_rare_loci.py", checkIfExists: true),
        file("${repoDir}/bin/m33_safe_bridge_core.py", checkIfExists: true),
    ]
    def inputs = Channel.value(tuple(
        file(params.m38_selected_loci, checkIfExists: true),
        file(params.m38_reference_summary, checkIfExists: true),
        file(params.m38_audit_tsv, checkIfExists: true),
        file(params.m38_audit_summary, checkIfExists: true),
    ))
    M38_RARE_LOCUS_STRATIFICATION(inputs, sources)
}
