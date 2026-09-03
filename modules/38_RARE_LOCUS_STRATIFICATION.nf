nextflow.enable.dsl=2

process M38_RARE_LOCUS_STRATIFICATION {
    tag { params.m38_run_id }
    publishDir { "${params.m38_results_dir}/${params.m38_run_id}/stratification" },
        mode: 'copy', overwrite: false
    cpus params.m38_cpus
    memory params.m38_memory
    time params.m38_time

    input:
    tuple path(selected), path(reference), path(audit_tsv), path(audit_summary)
    path source_files

    output:
    path 'm38_rare_locus_strata.tsv', emit: per_locus
    path 'm38_rare_locus_strata.npz', emit: strata
    path 'm38_rare_locus_stratification.summary.json', emit: summary
    path 'm38_rare_locus_stratification.receipt.json', emit: receipt

    script:
    def f0OverlapFlag = params.m38_f0_contains_selected_rare_loci ?
        '--f0-contains-selected-rare-loci' : ''
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38_stratify_rare_loci.py \
      --selected '${selected}' \
      --reference '${reference}' \
      --audit-tsv '${audit_tsv}' \
      --audit-summary '${audit_summary}' \
      --selected-sha256 '${params.m38_selected_sha256}' \
      --reference-sha256 '${params.m38_reference_sha256}' \
      --audit-tsv-sha256 '${params.m38_audit_tsv_sha256}' \
      --audit-summary-sha256 '${params.m38_audit_summary_sha256}' \
      --expected-loci '${params.m38_expected_loci}' \
      --expected-ancestry-an '${params.m38_expected_ancestry_an}' \
      --beta-priors '${params.m38_beta_priors}' \
      --rare-af-cutoff '${params.m38_rare_af_cutoff}' \
      --q-top-thresholds '${params.m38_q_top_thresholds}' \
      --q-rare-thresholds '${params.m38_q_rare_thresholds}' \
      --unit-thresholds '${params.m38_unit_thresholds}' \
      --q-top-draws '${params.m38_q_top_draws}' \
      --seed '${params.m38_seed}' \
      ${f0OverlapFlag} \
      --f0-overlap-assertion-source '${params.m38_f0_overlap_assertion_source}' \
      --output-tsv m38_rare_locus_strata.tsv \
      --output-npz m38_rare_locus_strata.npz \
      --output-summary m38_rare_locus_stratification.summary.json \
      --output-receipt m38_rare_locus_stratification.receipt.json
    """

    stub:
    """
    set -euo pipefail
    touch m38_rare_locus_strata.tsv \
          m38_rare_locus_strata.npz \
          m38_rare_locus_stratification.summary.json \
          m38_rare_locus_stratification.receipt.json
    """
}
