nextflow.enable.dsl=2

process M38B_FREEZE_LOO_SUBSET {
    tag { 'chr22_REF_TRAIN_leave_one_NAM_unit_out' }
    publishDir { "${params.m38b_results_dir}/${params.m38b_run_id}/selection" }, mode: 'copy', overwrite: false
    cpus params.m38b_loo_cpus
    memory params.m38b_loo_memory
    time params.m38b_loo_time

    input:
    tuple path(panel_vcf), path(split_tsv), path(selected_loci)
    path source_files

    output:
    tuple path('m38b_loo_subset.tsv'), path('m38b_loo_subset.npz'),
          path('m38b_loo_subset.receipt.json'), emit: bundle

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${source_files} staged/bin/
    PYTHONPATH=staged/bin python3 staged/bin/m38b_build_loo_subset.py \
      --panel-vcf ${panel_vcf} \
      --split-tsv ${split_tsv} \
      --selected-loci ${selected_loci} \
      --expected-panel-sha256 '${params.m38b_panel_sha256}' \
      --expected-split-sha256 '${params.m38b_split_sha256}' \
      --expected-selected-sha256 '${params.m38b_selected_sha256}' \
      --expected-chromosome 22 --expected-loci 660 --expected-nam-units 4 \
      --beta-priors '0.5,1.0' --q-top-threshold 0.8 \
      --min-remaining-nam-units 2 \
      --posterior-draws '${params.m38b_loo_posterior_draws}' \
      --seed '${params.m38b_loo_seed}' \
      --output-tsv m38b_loo_subset.tsv \
      --output-npz m38b_loo_subset.npz \
      --output-receipt m38b_loo_subset.receipt.json
    """
}
