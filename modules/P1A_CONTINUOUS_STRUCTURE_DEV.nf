nextflow.enable.dsl=2

process P1A_CONTINUOUS_STRUCTURE_DEV {
    tag 'p1a_continuous_structure_dev'
    publishDir params.p1a_results_dir, mode: 'copy', overwrite: false
    container params.p1a_container_image
    cpus params.p1a_cpus
    memory params.p1a_memory
    time params.p1a_time

    input:
    path pairs
    path global_summary
    path metadata
    path burden
    path feature_store
    path modeling_master
    path split_manifest
    path preflight_report
    path preregistration
    path runner_py
    path manifest_py
    val code_commit
    val provenance_b64

    output:
    path 'p1a_dev/p1a_dev_report.json', emit: report
    path 'p1a_dev/p1a_inner_selection.tsv', emit: inner_selection
    path 'p1a_dev/p1a_outer_metrics.tsv', emit: outer_metrics
    path 'p1a_dev/p1a_region_metrics.tsv', emit: region_metrics
    path 'p1a_dev/p1a_region_state_counts.tsv', emit: region_state_counts
    path 'p1a_dev/p1a_region_projectability_counts.tsv', emit: region_projectability_counts
    path 'p1a_dev/p1a_graph_diagnostics.tsv', emit: graph_diagnostics
    path 'p1a_dev/p1a_graph_features.tsv.gz', emit: graph_features
    path 'p1a_dev/p1a_oof_predictions.tsv.gz', emit: predictions
    path 'p1a_dev/p1a_cluster_bootstrap.tsv.gz', emit: bootstrap
    path 'p1a_dev/p1a_run_provenance.json', emit: provenance
    path 'p1a_dev/p1a_dev.manifest.json', emit: manifest

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS=${task.cpus}
    export OPENBLAS_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=${task.cpus}
    export NUMEXPR_NUM_THREADS=${task.cpus}

    mkdir p1a_dev
    printf '%s' '${provenance_b64}' | base64 -d > p1a_dev/p1a_run_provenance.json

    python3 ${runner_py} \
      --pairs ${pairs} \
      --global-summary ${global_summary} \
      --metadata ${metadata} \
      --burden ${burden} \
      --feature-store ${feature_store} \
      --modeling-master ${modeling_master} \
      --split-manifest ${split_manifest} \
      --preflight-report ${preflight_report} \
      --preregistration ${preregistration} \
      --code-commit '${code_commit}' \
      --output-dir p1a_dev/results

    mv p1a_dev/results/* p1a_dev/
    rmdir p1a_dev/results

    python3 ${manifest_py} \
      --stage P1A_CONTINUOUS_STRUCTURE_DEV \
      --input ${pairs} --input ${global_summary} --input ${metadata} \
      --input ${burden} --input ${feature_store} --input ${modeling_master} \
      --input ${split_manifest} --input ${preflight_report} --input ${preregistration} \
      --input ${runner_py} --input ${manifest_py} \
      --output p1a_dev/p1a_dev_report.json \
      --output p1a_dev/p1a_inner_selection.tsv \
      --output p1a_dev/p1a_outer_metrics.tsv \
      --output p1a_dev/p1a_region_metrics.tsv \
      --output p1a_dev/p1a_region_state_counts.tsv \
      --output p1a_dev/p1a_region_projectability_counts.tsv \
      --output p1a_dev/p1a_graph_diagnostics.tsv \
      --output p1a_dev/p1a_graph_features.tsv.gz \
      --output p1a_dev/p1a_oof_predictions.tsv.gz \
      --output p1a_dev/p1a_cluster_bootstrap.tsv.gz \
      --output p1a_dev/p1a_run_provenance.json \
      --params-json '{"scope":"DEV_transductive_no_fold3","anchor_scope":"all_dnabr_train_other_folds_both_modes","primary_graph":"binary","sensitivity":"log1p_bp_weights_only","graph_nulls":false,"team":"frank"}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref p1a_run_provenance.json \
      --out p1a_dev/p1a_dev.manifest.json
    """
}
