nextflow.enable.dsl=2

process WRITE_M16_5_RUN_PROVENANCE {
    tag "run_provenance"
    publishDir "${params.ibd_enhanced_results_dir}", mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '10m'

    input:
    val provenance_b64

    output:
    path "run_provenance.json"

    script:
    """
    set -euo pipefail
    printf '%s' '${provenance_b64}' | base64 -d > run_provenance.json
    """
}


process WRITE_M16_5_MANIFEST {
    tag "manifest"
    publishDir "${params.ibd_enhanced_results_dir}", mode: 'copy', overwrite: false
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    path all_segments
    path pair_summary
    path individual_summary
    path graph_edges
    path graph_summary
    path leiden_assignments
    path global_summary
    path ibd_script
    path write_stage_manifest_py
    val provenance_b64

    output:
    path "m16_5.manifest.json"

    script:
    """
    set -euo pipefail
    python3 ${write_stage_manifest_py} \
      --stage IBD_COMMUNITY_ENHANCED \
      --input ${all_segments} --input ${pair_summary} --input ${individual_summary} \
      --input ${ibd_script} \
      --output ${graph_edges} --output ${graph_summary} \
      --output ${leiden_assignments} --output ${global_summary} \
      --provenance-b64 '${provenance_b64}' \
      --params-json '{"min_edge_bp":5000000,"min_max_segment_bp":500000,"edge_weight_transform":"log1p","seed":42,"laplacian_normalize":true,"nmf_init_mode":"random-cophenetic","nmf_operational_k":8}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --out m16_5.manifest.json
    """
}


process COMPARE_M16_5_ORIENTATION {
    tag "alt_vs_minor"
    publishDir "${params.ibd_enhanced_results_dir}/comparison", mode: 'copy', overwrite: false
    cpus 2
    memory '8 GB'
    time '30m'

    input:
    path historical_assignments, stageAs: 'historical/leiden_assignments.tsv'
    path historical_edges, stageAs: 'historical/graph_edges.tsv.gz'
    path historical_graph_summary, stageAs: 'historical/graph_summary.json'
    path minor_assignments, stageAs: 'minor/leiden_assignments.tsv'
    path minor_edges, stageAs: 'minor/graph_edges.tsv.gz'
    path minor_graph_summary, stageAs: 'minor/graph_summary.json'
    path cohort_summary
    path compare_py
    path write_stage_manifest_py
    val provenance_b64

    output:
    path "m16_5_orientation_comparison.json", emit: report
    path "m16_5_status_transitions.tsv", emit: transitions
    path "m16_5_community_matches.tsv", emit: matches
    path "m16_5_orientation_comparison.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${compare_py} \
      --historical-dir historical \
      --minor-dir minor \
      --cohort-summary ${cohort_summary} \
      --outdir .

    python3 ${write_stage_manifest_py} \
      --stage COMPARE_M16_5_ORIENTATION \
      --input historical/leiden_assignments.tsv \
      --input historical/graph_edges.tsv.gz \
      --input historical/graph_summary.json \
      --input minor/leiden_assignments.tsv \
      --input minor/graph_edges.tsv.gz \
      --input minor/graph_summary.json \
      --input ${cohort_summary} --input ${compare_py} \
      --output m16_5_orientation_comparison.json \
      --output m16_5_status_transitions.tsv \
      --output m16_5_community_matches.tsv \
      --provenance-b64 '${provenance_b64}' \
      --params-json '{"primary_resolution":1.0,"full_cohort_n":2619,"label_matching":"hungarian_max_overlap"}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --out m16_5_orientation_comparison.manifest.json
    """
}
