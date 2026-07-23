nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Module 18 — Common-variant asIBD comparator vs rare-variant communities
// ---------------------------------------------------------------------------
// Builds the COMMON-variant community structure, AT EQUAL LOCAL ANCESTRY, from
// Nunes' asIBD (Refined IBD on the rephased WGS, stratified by local ancestry:
// anc1/anc2/anc3 .gapfilled_ibd) using the SAME Leiden as M16.5, and compares it
// against the rare-variant communities (M16.5 leiden_assignments.tsv).
//
// Reports ALL three ancestries and BOTH faces: concordance (reaffirms Nunes) AND
// complementarity (rare resolves what common collapses), plus fails — never only
// the divergence that confirms the thesis (feedback_report_all_validated_findings,
// feedback_all_communities_not_just_nam). Pre-registered JOINT discriminant
// (decision 2026-06-03): complement = mtDNA enrichment beats permutation null
// (fixed orthogonal axis) AND not contained in the >2cM per-ancestry arbiter.
//
// Single process (like M16.5), not per-chr: the asIBD files are genome-wide.
// NOTE: bin/asibd_comparator.py imports bin/ibd_community_enhanced.py to reuse the
// EXACT M16.5 Leiden, so BOTH scripts are staged into the work dir as inputs.

process COMPARE_ASIBD_COMMON {
    tag "asibd_comparator"

    publishDir "${params.asibd_results_dir}", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple path(asibd_files), path(leiden), path(metadata), path(comparator_py), path(leiden_lib_py)

    output:
    path "asibd_comparison.ari_by_ancestry.tsv",             emit: ari
    path "asibd_comparison.concordance_complementarity.tsv", emit: concordance
    path "asibd_comparison.summary.json",                    emit: summary

    script:
    """
    set -euo pipefail
    export MPLCONFIGDIR="\$PWD/.mplcache"   # \$HOME/.config is read-only on the worker

    python3 ${comparator_py} \
      --leiden ${leiden} \
      --metadata ${metadata} \
      --asibd_dir . \
      --asibd_glob '${params.asibd_glob}' \
      --anc_map '${params.asibd_anc_map}' \
      --resolution_col ${params.asibd_resolution_col} \
      --leiden_resolutions '${params.asibd_leiden_resolutions}' \
      --leiden_n_seeds ${params.asibd_leiden_n_seeds} \
      --leiden_min_community_size ${params.asibd_leiden_min_community_size} \
      --leiden_consensus_resolution ${params.asibd_leiden_consensus_resolution} \
      --seed ${params.asibd_seed} \
      --affinity_weight ${params.asibd_affinity_weight} \
      --normalize_by_dosage ${params.asibd_normalize_by_dosage} \
      --arbiter_min_cm ${params.asibd_arbiter_min_cm} \
      --ortho_perm_n ${params.asibd_ortho_perm_n} \
      --ortho_perm_alpha ${params.asibd_ortho_perm_alpha} \
      --arbiter_containment ${params.asibd_arbiter_containment} \
      --out_prefix asibd_comparison
    """
}
