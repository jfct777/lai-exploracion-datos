nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Módulo 18: comparación de asIBD común con comunidades de variantes raras
// ---------------------------------------------------------------------------
// Construye comunidades de variantes comunes a igual ancestría local a partir del asIBD de Nunes.
// Los archivos contienen Refined IBD sobre WGS refaseado y están estratificados como anc1, anc2 y
// anc3. Se usa la misma configuración de Leiden de M16.5 y se compara con
// leiden_assignments.tsv de variantes raras.
//
// El informe incluye las tres ancestrías, la concordancia, la complementariedad y los casos sin
// respaldo. La complementariedad exige enriquecimiento de mtDNA frente al null de permutaciones y
// ausencia de contención en el árbitro de más de 2 cM de la misma ancestría.
//
// Se ejecuta como un solo proceso porque los archivos asIBD cubren todo el genoma.
// asibd_comparator.py importa ibd_community_enhanced.py para reutilizar Leiden, por lo que ambos
// scripts se preparan como entradas del proceso.

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
    export MPLCONFIGDIR="\$PWD/.mplcache"   # En el worker, \$HOME/.config es de solo lectura.

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
