nextflow.enable.dsl=2

process M31_ORDERED_RARE_PREFLIGHT {
    tag { "m31_ordered_rare_${root_label}_${root_seed}" }
    publishDir { "${params.m31_preflight_results_dir}/root-${root_seed}" }, mode: 'copy', overwrite: false
    container params.m31_preflight_container_image
    containerOptions params.m31_preflight_container_options
    cpus params.m31_preflight_cpus
    memory params.m31_preflight_memory
    time params.m31_preflight_time

    input:
    tuple val(root_label), val(root_seed), path(tree), path(pools), path(catalog), path(haplotypes)
    path preregistration
    path preflight_py
    val git_commit

    output:
    tuple val(root_seed),
        path('m31_preflight/m31_ordered_rare.preflight.json'),
        path('m31_preflight/m31_ordered_rare.samples.tsv'),
        path('m31_preflight/m31_ordered_rare.sites.tsv.gz'),
        path('m31_preflight/m31_ordered_rare.target.tsv.gz'),
        path('m31_preflight/m31_ordered_rare.manifest.json'), emit: materialized

    script:
    """
    set -euo pipefail
    python3 ${preflight_py} \
      --preregistration ${preregistration} \
      --root-label ${root_label} --root-seed ${root_seed} \
      --git-commit ${git_commit} \
      --tree ${tree} --pools ${pools} \
      --catalog ${catalog} --haplotypes ${haplotypes} \
      --outdir m31_preflight
    """
}
