nextflow.enable.dsl=2

process RUN_M29_SAME_LOCUS_DEV {
    tag "m29_same_locus_dev"
    publishDir params.m29_results_dir, mode: 'copy', overwrite: false
    container params.m29_container_image
    containerOptions params.m29_container_options
    cpus params.m29_cpus
    memory params.m29_memory
    time params.m29_time

    input:
    path preregistration
    path genetic_map
    tuple val(root_a_seed), path(root_a_tree, stageAs: 'root_a/*'), path(root_a_pools, stageAs: 'root_a/*'), path(root_a_report, stageAs: 'root_a/*'), path(root_a_manifest, stageAs: 'root_a/*'), path(root_a_catalog, stageAs: 'root_a/*'), path(root_a_haplotypes, stageAs: 'root_a/*'), path(root_a_truth, stageAs: 'root_a/*'), path(root_a_fb, stageAs: 'root_a/*'), path(root_a_msp, stageAs: 'root_a/*'), path(root_a_binding, stageAs: 'root_a/*')
    tuple val(root_b_seed), path(root_b_tree, stageAs: 'root_b/*'), path(root_b_pools, stageAs: 'root_b/*'), path(root_b_report, stageAs: 'root_b/*'), path(root_b_manifest, stageAs: 'root_b/*'), path(root_b_catalog, stageAs: 'root_b/*'), path(root_b_haplotypes, stageAs: 'root_b/*'), path(root_b_truth, stageAs: 'root_b/*'), path(root_b_fb, stageAs: 'root_b/*'), path(root_b_msp, stageAs: 'root_b/*'), path(root_b_binding, stageAs: 'root_b/*')
    path m29_script
    path m28d_scorer
    val git_commit

    output:
    path "m29_dev/m29_dev_summary.public.json", emit: summary
    path "m29_dev/m29_dev_metrics.tsv", emit: metrics
    path "m29_dev/m29_dev_individual_errors.tsv.gz", emit: individual_errors
    path "m29_dev/m29_dev.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    test '${root_a_seed}' = '20260817'
    test '${root_b_seed}' = '20260818'
    python3 ${m29_script} \
      --preregistration ${preregistration} --genetic-map ${genetic_map} \
      --git-commit ${git_commit} \
      --root-a-tree ${root_a_tree} --root-a-pools ${root_a_pools} \
      --root-a-report ${root_a_report} --root-a-manifest ${root_a_manifest} \
      --root-a-catalog ${root_a_catalog} --root-a-haplotypes ${root_a_haplotypes} \
      --root-a-truth ${root_a_truth} --root-a-fb ${root_a_fb} --root-a-msp ${root_a_msp} \
      --root-a-binding ${root_a_binding} \
      --root-b-tree ${root_b_tree} --root-b-pools ${root_b_pools} \
      --root-b-report ${root_b_report} --root-b-manifest ${root_b_manifest} \
      --root-b-catalog ${root_b_catalog} --root-b-haplotypes ${root_b_haplotypes} \
      --root-b-truth ${root_b_truth} --root-b-fb ${root_b_fb} --root-b-msp ${root_b_msp} \
      --root-b-binding ${root_b_binding} --outdir m29_dev
    """
}
