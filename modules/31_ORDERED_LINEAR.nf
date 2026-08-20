nextflow.enable.dsl=2

process M31_ORDERED_LINEAR_DEV {
    tag 'm31_ordered_linear_root17_root18'
    publishDir params.m31_ordered_linear_results_dir, mode: 'copy', overwrite: false
    container params.m31_ordered_linear_container_image
    containerOptions params.m31_ordered_linear_container_options
    cpus params.m31_ordered_linear_cpus
    memory params.m31_ordered_linear_memory
    time params.m31_ordered_linear_time

    input:
    path preregistration
    path genetic_map
    tuple val(root17_label), val(root17_seed),
        path(root17_sites, stageAs: 'root17/preflight/*'),
        path(root17_target, stageAs: 'root17/preflight/*'),
        path(root17_tree, stageAs: 'root17/m28/*'),
        path(root17_pools, stageAs: 'root17/m28/*'),
        path(root17_truth, stageAs: 'root17/m28/*'),
        path(root17_flare_vcf, stageAs: 'root17/m30/*'),
        path(root17_flare_audit, stageAs: 'root17/m30/*')
    tuple val(root18_label), val(root18_seed),
        path(root18_sites, stageAs: 'root18/preflight/*'),
        path(root18_target, stageAs: 'root18/preflight/*'),
        path(root18_tree, stageAs: 'root18/m28/*'),
        path(root18_pools, stageAs: 'root18/m28/*'),
        path(root18_truth, stageAs: 'root18/m28/*'),
        path(root18_flare_vcf, stageAs: 'root18/m30/*'),
        path(root18_flare_audit, stageAs: 'root18/m30/*')
    path ordered_linear_py
    val provenance_b64

    output:
    path 'm31_ordered_linear/m31_ordered_linear.selftest.json', emit: selftest
    path 'm31_ordered_linear/m31_ordered_linear.input_sha256.tsv', emit: input_hashes
    path 'm31_ordered_linear/m31_ordered_linear.provenance.json', emit: provenance

    script:
    """
    set -euo pipefail
    test '${root17_label}' = 'root17'
    test '${root17_seed}' = '20260817'
    test '${root18_label}' = 'root18'
    test '${root18_seed}' = '20260818'
    mkdir -p m31_ordered_linear
    printf '%s' '${provenance_b64}' | base64 -d \
      > m31_ordered_linear/m31_ordered_linear.provenance.json
    sha256sum \
      ${preregistration} ${genetic_map} ${ordered_linear_py} \
      ${root17_sites} ${root17_target} ${root17_tree} ${root17_pools} \
      ${root17_truth} ${root17_flare_vcf} ${root17_flare_audit} \
      ${root18_sites} ${root18_target} ${root18_tree} ${root18_pools} \
      ${root18_truth} ${root18_flare_vcf} ${root18_flare_audit} \
      | LC_ALL=C sort -k2,2 \
      > m31_ordered_linear/m31_ordered_linear.input_sha256.tsv
    python3 ${ordered_linear_py} \
      --contract ${preregistration} --selftest \
      --genetic-map ${genetic_map} \
      --root17-sites ${root17_sites} --root17-target ${root17_target} \
      --root17-tree ${root17_tree} --root17-pools ${root17_pools} \
      --root17-truth ${root17_truth} --root17-flare-vcf ${root17_flare_vcf} \
      --root17-flare-audit ${root17_flare_audit} \
      --root18-sites ${root18_sites} --root18-target ${root18_target} \
      --root18-tree ${root18_tree} --root18-pools ${root18_pools} \
      --root18-truth ${root18_truth} --root18-flare-vcf ${root18_flare_vcf} \
      --root18-flare-audit ${root18_flare_audit} \
      --output m31_ordered_linear/m31_ordered_linear.selftest.json
    """
}
