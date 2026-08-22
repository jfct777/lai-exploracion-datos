nextflow.enable.dsl=2

process M33_A0_AUTHENTICATE_SOURCES {
    tag 'm33_a0_source_auth'
    cpus 1
    memory '256 MB'
    time '5m'

    input:
    path source_auth_py
    path adapter_py
    path tabix_audit_py
    path ordered_linear_py
    path rare_preflight_py
    path asset_registry
    path config_nf
    path preregistration
    path module_nf
    path workflow_nf
    path adapter_test_py
    path nextflow_test_py
    val git_commit
    val repository_root

    output:
    path 'm33_a0_source_auth.json', emit: auth

    script:
    """
    set -euo pipefail
    python3 ${source_auth_py} \
      --repository-root ${repository_root} --git-commit ${git_commit} \
      --source bin/m33_a0_real_adapter.py=${adapter_py} \
      --source bin/m33_a0_source_auth.py=${source_auth_py} \
      --source bin/m33_a0_tabix_audit.py=${tabix_audit_py} \
      --source bin/m31_ordered_linear.py=${ordered_linear_py} \
      --source bin/m31_ordered_rare_preflight.py=${rare_preflight_py} \
      --source conf/m33_a0_legacy_assets.json=${asset_registry} \
      --source conf/m33_a0_real_adapter.config=${config_nf} \
      --source conf/m33_a0_real_adapter_preregistration.json=${preregistration} \
      --source modules/33_A0_REAL_ADAPTER.nf=${module_nf} \
      --source workflows/m33_a0_real_adapter.nf=${workflow_nf} \
      --source tests/test_m33_a0_real_adapter.py=${adapter_test_py} \
      --source tests/test_m33_a0_real_adapter_nextflow.py=${nextflow_test_py} \
      --output m33_a0_source_auth.json
    """
}

process M33_A0_VALIDATE_INDEXES {
    tag { "m33_a0_tabix_${root_label}" }
    container params.m33_a0_tabix_container_image
    containerOptions params.m33_a0_container_options
    cpus 1
    memory '1 GB'
    time '10m'

    input:
    tuple val(root_label), path(ref_vcf), path(ref_tbi), path(target_vcf), path(target_tbi)
    path tabix_audit_py
    path source_auth
    val git_commit
    val expected_image_id

    output:
    tuple val(root_label), path('m33_a0_index_audit.json'), emit: audit

    script:
    """
    set -euo pipefail
    python3 ${tabix_audit_py} \
      --ref-vcf ${ref_vcf} --ref-tbi ${ref_tbi} \
      --target-vcf ${target_vcf} --target-tbi ${target_tbi} \
      --source-auth ${source_auth} --git-commit ${git_commit} \
      --expected-image-id ${expected_image_id} --output m33_a0_index_audit.json
    """
}

process M33_A0_AUDIT_LEGACY_ROOT {
    tag { "m33_a0_${root_label}" }
    publishDir { "${params.m33_a0_results_dir}/${params.m33_a0_run_id}/${root_label}" }, mode:'copy', overwrite:false
    container params.m33_a0_container_image
    containerOptions params.m33_a0_container_options
    cpus params.m33_a0_cpus
    memory params.m33_a0_memory
    time params.m33_a0_time

    input:
    tuple val(root_label), val(root_seed), path(tree_sequence), path(pools), path(rare_catalog),
        path(rare_haplotypes), path(m31_sites), path(m31_target),
        path(ref_vcf), path(ref_tbi), path(target_vcf), path(target_tbi), path(ref_pairs),
        path(panel_map), path(flare_anc), path(genetic_map)
    path preregistration
    path asset_registry
    path source_auth
    tuple val(index_root_label), path(index_audit)
    path adapter_py
    path source_auth_py
    path tabix_audit_py
    path ordered_linear_py
    path rare_preflight_py
    path config_nf
    path module_nf
    path workflow_nf
    path adapter_test_py
    path nextflow_test_py
    val git_commit
    val nextflow_version
    val adapter_image_id

    output:
    tuple val(root_label), path('m33_a0.receipt.json'), path('m33_a0_source_auth.json'),
        path('m33_a0_index_audit.json'), emit: receipts

    script:
    """
    set -euo pipefail
    test "${root_label}" = "${index_root_label}"
    export PYTHONPATH="\$PWD"
    python3 ${adapter_py} \
      --preregistration ${preregistration} --asset-registry ${asset_registry} \
      --source-auth ${source_auth} --git-commit ${git_commit} \
      --index-audit ${index_audit} \
      --nextflow-version ${nextflow_version} --adapter-image-id ${adapter_image_id} \
      --source bin/m33_a0_real_adapter.py=${adapter_py} \
      --source bin/m33_a0_source_auth.py=${source_auth_py} \
      --source bin/m33_a0_tabix_audit.py=${tabix_audit_py} \
      --source bin/m31_ordered_linear.py=${ordered_linear_py} \
      --source bin/m31_ordered_rare_preflight.py=${rare_preflight_py} \
      --source conf/m33_a0_legacy_assets.json=${asset_registry} \
      --source conf/m33_a0_real_adapter.config=${config_nf} \
      --source conf/m33_a0_real_adapter_preregistration.json=${preregistration} \
      --source modules/33_A0_REAL_ADAPTER.nf=${module_nf} \
      --source workflows/m33_a0_real_adapter.nf=${workflow_nf} \
      --source tests/test_m33_a0_real_adapter.py=${adapter_test_py} \
      --source tests/test_m33_a0_real_adapter_nextflow.py=${nextflow_test_py} \
      --root-label ${root_label} --root-seed ${root_seed} \
      --tree-sequence ${tree_sequence} --pools ${pools} \
      --rare-catalog ${rare_catalog} --rare-haplotypes ${rare_haplotypes} \
      --m31-sites ${m31_sites} --m31-target ${m31_target} \
      --ref-vcf ${ref_vcf} --ref-tbi ${ref_tbi} \
      --target-vcf ${target_vcf} --target-tbi ${target_tbi} \
      --ref-pairs ${ref_pairs} --panel-map ${panel_map} --flare-anc ${flare_anc} \
      --genetic-map ${genetic_map} \
      --output m33_a0.receipt.json
    """
}
