process M33_SAFE_BRIDGE_TECHNICAL_KAT_ROOT {
    tag "${root_label}"
    cache false
    cpus params.m33_safe_bridge_technical_cpus
    memory params.m33_safe_bridge_technical_memory
    time params.m33_safe_bridge_technical_time
    container params.m33_safe_bridge_technical_container_image
    containerOptions params.m33_safe_bridge_technical_container_options

    input:
    tuple val(root_label), val(root_seed),
          path(tree_sequence), path(pools), path(rare_catalog), path(rare_haplotypes),
          path(m31_sites), path(m31_target), path(ref_vcf), path(ref_tbi),
          path(ref_pairs), path(panel_map), path(genetic_map), path(flare_anc), path(flare_tbi)
    path contract
    path authorization
    path source_auth
    path runner
    path core
    path a0_adapter
    path ordered_linear
    path rare_preflight
    path config_nf
    path module_nf
    path workflow_nf
    path runner_test
    path nextflow_test
    val git_commit

    output:
    tuple val(root_label), path("${root_label}.technical_kat"), emit: technical_kat

    script:
    def sources = [
        'bin/m33_safe_bridge_technical_kat.py': runner,
        'bin/m33_safe_bridge_core.py': core,
        'bin/m33_a0_real_adapter.py': a0_adapter,
        'bin/m31_ordered_linear.py': ordered_linear,
        'bin/m31_ordered_rare_preflight.py': rare_preflight,
        'conf/m33_safe_bridge_technical_kat_contract.json': contract,
        'conf/m33_safe_bridge_technical_kat_authorization.json': authorization,
        'conf/m33_safe_bridge_technical_kat.config': config_nf,
        'modules/33_SAFE_BRIDGE_TECHNICAL_KAT.nf': module_nf,
        'workflows/m33_safe_bridge_technical_kat.nf': workflow_nf,
        'tests/test_m33_safe_bridge_technical_kat.py': runner_test,
        'tests/test_m33_safe_bridge_technical_kat_nextflow.py': nextflow_test,
    ].collect { relative, staged -> "--source '${relative}=${staged}'" }.join(' ')
    """
    mkdir '${root_label}.technical_kat'
    chmod 0777 '${root_label}.technical_kat'
    chmod 0444 '${tree_sequence}' '${pools}' '${rare_catalog}' '${rare_haplotypes}' \
      '${m31_sites}' '${m31_target}' '${ref_vcf}' '${ref_tbi}' '${ref_pairs}' \
      '${panel_map}' '${genetic_map}' '${flare_anc}' '${flare_tbi}' \
      '${contract}' '${authorization}' '${source_auth}' '${runner}' '${core}' \
      '${a0_adapter}' '${ordered_linear}' '${rare_preflight}' '${config_nf}' \
      '${module_nf}' '${workflow_nf}' '${runner_test}' '${nextflow_test}'
    env -u HOME -u GOOGLE_APPLICATION_CREDENTIALS -u CLOUDSDK_CONFIG \
      PYTHONPATH=. setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs \
      python '${runner}' \
        --contract '${contract}' \
        --authorization '${authorization}' \
        --source-auth '${source_auth}' \
        ${sources} \
        --git-commit '${git_commit}' \
        --root-label '${root_label}' \
        --root-seed '${root_seed}' \
        --nextflow-version '${workflow.nextflow.version}' \
        --oci-digest '${params.m33_safe_bridge_technical_container_image}' \
        --tree-sequence '${tree_sequence}' \
        --pools '${pools}' \
        --rare-catalog '${rare_catalog}' \
        --rare-haplotypes '${rare_haplotypes}' \
        --m31-sites '${m31_sites}' \
        --m31-target '${m31_target}' \
        --ref-vcf '${ref_vcf}' \
        --ref-tbi '${ref_tbi}' \
        --ref-pairs '${ref_pairs}' \
        --panel-map '${panel_map}' \
        --genetic-map '${genetic_map}' \
        --flare-anc '${flare_anc}' \
        --flare-tbi '${flare_tbi}' \
        --output-dir '${root_label}.technical_kat'
    """
}
