nextflow.enable.dsl=2

process M33_SANITIZE_FLARE_F0 {
    tag "root_${root_seed}"
    cache false
    cpus params.m33_f0_sanitize_cpus
    memory params.m33_f0_sanitize_memory
    time params.m33_f0_sanitize_time
    container params.m33_f0_sanitize_container_image
    containerOptions params.m33_f0_sanitize_container_options

    input:
    tuple val(root_seed), path(flare_anc), path(target_rare_diploid)
    path runner
    path core
    path contract
    path source_auth
    path config_nf
    path module_nf
    path workflow_nf
    path runner_test
    path nextflow_test
    val git_commit

    output:
    tuple val(root_seed), path("root-${root_seed}.f0_sanitized"), emit: sanitized

    script:
    def sources = [
        'bin/m33_f0_sanitize.py': runner,
        'bin/m33_safe_bridge_core.py': core,
        'conf/m33_m0_f0_sanitized_amendment_contract.json': contract,
        'conf/m33_f0_sanitize.config': config_nf,
        'modules/33_SANITIZE_FLARE_F0.nf': module_nf,
        'workflows/m33_f0_sanitize.nf': workflow_nf,
        'tests/test_m33_f0_sanitize.py': runner_test,
        'tests/test_m33_f0_sanitize_nextflow.py': nextflow_test,
    ].collect { relative, staged -> "--source '${relative}=${staged}'" }.join(' ')
    """
    set -euo pipefail
    mkdir 'root-${root_seed}.f0_sanitized'
    PYTHONPATH=. python '${runner}' \
      --flare-anc '${flare_anc}' \
      --target-rare-diploid '${target_rare_diploid}' \
      --root-seed '${root_seed}' \
      --source-auth '${source_auth}' \
      --git-commit '${git_commit}' \
      ${sources} \
      --output-dir 'root-${root_seed}.f0_sanitized'
    """
}
