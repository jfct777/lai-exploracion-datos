nextflow.enable.dsl=2

process M33_AUTHENTICATE_CONTRACT_SOURCES {
    tag 'm33_pre4_source_auth'
    publishDir { "${params.m33_pre4_results_dir}/${params.m33_pre4_run_id}" }, mode: 'copy', overwrite: false
    cpus 1
    memory '256 MB'
    time '5m'

    input:
    path source_auth_py
    path contract_py
    path preregistration
    path config_nf
    path module_nf
    path workflow_nf
    path contract_tests
    path nextflow_tests
    val git_commit
    val repository_root

    output:
    path 'm33_pre4_source_auth.json', emit: auth

    script:
    """
    set -euo pipefail
    python3 ${source_auth_py} --repository-root ${repository_root} --git-commit ${git_commit} \
      --source bin/m33_pre4_source_auth.py=${source_auth_py} \
      --source bin/m33_pre4_contract.py=${contract_py} \
      --source conf/m33_pre4_preregistration.json=${preregistration} \
      --source conf/m33_pre4_contract.config=${config_nf} \
      --source modules/33_PRE4_CONTRACT.nf=${module_nf} \
      --source workflows/m33_pre4_contract.nf=${workflow_nf} \
      --source tests/test_m33_pre4_contract.py=${contract_tests} \
      --source tests/test_m33_pre4_nextflow.py=${nextflow_tests} \
      --output m33_pre4_source_auth.json
    """
}

process M33_VALIDATE_PRE4_CONTRACT {
    tag 'm33_pre4_contract_only'
    publishDir { "${params.m33_pre4_results_dir}/${params.m33_pre4_run_id}" }, mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    path source_auth_py
    path contract_py
    path preregistration
    path config_nf
    path module_nf
    path workflow_nf
    path contract_tests
    path nextflow_tests
    path source_auth
    val git_commit
    val repository_root
    val container_digest
    val nextflow_version

    output:
    path 'm33_pre4_contract.receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    python3 ${contract_py} --contract ${preregistration} --source-auth ${source_auth} \
      --staged-source bin/m33_pre4_source_auth.py=${source_auth_py} \
      --staged-source bin/m33_pre4_contract.py=${contract_py} \
      --staged-source conf/m33_pre4_preregistration.json=${preregistration} \
      --staged-source conf/m33_pre4_contract.config=${config_nf} \
      --staged-source modules/33_PRE4_CONTRACT.nf=${module_nf} \
      --staged-source workflows/m33_pre4_contract.nf=${workflow_nf} \
      --staged-source tests/test_m33_pre4_contract.py=${contract_tests} \
      --staged-source tests/test_m33_pre4_nextflow.py=${nextflow_tests} \
      --git-commit ${git_commit} --repository-root ${repository_root} \
      --container-digest ${container_digest} \
      --nextflow-version ${nextflow_version} --output m33_pre4_contract.receipt.json
    """
}
