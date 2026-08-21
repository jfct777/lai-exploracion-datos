nextflow.enable.dsl=2

process M33_AUTHENTICATE_ASSET_CONTRACT_SOURCES {
    tag 'm33_asset_manifest_source_auth'
    publishDir { "${params.m33_asset_contract_results_dir}/${params.m33_asset_contract_run_id}" }, mode: 'copy', overwrite: false
    cpus 1
    memory '256 MB'
    time '5m'

    input:
    path source_auth_py
    path contract_py
    path contract_json
    path config_nf
    path module_nf
    path workflow_nf
    path contract_tests
    path nextflow_tests
    val git_commit
    val repository_root

    output:
    path 'm33_asset_manifest_source_auth.json', emit: auth

    script:
    """
    set -euo pipefail
    python3 ${source_auth_py} --repository-root ${repository_root} --git-commit ${git_commit} \
      --source bin/m33_asset_manifest_source_auth.py=${source_auth_py} \
      --source bin/m33_asset_manifest_contract.py=${contract_py} \
      --source conf/m33_asset_manifest_contract.json=${contract_json} \
      --source conf/m33_asset_manifest_contract.config=${config_nf} \
      --source modules/33_ASSET_MANIFEST_CONTRACT.nf=${module_nf} \
      --source workflows/m33_asset_manifest_contract.nf=${workflow_nf} \
      --source tests/test_m33_asset_manifest_contract.py=${contract_tests} \
      --source tests/test_m33_asset_manifest_nextflow.py=${nextflow_tests} \
      --output m33_asset_manifest_source_auth.json
    """
}

process M33_VALIDATE_ASSET_MANIFEST_CONTRACT {
    tag 'm33_asset_manifest_contract_only'
    publishDir { "${params.m33_asset_contract_results_dir}/${params.m33_asset_contract_run_id}" }, mode: 'copy', overwrite: false
    cpus 1
    memory '512 MB'
    time '5m'

    input:
    path source_auth_py
    path contract_py
    path contract_json
    path config_nf
    path module_nf
    path workflow_nf
    path contract_tests
    path nextflow_tests
    path source_auth
    val git_commit
    val repository_root
    val nextflow_version

    output:
    path 'm33_asset_manifest_contract.receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    python3 ${contract_py} --contract ${contract_json} --source-auth ${source_auth} \
      --staged-source bin/m33_asset_manifest_source_auth.py=${source_auth_py} \
      --staged-source bin/m33_asset_manifest_contract.py=${contract_py} \
      --staged-source conf/m33_asset_manifest_contract.json=${contract_json} \
      --staged-source conf/m33_asset_manifest_contract.config=${config_nf} \
      --staged-source modules/33_ASSET_MANIFEST_CONTRACT.nf=${module_nf} \
      --staged-source workflows/m33_asset_manifest_contract.nf=${workflow_nf} \
      --staged-source tests/test_m33_asset_manifest_contract.py=${contract_tests} \
      --staged-source tests/test_m33_asset_manifest_nextflow.py=${nextflow_tests} \
      --git-commit ${git_commit} --repository-root ${repository_root} \
      --nextflow-version ${nextflow_version} \
      --output m33_asset_manifest_contract.receipt.json
    """
}
