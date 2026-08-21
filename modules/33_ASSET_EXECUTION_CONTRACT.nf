nextflow.enable.dsl=2

process M33_AUTHENTICATE_ASSET_EXECUTION_SOURCES {
    tag 'm33_asset_execution_source_auth'
    publishDir { "${params.m33_asset_execution_results_dir}/${params.m33_asset_execution_run_id}" }, mode: 'copy', overwrite: false
    cpus 1
    memory '256 MB'
    time '5m'

    input:
    path source_auth_py
    path execution_py
    path base_helper_py
    path asset_contract_json
    path amendment_json
    path config_nf
    path module_nf
    path workflow_nf
    path contract_tests
    path nextflow_tests
    val git_commit
    val repository_root

    output:
    path 'm33_asset_execution_source_auth.json', emit: auth

    script:
    """
    set -euo pipefail
    python3 ${source_auth_py} --repository-root ${repository_root} --git-commit ${git_commit} \
      --source bin/m33_asset_execution_source_auth.py=${source_auth_py} \
      --source bin/m33_asset_execution_contract.py=${execution_py} \
      --source bin/m33_asset_manifest_contract.py=${base_helper_py} \
      --source conf/m33_asset_manifest_contract.json=${asset_contract_json} \
      --source conf/m33_asset_execution_amendment.json=${amendment_json} \
      --source conf/m33_asset_execution_contract.config=${config_nf} \
      --source modules/33_ASSET_EXECUTION_CONTRACT.nf=${module_nf} \
      --source workflows/m33_asset_execution_contract.nf=${workflow_nf} \
      --source tests/test_m33_asset_execution_contract.py=${contract_tests} \
      --source tests/test_m33_asset_execution_nextflow.py=${nextflow_tests} \
      --output m33_asset_execution_source_auth.json
    """
}

process M33_VALIDATE_ASSET_EXECUTION_CONTRACT {
    tag 'm33_asset_execution_contract_fixtures_only'
    publishDir { "${params.m33_asset_execution_results_dir}/${params.m33_asset_execution_run_id}" }, mode: 'copy', overwrite: false
    cpus 1
    memory '512 MB'
    time '5m'

    input:
    path source_auth_py
    path execution_py
    path base_helper_py
    path asset_contract_json
    path amendment_json
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
    path 'm33_asset_execution_contract.receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin staged/conf staged/modules staged/workflows staged/tests
    cp ${source_auth_py} staged/bin/m33_asset_execution_source_auth.py
    cp ${execution_py} staged/bin/m33_asset_execution_contract.py
    cp ${base_helper_py} staged/bin/m33_asset_manifest_contract.py
    cp ${asset_contract_json} staged/conf/m33_asset_manifest_contract.json
    cp ${amendment_json} staged/conf/m33_asset_execution_amendment.json
    cp ${config_nf} staged/conf/m33_asset_execution_contract.config
    cp ${module_nf} staged/modules/33_ASSET_EXECUTION_CONTRACT.nf
    cp ${workflow_nf} staged/workflows/m33_asset_execution_contract.nf
    cp ${contract_tests} staged/tests/test_m33_asset_execution_contract.py
    cp ${nextflow_tests} staged/tests/test_m33_asset_execution_nextflow.py
    PYTHONPATH=staged/bin python3 -m unittest discover -s staged/tests -p 'test_m33_asset_execution_*.py'
    PYTHONPATH=staged/bin python3 staged/bin/m33_asset_execution_contract.py \
      --asset-contract staged/conf/m33_asset_manifest_contract.json \
      --amendment staged/conf/m33_asset_execution_amendment.json \
      --source-auth ${source_auth} \
      --staged-source bin/m33_asset_execution_source_auth.py=staged/bin/m33_asset_execution_source_auth.py \
      --staged-source bin/m33_asset_execution_contract.py=staged/bin/m33_asset_execution_contract.py \
      --staged-source bin/m33_asset_manifest_contract.py=staged/bin/m33_asset_manifest_contract.py \
      --staged-source conf/m33_asset_manifest_contract.json=staged/conf/m33_asset_manifest_contract.json \
      --staged-source conf/m33_asset_execution_amendment.json=staged/conf/m33_asset_execution_amendment.json \
      --staged-source conf/m33_asset_execution_contract.config=staged/conf/m33_asset_execution_contract.config \
      --staged-source modules/33_ASSET_EXECUTION_CONTRACT.nf=staged/modules/33_ASSET_EXECUTION_CONTRACT.nf \
      --staged-source workflows/m33_asset_execution_contract.nf=staged/workflows/m33_asset_execution_contract.nf \
      --staged-source tests/test_m33_asset_execution_contract.py=staged/tests/test_m33_asset_execution_contract.py \
      --staged-source tests/test_m33_asset_execution_nextflow.py=staged/tests/test_m33_asset_execution_nextflow.py \
      --git-commit ${git_commit} --repository-root ${repository_root} \
      --nextflow-version ${nextflow_version} \
      --output m33_asset_execution_contract.receipt.json
    """
}
