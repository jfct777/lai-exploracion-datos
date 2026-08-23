nextflow.enable.dsl=2

process M33_VALIDATE_FACTORIZED_LAZY_CONTRACT {
    tag 'm33_factorized_lazy_contract_and_synthetic_equivalence'
    publishDir { "${params.m33_factorized_lazy_results_dir}/${params.m33_factorized_lazy_run_id}" }, mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '10m'

    input:
    path validator_py
    path materialize_py
    path m0_contract_py
    path amendment_json
    path materializer_json
    path sanitized_f0_json
    path pre4_json
    path amendment_tests
    path materialize_tests

    output:
    path 'm33_m0_factorized_lazy_contract.receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin staged/conf staged/tests
    cp ${validator_py} staged/bin/m33_m0_factorized_lazy_amendment.py
    cp ${materialize_py} staged/bin/m33_materialize.py
    cp ${m0_contract_py} staged/bin/m33_m0_contract.py
    cp ${amendment_json} staged/conf/m33_m0_factorized_lazy_amendment_contract.json
    cp ${materializer_json} staged/conf/m33_m0_materializer_contract.json
    cp ${sanitized_f0_json} staged/conf/m33_m0_f0_sanitized_amendment_contract.json
    cp ${pre4_json} staged/conf/m33_pre4_preregistration.json
    cp ${amendment_tests} staged/tests/test_m33_m0_factorized_lazy_amendment.py
    cp ${materialize_tests} staged/tests/test_m33_materialize.py
    PYTHONPATH=staged/bin python3 -m unittest discover -s staged/tests -p 'test_m33_*factorized_lazy*.py'
    PYTHONPATH=staged/bin python3 -m unittest discover -s staged/tests -p 'test_m33_materialize.py'
    PYTHONPATH=staged/bin python3 staged/bin/m33_m0_factorized_lazy_amendment.py \
      --contract staged/conf/m33_m0_factorized_lazy_amendment_contract.json \
      --materializer-contract staged/conf/m33_m0_materializer_contract.json \
      --sanitized-f0-contract staged/conf/m33_m0_f0_sanitized_amendment_contract.json \
      --pre4-preregistration staged/conf/m33_pre4_preregistration.json \
      --output m33_m0_factorized_lazy_contract.receipt.json
    """
}
