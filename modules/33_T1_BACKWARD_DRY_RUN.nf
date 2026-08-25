nextflow.enable.dsl=2

process M33_T1_PREFLIGHT {
    publishDir { "${params.m33_t1_results_dir}/${params.m33_t1_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '10m'

    input:
    path preflight_py
    path contract
    path source_auth
    val implementation_commit
    val oci_image
    path t0b_aggregate
    path t0b_child_receipts

    output:
    path 'm33_t1.preflight.receipt.json', emit: receipt

    script:
    def childArgs = t0b_child_receipts.collect { "--t0b-child-receipt ${it}" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin staged/conf
    cp ${preflight_py} staged/bin/m33_t1_preflight.py
    cp ${contract} staged/conf/m33_t1_contract.json
    python3 staged/bin/m33_t1_preflight.py \
      --contract staged/conf/m33_t1_contract.json \
      --source-auth ${source_auth} --source-root staged \
      --implementation-commit ${implementation_commit} --oci-image '${oci_image}' \
      --t0b-aggregate ${t0b_aggregate} ${childArgs} \
      --output m33_t1.preflight.receipt.json
    """
}

process M33_T1_BACKWARD {
    tag { "${model_family}_rep${repetition}" }
    publishDir { "${params.m33_t1_results_dir}/${params.m33_t1_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '8 GB'
    time '30m'
    maxForks 1

    input:
    tuple val(model_family), val(repetition)
    path preflight_receipt
    path backward_py
    path preflight_py
    path models_py
    path source_auth
    path pre4_contract
    path contract
    val implementation_commit
    val oci_image

    output:
    path "${model_family}.rep${repetition}.t1.receipt.json", emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin staged/conf
    cp ${backward_py} staged/bin/m33_t1_backward.py
    cp ${preflight_py} staged/bin/m33_t1_preflight.py
    cp ${models_py} staged/bin/m33_t0a_models.py
    cp ${pre4_contract} staged/conf/m33_pre4_preregistration.json
    cp ${contract} staged/conf/m33_t1_contract.json
    PYTHONPATH=staged/bin python3 staged/bin/m33_t1_backward.py \
      --model-family ${model_family} --repetition ${repetition} \
      --preflight-receipt ${preflight_receipt} \
      --source-auth ${source_auth} --source-root staged \
      --pre4-contract staged/conf/m33_pre4_preregistration.json \
      --contract staged/conf/m33_t1_contract.json \
      --implementation-commit ${implementation_commit} --oci-image '${oci_image}' \
      --output ${model_family}.rep${repetition}.t1.receipt.json
    """
}

process M33_T1_COMPARE {
    publishDir { "${params.m33_t1_results_dir}/${params.m33_t1_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    path receipts
    path preflight_receipt
    path compare_py
    path preflight_py
    path source_auth
    path contract
    val implementation_commit
    val oci_image

    output:
    path 'm33_t1.backward_dry_run.receipt.json', emit: receipt

    script:
    def receiptArgs = receipts.collect { "--receipt ${it}" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${compare_py} staged/bin/m33_t1_compare.py
    cp ${preflight_py} staged/bin/m33_t1_preflight.py
    mkdir -p staged/conf
    cp ${contract} staged/conf/m33_t1_contract.json
    PYTHONPATH=staged/bin python3 staged/bin/m33_t1_compare.py ${receiptArgs} \
      --source-auth ${source_auth} --source-root staged \
      --implementation-commit ${implementation_commit} --oci-image '${oci_image}' \
      --contract staged/conf/m33_t1_contract.json \
      --preflight-receipt ${preflight_receipt} \
      --output m33_t1.backward_dry_run.receipt.json
    """
}
