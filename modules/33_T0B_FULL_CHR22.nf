nextflow.enable.dsl=2

process M33_T0B_PREFLIGHT {
    publishDir { "${params.m33_t0b_results_dir}/${params.m33_t0b_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '10m'

    input:
    path preflight_py
    path contract
    path source_auth
    val implementation_commit
    path t0a_aggregate
    path t0a_source_auth
    path t0a_child_receipts
    path root17_technical_dir
    path root18_technical_dir
    path root17_verify
    path root18_verify
    path root17_map, stageAs: 'root17-map/*'
    path root18_map, stageAs: 'root18-map/*'

    output:
    path 'm33_t0b.preflight.receipt.json', emit: receipt

    script:
    def childArgs = t0a_child_receipts.collect { "--t0a-child-receipt ${it}" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${preflight_py} staged/bin/m33_t0b_preflight.py
    python3 staged/bin/m33_t0b_preflight.py \
      --contract ${contract} --source-auth ${source_auth} --source-root staged \
      --implementation-commit ${implementation_commit} \
      --t0a-aggregate ${t0a_aggregate} --t0a-source-auth ${t0a_source_auth} \
      ${childArgs} --root17-technical-dir ${root17_technical_dir} \
      --root18-technical-dir ${root18_technical_dir} \
      --root17-verify ${root17_verify} --root18-verify ${root18_verify} \
      --root17-map ${root17_map} --root18-map ${root18_map} \
      --output m33_t0b.preflight.receipt.json
    """
}

process M33_T0B_FORWARD {
    tag { "${root_label}_${model_family}_rep${repetition}" }
    publishDir { "${params.m33_t0b_results_dir}/${params.m33_t0b_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '6 GB'
    time '10h'
    maxForks 3

    input:
    tuple val(root_label), val(root_seed), val(model_family), val(repetition),
          path(technical_dir), path(verify_receipt), path(genetic_map)
    path preflight_receipt
    path forward_py
    path t0a_forward_py
    path models_py
    path materialize_py
    path technical_kat_py
    path m0_contract_py
    path ordered_linear_py
    path source_auth
    path pre4_contract
    path contract
    val implementation_commit
    val oci_image

    output:
    path "${root_label}.${model_family}.rep${repetition}.t0b.receipt.json", emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin staged/conf
    cp ${forward_py} staged/bin/m33_t0b_forward.py
    cp ${t0a_forward_py} staged/bin/m33_t0a_forward.py
    cp ${models_py} staged/bin/m33_t0a_models.py
    cp ${materialize_py} staged/bin/m33_materialize.py
    cp ${technical_kat_py} staged/bin/m33_m0_factorized_lazy_technical_kat.py
    cp ${m0_contract_py} staged/bin/m33_m0_contract.py
    cp ${ordered_linear_py} staged/bin/m31_ordered_linear.py
    cp ${pre4_contract} staged/conf/m33_pre4_preregistration.json
    cp ${contract} staged/conf/m33_t0b_contract.json
    PYTHONPATH=staged/bin python3 staged/bin/m33_t0b_forward.py \
      --root-label ${root_label} --root-seed ${root_seed} \
      --technical-dir ${technical_dir} \
      --independent-verify-receipt ${verify_receipt} \
      --genetic-map ${genetic_map} --model-family ${model_family} \
      --repetition ${repetition} --marker-count ${params.m33_t0b_marker_count} \
      --preflight-receipt ${preflight_receipt} \
      --source-auth ${source_auth} --source-root staged \
      --pre4-contract staged/conf/m33_pre4_preregistration.json \
      --contract staged/conf/m33_t0b_contract.json \
      --implementation-commit ${implementation_commit} --oci-image '${oci_image}' \
      --output ${root_label}.${model_family}.rep${repetition}.t0b.receipt.json
    """
}

process M33_T0B_COMPARE {
    publishDir { "${params.m33_t0b_results_dir}/${params.m33_t0b_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    path receipts
    path preflight_receipt
    path compare_py
    path source_auth
    path contract
    val implementation_commit
    val oci_image

    output:
    path 'm33_t0b.full_chr22.receipt.json', emit: receipt

    script:
    def receiptArgs = receipts.collect { "--receipt ${it}" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${compare_py} staged/bin/m33_t0b_compare.py
    python3 staged/bin/m33_t0b_compare.py ${receiptArgs} \
      --source-auth ${source_auth} --source-root staged \
      --implementation-commit ${implementation_commit} --oci-image '${oci_image}' \
      --contract ${contract} --preflight-receipt ${preflight_receipt} \
      --output m33_t0b.full_chr22.receipt.json
    """
}
