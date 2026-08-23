nextflow.enable.dsl=2

process M33_T0A_FORWARD {
    tag { "${root_label}_${model_family}_rep${repetition}" }
    publishDir { "${params.m33_t0a_results_dir}/${params.m33_t0a_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '4 GB'
    time '45m'

    input:
    tuple val(root_label), val(root_seed), val(model_family), val(repetition),
          path(technical_dir), path(verify_receipt), path(genetic_map)
    path forward_py
    path models_py
    path materialize_py
    path technical_kat_py
    path m0_contract_py
    path ordered_linear_py
    path source_auth
    path pre4_contract
    val implementation_commit
    val oci_image

    output:
    path "${root_label}.${model_family}.rep${repetition}.t0a.receipt.json", emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin staged/conf
    cp ${forward_py} staged/bin/m33_t0a_forward.py
    cp ${models_py} staged/bin/m33_t0a_models.py
    cp ${materialize_py} staged/bin/m33_materialize.py
    cp ${technical_kat_py} staged/bin/m33_m0_factorized_lazy_technical_kat.py
    cp ${m0_contract_py} staged/bin/m33_m0_contract.py
    cp ${ordered_linear_py} staged/bin/m31_ordered_linear.py
    cp ${pre4_contract} staged/conf/m33_pre4_preregistration.json
    PYTHONPATH=staged/bin python3 staged/bin/m33_t0a_forward.py \
      --root-label ${root_label} --root-seed ${root_seed} \
      --technical-dir ${technical_dir} \
      --independent-verify-receipt ${verify_receipt} \
      --genetic-map ${genetic_map} --model-family ${model_family} \
      --repetition ${repetition} --marker-count ${params.m33_t0a_marker_count} \
      --source-auth ${source_auth} --source-root staged \
      --pre4-contract staged/conf/m33_pre4_preregistration.json \
      --implementation-commit ${implementation_commit} --oci-image '${oci_image}' \
      --output ${root_label}.${model_family}.rep${repetition}.t0a.receipt.json
    """
}

process M33_T0A_COMPARE {
    publishDir { "${params.m33_t0a_results_dir}/${params.m33_t0a_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    path receipts
    path stress_receipts
    path compare_py
    path source_auth
    val implementation_commit
    val oci_image

    output:
    path 'm33_t0a.cross_process.receipt.json', emit: receipt

    script:
    def receiptArgs = receipts.collect { "--receipt ${it}" }.join(' ')
    def stressArgs = stress_receipts.collect { "--stress-receipt ${it}" }.join(' ')
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${compare_py} staged/bin/m33_t0a_compare.py
    python3 staged/bin/m33_t0a_compare.py ${receiptArgs} ${stressArgs} \
      --source-auth ${source_auth} --source-root staged \
      --implementation-commit ${implementation_commit} \
      --oci-image '${oci_image}' \
      --output m33_t0a.cross_process.receipt.json
    """
}

process M33_T0A_STRESS {
    tag { "${model_family}_stress_rep${repetition}" }
    publishDir { "${params.m33_t0a_results_dir}/${params.m33_t0a_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '4 GB'
    time '45m'

    input:
    tuple val(model_family), val(repetition)
    path stress_py
    path forward_py
    path models_py
    path materialize_py
    path technical_kat_py
    path m0_contract_py
    path ordered_linear_py
    path source_auth
    path pre4_contract
    val implementation_commit
    val oci_image

    output:
    path "${model_family}.rep${repetition}.t0a_stress.receipt.json", emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin staged/conf
    cp ${stress_py} staged/bin/m33_t0a_stress.py
    cp ${forward_py} staged/bin/m33_t0a_forward.py
    cp ${models_py} staged/bin/m33_t0a_models.py
    cp ${materialize_py} staged/bin/m33_materialize.py
    cp ${technical_kat_py} staged/bin/m33_m0_factorized_lazy_technical_kat.py
    cp ${m0_contract_py} staged/bin/m33_m0_contract.py
    cp ${ordered_linear_py} staged/bin/m31_ordered_linear.py
    cp ${pre4_contract} staged/conf/m33_pre4_preregistration.json
    PYTHONPATH=staged/bin python3 staged/bin/m33_t0a_stress.py \
      --model-family ${model_family} --repetition ${repetition} \
      --source-auth ${source_auth} --source-root staged \
      --pre4-contract staged/conf/m33_pre4_preregistration.json \
      --implementation-commit ${implementation_commit} --oci-image '${oci_image}' \
      --output ${model_family}.rep${repetition}.t0a_stress.receipt.json
    """
}
