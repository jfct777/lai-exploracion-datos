nextflow.enable.dsl=2

process M33_M0_FACTORIZED_LAZY_TECHNICAL_KAT {
    tag { "${root_label}_factorized_lazy_technical_kat" }
    publishDir { "${params.m33_lazy_kat_results_dir}/${params.m33_lazy_kat_run_id}" }, mode: 'copy', overwrite: false
    cpus 2
    memory '2 GB'
    time '30m'

    input:
    tuple val(root_label), val(root_seed), path(technical_dir), path(verify_receipt), path(genetic_map)
    path kat_py
    path materialize_py
    path m0_contract_py
    path ordered_linear_py
    path source_auth
    val implementation_commit

    output:
    path "${root_label}.factorized_lazy_technical_kat.receipt.json", emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${kat_py} staged/bin/m33_m0_factorized_lazy_technical_kat.py
    cp ${materialize_py} staged/bin/m33_materialize.py
    cp ${m0_contract_py} staged/bin/m33_m0_contract.py
    cp ${ordered_linear_py} staged/bin/m31_ordered_linear.py
    PYTHONPATH=staged/bin python3 staged/bin/m33_m0_factorized_lazy_technical_kat.py \
      --root-label ${root_label} --root-seed ${root_seed} \
      --technical-dir ${technical_dir} \
      --independent-verify-receipt ${verify_receipt} \
      --genetic-map ${genetic_map} --marker-count ${params.m33_lazy_kat_marker_count} \
      --source-auth ${source_auth} --implementation-commit ${implementation_commit} \
      --source-root staged \
      --output ${root_label}.factorized_lazy_technical_kat.receipt.json
    """
}
