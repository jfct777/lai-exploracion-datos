nextflow.enable.dsl=2

process M33_REF_LABEL_SHAM_KAT {
    publishDir { "${params.m33_ref_label_sham_results_dir}/${params.m33_ref_label_sham_run_id}" },
               mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '10m'
    maxForks 1
    cache false

    input:
    path runner
    path core
    path source_auth_py
    path contract
    path pre4_contract
    path m0_contract
    path config_nf
    path module_nf
    path workflow_nf
    path test_py
    path source_auth
    val implementation_commit
    val oci_image

    output:
    path 'm33_ref_label_sham.kat.receipt.json', emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin staged/conf staged/modules staged/workflows staged/tests
    cp ${runner} staged/bin/m33_ref_label_sham_kat.py
    cp ${core} staged/bin/m33_safe_bridge_core.py
    cp ${source_auth_py} staged/bin/m33_ref_label_sham_source_auth.py
    cp ${contract} staged/conf/m33_ref_label_sham_contract.json
    cp ${pre4_contract} staged/conf/m33_pre4_preregistration.json
    cp ${m0_contract} staged/conf/m33_m0_materializer_contract.json
    cp ${config_nf} staged/conf/m33_ref_label_sham.config
    cp ${module_nf} staged/modules/33_REF_LABEL_SHAM_KAT.nf
    cp ${workflow_nf} staged/workflows/m33_ref_label_sham.nf
    cp ${test_py} staged/tests/test_m33_ref_label_sham.py
    PYTHONPATH=staged/bin python3 staged/bin/m33_ref_label_sham_kat.py \
      --contract staged/conf/m33_ref_label_sham_contract.json \
      --pre4-preregistration staged/conf/m33_pre4_preregistration.json \
      --m0-materializer-contract staged/conf/m33_m0_materializer_contract.json \
      --source-auth ${source_auth} --source-root staged \
      --implementation-commit ${implementation_commit} --oci-image '${oci_image}' \
      --output m33_ref_label_sham.kat.receipt.json
    """
}
