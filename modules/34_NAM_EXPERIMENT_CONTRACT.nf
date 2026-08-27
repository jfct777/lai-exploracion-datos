nextflow.enable.dsl=2

process M34_NAM_VALIDATE_EXPERIMENT_CONTRACT {
    tag 'm34_exact_experiment_contract'
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/contract"
    }, mode: 'copy', overwrite: false,
       saveAs: { filename -> filename == 'm34_experiment_contract.receipt.json' ? filename : null }
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_contract_cpus }
    memory { params.m34_inputs_contract_memory }
    time { params.m34_inputs_contract_time }

    input:
    path experimentContract
    path validatorPy
    val expectedSha256
    val experimentRoot
    val targetSize

    output:
    tuple path(experimentContract),
          path('m34_experiment_contract.receipt.json'),
          val(expectedSha256),
          emit: validated

    script:
    """
    set -euo pipefail
    python3 ${validatorPy} \
      --contract ${experimentContract} \
      --expected-sha256 ${expectedSha256} \
      --root ${experimentRoot} \
      --target-size ${targetSize} \
      --receipt m34_experiment_contract.receipt.json
    """

    stub:
    """
    set -euo pipefail
    touch m34_experiment_contract.receipt.json
    """
}
