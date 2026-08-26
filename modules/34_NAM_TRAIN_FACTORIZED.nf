nextflow.enable.dsl=2

process M34_NAM_TRAIN_FACTORIZED {
    tag { "m34_train_${family}_${configId}_${arm}" }
    publishDir {
        "${params.m34_inputs_results_dir}/${params.m34_inputs_run_id}/models/${family}/${configId}/${arm}"
    }, mode: 'copy', overwrite: false
    container params.m34_inputs_pytorch_image
    containerOptions { "--network none --user ${params.m34_inputs_container_user}" }
    cpus { params.m34_inputs_train_cpus }
    memory { params.m34_inputs_train_memory }
    time { params.m34_inputs_train_time }
    maxForks params.m34_inputs_train_max_forks
    stageInMode 'symlink'

    input:
    tuple val(family), val(configId), val(arm), val(taskBase64)
    path factorBundle
    path adaptiveContract
    path trainerPy
    path adaptiveSweepPy
    path materializePy
    path modelsPy
    path packedTrainPy
    path bridgeCorePy
    path m33MaterializePy
    path m33ContractPy

    output:
    tuple val(family), val(configId), val(arm),
          path("m34_train_${family}_${configId}_${arm}/valid.prediction.npz"),
          path("m34_train_${family}_${configId}_${arm}/train.receipt.json"),
          path("m34_train_${family}_${configId}_${arm}/model.pt"),
          emit: trained

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    cp ${trainerPy} staged/bin/m34_train_factorized.py
    cp ${adaptiveSweepPy} staged/bin/m34_adaptive_sweep.py
    cp ${materializePy} staged/bin/m34_materialize.py
    cp ${modelsPy} staged/bin/m34_models.py
    cp ${packedTrainPy} staged/bin/m34_train_packed.py
    cp ${bridgeCorePy} staged/bin/m33_safe_bridge_core.py
    cp ${m33MaterializePy} staged/bin/m33_materialize.py
    cp ${m33ContractPy} staged/bin/m33_m0_contract.py
    python3 -c "import base64; open('task.json','xb').write(base64.b64decode('${taskBase64}'))"
    export USER=m34 LOGNAME=m34 TORCHINDUCTOR_CACHE_DIR=\$PWD/.torch-cache
    PYTHONPATH=staged/bin python3 staged/bin/m34_train_factorized.py \
      --contract ${adaptiveContract} \
      --manifest ${factorBundle}/factorized.manifest.json \
      --task task.json \
      --outdir m34_train_${family}_${configId}_${arm} \
      --device ${params.m34_inputs_train_device} \
      --threads ${params.m34_inputs_train_cpus} \
      --sample-shard-size ${params.m34_inputs_train_sample_shard_size} \
      --marker-shard-size ${params.m34_inputs_train_marker_shard_size} \
      --maximum-rows-per-batch ${params.m34_inputs_train_maximum_rows_per_batch} \
      --maximum-tokens-per-batch ${params.m34_inputs_train_maximum_tokens_per_batch} \
      --validation-every ${params.m34_inputs_train_validation_every}
    """

    stub:
    """
    set -euo pipefail
    mkdir -p m34_train_${family}_${configId}_${arm}
    touch \
      m34_train_${family}_${configId}_${arm}/valid.prediction.npz \
      m34_train_${family}_${configId}_${arm}/train.receipt.json \
      m34_train_${family}_${configId}_${arm}/model.pt
    """
}
