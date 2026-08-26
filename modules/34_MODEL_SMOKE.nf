nextflow.enable.dsl=2

process M34_MODEL_SMOKE {
    tag { "${family}_${configId}" }
    publishDir { "${params.m34_smoke_results_dir}/${params.m34_smoke_run_id}" },
               mode: 'copy', overwrite: false
    cpus { params.m34_smoke_cpus }
    memory { params.m34_smoke_memory }
    time { params.m34_smoke_time }
    maxForks params.m34_smoke_max_forks

    input:
    tuple val(family), val(configId)
    path contract
    path smokePy
    path modelsPy
    path sweepPy

    output:
    path "${family}.${configId}.smoke.json", emit: receipts

    script:
    """
    set -euo pipefail
    mkdir -p staged/bin
    export USER=m34 LOGNAME=m34 TORCHINDUCTOR_CACHE_DIR=\$PWD/.torch-cache
    cp ${smokePy} staged/bin/m34_model_smoke.py
    cp ${modelsPy} staged/bin/m34_models.py
    cp ${sweepPy} staged/bin/m34_adaptive_sweep.py
    PYTHONPATH=staged/bin python3 staged/bin/m34_model_smoke.py \
      --contract ${contract} \
      --family ${family} --config-id ${configId} \
      --contract-stage triage --arm both \
      --channels ${params.m34_smoke_channels} \
      --ancestries ${params.m34_smoke_ancestries} \
      --batch-size ${params.m34_smoke_batch_size} \
      --context-length ${params.m34_smoke_context_length} \
      --haplotypes ${params.m34_smoke_haplotypes} \
      --output ${family}.${configId}.smoke.json
    """
}
