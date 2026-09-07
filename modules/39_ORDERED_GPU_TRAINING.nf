nextflow.enable.dsl = 2

process M39_ORDERED_GPU_TRAINING {
    tag group_id
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false

    input:
    tuple val(group_id), path(plan_files, stageAs: 'training-plan/*')
    path train_store, stageAs: 'train-store'
    path select_store, stageAs: 'select-store'
    path development, stageAs: 'development.npz'
    path source_seal, stageAs: 'source-seal.json'
    path sources

    output:
    path "training-${group_id}", emit: results

    script:
    """
    umask 077
    CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCHINDUCTOR_CACHE_DIR=/tmp/m39-torch-cache \\
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
      python3 m39_ordered_gpu_worker.py --train-store '${train_store}' --select-store '${select_store}' \\
      --development '${development}' --plan training-plan/plan.json --source-seal '${source_seal}' \\
      --group-id '${group_id}' --outdir 'training-${group_id}' --preserve-failure-output
    """

    stub:
    """
    mkdir 'training-${group_id}'
    printf '%s\\n' 'STUB_ONLY_NOT_TRAINING' > 'training-${group_id}/STUB_ONLY.txt'
    printf '%s\\n' '{"status":"STUB_ONLY_NOT_TRAINING","group_id":"${group_id}","SCORE_opened":false}' > 'training-${group_id}/group.completion.json'
    """
}
