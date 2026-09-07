nextflow.enable.dsl = 2

process M39_GPU_SERIAL_PROFILE {
    tag 'six-frozen-cases-one-l4'
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false

    input:
    path store, stageAs: 'ordered-store'
    path parent_receipt, stageAs: 'parent-receipt.json'
    path folds, stageAs: 'historical-folds.npz'
    path profile_config, stageAs: 'profile-config.json'
    path source_seal, stageAs: 'source-seal.json'
    path sources
    val source_commit

    output:
    path 'gpu-profile', emit: profiles

    script:
    """
    umask 077
    CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCHINDUCTOR_CACHE_DIR=/tmp/m39-torch-cache \\
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
      python3 m39_gpu_serial_profile.py --store-dir '${store}' --parent-receipt '${parent_receipt}' \\
      --folds '${folds}' --profile-config '${profile_config}' --source-seal '${source_seal}' \\
      --output-dir gpu-profile --source-commit '${source_commit}' --max-seconds ${params.m39_max_seconds}
    """

    stub:
    """
    mkdir gpu-profile
    printf '%s\\n' 'STUB_ONLY_NOT_A_GPU_RESULT' > gpu-profile/STUB_ONLY.txt
    """
}
