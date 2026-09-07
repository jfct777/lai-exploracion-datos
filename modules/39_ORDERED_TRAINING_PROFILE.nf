nextflow.enable.dsl = 2

process M39_ORDERED_TRAINING_PROFILE {
    tag "technical-${case_id}"
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false
    container params.m39_image
    cpus 2
    memory '8 GB'
    time '20m'
    maxForks 2

    input:
    val case_id
    val store_dir
    path parent_receipt, stageAs: 'parent-receipt.json'
    path folds, stageAs: 'historical-folds.npz'
    path profile_config, stageAs: 'profile-config.json'
    path sources
    val source_commit

    output:
    tuple val(case_id), path("${case_id}"), emit: profiles

    script:
    """
    umask 077
    OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus} \\
      TORCHINDUCTOR_CACHE_DIR=/tmp/m39-torch-cache \\
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 m39_profile_ordered_training.py \\
      --store-dir '/m39-ordered-store' --parent-receipt '${parent_receipt}' \\
      --folds '${folds}' --profile-config '${profile_config}' --case-id '${case_id}' \\
      --output-dir '${case_id}' --source-commit '${source_commit}'
    """
}
