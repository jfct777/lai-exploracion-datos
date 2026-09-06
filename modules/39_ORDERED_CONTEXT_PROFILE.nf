nextflow.enable.dsl=2

process M39_ORDERED_CONTEXT_PROFILE {
    tag 'chr22-inner-TRAIN-ordered-profile'
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false
    container params.m39_image
    cpus 2
    memory '6 GB'
    time '1h'
    maxForks 1
    input:
    path inputs
    path bridge_contract
    path profile_contract
    path sources
    output:
    path 'ordered-profile', emit: materialized
    script:
    """
    OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} PYTHONPATH=. \\
      python3 m39_profile_ordered.py --bridge-config '${bridge_contract}' \\
      --profile-config '${profile_contract}' --input-dir . --outdir ordered-profile
    """
}
