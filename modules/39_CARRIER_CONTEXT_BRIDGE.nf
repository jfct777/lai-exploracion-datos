nextflow.enable.dsl=2

process M39_CARRIER_CONTEXT_BRIDGE {
    tag 'chr22-FIT-carrier-context'
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false
    container params.m39_image
    cpus 2
    memory '6 GB'
    time '30m'
    maxForks 1
    input:
    path inputs
    path contract
    path sources
    output:
    path 'bridge', emit: materialized
    script:
    """
    PYTHONPATH=. python3 m39_carrier_context.py \\
      --config '${contract}' --input-dir . --outdir bridge
    """
}
