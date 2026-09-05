nextflow.enable.dsl=2

process M39_CAPACITY_CASE {
    tag "${variant}-w${width}-lr${rate}"
    publishDir params.m39_output_dir, mode: 'copy', overwrite: false
    container params.m39_image
    cpus 2
    memory '3 GB'
    time '15m'
    maxForks 2
    input:
    tuple val(variant), val(width), val(rate)
    path sources
    output:
    path "${variant}-w${width}-lr${rate}", emit: evaluated
    script:
    """
    PYTHONPATH=. python3 m39_capacity_screen.py \\
      --variant '${variant}' --width '${width}' --learning-rate '${rate}' \\
      --steps '${params.m39_capacity_steps}' --seed '${params.m39_capacity_seed}' \\
      --outdir '${variant}-w${width}-lr${rate}'
    """
}

workflow {
    if (!params.m39_output_dir) error '--m39_output_dir is required'
    if (file(params.m39_output_dir).exists() && !workflow.resume)
        error 'Use a fresh output directory'
    if (params.m39_capacity_steps < 1 || params.m39_capacity_steps > 2000)
        error 'Capacity steps outside the diagnostic budget'
    if (!(params.m39_capacity_widths instanceof List) ||
        !params.m39_capacity_widths.every { it in [16, 32, 64] })
        error 'Capacity widths must be 16, 32 or 64'
    if (!(params.m39_capacity_rates instanceof List) ||
        !params.m39_capacity_rates.every { it in [0.001, 0.003] })
        error 'Capacity learning rates must be 0.001 or 0.003'
    def cases = ['gated_deepset', 'carrier_cross_attention', 'bilinear_context'].collectMany { variant ->
        params.m39_capacity_widths.collectMany { width ->
            params.m39_capacity_rates.collect { rate -> tuple(variant, width, rate) }
        }
    }
    if (cases.size() > 12) error 'At most twelve diagnostic configurations per run'
    def repoDir = projectDir.resolve('..')
    def sources = ['m39_capacity_screen.py', 'm39_carrier_models.py'].collect {
        file("${repoDir}/bin/${it}", checkIfExists: true)
    }
    M39_CAPACITY_CASE(channel.fromList(cases), channel.value(sources))
}
