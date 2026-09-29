nextflow.enable.dsl=2

// Render saved M16.5 results only: no genotypes, community fitting or NMF.
process RENDER_M165_SWEEP_FIGURES {
    tag 'm165_saved_sweep_figures'
    label 'm165_sweep_figures'
    cpus 2
    memory '4 GB'
    time '30m'
    maxForks 1
    // A complete, task-local copy also resolves symlinks before container entry.
    stageInMode 'copy'
    publishDir params.m165_figures_output_dir, mode: 'copy', overwrite: false

    input:
    path results_dir, stageAs: 'input_results'
    path renderer, stageAs: 'm165_sweep_figures.py'

    output:
    path 'figures', emit: figures

    script:
    def flags = [dpi: '--dpi', layout_seed: '--layout-seed', layout_iterations: '--layout-iterations']
    def numeric = flags.collect { name, flag ->
        def value = params["m165_figures_${name}"]
        if (value == null || !(value.toString() ==~ /[0-9]+/) ||
            (name != 'layout_seed' && value.toString().toLong() < 1))
            error "Invalid m165_figures_${name}: expected ${name == 'layout_seed' ? 'nonnegative' : 'positive'} integer"
        "${flag} ${value}"
    }.join(' ')
    def prefix = params.m165_figures_prefix?.toString()
    if (!prefix || !(prefix ==~ /[A-Za-z0-9_-]+/))
        error 'Invalid m165_figures_prefix: use letters, digits, underscores or hyphens'
    def networkIds = params.m165_figures_network_config_ids?.toString()?.split(',') as List
    if (!networkIds || networkIds.toSet().size() != networkIds.size() ||
        !networkIds.every { it ==~ /L[0-9]+_G[0-9]+_N[0-9]+_T[0-9]+_U[0-9]+/ })
        error 'Invalid m165_figures_network_config_ids: use a comma-separated list of distinct graph IDs'
    def resolution = params.m165_figures_network_resolution
    if (resolution == null || !(resolution.toString() ==~ /[0-9]+(\.[0-9]+)?/) ||
        !Double.isFinite(resolution.toString().toDouble()) || resolution.toString().toDouble() <= 0)
        error 'Invalid m165_figures_network_resolution: expected a positive finite number'
    """
    set -euo pipefail
    test ! -L '${results_dir}' || { echo 'Figure inputs require stageInMode copy' >&2; exit 1; }
    if find '${results_dir}' -type l -print -quit | grep -q .; then
        echo 'Unresolved symlink in staged figure inputs; refusing to change permissions' >&2
        exit 1
    fi
    chmod -R a-w '${results_dir}'
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    python3 '${renderer}' --results-dir '${results_dir}' --output-dir figures \\
      --prefix '${prefix}' ${numeric} \\
      --network-config-ids '${networkIds.join(',')}' --network-resolution '${resolution}'
    """
}
