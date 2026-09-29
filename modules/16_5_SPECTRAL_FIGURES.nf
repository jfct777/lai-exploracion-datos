nextflow.enable.dsl=2

// Display saved graphs with the workflow's existing spectral/static functions.
// No segment detection, graph filtering, Leiden fitting or NMF is invoked.
process RENDER_M165_SPECTRAL_FIGURES {
    tag 'm165_saved_spectral_umap'
    label 'm165_spectral_figures'
    cpus 2
    memory '8 GB'
    time '30m'
    maxForks 1
    // A directory path does not reliably invalidate on nested NPZ changes.
    // Always revalidate saved-coordinate payloads, including on -resume.
    cache false
    stageInMode 'copy'
    publishDir params.m165_spectral_output_dir, mode: 'copy', overwrite: false

    input:
    path results_dir, stageAs: 'input_results'
    path scripts
    path metadata, stageAs: 'metadata.tsv'
    path coordinates, stageAs: 'saved_coordinates'
    val settings

    output:
    path 'figures', emit: figures

    script:
    def quote = { value -> "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    def settingsText = quote(groovy.json.JsonOutput.toJson(settings))
    def metadataFlags = metadata ? "--metadata-file metadata.tsv --metadata-sample-column ${quote(params.m165_spectral_metadata_sample_column)} --metadata-color-column ${quote(params.m165_spectral_metadata_color_column)}" : ''
    def coordinateFlags = coordinates ? "--coordinates-source-dir saved_coordinates --coordinates-source-manifest-sha256 ${quote(params.m165_spectral_coordinates_source_manifest_sha256)}" : ''
    """
    set -euo pipefail
    test ! -L '${results_dir}' || { echo 'Inputs must be task-local copies' >&2; exit 1; }
    if find '${results_dir}' -type l -print -quit | grep -q .; then
        echo 'Unresolved input symlink; refusing to change permissions' >&2
        exit 1
    fi
    chmod -R a-w '${results_dir}'
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    export NUMBA_CACHE_DIR="\$PWD/.numba"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
    python3 m165_spectral_figures.py --results-dir '${results_dir}' --output-dir figures \\
      --parameters-json-text ${settingsText} ${metadataFlags} ${coordinateFlags}
    """
}
