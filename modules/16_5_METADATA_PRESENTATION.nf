nextflow.enable.dsl=2

// Supplemental legends/captions from authenticated summaries; geometry is read-only.
process PRESENT_M165_METADATA {
    tag 'm165_correspondence_and_interpretation'
    cpus 2
    memory '8 GB'
    time '30m'
    maxForks 1
    cache false
    stageInMode 'copy'
    publishDir params.m165_metadata_presentation_output_dir, mode: 'copy', overwrite: false
    input:
    path figures, stageAs: 'saved_figures'
    path metadata, stageAs: 'metadata_analysis'
    path kinship, stageAs: 'kinship_analysis'
    path scripts
    output:
    path 'interpretation', emit: interpretation
    script:
    """
    set -euo pipefail
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 m165_metadata_presentation.py --figures-dir saved_figures \\
      --metadata-dir metadata_analysis --kinship-dir kinship_analysis --output-dir interpretation
    """
}
