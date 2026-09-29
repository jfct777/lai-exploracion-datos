nextflow.enable.dsl=2

// Descriptive annotation of saved graphs; no clustering or genetic estimates.
process SUMMARIZE_M165_METADATA {
    tag 'm165_metadata_saved_partitions'
    cpus 1
    memory '4 GB'
    time '15m'
    cache false
    stageInMode 'copy'
    publishDir params.m165_spectral_output_dir, mode: 'copy', overwrite: false
    input:
    path results, stageAs: 'input_results'
    path metadata, stageAs: 'metadata.tsv'
    path scripts
    val settings
    output:
    path 'metadata_analysis', emit: analysis
    script:
    def quote = { value -> "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    def settingsText = quote(groovy.json.JsonOutput.toJson(settings))
    """
    set -euo pipefail
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 m165_metadata_summary.py --results-dir input_results --metadata-file metadata.tsv \\
      --sample-column ${quote(params.m165_spectral_metadata_sample_column)} \\
      --settings-json-text ${settingsText} --output-dir metadata_analysis
    """
}

process SUMMARIZE_M165_GRAPH_KINSHIP {
    tag 'm165_existing_pcrelate_saved_edges'
    cpus 1
    memory '4 GB'
    time '15m'
    cache false
    stageInMode 'copy'
    publishDir params.m165_spectral_output_dir, mode: 'copy', overwrite: false
    input:
    path results, stageAs: 'input_results'
    path kinship, stageAs: 'pcrelate.tsv'
    path scripts
    val expected_sha256
    val threshold
    output:
    path 'kinship_analysis', emit: analysis
    script:
    def quote = { value -> "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    """
    set -euo pipefail
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 m165_graph_kinship.py --results-dir input_results --pcrelate-file pcrelate.tsv \\
      --expected-sha256 ${quote(expected_sha256)} --threshold ${quote(threshold)} --output-dir kinship_analysis
    """
}
