nextflow.enable.dsl=2

process ANALYZE_INDIVIDUAL_SNP_DISTANCE_MODES {
    tag "chr${chr}"

    publishDir "${params.distance_mode_results_dir}", mode: 'copy', saveAs: { filename -> filename }

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(individual_distance_modes_py)
    val(sample_ids_payload_b64)

    output:
    path "per_individual/*/*", emit: per_individual
    path "per_chr/*", emit: per_chr
    path "summary/*", emit: summary_files

    script:
    def sample_setup = sample_ids_payload_b64 ? """
    printf '%s' '${sample_ids_payload_b64}' | base64 -d > module13_selected_samples.txt
    """ : ""
    def sample_arg = sample_ids_payload_b64 ? "--sample-ids-file module13_selected_samples.txt" : ""
    """
    set -euo pipefail

    mkdir -p per_individual per_chr summary
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    mkdir -p "\$MPLCONFIGDIR"
    ${sample_setup}

    python3 ${individual_distance_modes_py} \
      --mode scan \
      --input ${vcf_gz} \
      --input-format ${params.distance_mode_input_format} \
      --chr ${chr} \
      ${sample_arg} \
      ${params.distance_mode_max_samples != null ? "--max-samples ${params.distance_mode_max_samples}" : ""} \
      --min-carrier-snps-per-individual-chr ${params.distance_mode_min_carrier_snps_per_individual_chr} \
      --distance-units ${params.distance_mode_distance_units} \
      --pair-selection ${params.distance_mode_pair_selection} \
      ${params.distance_mode_max_pair_distance_bp != null ? "--max-pair-distance-bp ${params.distance_mode_max_pair_distance_bp}" : ""} \
      --nearest-neighbor-k ${params.distance_mode_nearest_neighbor_k} \
      --pair-block-size-snps ${params.distance_mode_pair_block_size_snps} \
      --hist-min-log10-bp ${params.distance_mode_hist_min_log10_bp} \
      --hist-max-log10-bp ${params.distance_mode_hist_max_log10_bp} \
      --hist-n-bins ${params.distance_mode_hist_n_bins} \
      ${params.distance_mode_abort_if_pairs_exceed != null ? "--abort-if-pairs-exceed ${params.distance_mode_abort_if_pairs_exceed}" : ""} \
      --plot-hist ${params.distance_mode_plot_hist} \
      --plot-dpi ${params.distance_mode_plot_dpi} \
      --plot-width-inches ${params.distance_mode_plot_width_inches} \
      --plot-height-inches ${params.distance_mode_plot_height_inches} \
      --plot-palette ${params.distance_mode_plot_palette} \
      --plot-font-family '${params.distance_mode_plot_font_family}' \
      --plot-export-pdf ${params.distance_mode_plot_export_pdf} \
      --plot-export-svg ${params.distance_mode_plot_export_svg} \
      --output-dir ./
    """
}

process AGGREGATE_INDIVIDUAL_SNP_DISTANCE_MODES {
    tag "aggregate"

    publishDir "${params.distance_mode_results_dir}", mode: 'copy', saveAs: { filename -> filename }

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple path(individual_summary_files), path(cohort_summary_files), path(individual_distance_modes_py)

    output:
    path "summary/*", emit: summary
    path "report.html", emit: report

    script:
    def individual_args = individual_summary_files.collect { f -> "--individual-summary ${f}" }.join(' ')
    def cohort_args = cohort_summary_files.collect { f -> "--cohort-summary ${f}" }.join(' ')
    """
    set -euo pipefail

    mkdir -p summary
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    mkdir -p "\$MPLCONFIGDIR"

    python3 ${individual_distance_modes_py} \
      --mode aggregate \
      ${individual_args} \
      ${cohort_args} \
      --input-dir ${params.distance_mode_input_dir} \
      --output-dir ./
    """
}
