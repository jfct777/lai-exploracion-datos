nextflow.enable.dsl=2

process ANALYZE_RARE_SNP_TRACTS {
    tag "chr${chr}"

    publishDir "${params.rare_tract_results_dir}/per_chr", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(rare_snp_tract_py)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.window_scores.tsv.gz"), emit: window_scores
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.rare_scan.summary.json"), emit: scan_summaries

    script:
    def out_prefix = "dnabr.hg38.2723.chr${chr}"
    """
    set -euo pipefail

    python3 ${rare_snp_tract_py} \
      --mode scan \
      --input ${vcf_gz} \
      --input-format ${params.rare_tract_input_format} \
      --chr ${chr} \
      --maf-threshold ${params.rare_tract_maf_threshold} \
      --window-size-snps ${params.rare_tract_window_size_snps} \
      --step-size-snps ${params.rare_tract_step_size_snps} \
      --min-chrom-rare-snps ${params.rare_tract_min_chrom_rare_snps} \
      --out-window-scores ${out_prefix}.window_scores.tsv.gz \
      --out-summary-json ${out_prefix}.rare_scan.summary.json
    """
}

process AGGREGATE_RARE_SNP_TRACTS {
    tag "aggregate"

    publishDir "${params.rare_tract_results_dir}", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple path(window_score_files), path(scan_summary_files), path(rare_snp_tract_py)
    path(metadata_file, stageAs: 'metadata.tsv')
    path(genetic_map_file, stageAs: 'genetic_map.tsv')

    output:
    path "tracts.tsv", emit: tracts_table
    path "tract_summary.json", emit: summary
    path "report.html", emit: report
    path "plots/*.png", emit: plots

    script:
    def window_args = window_score_files.collect { f -> "--window-scores ${f}" }.join(' ')
    def summary_args = scan_summary_files.collect { f -> "--per-chr-summary ${f}" }.join(' ')
    def metadata_arg = metadata_file.size() > 0 ? "--metadata ${metadata_file}" : ""
    def map_arg = genetic_map_file.size() > 0 ? "--genetic-map ${genetic_map_file}" : ""
    """
    set -euo pipefail

    mkdir -p plots

    python3 ${rare_snp_tract_py} \
      --mode aggregate \
      ${window_args} \
      ${summary_args} \
      ${metadata_arg} \
      ${map_arg} \
      --input-dir ${params.rare_tract_input_dir} \
      --maf-threshold ${params.rare_tract_maf_threshold} \
      --enrichment-percentile ${params.rare_tract_enrichment_percentile} \
      --threshold-scope ${params.rare_tract_threshold_scope} \
      --max-gap-windows ${params.rare_tract_max_gap_windows} \
      --min-tract-snps ${params.rare_tract_min_tract_snps} \
      --min-tract-bp ${params.rare_tract_min_tract_bp} \
      --use-cm-if-available ${params.rare_tract_use_cm_if_available} \
      --plot-dpi ${params.rare_tract_plot_dpi} \
      --plot-palette ${params.rare_tract_plot_palette} \
      --plot-bins-method ${params.rare_tract_plot_bins_method} \
      --plot-kde-bandwidth-scale ${params.rare_tract_plot_kde_bandwidth_scale} \
      --plot-ecdf ${params.rare_tract_plot_ecdf} \
      --plot-kde ${params.rare_tract_plot_kde} \
      --plot-scatter ${params.rare_tract_plot_scatter} \
      --plot-boxplot-by-chrom ${params.rare_tract_plot_boxplot_by_chrom} \
      --plot-violin-by-chrom ${params.rare_tract_plot_violin_by_chrom} \
      --plot-chrom-metric ${params.rare_tract_plot_chrom_metric} \
      --plot-chrom-min-tracts ${params.rare_tract_plot_chrom_min_tracts} \
      --plot-chrom-max ${params.rare_tract_plot_chrom_max} \
      --plot-scatter-log-x ${params.rare_tract_plot_scatter_log_x} \
      --plot-scatter-log-y ${params.rare_tract_plot_scatter_log_y} \
      --output-dir ./
    """
}
