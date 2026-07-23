nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Module 14 — Rare Allele Sharing Painter
// ---------------------------------------------------------------------------
//
// Per-chromosome SCAN process detects pairwise shared rare-variant segments;
// the AGGREGATE process collapses 24 per-chromosome outputs into a single
// pair / individual / chromosome summary plus an HTML report.
//
// ---------------------------------------------------------------------------
// Flag-passing convention (single source of truth)
// ---------------------------------------------------------------------------
//
// All non-conditional CLI flags forwarded to bin/rare_allele_sharing_painter.py
// are declared in ``CLI_FLAG_MAP_SCAN`` and ``CLI_FLAG_MAP_AGGREGATE`` below.
// Each entry maps a Nextflow parameter name (without the ``painting_``
// prefix) to a 2-element list:
//
//     [ '--cli-flag-name', shouldQuoteValue ]
//
// where ``shouldQuoteValue`` is ``true`` for strings that must reach Python
// argparse as a single shell argument (e.g. ``--plot-font-family 'DejaVu Sans'``).
// Adding a new flag therefore touches three places only:
//
//   1. ``bin/rare_allele_sharing_painter.py`` — argparse declaration
//   2. ``nextflow.config`` — ``painting_<param_name>`` default
//   3. This file — one line in the appropriate ``CLI_FLAG_MAP``
//
// Conditional / dynamic flags (``--sample-ids-file``, ``--max-samples``,
// ``--n-jobs``, ``--region``, ``--region-size-bp``) stay outside the map
// because their value depends on task-local state (Slurm cpus reservation,
// staged file, null sentinels), not directly on a single config param.
//
// Escape hatch: set ``--painting_extra_args '--my-new-flag value'`` on the
// ``nextflow run`` command line to forward any extra CLI argument without
// modifying this module.
// ---------------------------------------------------------------------------

// Shared helper: render a CLI_FLAG_MAP into a backslash-newline-joined block
// of ``--flag value`` strings.  Missing params raise a clear error so config
// drift is caught at run time, not silently dropped.  Optional string params
// whose value is null/empty are skipped (emitting ``--flag`` alone would let
// bash word-splitting feed argparse the next flag as its value).
def renderFlagMap(Map flagMap) {
    flagMap.collect { entry ->
        def paramSuffix = entry.key
        def cliFlag     = entry.value[0]
        def quote       = entry.value[1]
        def fullName    = "painting_${paramSuffix}".toString()
        if( !params.containsKey(fullName) ) {
            throw new IllegalStateException(
                "M14 module references params.${fullName} but it is "
                + "not declared in nextflow.config."
            )
        }
        def value = params[fullName]
        if( value == null
                || (value instanceof CharSequence && value.toString().isEmpty()) ) {
            return null
        }
        return quote ? "${cliFlag} '${value}'" : "${cliFlag} ${value}"
    }.findAll { it != null }.join(' \\\n      ')
}


process ANALYZE_RARE_ALLELE_SHARING {
    tag "chr${chr}"

    publishDir "${params.painting_results_dir}/per_chr", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(rare_allele_sharing_painter_py)
    val(sample_ids_payload_b64)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.sharing_windows.tsv.gz"), emit: sharing_windows
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.pairwise_segments.tsv.gz"), emit: pairwise_segments
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.sharing_scan.summary.json"), emit: scan_summaries
    path "plots/*.png", emit: plots

    script:
    // Declarative param → CLI flag mapping for the per-chromosome scan.
    // Order is preserved in the rendered command (helps log readability).
    def CLI_FLAG_MAP_SCAN = [
        input_format:               ['--input-format',            false],
        window_size_bp:             ['--window-size-bp',          false],
        step_size_bp:               ['--step-size-bp',            false],
        min_shared_variants:        ['--min-shared-variants',     false],
        min_jaccard:                ['--min-jaccard',             false],
        max_gap_bp:                 ['--max-gap-bp',              false],
        min_segment_bp:             ['--min-segment-bp',          false],
        max_block_gap_bp:           ['--max-block-gap-bp',        false],
        min_block_snps:             ['--min-block-snps',          false],
        plot_dpi:                   ['--plot-dpi',                false],
        plot_palette:               ['--plot-palette',            false],
        plot_font_family:           ['--plot-font-family',        true ],
        plot_width_inches:          ['--plot-width-inches',       false],
        plot_height_inches:         ['--plot-height-inches',      false],
        plot_max_height_inches:     ['--plot-max-height-inches',  false],
        plot_raster_bp_per_col:     ['--plot-raster-bp-per-col',  false],
        plot_export_pdf:            ['--plot-export-pdf',         false],
        plot_export_svg:            ['--plot-export-svg',         false],
        plot_mode:                  ['--plot-mode',               false],
        plot_max_pairs_legend:      ['--plot-max-pairs-legend',   false],
        skip_plots:                 ['--skip-plots',              false],
    ]
    def fixedFlagsStr = renderFlagMap(CLI_FLAG_MAP_SCAN)

    def out_prefix = "dnabr.hg38.2723.chr${chr}"

    // --- Conditional / dynamic flags ---------------------------------------
    // sample-ids-file: payload is base64-encoded so it survives Nextflow's
    // val(String) channel without re-escaping; empty string ⇒ no filter.
    def sample_setup = sample_ids_payload_b64 ? """
    printf '%s' '${sample_ids_payload_b64}' | base64 -d > module14_selected_samples.txt
    """ : ""
    def sample_arg = sample_ids_payload_b64 ? "--sample-ids-file module14_selected_samples.txt" : ""

    // max-samples: null sentinel means "use all loaded samples" (no flag).
    def max_samples_arg = params.painting_max_samples != null \
                          ? "--max-samples ${params.painting_max_samples}" : ""

    // n-jobs: null in config → auto = task.cpus (Slurm reservation).
    // Explicit override via --painting_n_jobs wins (e.g. 1 for debugging).
    def n_jobs_value = params.painting_n_jobs ?: task.cpus

    // region / region-size-bp: optional sub-chromosomal windows.
    def region_arg = params.painting_region != null \
                     ? "--region ${params.painting_region}" : ""
    def region_size_arg = params.painting_region_size_bp != null \
                          ? "--region-size-bp ${params.painting_region_size_bp}" : ""

    // Escape hatch for ad-hoc CLI args (empty by default).
    def extra_args = (params.containsKey('painting_extra_args')
                      && params.painting_extra_args) \
                     ? params.painting_extra_args : ''

    """
    set -euo pipefail

    mkdir -p plots
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    mkdir -p "\$MPLCONFIGDIR"
    ${sample_setup}

    python3 ${rare_allele_sharing_painter_py} \\
      --mode scan \\
      --input ${vcf_gz} \\
      --chr ${chr} \\
      ${fixedFlagsStr} \\
      --n-jobs ${n_jobs_value} \\
      --out-sharing-windows ${out_prefix}.sharing_windows.tsv.gz \\
      --out-pairwise-segments ${out_prefix}.pairwise_segments.tsv.gz \\
      --out-summary-json ${out_prefix}.sharing_scan.summary.json \\
      ${sample_arg} ${max_samples_arg} ${region_arg} ${region_size_arg} \\
      --output-dir ./ \\
      ${extra_args}
    """
}

process AGGREGATE_RARE_ALLELE_SHARING {
    tag "aggregate"

    publishDir "${params.painting_results_dir}", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple path(pairwise_segment_files), path(scan_summary_files), path(rare_allele_sharing_painter_py)

    output:
    path "all_pairwise_segments.tsv.gz", emit: all_segments
    path "chromosome_sharing_summary.tsv", emit: chrom_summary
    path "pair_sharing_summary.tsv", emit: pair_summary, optional: true
    path "individual_sharing_summary.tsv", emit: individual_summary, optional: true
    path "global_sharing_summary.json", emit: global_summary
    path "report.html", emit: report
    path "plots/*.png", emit: plots, optional: true

    script:
    // Declarative param → CLI flag mapping for the aggregate step.  Only the
    // plot-related flags + min_block_snps + skip_plots are meaningful here;
    // the rest (segmentation thresholds, n-jobs) belong to the scan stage.
    def CLI_FLAG_MAP_AGGREGATE = [
        plot_dpi:                   ['--plot-dpi',                false],
        plot_palette:               ['--plot-palette',            false],
        plot_font_family:           ['--plot-font-family',        true ],
        plot_width_inches:          ['--plot-width-inches',       false],
        plot_height_inches:         ['--plot-height-inches',      false],
        plot_max_height_inches:     ['--plot-max-height-inches',  false],
        plot_raster_bp_per_col:     ['--plot-raster-bp-per-col',  false],
        plot_export_pdf:            ['--plot-export-pdf',         false],
        plot_export_svg:            ['--plot-export-svg',         false],
        plot_mode:                  ['--plot-mode',               false],
        plot_max_pairs_legend:      ['--plot-max-pairs-legend',   false],
        min_block_snps:             ['--min-block-snps',          false],
        skip_plots:                 ['--skip-plots',              false],
    ]
    def fixedFlagsStr = renderFlagMap(CLI_FLAG_MAP_AGGREGATE)

    def segment_args = pairwise_segment_files.collect { f -> "--pairwise-segments ${f}" }.join(' ')
    def summary_args = scan_summary_files.collect { f -> "--per-chr-summary ${f}" }.join(' ')

    def extra_args = (params.containsKey('painting_extra_args')
                      && params.painting_extra_args) \
                     ? params.painting_extra_args : ''

    """
    set -euo pipefail

    mkdir -p plots
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    mkdir -p "\$MPLCONFIGDIR"

    python3 ${rare_allele_sharing_painter_py} \\
      --mode aggregate \\
      ${segment_args} \\
      ${summary_args} \\
      --input-dir ${params.painting_input_dir} \\
      ${fixedFlagsStr} \\
      --output-dir ./ \\
      ${extra_args}
    """
}
