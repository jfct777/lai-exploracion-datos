nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Module 16.5 — IBD Community Detection Enhanced (biological interpretability)
// ---------------------------------------------------------------------------
//
// Fork of Module 16 with targeted improvements oriented at the DNABR
// biological questions:
//   1. Macro-structure  — communities vs Brazilian region/UF
//   2. Cryptic kinship  — small dense communities = extended families
//   3. Founder effects  — isolated subpopulations with high intra/low inter
//
// Consumes the same three Module 14 aggregate outputs as Module 16:
//   - all_pairwise_segments.tsv.gz
//   - pair_sharing_summary.tsv
//   - individual_sharing_summary.tsv
//
// M16 is left untouched for reproducibility; M16.5 runs independently and
// can be A/B-compared against M16 on identical inputs.
//
// Metadata is OPTIONAL.  When a TSV with sample_id + (region|UF|...) is
// supplied via params.ibd_enhanced_metadata_file, the module adds Fisher
// enrichment tables and metadata-coloured sidebars to the plots.  Without
// metadata it still runs every statistical-robustness step and emits the
// core figures.
//
// ---------------------------------------------------------------------------
// Flag-passing convention (single source of truth)
// ---------------------------------------------------------------------------
//
// All non-conditional CLI flags forwarded to bin/ibd_community_enhanced.py
// are declared in the ``CLI_FLAG_MAP`` below.  Each entry maps a Nextflow
// parameter name (without the ``ibd_enhanced_`` prefix) to a 2-element list:
//
//     [ '--cli-flag-name', shouldQuoteValue ]
//
// where ``shouldQuoteValue`` is ``true`` for comma-separated lists or
// strings that must be passed as a single shell argument.  Adding a new
// flag therefore touches three places only:
//
//   1. ``bin/ibd_community_enhanced.py`` — argparse declaration
//   2. ``nextflow.config`` — ``ibd_enhanced_<param_name>`` default
//   3. This file — one line in ``CLI_FLAG_MAP``
//
// For ad-hoc experiments without modifying the module (e.g. trying a flag
// that exists in the Python script but not yet in this map), set
// ``--ibd_enhanced_extra_args '--my-new-flag value'`` on the
// ``nextflow run`` command line.
// ---------------------------------------------------------------------------

process IBD_COMMUNITY_ENHANCED {
    tag "ibd_community_enhanced"

    publishDir "${params.ibd_enhanced_results_dir}", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple path(all_segments), path(pair_summary), path(individual_summary), path(ibd_script)
    path metadata_file

    output:
    // Core tables
    path "graph_edges.tsv.gz",                       emit: graph_edges
    path "graph_nodes.tsv",                          emit: graph_nodes
    path "graph_summary.json",                       emit: graph_summary
    path "graph_sharing_matrix.npz",                 emit: graph_matrix
    path "leiden_assignments.tsv",                   emit: leiden_assignments
    path "leiden_modularity.tsv",                    emit: leiden_modularity
    path "leiden_consensus_res_*.tsv.gz",            emit: leiden_consensus, optional: true
    path "nmf_soft_memberships_k*.tsv",              emit: nmf_soft
    path "nmf_reconstruction_error.tsv",             emit: nmf_err
    path "validation_intra_vs_inter.tsv",            emit: validation
    path "global_community_summary.json",            emit: global_summary
    path "report.html",                              emit: report
    path "m16_5.log",                                emit: log, optional: true
    // Sprint 2+ outputs (emitted optionally so sprint 1 still passes)
    path "leiden_ari_by_resolution.tsv",             emit: leiden_ari,                optional: true
    path "community_silhouette.tsv",                 emit: silhouette,                optional: true
    path "nmf_cophenetic_by_k.tsv",                  emit: cophenetic,                optional: true
    path "cryptic_kinship_candidates.tsv",           emit: kinship_candidates,        optional: true
    path "founder_effect_candidates.tsv",            emit: founder_candidates,        optional: true
    path "community_metadata_enrichment.tsv",        emit: enrichment,                optional: true
    path "community_labels.tsv",                     emit: community_labels,          optional: true
    path "community_auto_annotations.tsv",           emit: auto_annotations,          optional: true
    // Plots
    path "plots/*.png",                              emit: plots,     optional: true
    path "plots/*.pdf",                              emit: plots_pdf, optional: true
    path "plots/*.html",                             emit: plots_html, optional: true

    script:
    // -----------------------------------------------------------------
    // Declarative param → CLI flag mapping.  See header comment for
    // the conventions; ``true`` in column 3 means single-quote the value
    // (required for comma-separated lists and any string the shell
    // would otherwise re-tokenise).
    // -----------------------------------------------------------------
    def CLI_FLAG_MAP = [
        seed:                                ['--seed',                                false],
        min_edge_bp:                         ['--min-edge-bp',                         false],
        min_max_segment_bp:                  ['--min-max-segment-bp',                  false],
        edge_weight_transform:               ['--edge-weight-transform',               false],
        segments_chunk_rows:                 ['--segments-chunk-rows',                 false],
        leiden_resolutions:                  ['--leiden-resolutions',                  true ],
        leiden_n_seeds:                      ['--leiden-n-seeds',                      false],
        leiden_min_community_size:           ['--leiden-min-community-size',           false],
        leiden_consensus_resolution:         ['--leiden-consensus-resolution',         false],
        nmf_k_values:                        ['--nmf-k-values',                        true ],
        nmf_inits:                           ['--nmf-inits',                           false],
        nmf_init_mode:                       ['--nmf-init-mode',                       false],
        nmf_max_iter:                        ['--nmf-max-iter',                        false],
        nmf_tol:                             ['--nmf-tol',                             false],
        nmf_operational_k:                   ['--nmf-operational-k',                   false],
        nmf_dispersion_threshold:            ['--nmf-dispersion-threshold',            false],
        nmf_cophenetic_floor:                ['--nmf-cophenetic-floor',                false],
        nmf_min_marginal_coph:               ['--nmf-min-marginal-coph',               false],
        laplacian_normalize:                 ['--laplacian-normalize',                 false],
        kinship_segment_mb:                  ['--kinship-segment-mb',                  false],
        kinship_max_size:                    ['--kinship-max-size',                    false],
        founder_intra_inter_ratio:           ['--founder-intra-inter-ratio',           false],
        founder_min_silhouette:              ['--founder-min-silhouette',              false],
        founder_min_size:                    ['--founder-min-size',                    false],
        founder_min_size_for_report:         ['--founder-min-size-for-report',         false],
        validation_resolution:               ['--validation-resolution',               false],
        top_samples_per_community:           ['--top-samples-per-community',           false],
        plot_dpi:                            ['--plot-dpi',                            false],
        plot_width_inches:                   ['--plot-width-inches',                   false],
        plot_height_inches:                  ['--plot-height-inches',                  false],
        plot_export_pdf:                     ['--plot-export-pdf',                     false],
        plot_export_svg:                     ['--plot-export-svg',                     false],
        plot_network_max_nodes:              ['--plot-network-max-nodes',              false],
        plot_heatmap_max_nodes:              ['--plot-heatmap-max-nodes',              false],
        plot_cluster_label_min_size:         ['--plot-cluster-label-min-size',         false],
        plot_auto_annotate_by:               ['--plot-auto-annotate-by',               false],
        plot_auto_annotate_secondary:        ['--plot-auto-annotate-secondary',        false],
        plot_adjust_labels:                  ['--plot-adjust-labels',                  false],
        plot_palette:                        ['--plot-palette',                        false],
        plot_umap_3d:                        ['--plot-umap-3d',                        false],
        extra_color_columns:                 ['--extra-color-columns',                 true ],
    ]

    // Compose the forwarded-flags block.  Missing params raise a clear
    // error so config drift is caught at run time, not silently dropped.
    def fixedFlagsStr = CLI_FLAG_MAP.collect { entry ->
        def paramSuffix = entry.key
        def cliFlag     = entry.value[0]
        def quote       = entry.value[1]
        def fullName    = "ibd_enhanced_${paramSuffix}".toString()
        if( !params.containsKey(fullName) ) {
            throw new IllegalStateException(
                "M16.5 module references params.${fullName} but it is "
                + "not declared in nextflow.config."
            )
        }
        def value = params[fullName]
        // Skip optional string flags whose default is empty/null — emitting
        // `--flag` with no value would let bash word-splitting feed argparse
        // the *next* flag as the argument (or error out).  Numeric 0 and
        // boolean false are intentionally kept (they're meaningful values).
        if( value == null
                || (value instanceof CharSequence && value.toString().isEmpty()) ) {
            return null
        }
        return quote ? "${cliFlag} '${value}'" : "${cliFlag} ${value}"
    }.findAll { it != null }.join(' \\\n        ')

    // Conditional flags (semantics differ from the simple map above):
    //   * --metadata-file: skipped when no metadata TSV is provided.
    //   * --plot-color-by: only meaningful when metadata is present.
    //   * --plot-community-annotations-file: optional TSV; only forwarded
    //     when the param is non-empty AND the file exists.
    def meta_arg = metadata_file.name == 'empty.txt' ? '' : "--metadata-file ${metadata_file}"
    def color_arg = (metadata_file.name != 'empty.txt' && params.ibd_enhanced_plot_color_by) \
                    ? "--plot-color-by ${params.ibd_enhanced_plot_color_by}" : ''
    def annot_arg = (params.containsKey('ibd_enhanced_plot_community_annotations_file')
                     && params.ibd_enhanced_plot_community_annotations_file) \
                    ? "--plot-community-annotations-file ${params.ibd_enhanced_plot_community_annotations_file}" : ''

    // Escape hatch for ad-hoc experiments: any extra CLI arguments are
    // appended verbatim.  Empty by default.
    def extra_args = (params.containsKey('ibd_enhanced_extra_args')
                      && params.ibd_enhanced_extra_args) \
                     ? params.ibd_enhanced_extra_args : ''

    def segments_stage_name    = "all_pairwise_segments.tsv.gz"
    def pair_stage_name        = "pair_sharing_summary.tsv"
    def individual_stage_name  = "individual_sharing_summary.tsv"

    """
    set -euo pipefail

    mkdir -p plots
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    mkdir -p "\$MPLCONFIGDIR"

    # Numba JIT cache dir: UMAP imports numba, which by default tries to
    # write its AOT cache next to its source file.  That path is read-only
    # inside the Singularity image, so we redirect the cache to the
    # per-task workdir.  Also redirect HOME so any library that caches under
    # \$HOME (joblib, scikit-learn kernels) also writes to the workdir.
    export NUMBA_CACHE_DIR="\$PWD/.numba_cache"
    mkdir -p "\$NUMBA_CACHE_DIR"
    export HOME="\$PWD"

    # Re-stage M14 inputs under expected filenames (same pattern as M16).
    mkdir -p m14_inputs
    ln -sf "\$(readlink -f ${all_segments})"       m14_inputs/${segments_stage_name}
    ln -sf "\$(readlink -f ${pair_summary})"       m14_inputs/${pair_stage_name}
    ln -sf "\$(readlink -f ${individual_summary})" m14_inputs/${individual_stage_name}

    python3 ${ibd_script} \\
        --mode all \\
        --input-dir m14_inputs \\
        --output-dir ./ \\
        --threads ${task.cpus} \\
        ${fixedFlagsStr} \\
        ${meta_arg} ${color_arg} ${annot_arg} ${extra_args}
    """
}


// ---------------------------------------------------------------------------
// Module 16.5 — REPLOT process (multi-metadata-coloured plots)
// ---------------------------------------------------------------------------
//
// Re-renders the M16.5 plot set on outputs already computed by
// IBD_COMMUNITY_ENHANCED, using a different metadata column for sidebar /
// colouring on each task.  The graph, Leiden assignments, NMF
// memberships, and validation tables are all reused — only plotting runs.
//
// Activated when ``params.ibd_enhanced_extra_color_columns`` is non-empty.
// The wiring in main.nf fans the comma-separated column list out into one
// task per column, so the wall-clock cost is roughly that of a single
// re-plot regardless of the number of columns requested.
//
// Each task publishes to ``replot_by_<column>/`` to avoid collisions with
// the primary plot directory and with sibling re-plot tasks.
// ---------------------------------------------------------------------------

process IBD_ENHANCED_REPLOT {
    tag "ibd_enhanced_replot:${color_column}"

    publishDir "${params.ibd_enhanced_results_dir}/replot_by_${color_column}",
               mode: 'copy'

    // Resources come from the withName: 'IBD_ENHANCED_REPLOT' block in
    // conf/auto_resources.config (cpus=4, memory=48 GB).  16 GB of the
    // global default is OOM-killed at N=2723 by the Plotly heatmap's
    // 2723²×9 customdata array; do NOT silently fall back to the default.
    time params.time

    input:
    // First 8 paths come from IBD_COMMUNITY_ENHANCED's named outputs
    // (computed graph + leiden + nmf + validation).  The next 3 are the
    // M14 aggregate outputs that the python script's plot mode also
    // requires to reconstruct the sparse sharing matrix without
    // recomputing anything. Without them the script stops with
    // "Missing required Module 14 output".
    tuple path(graph_edges), path(graph_nodes), path(graph_matrix),
          path(leiden_assignments), path(leiden_modularity),
          path(nmf_err), path(global_summary),
          path(validation),
          path(m14_segments), path(m14_pair_summary),
          path(m14_individual_summary),
          path(ibd_script)
    path metadata_file
    each color_column

    output:
    path "plots/*.png",                    emit: plots,            optional: true
    path "plots/*.pdf",                    emit: plots_pdf,        optional: true
    path "plots/*.html",                   emit: plots_html,       optional: true
    path "community_auto_annotations.tsv", emit: auto_annotations, optional: true
    path "m16_5.replot.log",               emit: log,              optional: true

    script:
    // Re-use the same declarative flag map as the primary process so
    // adding a new flag in IBD_COMMUNITY_ENHANCED automatically propagates.
    // The replot still calls ``--mode plot`` on the Python script, so flags
    // affecting compute (Leiden, NMF, validation) are accepted but no-ops.
    def CLI_FLAG_MAP = [
        seed:                        ['--seed',                        false],
        leiden_resolutions:          ['--leiden-resolutions',          true ],
        validation_resolution:       ['--validation-resolution',       false],
        plot_dpi:                    ['--plot-dpi',                    false],
        plot_width_inches:           ['--plot-width-inches',           false],
        plot_height_inches:          ['--plot-height-inches',          false],
        plot_export_pdf:             ['--plot-export-pdf',             false],
        plot_export_svg:             ['--plot-export-svg',             false],
        plot_network_max_nodes:      ['--plot-network-max-nodes',      false],
        plot_heatmap_max_nodes:      ['--plot-heatmap-max-nodes',      false],
        plot_cluster_label_min_size: ['--plot-cluster-label-min-size', false],
        plot_auto_annotate_by:       ['--plot-auto-annotate-by',       false],
        plot_auto_annotate_secondary:['--plot-auto-annotate-secondary',false],
        plot_adjust_labels:          ['--plot-adjust-labels',          false],
        plot_palette:                ['--plot-palette',                false],
        plot_umap_3d:                ['--plot-umap-3d',                false],
        extra_color_columns:         ['--extra-color-columns',         true ],
    ]
    def fixedFlagsStr = CLI_FLAG_MAP.collect { entry ->
        def paramSuffix = entry.key
        def cliFlag     = entry.value[0]
        def quote       = entry.value[1]
        def fullName    = "ibd_enhanced_${paramSuffix}".toString()
        if( !params.containsKey(fullName) ) {
            throw new IllegalStateException(
                "M16.5 REPLOT references params.${fullName} but it is "
                + "not declared in nextflow.config."
            )
        }
        def value = params[fullName]
        // Skip optional string flags whose default is empty/null — emitting
        // `--flag` with no value would let bash word-splitting feed argparse
        // the *next* flag as the argument (or error out).  Numeric 0 and
        // boolean false are intentionally kept (they're meaningful values).
        if( value == null
                || (value instanceof CharSequence && value.toString().isEmpty()) ) {
            return null
        }
        return quote ? "${cliFlag} '${value}'" : "${cliFlag} ${value}"
    }.findAll { it != null }.join(' \\\n        ')

    def meta_arg = metadata_file.name == 'empty.txt' ? '' : "--metadata-file ${metadata_file}"
    def annot_arg = (params.containsKey('ibd_enhanced_plot_community_annotations_file')
                     && params.ibd_enhanced_plot_community_annotations_file) \
                    ? "--plot-community-annotations-file ${params.ibd_enhanced_plot_community_annotations_file}" : ''

    """
    set -euo pipefail

    mkdir -p plots
    export MPLCONFIGDIR="\$PWD/.matplotlib"
    mkdir -p "\$MPLCONFIGDIR"
    export NUMBA_CACHE_DIR="\$PWD/.numba_cache"
    mkdir -p "\$NUMBA_CACHE_DIR"
    export HOME="\$PWD"

    # Stage all compute outputs from the primary process under their
    # canonical filenames so --mode plot finds them.  The three M14
    # aggregate files are also required because the python script
    # reconstructs the pair-segment table at plot startup; without them
    # it aborts with "Missing required Module 14 output".
    ln -sf "\$(readlink -f ${graph_edges})"             graph_edges.tsv.gz
    ln -sf "\$(readlink -f ${graph_nodes})"             graph_nodes.tsv
    ln -sf "\$(readlink -f ${graph_matrix})"            graph_sharing_matrix.npz
    ln -sf "\$(readlink -f ${leiden_assignments})"      leiden_assignments.tsv
    ln -sf "\$(readlink -f ${leiden_modularity})"       leiden_modularity.tsv
    ln -sf "\$(readlink -f ${nmf_err})"                 nmf_reconstruction_error.tsv
    ln -sf "\$(readlink -f ${global_summary})"          global_community_summary.json
    ln -sf "\$(readlink -f ${validation})"              validation_intra_vs_inter.tsv
    ln -sf "\$(readlink -f ${m14_segments})"            all_pairwise_segments.tsv.gz
    ln -sf "\$(readlink -f ${m14_pair_summary})"        pair_sharing_summary.tsv
    ln -sf "\$(readlink -f ${m14_individual_summary})"  individual_sharing_summary.tsv

    python3 ${ibd_script} \\
        --mode plot \\
        --input-dir ./ \\
        --output-dir ./ \\
        --threads ${task.cpus} \\
        ${fixedFlagsStr} \\
        ${meta_arg} \\
        --plot-color-by ${color_column} \\
        ${annot_arg}

    # Rename the log so it's easy to identify per-replot.
    if [[ -f m16_5.log ]]; then mv m16_5.log m16_5.replot.log; fi
    """
}
