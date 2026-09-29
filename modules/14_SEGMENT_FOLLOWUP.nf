nextflow.enable.dsl=2

// Views replay stored intervals; sensitivity alone reopens the rare VCF.
// Neither task reruns normalization or changes the historical painting.
def sensitivityFlags() {
    def flags = [
        expected_samples: '--expected-samples', gaps_bp: '--gaps-bp',
        lengths_bp: '--lengths-bp', min_shared: '--min-shared',
        max_pair_events: '--max-pair-events', max_memory_mb: '--max-memory-mb',
        max_output_rows: '--max-output-rows'
    ]
    flags.collect { suffix, flag ->
        def value = params["m14_followup_${suffix}"]
        if (value == null || !(value.toString() ==~ /[0-9,]+/))
            throw new IllegalArgumentException("Invalid m14_followup_${suffix}")
        "${flag} '${value}'"
    }.join(' ')
}

process RENDER_RARE_SEGMENTS {
    tag "chr${chrom}"
    cpus 1
    memory '4 GB'
    time '30m'
    publishDir "${params.m14_followup_results_dir}", mode: 'copy', overwrite: false

    input:
    tuple val(chrom), path(segments), path(sample_ids), path(renderer)

    output:
    tuple val(chrom), path("segment_views"), emit: views

    script:
    def keep = sample_ids.size() > 0 ? "--sample-ids-file '${sample_ids}'" : ''
    """
    python3 '${renderer}' --segments '${segments}' --chr '${chrom}' \
      ${keep} --dpi '${params.m14_followup_dpi}' --output-dir segment_views
    """
}

process MEASURE_RARE_SEGMENT_SENSITIVITY {
    tag "chr${chrom}"
    cpus 4
    memory '16 GB'
    time '3h'
    publishDir "${params.m14_followup_results_dir}", mode: 'copy', overwrite: false

    input:
    tuple val(chrom), path(rare_vcf), path(sample_ids), path(anchor), path(anchor_summary), path(scripts)

    output:
    tuple val(chrom), path("sensitivity"), emit: analysis

    script:
    def flags = sensitivityFlags()
    """
    python3 rare_segment_sensitivity.py --input '${rare_vcf}' --chr '${chrom}' \
      --sample-ids-file '${sample_ids}' --anchor-segments '${anchor}' \
      --anchor-summary '${anchor_summary}' --output-dir sensitivity ${flags}
    """
}

process PLOT_RARE_SEGMENT_SENSITIVITY {
    tag "chr${chrom}"
    cpus 1
    memory '4 GB'
    time '30m'
    publishDir "${params.m14_followup_results_dir}", mode: 'copy', overwrite: false

    input:
    tuple val(chrom), path(analysis), path(plot_scripts)

    output:
    tuple val(chrom), path("sensitivity_plots"), emit: figures

    script:
    """
    python3 rare_segment_sensitivity_plots.py --distance-summary '${analysis}/distance_summary.tsv' \
      --distance-histogram '${analysis}/distance_histogram.tsv' \
      --configuration-summary '${analysis}/configuration_summary.tsv' \
      --chain-histogram '${analysis}/chain_histogram.tsv' \
      --output-dir sensitivity_plots --chr '${chrom}' --dpi '${params.m14_followup_dpi}'
    """
}

process SUMMARIZE_RARE_SEGMENT_KINSHIP {
    tag "existing_PCRelate"
    cpus 1
    memory '6 GB'
    time '30m'
    publishDir "${params.m14_followup_results_dir}", mode: 'copy', overwrite: false

    input:
    tuple path(configurations), path(pair_configurations), path(pcrelate), path(sample_ids), path(analyzer)

    output:
    path 'relatedness', emit: diagnostics

    script:
    def thresholds = params.m14_followup_kinship_thresholds.toString()
    if (!(thresholds ==~ /[0-9.,]+/)) error 'Invalid kinship thresholds'
    """
    python3 '${analyzer}' --configuration-summary '${configurations}' \
      --pair-configuration-summary '${pair_configurations}' --pcrelate '${pcrelate}' \
      --sample-ids-file '${sample_ids}' --output-dir relatedness \
      --expected-samples '${params.m14_followup_expected_samples}' --thresholds '${thresholds}' \
      --max-unique-pairs '${params.m14_followup_kinship_max_pairs}' \
      --max-memory-mb '${params.m14_followup_kinship_memory_mb}'
    """
}
