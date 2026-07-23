nextflow.enable.dsl=2

process QC_AGGREGATE_REPORT_PY {
    tag "aggregate"

    publishDir "${params.outdir}/06_report", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple path(qc_files), path(stats_files), path(counts_files), path(aggregate_qc_report_py)

    output:
    path "merged_per_sample.tsv"
    path "flags_per_sample.tsv"
    path "summary.json"
    path "report.html"
    path "plots"

    script:
    def qc_args = qc_files.collect { f -> "--qc ${f}" }.join(' ')
    def stats_args = stats_files.collect { f -> "--stats ${f}" }.join(' ')
    def counts_args = counts_files.collect { f -> "--counts ${f}" }.join(' ')

    """
    set -euo pipefail

    mkdir -p plots

    python3 ${aggregate_qc_report_py} --mode aggregate \
      ${qc_args} \
      ${stats_args} \
      ${counts_args} \
      --outdir ./ \
      --sample_missing_fail ${params.sample_missing_fail} \
      --het_sd_fail ${params.het_sd_fail}
    """
}
