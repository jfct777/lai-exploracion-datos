nextflow.enable.dsl=2

params.report = null
params.report_sha256 = null
params.operator = "${projectDir}/../bin/r02_k06_pair_truth_territory.py"
params.outdir = null
params.max_rows = 20000
params.timeout_seconds = 120
params.max_forks = 1

process PAIR_TRUTH_TERRITORY {
    tag "${case_id}"
    cpus 1
    memory '1 GB'
    time '3m'
    maxForks params.max_forks
    cache 'deep'
    errorStrategy 'terminate'
    publishDir params.outdir, mode: 'copy', overwrite: false
    input:
    tuple val(case_id), path(source_files, stageAs: 'sources/*')
    path source_report
    path operator_code
    path dependencies
    output:
    path "${case_id}.pair-truth-territory"
    script:
    """
    python3 '${operator_code}' --report '${source_report}' \
      --report-sha256 '${params.report_sha256}' --case-id '${case_id}' \
      --output '${case_id}.pair-truth-territory' --max-rows '${params.max_rows}' \
      --timeout-seconds '${params.timeout_seconds}' --staged-input-dir sources
    """
}

workflow {
    if (!params.report || !params.report_sha256 || !params.outdir)
        error 'Authenticated BM report and output directory required'
    def report_file = file(params.report, checkIfExists: true)
    def report = new groovy.json.JsonSlurper().parse(report_file.toFile())
    def cases = Channel.fromList(report.cases.collect { c ->
        tuple(c.case_id, ['manifest','simulation_truth','genetic_map','positive_territory'].collect {
            file(c.files[it].path, checkIfExists: true)
        })
    })
    def operator_file = file(params.operator, checkIfExists: true)
    def dependencies = Channel.value(['r02_donor_copy_truth.py','r02_k06_score_refined_ibd_positive.py'].collect {
        file("${operator_file.parent}/${it}", checkIfExists: true)
    })
    PAIR_TRUTH_TERRITORY(cases, Channel.value(report_file), Channel.value(operator_file), dependencies)
}
