nextflow.enable.dsl=2

params.cases = null
params.report = null
params.report_sha256 = null
params.operator = "${projectDir}/../bin/r02_k06_positive_coverage.py"
params.outdir = null

process POSITIVE_COVERAGE {
    tag "${case_id}"
    cpus 1
    memory '1 GB'
    maxForks 2
    publishDir params.outdir, mode: 'copy', overwrite: false
    input:
    val case_id
    path operator_code
    path dependencies
    output:
    path "${case_id}.positive-coverage.json"
    script:
    """
    python3 '${operator_code}' --report '${params.report}' \
      --report-sha256 '${params.report_sha256}' --case-id '${case_id}' \
      --output '${case_id}.positive-coverage.json'
    """
}

workflow {
    if (!params.cases || !params.report || !params.report_sha256 || !params.outdir)
        error 'Explicit case list, authenticated report and output directory required'
    def inputs = Channel.fromPath(params.cases, checkIfExists: true).splitText().map { it.trim() }.filter { it }
    def code = Channel.value(file(params.operator, checkIfExists: true))
    def base = file(params.operator).parent
    def dependencies = Channel.value(['r02_k06_observability.py', 'r02_segment_evidence.py',
        'r02_genomic_pair_evidence.py'].collect { file("${base}/${it}", checkIfExists: true) })
    POSITIVE_COVERAGE(inputs, code, dependencies)
}
