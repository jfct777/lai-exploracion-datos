nextflow.enable.dsl=2

process IDENTIFIED_DONOR_COPY_OVERLAPS {
    tag "${job_id}"
    cpus 1
    memory '2 GB'
    time '30m'
    maxForks 2
    container params.container
    stageInMode 'copy'
    publishDir "${params.output}/${job_id}", mode: 'copy', overwrite: false

    input:
    tuple val(job_id), path(manifest, stageAs: 'simulation-manifest.json'), val(manifest_sha), path(truth, stageAs: 'source-truth.tsv.gz')
    path worker

    output:
    tuple val(job_id), path('result')

    script:
    """
    python3 '${worker}' --manifest '${manifest}' --manifest-sha256 '${manifest_sha}' \
        --truth '${truth}' --out result
    """
}

workflow {
    def request = new groovy.json.JsonSlurper().parse(file(params.request).toFile())
    assert request.schema == 'r02_donor_copy_truth_campaign_v1'
    assert request.demographic_inference == false
    assert request.tasks.size() == request.expected_cases
    assert request.tasks.collect{ it.job_id }.unique().size() == request.expected_cases
    def tasks = Channel.fromList(request.tasks).map { r ->
        assert r.job_id ==~ /[A-Za-z0-9_.-]+/
        assert r.manifest_sha256 ==~ /[a-f0-9]{64}/
        tuple(r.job_id, file(r.manifest, checkIfExists: true), r.manifest_sha256, file(r.truth, checkIfExists: true))
    }
    IDENTIFIED_DONOR_COPY_OVERLAPS(tasks, file(params.worker, checkIfExists: true))
}
