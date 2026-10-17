nextflow.enable.dsl=2

process PREPARE_POSITIVE_TERRITORY {
    tag { case_id }
    cpus 1
    memory '2 GB'
    time '15m'
    maxForks params.max_forks
    container params.container
    cache 'deep'
    stageInMode 'symlink'
    publishDir { "${params.output_dir}/${case_id}" }, mode:'copy', overwrite:false
    input:
    tuple val(case_id), path(request), val(pin), path(inputs)
    path code
    output:
    tuple val(case_id), path('result')
    script:
    """
    python3 '${code}/r02_k06_positive_territory.py' --request '${request}' \
        --request-sha256 '${pin}' --output result
    """
}

workflow {
    def manifest=new groovy.json.JsonSlurper().parse(file(params.batch).toFile())
    assert manifest.schema=='r02_k06_positive_territory_batch_v1'
    assert manifest.tasks.size()==manifest.expected_cases
    assert manifest.tasks.collect{it.id}.unique().size()==manifest.expected_cases
    assert params.max_forks.toInteger()>0
    def tasks=Channel.fromList(manifest.tasks).map{ t ->
        assert t.id ==~ /[A-Za-z0-9_.-]+/
        def request=file(t.request,checkIfExists:true)
        assert java.security.MessageDigest.getInstance('SHA-256').digest(request.bytes).encodeHex().toString()==t.sha256
        def r=new groovy.json.JsonSlurper().parse(request.toFile())
        assert r.job_id==t.id
        def inputs=[r.summary,r.overlaps,r.map].collect{ x ->
            def p=file(x.path,checkIfExists:true)
            assert java.security.MessageDigest.getInstance('SHA-256').digest(p.bytes).encodeHex().toString()==x.sha256
            p
        }
        tuple(t.id,request,t.sha256,inputs)
    }
    PREPARE_POSITIVE_TERRITORY(tasks,file(params.code_dir,checkIfExists:true))
}
