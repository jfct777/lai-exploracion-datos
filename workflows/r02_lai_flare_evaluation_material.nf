nextflow.enable.dsl=2

process R02_LAI_FLARE_EVALUATION_MATERIAL {
    tag "$job_id"
    cpus 1
    memory { limits.max_working_bytes + ' B' }
    time { (limits.timeout_seconds+60L)+' sec' }
    maxForks params.max_forks
    cache 'deep'
    container { params.container }
    publishDir { params.output_dir+'/'+job_id+'/evaluation-material' },mode:'copy',overwrite:false
    input:
    tuple val(job_id),path(contract,stageAs:'contract.json'),val(digest),val(limits),path(inputs,stageAs:'inputs/input????????')
    path code
    output:
    tuple val(job_id),path('material'),emit: prepared
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_flare_evaluation_material.py --contract contract.json --contract-sha256 '$digest' --output-dir material
    """
}

process R02_LAI_FLARE_EVALUATION_SCORE {
    tag "$job_id:gap=$gap"
    cpus 1
    memory { (limits.max_working_bytes+256L*1024*1024)+' B' }
    time { (limits.timeout_seconds+60L)+' sec' }
    maxForks params.max_forks
    cache 'deep'
    container { params.container }
    publishDir { params.output_dir+'/'+job_id+'/evaluation-gap-'+gap },mode:'copy',overwrite:false
    input:
    tuple val(job_id),val(gap),path(contract,stageAs:'source-input.json'),val(digest),val(limits),path(inputs,stageAs:'inputs/input????????')
    path code
    output:
    tuple val(job_id),val(gap),path('evaluation'),path('resolved-input.json'),emit: reports
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 - <<'PY'
    import hashlib,json,pathlib
    p=pathlib.Path('source-input.json')
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='$digest'
    c=json.loads(p.read_text());c['max_query_gap_cm']=$gap
    pathlib.Path('resolved-input.json').write_text(json.dumps(c,sort_keys=True)+'\\n')
    PY
    digest=\$(sha256sum resolved-input.json | cut -d' ' -f1)
    python3 r02_lai_evaluate.py --contract resolved-input.json --contract-sha256 "\$digest" --output-dir evaluation
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.campaign,params.source_bin,params.output_dir].every{safe(it)})error 'Safe absolute paths required'
    if(!(params.campaign_sha256 ==~ /[0-9a-f]{64}/) || !(params.max_forks instanceof Integer) || params.max_forks<1 || params.max_forks>4)error 'Explicit digest and bounded concurrency required'
    def hash={p ->
        def d=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream{s -> byte[] b=new byte[1048576];int n;while((n=s.read(b))>0)d.update(b,0,n)}
        d.digest().encodeHex().toString()
    }
    def descriptor={d ->
        if(!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/))error 'Invalid evidence descriptor'
        def p=file(d.path,checkIfExists:true)
        if(hash(p)!=d.sha256)error 'Evidence differs before cache lookup'
        p
    }
    def cp=descriptor([path:params.campaign,sha256:params.campaign_sha256])
    def campaign=new groovy.json.JsonSlurper().parse(cp.toFile())
    if(campaign.schema!='r02_lai_flare_evaluation_campaign_v1' || !campaign.jobs || campaign.jobs.collect{it.job_id}.unique().size()!=campaign.jobs.size())error 'Unique paired evaluation jobs required'
    def names=['r02_lai_flare_evaluation_material.py','r02_lai_evaluate.py','m39_calibration.py','m34_score_predictions.py','m37_trace_score.py','m37_trace_core.py','r02_lai_flare_plan.py','m34_run_flare.py','r02_lai_simulate.py','r02_lai_support_review.py']
    def code=names.collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    def jobs=campaign.jobs.collect{j ->
        if(!(j.job_id ==~ /[A-Za-z0-9_.-]+/))error 'Unsafe job identity'
        def p=descriptor(j.input);def c=new groovy.json.JsonSlurper().parse(p.toFile())
        def dependencies=c.inputs.values().collect{descriptor(it)}
        ['baseline_config','full_config'].each{label ->
            def config=new groovy.json.JsonSlurper().parse(file(c.inputs[label].path).toFile())
            dependencies.addAll(config.inputs.values().collect{descriptor(it)})
        }
        tuple(j.job_id,p,j.input.sha256,c.resources,dependencies.unique())
    }
    R02_LAI_FLARE_EVALUATION_MATERIAL(Channel.fromList(jobs),code)
    def scores=R02_LAI_FLARE_EVALUATION_MATERIAL.out.prepared.flatMap{job,dir ->
        def p=dir.resolve('evaluation-input.json');def c=new groovy.json.JsonSlurper().parse(p.toFile())
        def deps=c.inputs.values().collect{descriptor(it)}
        c.predictions.each{row -> deps.add(descriptor(row.probabilities));deps.add(descriptor(row.expected_dosage));deps.add(descriptor(row.receipt))}
        def truth=new groovy.json.JsonSlurper().parse(file(c.inputs.truth_provenance.path).toFile())
        deps.add(descriptor(truth.recorded_truth));deps.add(descriptor(truth.simulation_manifest))
        [0.01,0.005].collect{gap ->tuple(job,gap,p,hash(p),c.resources,deps.unique())}
    }
    R02_LAI_FLARE_EVALUATION_SCORE(scores,code)
}
