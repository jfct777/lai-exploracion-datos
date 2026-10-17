nextflow.enable.dsl=2

process R02_LAI_FLARE_MATERIAL_PAIR {
    tag "$job_id"
    cpus 1
    memory '2 GB'
    time { (settings.resources.timeout_seconds + 60L) + ' sec' }
    maxForks params.max_forks
    cache 'deep'
    container { params.material_container }
    publishDir { params.output_dir+'/'+job_id+'/material' }, mode:'copy', overwrite:false
    input:
    tuple val(job_id), path(contract, stageAs:'contract.json'), val(digest), val(settings), path(dependencies, stageAs:'inputs/input????????')
    path code_files
    output:
    tuple val(job_id), path('material'), emit: prepared
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_flare_material.py --contract contract.json --contract-sha256 '$digest' --output-dir material
    """
}

process R02_LAI_FLARE_PAIRED_INFERENCE {
    tag "$job_id:$arm"
    cpus { settings.parameters.nthreads }
    memory { (settings.resources.java_heap_mib + 1024) + ' MB' }
    time { (2L*settings.resources.max_wall_seconds + 60L) + ' sec' }
    maxForks params.max_forks
    cache 'deep'
    container { params.inference_container }
    publishDir { params.output_dir+'/'+job_id+'/models/'+arm }, mode:'copy', overwrite:false
    input:
    tuple val(job_id), val(arm), path(contract, stageAs:'contract.json'), val(digest), val(settings), path(dependencies, stageAs:'inputs/input????????')
    path code_files
    output:
    tuple val(job_id), val(arm), path('preflight'), path('inference'), emit: predictions
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_flare_plan.py --contract contract.json --contract-sha256 '$digest' --output-dir preflight
    plan_digest=\$(sha256sum preflight/manifest.json | cut -d' ' -f1)
    python3 r02_lai_flare_run.py --contract contract.json --contract-sha256 '$digest' \\
      --plan-dir preflight --plan-manifest-sha256 "\$plan_digest" --output-dir inference
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.campaign,params.source_bin,params.output_dir].every{safe(it)}) error 'Absolute safe paths required'
    if(!(params.campaign_sha256 ==~ /[0-9a-f]{64}/)) error 'Campaign SHA256 required'
    if(!(params.max_forks instanceof Integer) || params.max_forks<1 || params.max_forks>4) error 'Explicit max_forks1–4 required'
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {stream -> byte[] buffer=new byte[1048576];int n;while((n=stream.read(buffer))>0)md.update(buffer,0,n)}
        md.digest().encodeHex().toString()
    }
    def descriptor={d ->
        if(!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid input descriptor'
        def p=file(d.path,checkIfExists:true)
        if(hash(p)!=d.sha256) error 'Source SHA256 mismatch before cache lookup'
        p
    }
    def campaign=file(params.campaign,checkIfExists:true)
    if(hash(campaign)!=params.campaign_sha256) error 'Campaign changed before cache lookup'
    def c=new groovy.json.JsonSlurper().parse(campaign.toFile())
    if(c.schema!='r02_lai_flare_pair_campaign_v1' || !c.jobs || c.jobs.collect{it.job_id}.unique().size()!=c.jobs.size())error 'Explicit unique paired jobs required'
    def jobs=c.jobs.collect{j ->
        if(!(j.job_id ==~ /[A-Za-z0-9_.-]+/))error 'Unsafe job ID'
        def p=descriptor(j.input);def settings=new groovy.json.JsonSlurper().parse(p.toFile())
        if(settings.schema!='r02_lai_flare_material_input_v1')error 'Wrong material schema'
        tuple(j.job_id,p,j.input.sha256,settings,settings.inputs.values().collect{descriptor(it)})
    }
    def names=['r02_lai_flare_material.py','r02_lai_flare_plan.py','r02_lai_flare_run.py','m34_run_flare.py','r02_lai_simulate.py','r02_lai_support_review.py']
    def code=names.collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    R02_LAI_FLARE_MATERIAL_PAIR(Channel.fromList(jobs),code)
    def plans=R02_LAI_FLARE_MATERIAL_PAIR.out.prepared.flatMap{job,p ->
        def m=new groovy.json.JsonSlurper().parse(p.resolve('manifest.json').toFile())
        if(m.status!='PAIRED_PREDICTOR_INPUTS_PREPARED_NO_INFERENCE' || m.plans.size()!=2)error 'Incomplete paired material'
        (m.inference_plans ?: m.plans).collect{d ->
            def cp=descriptor(d);def settings=new groovy.json.JsonSlurper().parse(cp.toFile())
            def label=settings.marker_condition+(settings.initialization_label ? '.'+settings.initialization_label : '')
            tuple(job,label,cp,d.sha256,settings,settings.inputs.values().collect{descriptor(it)})
        }.findAll{row -> !params.admitted_initial_generations || params.admitted_initial_generations.contains(row[4].parameters.gen)}
    }
    R02_LAI_FLARE_PAIRED_INFERENCE(plans,code)
}
