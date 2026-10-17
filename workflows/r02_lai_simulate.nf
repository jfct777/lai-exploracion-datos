nextflow.enable.dsl=2

process R02_LAI_SIMULATE {
    tag "${job_id}"
    cpus 1
    memory "${memory_gib} GB"
    time "${timeout_seconds + 300}s"
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/${job_id}", mode:'copy', overwrite:false
    input:
    tuple val(job_id), path(contract,stageAs:'contract.json'), val(digest), val(memory_gib), val(timeout_seconds), path(dependencies,stageAs:'inputs/input????????')
    path code_files
    output:
    tuple val(job_id), path('simulation'), emit: simulations
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_simulate.py --contract contract.json --contract-sha256 '${digest}' --output-dir simulation
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    if (![params.campaign_contract,params.source_bin,params.output_dir].every{safe(it)}) error 'Absolute safe paths required'
    if (!(params.campaign_contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Campaign SHA256 required'
    if (!(params.max_forks?.toString() ==~ /[1-9][0-9]*/)) error 'Explicit positive max_forks required'
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {stream ->
            byte[] buffer=new byte[1048576]; int count
            while((count=stream.read(buffer))>0) md.update(buffer,0,count)
        }
        md.digest().encodeHex().toString()
    }
    def campaign_path=file(params.campaign_contract,checkIfExists:true)
    if(hash(campaign_path)!=params.campaign_contract_sha256) error 'Campaign SHA256 mismatch'
    def parse={p -> new groovy.json.JsonSlurper().parse(p.toFile())}
    def descriptor={d ->
        if (!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid input descriptor'
        file(d.path,checkIfExists:true)
    }
    def campaign=parse(campaign_path)
    if(campaign.schema!='r02_lai_simulate_campaign_v1' || !(campaign.jobs instanceof List) || !campaign.jobs) error 'Invalid simulation campaign'
    def seen=[];def namespaces=[]
    def jobs=campaign.jobs.collect{j ->
        def p=descriptor(j);if(hash(p)!=j.sha256) error 'Job contract SHA256 mismatch'
        def c=parse(p)
        if(!(c.schema in ['r02_lai_simulate_input_v1','r02_lai_simulate_input_v2']) || !(c.job_id ==~ /[A-Za-z][A-Za-z0-9_.-]{0,79}/) || seen.contains(c.job_id)) error 'Invalid/duplicate simulation job'
        seen.add(c.job_id)
        if(namespaces.contains(c.sample_namespace)) error 'Unique sample namespace per independent simulation job required'
        namespaces.add(c.sample_namespace)
        if(c.resources.cpus!=1 || !(c.resources.timeout_seconds instanceof Integer) || c.resources.timeout_seconds<1 || !(c.resources.memory_gib instanceof Number) || c.resources.memory_gib<=0) error 'Invalid task resources'
        def deps=['donor_vcf','donor_index','roles','model','admission'].collect{descriptor(c.inputs[it])}
        deps+=c.inputs.maps.values().collect{descriptor(it)}
        deps+=(c.preparation_provenance ?: [:]).values().collect{descriptor(it)}
        def admission=parse(file(c.inputs.admission.path,checkIfExists:true))
        if(c.scope=='DNABR_AUTHORIZED_SIMULATION') {
            if(admission.status!='REVIEWED_FOR_DECLARED_SCENARIO') error 'Real simulation admission not reviewed'
            deps+=admission.checks.values().collect{descriptor(it.evidence)}
        } else if(c.scope!='SYNTHETIC_TEST') error 'Explicit simulation scope required'
        // Authenticate dependencies before cache lookup, not only in Python.
        def evidence = [c.inputs.donor_vcf,c.inputs.donor_index,c.inputs.roles,c.inputs.model,c.inputs.admission]
        evidence += c.inputs.maps.values()
        evidence += (c.preparation_provenance ?: [:]).values()
        if(c.scope=='DNABR_AUTHORIZED_SIMULATION') evidence += admission.checks.values().collect{it.evidence}
        evidence.each {d -> if(hash(file(d.path,checkIfExists:true))!=d.sha256) error 'Simulation dependency SHA256 mismatch before cache lookup'}
        tuple(c.job_id,p,j.sha256,c.resources.memory_gib,c.resources.timeout_seconds,deps)
    }
    def code=['r02_lai_simulate.py','r02_lai_support_review.py'].collect{file("${params.source_bin}/${it}",checkIfExists:true)}
    R02_LAI_SIMULATE(Channel.fromList(jobs),code)
}
