nextflow.enable.dsl=2

process R02_LAI_SIMULATION_PLAN {
    cpus 1
    memory '2 GB'
    time '10 min'
    maxForks 1
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path contract, stageAs:'contract.json'
    val digest
    path dependencies, stageAs:'inputs/input????????'
    path code_files
    output:
    path 'plan', emit: plan
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_simulation_plan.py --contract contract.json --contract-sha256 '$digest' --output-dir plan
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.contract,params.source_bin,params.output_dir].every{safe(it)}) error 'Absolute safe paths required'
    if(!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Plan contract SHA256 required'
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {stream ->
            byte[] buffer=new byte[1048576]; int count
            while((count=stream.read(buffer))>0) md.update(buffer,0,count)
        }
        md.digest().encodeHex().toString()
    }
    def descriptor={d ->
        if(!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid source descriptor'
        def p=file(d.path,checkIfExists:true)
        if(hash(p)!=d.sha256) error 'Plan source SHA256 mismatch before cache lookup'
        p
    }
    def parse={p -> new groovy.json.JsonSlurper().parse(p.toFile())}
    def cpath=file(params.contract,checkIfExists:true)
    if(hash(cpath)!=params.contract_sha256) error 'Plan contract SHA256 mismatch before cache lookup'
    def c=parse(cpath)
    if(c.schema!='r02_lai_simulation_plan_input_v1') error 'Invalid plan schema'
    def report=descriptor(c.material_report); def scenario=descriptor(c.scenario)
    def dependencies=[report,scenario,descriptor(c.original_roles)]
    def material=parse(report)
    material.outputs_sha256.each {name,digest ->
        if(!(name ==~ /[A-Za-z0-9][A-Za-z0-9._-]*/) || name=='manifest.json') error 'Unsafe material output'
        dependencies+=descriptor([path:report.parent.resolve(name).toString(),sha256:digest])
    }
    def job=parse(scenario)
    dependencies+=['donor_vcf','donor_index','roles','model'].collect{descriptor(job.inputs[it])}
    dependencies+=job.inputs.maps.values().collect{descriptor(it)}
    if(c.review!=null) {
        def review=descriptor(c.review); dependencies+=review
        def decision=parse(review)
        dependencies+=(decision.checks ?: [:]).values().findAll{it.evidence!=null}.collect{descriptor(it.evidence)}
    }
    def code=['r02_lai_simulation_plan.py','r02_lai_simulate.py','r02_lai_support_review.py'].collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    R02_LAI_SIMULATION_PLAN(cpath,params.contract_sha256,dependencies,code)
}
