nextflow.enable.dsl=2

process R02_LAI_FLARE_PLAN {
    cpus 1
    memory '2 GB'
    // Preserve the four-input process interface used by both entry workflows.
    // The per-task contract budget includes Python input hashes/audits; the
    // extra 30 seconds are executor cleanup allowance, not extra Java time.
    time {
        def seconds = new groovy.json.JsonSlurper().parse(contract.toFile()).resources.max_wall_seconds
        if (!(seconds instanceof Integer || seconds instanceof Long) || seconds <= 0)
            error 'FLARE max_wall_seconds must be a positive integer'
        (seconds + 30L) + ' sec'
    }
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
    python3 r02_lai_flare_plan.py --contract contract.json --contract-sha256 '$digest' --output-dir plan
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.contract,params.source_bin,params.output_dir].every{safe(it)}) error 'Absolute safe paths required'
    if(!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'FLARE plan contract SHA256 required'
    // These pre-cache driver hashes run outside any process time directive.
    // The caller must also bound the whole Nextflow controller/VM lifetime;
    // max_wall_seconds is a budget for each task, not the entire workflow.
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
        if(hash(p)!=d.sha256) error 'FLARE source SHA256 mismatch before cache lookup'
        p
    }
    def cpath=file(params.contract,checkIfExists:true)
    if(hash(cpath)!=params.contract_sha256) error 'FLARE plan contract SHA256 mismatch before cache lookup'
    def c=new groovy.json.JsonSlurper().parse(cpath.toFile())
    if(c.schema!='r02_lai_flare_plan_input_v1') error 'Invalid FLARE plan schema'
    def dependencies=c.inputs.values().collect{descriptor(it)}
    def code=['r02_lai_flare_plan.py','m34_run_flare.py','r02_lai_simulate.py','r02_lai_support_review.py'].collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    R02_LAI_FLARE_PLAN(cpath,params.contract_sha256,dependencies,code)
}
