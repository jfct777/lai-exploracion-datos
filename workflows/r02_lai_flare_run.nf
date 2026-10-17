nextflow.enable.dsl=2

include { R02_LAI_FLARE_PLAN } from './r02_lai_flare_plan.nf'

process R02_LAI_FLARE_INFERENCE {
    cpus { settings.parameters.nthreads }
    memory { (settings.resources.java_heap_mib + 1024) + ' MB' }
    // Includes executor-side input hashes and posterior audit. Java itself is
    // monitored against max_wall_seconds; 30 seconds allow process cleanup.
    time { (settings.resources.max_wall_seconds + 30L) + ' sec' }
    maxForks 1
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path contract, stageAs:'contract.json'
    val digest
    path dependencies, stageAs:'inputs/input????????'
    path code_files
    tuple path(preflight, stageAs:'preflight'), val(preflight_digest)
    val settings
    output:
    path 'inference', emit: inference
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_flare_run.py --contract contract.json --contract-sha256 '$digest' \
        --plan-dir preflight --plan-manifest-sha256 '$preflight_digest' --output-dir inference
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.contract,params.source_bin,params.output_dir].every{safe(it)}) error 'Absolute safe paths required'
    if(!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'FLARE execution contract SHA256 required'
    // Pre-cache driver hashes are not covered by task time directives. The
    // production launcher must bound the complete Nextflow/VM lifetime too.
    // Preparation and inference each receive their own per-task budget.
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {stream ->
            byte[] buffer=new byte[1048576]; int count
            while((count=stream.read(buffer))>0) md.update(buffer,0,count)
        }
        md.digest().encodeHex().toString()
    }
    def cpath=file(params.contract,checkIfExists:true)
    if(hash(cpath)!=params.contract_sha256) error 'FLARE contract changed before cache lookup'
    def c=new groovy.json.JsonSlurper().parse(cpath.toFile())
    if(c.schema!='r02_lai_flare_plan_input_v1') error 'Invalid FLARE input schema'
    def dependencies=c.inputs.values().collect{d ->
        if(!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid input descriptor'
        def p=file(d.path,checkIfExists:true)
        if(hash(p)!=d.sha256) error 'FLARE input changed before cache lookup'
        p
    }
    def code=['r02_lai_flare_plan.py','r02_lai_flare_run.py','m34_run_flare.py','r02_lai_simulate.py','r02_lai_support_review.py'].collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    R02_LAI_FLARE_PLAN(cpath,params.contract_sha256,dependencies,code)
    def reviewed=R02_LAI_FLARE_PLAN.out.plan.map{ p -> tuple(p,hash(p.resolve('manifest.json'))) }
    R02_LAI_FLARE_INFERENCE(cpath,params.contract_sha256,dependencies,code,reviewed,c)
}
