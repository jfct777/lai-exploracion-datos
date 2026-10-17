nextflow.enable.dsl=2

process R02_LAI_EVALUATE {
    cpus 1
    memory { limits.max_working_bytes + 256*1024*1024 + ' B' }
    time { limits.timeout_seconds + 60 + 's' }
    maxForks 1
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path contract, stageAs:'contract.json'
    val digest
    val limits
    path evidence, stageAs:'evidence/input????????'
    path code, stageAs:'code/*'
    output:
    path 'evaluation', emit: report
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_lai_evaluate.py --contract contract.json --contract-sha256 '${digest}' --output-dir evaluation
    """
}

workflow {
    def safe={ p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.contract, params.source_bin, params.output_dir].every { safe(it) }) error 'Safe absolute paths required'
    if (!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Contract SHA256 required'
    // Hash by streaming: prediction arrays may be larger than controller RAM.
    def hash={ p ->
        def d=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream { stream ->
            byte[] block=new byte[1048576]
            int n
            while ((n=stream.read(block)) != -1) d.update(block,0,n)
        }
        d.digest().encodeHex().toString()
    }
    def cp=file(params.contract,checkIfExists:true)
    if (cp.size()>1048576 || hash(cp)!=params.contract_sha256) error 'Contract hash/size differs before cache lookup'
    def c=new groovy.json.JsonSlurper().parse(cp.toFile())
    if (c.schema!='r02_lai_evaluate_input_v1') error 'Unsupported evaluation contract'
    def evidence=[]
    def checked=[] as Set
    def visit
    visit={ v ->
        if(v instanceof Map) {
            if(v.keySet()==['path','sha256'].toSet()) {
                if (!safe(v.path) || !(v.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid evidence path/SHA256'
                def p=file(v.path,checkIfExists:true)
                if(hash(p)!=v.sha256) error 'Evidence SHA256 differs before cache lookup'
                evidence.add(p)
                if(p.name.endsWith('.json') && checked.add(p.toString())) {
                    if(p.size()>1048576) error 'Metadata receipt exceeds one MiB'
                    visit(new groovy.json.JsonSlurper().parse(p.toFile()))
                }
            } else v.values().each { visit(it) }
        } else if(v instanceof List) v.each { visit(it) }
    }
    visit(c)
    def names=['r02_lai_evaluate.py','m39_calibration.py','m34_score_predictions.py','m37_trace_score.py','m37_trace_core.py']
    if(c.code_sha256.keySet()!=names.toSet()) error 'Metric dependency inventory differs'
    def code=names.collect { name ->
        def p=file("${params.source_bin}/${name}",checkIfExists:true)
        if(hash(p)!=c.code_sha256[name]) error 'Scoring code changed before cache lookup'
        p
    }
    R02_LAI_EVALUATE(cp,params.contract_sha256,c.resources,evidence.unique(),code)
}
