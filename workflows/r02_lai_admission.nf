nextflow.enable.dsl=2

process R02_LAI_ADMISSION {
    cpus 1
    memory '2 GB'
    maxForks 1
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path contract, stageAs:'contract.json'
    val digest
    path evidence, stageAs:'evidence/input????????'
    path code, stageAs:'code/*'
    output:
    path 'admission', emit: report
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python3 code/r02_lai_admission.py --contract contract.json --contract-sha256 '${digest}' --output-dir admission
    """
}

workflow {
    def safe={ p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.contract,params.source_bin,params.output_dir].every { safe(it) }) error 'Safe absolute paths required'
    if (!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Contract SHA256 required'
    def cp=file(params.contract,checkIfExists:true)
    def hash={ p -> java.security.MessageDigest.getInstance('SHA-256').digest(p.toFile().bytes).encodeHex().toString() }
    if (hash(cp)!=params.contract_sha256) error 'Contract SHA256 mismatch before cache lookup'
    def c=new groovy.json.JsonSlurper().parse(cp.toFile())
    if (c.schema!='r02_lai_admission_input_v1') error 'Unsupported admission contract'
    def evidence=[]
    def visit
    visit={ v ->
        if (v instanceof Map) {
            if (v.keySet()==['path','sha256'].toSet()) {
                if (!safe(v.path) || !(v.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid evidence descriptor'
                def p=file(v.path,checkIfExists:true)
                if(hash(p)!=v.sha256) error 'Evidence SHA256 mismatch before cache lookup'
                evidence.add(p)
                if(p.name=='manifest.json') {
                    def m=new groovy.json.JsonSlurper().parse(p.toFile())
                    m.outputs_sha256?.each { name,d ->
                        if (!(name ==~ /[A-Za-z0-9_.-]+/) || name=='manifest.json') error 'Unsafe/self-referential manifest output'
                        def artifact=file(p.parent.resolve(name),checkIfExists:true)
                        if(hash(artifact)!=d) error 'Artifact SHA256 mismatch before cache lookup'
                        evidence.add(artifact)
                    }
                }
            } else v.values().each { visit(it) }
        } else if(v instanceof List) v.each { visit(it) }
    }
    visit(c)
    def names=['r02_lai_admission.py','r02_lai_material_decision.py','r02_lai_simulation_material.py','r02_lai_support_review.py']
    R02_LAI_ADMISSION(cp,params.contract_sha256,evidence.unique(),names.collect { file("${params.source_bin}/${it}",checkIfExists:true) })
}
