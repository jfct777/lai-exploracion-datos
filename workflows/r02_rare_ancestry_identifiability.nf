nextflow.enable.dsl=2

// One existing event chunk. No discovery, whole-genome launch, or rephasing.
process R02_RARE_ANCESTRY_IDENTIFIABILITY {
    tag 'conditional_copy_bands'
    cpus 1
    cache 'deep'
    input:
    path contract, stageAs:'contract.json'
    val digest
    path inputs, stageAs:'inputs/*'
    path code
    output:
    path 'identifiability', emit:evidence
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    mkdir scratch
    python3 r02_rare_ancestry_identifiability.py --contract contract.json \\
      --contract-sha256 '$digest' --inputs-dir inputs --scratch scratch --output-dir identifiability
    """
}

workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') &&
        !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.contract,params.source_bin].every{safe.call(it)} || !(params.contract_sha256 ==~ /[0-9a-f]{64}/))
        error 'Absolute safe input paths and contract SHA256 required'
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {stream ->
            def hashing = new java.security.DigestInputStream(stream, md)
            hashing.transferTo(java.io.OutputStream.nullOutputStream())
        }
        md.digest().encodeHex().toString()
    }
    def descriptor={d ->
        if(!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe.call(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/))
            error 'Invalid dependency descriptor'
        def p=file(d.path,checkIfExists:true)
        if(hash.call(p)!=d.sha256) error 'Input hash differs before cache lookup'
        p
    }
    def contract=descriptor.call([path:params.contract,sha256:params.contract_sha256])
    def c=new groovy.json.JsonSlurper().parse(contract.toFile())
    def names=['ancestry_contract','event_manifest','material_contract','persons','msp','fb','crosswalk','event_chunk']
    if(c.schema!='r02_rare_ancestry_identifiability_v1' || c.inputs.keySet()!=names.toSet()) error 'Contract input schema differs'
    def inputs=names.collect{descriptor.call(c.inputs[it])}
    if(inputs.collect{it.name}.toSet().size()!=inputs.size()) error 'Unique input basenames required'
    def modules=['r02_rare_ancestry_identifiability.py','r02_ibd_local_ancestry.py','r02_ibd_compatibility.py',
                 'r02_ibd_prepare.py','r02_genomic_pair_evidence.py','r02_segment_evidence.py']
    def code=modules.collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    R02_RARE_ANCESTRY_IDENTIFIABILITY(contract,params.contract_sha256,inputs,code)
}
