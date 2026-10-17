nextflow.enable.dsl=2

process R02_RARE_ANCESTRY_PAIR_BOUNDS {
    tag "chr${chromosome}"
    cpus 2
    memory "${memory_gib} GB"
    time "${timeout_seconds+300}s"
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/chr${chromosome}", mode:'copy', overwrite:false
    input:
    tuple val(chromosome), path(contract,stageAs:'contract.json'), val(digest), val(memory_gib), val(timeout_seconds), path(inputs,stageAs:'inputs/*')
    path code
    output:
    tuple val(chromosome), path('identifiability'), emit: summaries
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    mkdir scratch
    python3 r02_rare_ancestry_pair_bounds.py --contract contract.json \\
      --contract-sha256 '$digest' --inputs-dir inputs --scratch scratch --output-dir identifiability
    """
}
workflow {
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') &&
        !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.campaign,params.source_bin,params.output_dir].every{safe.call(it)} || !(params.campaign_sha256 ==~ /[0-9a-f]{64}/))
        error 'Absolute safe input paths and campaign SHA256 required'
    if(!(params.max_forks?.toString() ==~ /[1-9][0-9]*/)) error 'Explicit positive concurrency required'
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {stream ->
            def hashing=new java.security.DigestInputStream(stream,md)
            hashing.transferTo(java.io.OutputStream.nullOutputStream())
        }
        md.digest().encodeHex().toString()
    }
    def descriptor={d ->
        if(!(d instanceof Map) || d.keySet()!=['path','sha256'].toSet() || !safe.call(d.path) || !(d.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid dependency descriptor'
        def p=file(d.path,checkIfExists:true)
        if(hash.call(p)!=d.sha256) error 'Input SHA256 differs before cache lookup'
        p
    }
    def path=descriptor.call([path:params.campaign,sha256:params.campaign_sha256])
    def parse={p -> new groovy.json.JsonSlurper().parse(p.toFile())}
    def c=parse(path)
    if(c.schema!='r02_rare_ancestry_campaign_v1' || !(c.jobs instanceof List) || !c.jobs) error 'Explicit chromosome campaign required'
    def names=['ancestry_contract','event_manifest','material_contract','persons','msp','fb','crosswalk','event_archive']
    def seen=[]
    def jobs=c.jobs.collect{j ->
        if(!(j.chromosome?.toString() ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Autosome required'
        if(seen.contains(j.chromosome)) error 'Duplicate chromosome'
        seen.add(j.chromosome)
        def p=descriptor.call(j.input);def spec=parse(p)
        if(spec.schema!='r02_rare_ancestry_chromosome_input_v1' || spec.inputs.keySet()!=names.toSet()) error 'Chromosome input inventory differs'
        if(!(j.memory_gib instanceof Number) || j.memory_gib<=0 || !(spec.resources.timeout_seconds instanceof Integer) || spec.resources.timeout_seconds<=0) error 'Explicit bounded resources required'
        def inputs=names.collect{descriptor.call(spec.inputs[it])}
        if(inputs.collect{it.name}.toSet().size()!=inputs.size()) error 'Unique input basenames required'
        if(parse(file(spec.inputs.ancestry_contract.path)).chromosome.toString()!=j.chromosome.toString()) error 'Chromosome identity differs'
        tuple(j.chromosome,p,j.input.sha256,j.memory_gib,spec.resources.timeout_seconds,inputs)
    }
    def modules=['r02_rare_ancestry_pair_bounds.py','r02_rare_ancestry_chromosome.py','r02_rare_ancestry_identifiability.py','r02_ibd_local_ancestry.py','r02_ibd_compatibility.py','r02_ibd_prepare.py','r02_genomic_pair_evidence.py','r02_segment_evidence.py']
    def code=modules.collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    R02_RARE_ANCESTRY_PAIR_BOUNDS(Channel.fromList(jobs),code)
}
