nextflow.enable.dsl=2

process RARE_INTERVAL_COUNTS {
    tag "rare:${chrom}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(request), val(pin), path(inputs, stageAs:'inputs/??/*')
    path metadata, stageAs:'metadata/*'
    path code
    output:
    tuple val(chrom), path("rare_counts_chr${chrom}")
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_rare_inputs.py --bundle '${params.bundle}' --request '${request}' --request-sha256 '${pin}' --output-dir rare_counts_chr${chrom}
    """
}

workflow {
    def safe={s -> s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') && !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {s -> new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())}
        md.digest().encodeHex().toString()
    }
    def pinned={name,pin ->
        if(!safe(name)||!(pin ==~ /[a-f0-9]{64}/)) error 'Safe pinned rare source path required'
        def p=file(name,checkIfExists:true)
        if(hash(p)!=pin) error 'Rare source changed before cache lookup'
        p
    }
    if(![params.batch,params.bundle,params.code_dir,params.output_dir].every{safe(it)}||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)||!(params.max_forks?.toString() ==~ /[1-9][0-9]*/)||
       !(params.task_time instanceof String)||!(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit bounded rare campaign required'
    def batch=pinned(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(b.schema!='r02_structure_rare_interval_batch_v1'||!b.chromosomes||b.chromosomes.collect{it.chromosome}.toSet().size()!=b.chromosomes.size()) error 'Unique rare chromosome batch required'
    def sourceFile=pinned(params.bundle+'/materialize.json',b.source_contract_sha256)
    def source=new groovy.json.JsonSlurper().parse(sourceFile.toFile())
    def metadata=[sourceFile]+source.metadata_sha256.collect{n,h -> pinned(params.bundle+'/'+n,h)}
    def code=b.implementation_sha256.collect{n,h ->
        if(!(n ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe rare consumer name'
        pinned(params.code_dir+'/'+n,h)
    }
    def jobs=b.chromosomes.collect{d ->
        if(!(d.chromosome ==~ /[1-9]|1[0-9]|2[0-2]/)) error 'Rare autosome required'
        def request=pinned(d.request,d.sha256)
        def r=new groovy.json.JsonSlurper().parse(request.toFile())
        if(r.schema!='r02_structure_rare_interval_inputs_v1'||r.chromosome!=d.chromosome||
           r.implementation_sha256!=b.implementation_sha256||r.source_contract_sha256!=b.source_contract_sha256) error 'Rare request/source/code differs'
        def inputs=[r.event_manifest,r.event_archive,r.tails.request,r.tails.manifest,r.tails.vcf,r.m02.manifest,r.m02.npz].collect{v -> pinned(v.path,v.sha256)}
        def tr=new groovy.json.JsonSlurper().parse(file(r.tails.request.path).toFile())
        inputs.addAll([tr.regions,tr.samples].collect{v -> pinned(v.path,v.sha256)})
        r.partials.each{v ->
            def mfile=pinned(v.directory+'/manifest.json',v.manifest_sha256)
            def m=new groovy.json.JsonSlurper().parse(mfile.toFile())
            inputs.add(mfile);inputs.add(pinned(v.directory+'/data.npz',m.outputs_sha256['data.npz']))
        }
        tuple(d.chromosome,request,d.sha256,inputs)
    }
    RARE_INTERVAL_COUNTS(Channel.fromList(jobs),metadata,code)
}
