nextflow.enable.dsl=2

process RARE_GLOBAL_COUNTS {
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir "${params.output_dir}/global", mode:'copy', overwrite:false
    input:
    tuple path(request), val(pin), path(inputs, stageAs:'inputs/??/*')
    path metadata, stageAs:'metadata/*'
    path code
    output:
    path 'rare_global'
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_rare_affinities.py --stage reduce --bundle '${params.bundle}' --request '${request}' --request-sha256 '${pin}' --output-dir rare_global
    """
}

process RARE_AFFINITY_VIEW {
    tag "${scope}:${block}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/views", mode:'copy', overwrite:false
    input:
    tuple val(scope), val(block), path(global)
    path metadata, stageAs:'metadata/*'
    path sources, stageAs:'inputs/??/*'
    path code
    output:
    path "affinities_${scope}_${block}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 - <<'PY'
    import r02_structure_rare_affinities as mod
    mod.view('${params.bundle}','${global}',mod.material.sha('${global}/manifest.json'),'${scope}','${block}','affinities_${scope}_${block}')
    PY
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
        if(!safe(name)||!(pin ==~ /[a-f0-9]{64}/)) error 'Safe pinned affinity path required'
        def p=file(name,checkIfExists:true)
        if(hash(p)!=pin) error 'Affinity source changed before cache lookup'
        p
    }
    if(![params.batch,params.bundle,params.code_dir,params.output_dir].every{safe(it)}||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)||!(params.max_forks?.toString() ==~ /[1-9][0-9]*/)||
       !(params.task_time instanceof String)||!(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit bounded affinity campaign required'
    def batch=pinned(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(b.schema!='r02_structure_rare_affinity_batch_v1') error 'Exact affinity batch required'
    def request=pinned(b.request,b.request_sha256)
    def r=new groovy.json.JsonSlurper().parse(request.toFile())
    if(r.schema!='r02_structure_rare_affinity_assembly_v1'||r.implementation_sha256!=b.implementation_sha256) error 'Affinity request/code differs'
    def sourceFile=pinned(params.bundle+'/materialize.json',r.source_contract_sha256)
    def source=new groovy.json.JsonSlurper().parse(sourceFile.toFile())
    def metadata=[sourceFile]+source.metadata_sha256.collect{n,h -> pinned(params.bundle+'/'+n,h)}
    def code=b.implementation_sha256.collect{n,h ->
        if(!(n ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe affinity consumer name'
        pinned(params.code_dir+'/'+n,h)
    }
    def inputs=[]
    r.chromosomes.each{chrom,d ->
        def manifest=pinned(d.directory+'/manifest.json',d.manifest_sha256);inputs.add(manifest)
        def m=new groovy.json.JsonSlurper().parse(manifest.toFile())
        m.tasks.each{key,row ->
            if(row.reused_original_m02) {
                inputs.add(pinned(row.reused_original_m02.manifest.path,row.reused_original_m02.manifest.sha256))
                inputs.add(pinned(row.reused_original_m02.npz.path,row.reused_original_m02.npz.sha256))
            } else inputs.add(pinned(d.directory+'/'+row.path,row.sha256))
        }
    }
    def blockIds=file(params.bundle+'/blocks.tsv').readLines().drop(1).collect{it.split('\t')[0]}
    def expected=['BLOCK_AND_LD_MARGIN','WHOLE_CHROMOSOME'].collectMany{s -> blockIds.collect{v -> s+':'+v}}
    if(!b.views||b.views.collect{it.scope+':'+it.block_id}.sort()!=expected.sort()) error 'Both geometries and every original target required'
    b.views.each{v -> if(!(v.block_id ==~ /[A-Za-z][A-Za-z0-9_]*/)) error 'Unsafe affinity block ID'}
    RARE_GLOBAL_COUNTS(Channel.value(tuple(request,b.request_sha256,inputs)),metadata,code)
    def views=RARE_GLOBAL_COUNTS.out.flatMap{g -> b.views.collect{v -> tuple(v.scope,v.block_id,g)}}
    RARE_AFFINITY_VIEW(views,metadata,[request]+inputs,code)
}
