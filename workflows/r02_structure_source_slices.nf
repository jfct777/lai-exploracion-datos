nextflow.enable.dsl=2

process INDEXED_SOURCE_SLICE {
    tag "source:${chrom}"
    label 'authenticated_network_source'
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/slices", mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(request), val(pin), path(inputs, stageAs:'inputs/??/*')
    path code
    output:
    tuple val(chrom), path("slice_chr${chrom}")
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1
    python3 r02_structure_source_slice.py --request ${request} --request-sha256 '${pin}' --output slice_chr${chrom}
    """
}

process EXACT_COMMON_SUPPLEMENT {
    tag "supplement:${chrom}"
    label 'offline_common_supplement'
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/supplements", mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(slice)
    path batch
    path requests, stageAs:'requests/*'
    path loci, stageAs:'loci/*'
    path metadata, stageAs:'metadata/*'
    path code
    output:
    tuple val(chrom), path("common_supplement_chr${chrom}")
    path "supplement_request_chr${chrom}.json"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 - <<'PY'
    import json
    from pathlib import Path
    import r02_structure_common_supplement as mod
    batch=json.loads(Path('${batch}').read_text())
    row=next(v for v in batch['slices']if v['chromosome']=='${chrom}')
    directory=Path('${slice}').resolve()
    receipt=json.loads((directory/'manifest.json').read_text())
    mod.require(receipt['request_sha256']==row['request_sha256'],'Recovered slice request differs')
    request=dict(schema=mod.SCHEMA,source_contract_sha256=batch['bundle']['sha256'],implementation_sha256=mod.implementation(),
      chromosome='${chrom}',slice_request=dict(path=row['request'],sha256=row['request_sha256']),
      slice_manifest=dict(path=str(directory/'manifest.json'),sha256=mod.shared.sha(directory/'manifest.json')),
      slice_vcf=dict(path=str(directory/'selected.vcf.gz'),sha256=mod.shared.sha(directory/'selected.vcf.gz')),
      loci=row['loci'],resources=batch['supplement_resources'])
    path=Path('supplement_request_chr${chrom}.json');path.write_text(json.dumps(request,indent=2)+'\\n')
    mod.extract(batch['bundle']['path'],path,mod.shared.sha(path),'common_supplement_chr${chrom}')
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
        if(!safe(name)||!(pin ==~ /[a-f0-9]{64}/)) error 'Safe pinned slice path required'
        def p=file(name,checkIfExists:true)
        if(hash(p)!=pin) error 'Source-slice input changed before cache lookup'
        p
    }
    if(![params.batch,params.code_dir,params.output_dir].every{safe(it)}||
       !(params.mode in ['extract_and_supplement','supplement_from_verified_slices'])||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)||!(params.max_forks?.toString() ==~ /[1-9][0-9]*/)||
       !(params.task_time instanceof String)||!(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit bounded source campaign required'
    def batch=pinned(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(b.schema!='r02_structure_source_slices_batch_v1') error 'Wrong source-slice batch'
    def sourceFile=pinned(b.bundle.path+'/materialize.json',b.bundle.sha256)
    def source=new groovy.json.JsonSlurper().parse(sourceFile.toFile())
    def metadata=[sourceFile]+source.metadata_sha256.collect{n,h -> pinned(b.bundle.path+'/'+n,h)}
    def code=b.implementation_sha256.collect{n,h ->
        if(!(n ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe source consumer name'
        pinned(params.code_dir+'/'+n,h)
    }
    def requests=[];def loci=[]
    def jobs=b.slices.collect{d ->
        if(!(d.chromosome ==~ /[1-9]|1[0-9]|2[0-2]/)) error 'Autosomal source required'
        def request=pinned(d.request,d.request_sha256);requests.add(request)
        def r=new groovy.json.JsonSlurper().parse(request.toFile())
        if(r.schema!='r02_structure_indexed_source_slice_v1'||r.implementation_sha256!=b.implementation_sha256['r02_structure_source_slice.py']||
           r.source.sha256!=source.sources[d.chromosome].common_vcf.sha256) error 'Slice source/code differs'
        loci.add(pinned(d.loci.path,d.loci.sha256))
        def inputs=['index','regions','samples'].collect{k -> pinned(r[k].path,r[k].sha256)}
        tuple(d.chromosome,request,d.request_sha256,inputs)
    }
    if(jobs.isEmpty()||jobs.collect{it[0]}.toSet().size()!=jobs.size()) error 'Duplicate/empty source chromosomes'
    if(params.mode=='extract_and_supplement') {
        if(params.network_source_reads_admitted!=true) error 'Explicit source network-read admission required'
        INDEXED_SOURCE_SLICE(Channel.fromList(jobs),code)
        EXACT_COMMON_SUPPLEMENT(INDEXED_SOURCE_SLICE.out,batch,requests,loci,metadata,code)
    } else {
        def recovered=b.slices.collect{d ->
            def manifest=pinned(d.completed_slice.directory+'/manifest.json',d.completed_slice.manifest_sha256)
            def m=new groovy.json.JsonSlurper().parse(manifest.toFile())
            if(m.request_sha256!=d.request_sha256) error 'Recovered source request differs'
            pinned(d.completed_slice.directory+'/selected.vcf.gz',m.output_sha256)
            tuple(d.chromosome,file(d.completed_slice.directory,checkIfExists:true))
        }
        EXACT_COMMON_SUPPLEMENT(Channel.fromList(recovered),batch,requests,loci,metadata,code)
    }
}
