nextflow.enable.dsl=2

process K03_PREPARE_FAMILY {
    tag "${viewId}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(viewId), path(request), val(requestPin), path(admission), val(admissionPin), path(inputs,stageAs:'inputs/??/*')
    path code
    output:
    path "prepared_${viewId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_k03_prepare.py --request '${request}' --request-sha256 '${requestPin}' \
      --admission '${admission}' --admission-sha256 '${admissionPin}' --output prepared_${viewId}
    """
}

workflow {
    def safe={s -> s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') && !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream{s -> new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())}
        md.digest().encodeHex().toString()
    }
    def pinned={name,pin ->
        if(!safe(name)||!(pin ==~ /[a-f0-9]{64}/)) error 'Safe absolute K03 preparation source required'
        def p=file(name,checkIfExists:true)
        if(hash(p)!=pin) error 'K03 preparation input changed before cache lookup'
        p
    }
    def read={p -> new groovy.json.JsonSlurper().parse(p.toFile())}
    if(![params.batch,params.code_dir,params.output_dir].every{safe(it)}||
       ![params.memory_gb,params.max_forks].every{it?.toString() ==~ /[1-9][0-9]*/}||
       !(params.task_time instanceof String)||!(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit bounded K03 preparation resources required'
    def b=read(pinned(params.batch,params.batch_sha256))
    if(b.schema!='r02_k03_prepare_batch_v1'||!b.views||b.views.collect{it.id}.toSet().size()!=b.views.size()) error 'Unique admitted K03 preparation views required'
    def code=b.implementation_sha256.collect{n,h ->
        if(!(n ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe K03 preparation module'
        pinned(params.code_dir+'/'+n,h)
    }
    def jobs=b.views.collect{d ->
        if(!(d.id ==~ /[A-Za-z0-9_]+/)) error 'Unsafe K03 preparation id'
        def request=pinned(d.request,d.request_sha256);def r=read(request)
        if(r.schema!='r02_k03_prepare_request_v1'||r.implementation_sha256!=b.implementation_sha256||
           r.scientific_spec_sha256!=b.scientific_spec_sha256) error 'K03 preparation family/source differs'
        def admission=pinned(d.admission,d.admission_sha256);def a=read(admission)
        if(a.schema!='r02_k03_prepare_admission_v1'||a.status!='FIT_SELECT_INPUTS_ADMITTED'||a.evaluate_allowed!=false||
           a.evidence_scope!=r.evidence_scope||a.request_sha256!=d.request_sha256||a.scientific_spec_sha256!=r.scientific_spec_sha256) error 'Exact K03 preparation admission required before cache'
        def inputs=[]
        r.sources.rare.each{n,v -> inputs.add(pinned(v.path,v.sha256))}
        ['common','profile'].each{k ->
            def desc=r.sources[k];def name=k=='common'?'manifest.json':'profile_manifest.json'
            def p=pinned(desc.directory+'/'+name,desc.manifest_sha256);inputs.add(p)
            read(p).outputs_sha256.each{n,h ->
                if(n.startsWith('/')||n.tokenize('/').contains('..')) error 'Unsafe K03 producer artifact'
                inputs.add(pinned(desc.directory+'/'+n,h))
            }
        }
        def pd=r.sources.prepared;def contract=pinned(pd.directory+'/contract.json',pd.contract_sha256);inputs.add(contract)
        def c=read(contract);c.sha256.each{n,h ->
            if(!(n in ['persons.tsv','blocks.tsv','block_index.json','protocol.md'])) error 'Unexpected K03 target catalogue file'
            inputs.add(pinned(pd.directory+'/'+n,h))
        }
        def index=read(file(pd.directory+'/block_index.json'));def matches=index.blocks.findAll{it.block_id==r.block_id}
        if(matches.size()!=1) error 'Exact K03 target block required'
        def shard=matches[0]
        if(shard.path.startsWith('/')||shard.path.tokenize('/').contains('..')) error 'Unsafe K03 target shard'
        def dir=pd.directory+'/'+shard.path
        inputs.add(pinned(dir+'/contract.json',shard.contract_sha256))
        def manifest=pinned(dir+'/manifest.json',shard.manifest_sha256);inputs.add(manifest)
        read(manifest).outputs_sha256.each{n,h ->
            if(n.contains('/')||n=='..') error 'Unsafe K03 shard product'
            inputs.add(pinned(dir+'/'+n,h))
        }
        inputs.add(pinned(r.sources.evaluation.path,r.sources.evaluation.sha256))
        tuple(d.id,request,d.request_sha256,admission,d.admission_sha256,inputs)
    }
    K03_PREPARE_FAMILY(Channel.fromList(jobs),code)
}
