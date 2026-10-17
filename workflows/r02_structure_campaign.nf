nextflow.enable.dsl=2

process SELECT_BLOCK {
    tag "select:${index}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    input:
    tuple val(index), path(shard)
    path metadata, stageAs:'metadata/*'
    path evaluation
    path request
    val requestHash
    path code
    output:
    path "select_block_${index}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_campaign.py --request ${request} --request-sha256 '${requestHash}' \
      --stage select-block --index ${index} --output select_block_${index}
    """
}

process MERGE_SELECT {
    tag 'freeze_complete_select'
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path blocks, stageAs:'blocks/*'
    path metadata, stageAs:'metadata/*'
    path evaluation
    path request
    val requestHash
    path code
    output:
    path 'selection'
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_campaign.py --request ${request} --request-sha256 '${requestHash}' \
      --stage merge-select --checkpoints blocks/* --output selection
    """
}

process ADMIT_EVALUATE {
    tag 'authenticate_explicit_review'
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    input:
    path selection
    val selectionHash
    path review
    val reviewHash
    path metadata, stageAs:'metadata/*'
    path evaluation
    path request
    val requestHash
    path code
    output:
    path 'admission'
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_campaign.py --request ${request} --request-sha256 '${requestHash}' \
      --stage admit --selection ${selection} --selection-sha256 '${selectionHash}' \
      --review ${review} --review-sha256 '${reviewHash}' --output admission
    """
}

process EVALUATE_BLOCK {
    tag "evaluate:${index}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    input:
    tuple val(index), path(shard)
    path admission
    path selection
    path review
    path metadata, stageAs:'metadata/*'
    path evaluation
    path request
    val requestHash
    path code
    output:
    path "evaluate_block_${index}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_campaign.py --request ${request} --request-sha256 '${requestHash}' \
      --stage evaluate-block --index ${index} --admission ${admission} --output evaluate_block_${index}
    """
}

process MERGE_EVALUATE {
    tag 'complete_conditional_comparison'
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path blocks, stageAs:'blocks/*'
    path admission
    path metadata, stageAs:'metadata/*'
    path evaluation
    path request
    val requestHash
    path code
    output:
    path 'evaluation'
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_campaign.py --request ${request} --request-sha256 '${requestHash}' \
      --stage merge-evaluate --checkpoints blocks/* --admission ${admission} --output evaluation
    """
}

workflow {
    def safe={s -> s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') &&
        !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)}
    def hash={p ->
        def md=java.security.MessageDigest.getInstance('SHA-256')
        p.toFile().withInputStream {s -> new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())}
        md.digest().encodeHex().toString()
    }
    if(![params.request,params.code_dir,params.output_dir].every{safe(it)} ||
       !(params.request_sha256 ==~ /[0-9a-f]{64}/)) error 'Pinned absolute campaign paths required'
    if(!(params.mode in ['select','evaluate']) || !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
       !(params.max_forks?.toString() ==~ /[1-9][0-9]*/) || !params.task_time) error 'Explicit mode and operational resources required'
    def request=file(params.request,checkIfExists:true)
    if(hash(request)!=params.request_sha256) error 'Campaign request changed before cache lookup'
    def r=new groovy.json.JsonSlurper().parse(request.toFile())
    if(r.schema!='r02_structure_campaign_v1' || ![r.bundle.path,r.evaluation.path].every{safe(it)}) error 'Invalid campaign source descriptors'
    def metadata=['contract.json','persons.tsv','blocks.tsv','block_index.json','protocol.md'].collect {file(r.bundle.path+'/'+it,checkIfExists:true)}
    def evaluation=file(r.evaluation.path,checkIfExists:true)
    if(hash(file(r.bundle.path+'/contract.json'))!=r.bundle.sha256 || hash(evaluation)!=r.evaluation.sha256)
        error 'Campaign source identity changed before cache lookup'
    def modules=['r02_structure_campaign.py','r02_structure_numerics.py','r02_structure_increment.py']
    if(r.implementation_sha256.keySet()!=modules.toSet()) error 'Exact consumer source inventory required'
    def code=modules.collect {n ->
        def p=file(params.code_dir+'/'+n,checkIfExists:true)
        if(hash(p)!=r.implementation_sha256[n]) error 'Consumer source changed before cache lookup'
        p
    }
    def catalogue=new groovy.json.JsonSlurper().parse(file(r.bundle.path+'/block_index.json').toFile())
    def prepared=new groovy.json.JsonSlurper().parse(file(r.bundle.path+'/contract.json').toFile())
    prepared.sha256.each {name,digest ->
        if(hash(file(r.bundle.path+'/'+name))!=digest) error 'Global metadata changed before cache lookup'
    }
    def targets=catalogue.blocks.withIndex().collect {item,k ->
        if(item.path.startsWith('/') || item.path.tokenize('/').contains('..')) error 'Unsafe shard path'
        def shard=file(r.bundle.path+'/'+item.path,checkIfExists:true)
        if(hash(shard.resolve('contract.json'))!=item.contract_sha256 || hash(shard.resolve('manifest.json'))!=item.manifest_sha256)
            error 'Shard metadata changed before cache lookup'
        tuple(k,shard)
    }
    if(params.mode=='select') {
        SELECT_BLOCK(Channel.fromList(targets),metadata,evaluation,request,params.request_sha256,code)
        MERGE_SELECT(SELECT_BLOCK.out.collect(),metadata,evaluation,request,params.request_sha256,code)
    } else {
        if(![params.selection,params.review].every{safe(it)} ||
           !(params.selection_sha256 ==~ /[0-9a-f]{64}/) || !(params.review_sha256 ==~ /[0-9a-f]{64}/))
            error 'EVALUATE requires explicit pinned selection/review'
        def selection=file(params.selection,checkIfExists:true)
        def review=file(params.review,checkIfExists:true)
        if(hash(file(params.selection+'/manifest.json'))!=params.selection_sha256 || hash(review)!=params.review_sha256)
            error 'Selection/review changed before cache lookup'
        ADMIT_EVALUATE(selection,params.selection_sha256,review,params.review_sha256,metadata,evaluation,request,params.request_sha256,code)
        def admission=ADMIT_EVALUATE.out.first()
        EVALUATE_BLOCK(Channel.fromList(targets),admission,selection,review,metadata,evaluation,request,params.request_sha256,code)
        MERGE_EVALUATE(EVALUATE_BLOCK.out.collect(),admission,metadata,evaluation,request,params.request_sha256,code)
    }
}
