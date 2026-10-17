nextflow.enable.dsl=2

include { MERGE_SELECT } from './r02_structure_campaign'

process REFINE_SELECT_BLOCK {
    tag "refine:${index}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    input:
    tuple val(index), path(shard), path(fitCache)
    path plan
    val planHash
    path metadata, stageAs:'metadata/*'
    path oldSelection, stageAs:'initial/*'
    path configurations, stageAs:'configurations/??/*'
    path code
    output:
    path "refined_block_${index}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_refine.py --stage block --plan ${plan} --plan-sha256 '${planHash}' \
      --index ${index} --output refined_block_${index}
    """
}

process REVIEW_REFINEMENT_BOUNDARY {
    tag 'review_boundary_no_acceptance'
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple path(selection), val(selectionHash)
    path plan
    val planHash
    path metadata, stageAs:'metadata/*'
    path oldSelection, stageAs:'initial/*'
    path configurations, stageAs:'configurations/??/*'
    path code
    output:
    path 'boundary-review-required.json'
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_refine.py --stage boundary --plan ${plan} --plan-sha256 '${planHash}' \
      --selection ${selection} --selection-sha256 '${selectionHash}' > boundary-review-required.json
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
    if(![params.plan,params.code_dir,params.output_dir].every{safe(it)} ||
       !(params.plan_sha256 ==~ /[0-9a-f]{64}/) || !params.task_time ||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/) || !(params.max_forks?.toString() ==~ /[1-9][0-9]*/))
        error 'Explicit pinned refinement paths/resources required'
    def plan=file(params.plan,checkIfExists:true)
    if(hash(plan)!=params.plan_sha256) error 'Refinement plan changed before cache lookup'
    def p=new groovy.json.JsonSlurper().parse(plan.toFile())
    if(p.schema!='r02_structure_select_refinement_v1' || p.iteration!=1 || p.evaluate_allowed!=false ||
       ![p.request.path,p.initial_request.path,p.initial_selection.path].every{safe(it)}) error 'Invalid refinement identity'
    def request=file(p.request.path,checkIfExists:true)
    def oldRequest=file(p.initial_request.path,checkIfExists:true)
    if(hash(request)!=p.request.sha256 || hash(oldRequest)!=p.initial_request.sha256 ||
       hash(file(p.initial_selection.path+'/manifest.json'))!=p.initial_selection.sha256)
        error 'Refinement inputs changed before cache lookup'
    def r=new groovy.json.JsonSlurper().parse(request.toFile())
    def prior=new groovy.json.JsonSlurper().parse(oldRequest.toFile())
    if(![r.bundle.path,r.evaluation.path,prior.evaluation.path].every{safe(it)} || r.bundle!=prior.bundle)
        error 'Refinement cannot change prepared data'
    def evaluation=file(r.evaluation.path,checkIfExists:true)
    def oldEvaluation=file(prior.evaluation.path,checkIfExists:true)
    if(hash(evaluation)!=r.evaluation.sha256 || hash(oldEvaluation)!=prior.evaluation.sha256 ||
       hash(file(r.bundle.path+'/contract.json'))!=r.bundle.sha256) error 'Refinement data/configuration changed'
    def modules=['r02_structure_refine.py','r02_structure_campaign.py','r02_structure_numerics.py','r02_structure_increment.py']
    if(p.implementation_sha256.keySet()!=modules.toSet()) error 'Refinement implementation inventory differs'
    def code=modules.collect {name ->
        def path=file(params.code_dir+'/'+name,checkIfExists:true)
        if(hash(path)!=p.implementation_sha256[name]) error 'Refinement source changed before cache lookup'
        path
    }
    def metadata=['contract.json','persons.tsv','blocks.tsv','block_index.json','protocol.md'].collect{file(r.bundle.path+'/'+it,checkIfExists:true)}
    def oldSelection=['manifest.json','complete.json','frozen_selection.json','support.json'].collect{file(p.initial_selection.path+'/'+it,checkIfExists:true)}
    def saved=new groovy.json.JsonSlurper().parse(file(p.initial_selection.path+'/manifest.json').toFile())
    def catalogue=new groovy.json.JsonSlurper().parse(file(r.bundle.path+'/block_index.json').toFile())
    def targets=catalogue.blocks.withIndex().collect {entry,k ->
        if(entry.path.startsWith('/') || entry.path.tokenize('/').contains('..')) error 'Unsafe refinement shard path'
        def shard=file(r.bundle.path+'/'+entry.path,checkIfExists:true)
        def name=String.format('fit_targets/fit_%04d.npz',k)
        def fit=file(p.initial_selection.path+'/'+name,checkIfExists:true)
        if(hash(shard.resolve('contract.json'))!=entry.contract_sha256 ||
           hash(shard.resolve('manifest.json'))!=entry.manifest_sha256 || hash(fit)!=saved.outputs_sha256[name])
            error 'Refinement shard/FIT cache changed before cache lookup'
        tuple(k,shard,fit)
    }
    def configurations=[request,evaluation,oldRequest,oldEvaluation]
    REFINE_SELECT_BLOCK(Channel.fromList(targets),plan,params.plan_sha256,metadata,oldSelection,configurations,code)
    MERGE_SELECT(REFINE_SELECT_BLOCK.out.collect(),metadata,evaluation,request,p.request.sha256,code)
    def selection=MERGE_SELECT.out.map {folder -> tuple(folder,hash(folder.resolve('manifest.json')))}
    REVIEW_REFINEMENT_BOUNDARY(selection,plan,params.plan_sha256,metadata,oldSelection,configurations,code)
}
