nextflow.enable.dsl=2

def safeAbsolute(s) {
    s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') &&
        !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

def fileHash(p) {
    def md=java.security.MessageDigest.getInstance('SHA-256')
    p.toFile().withInputStream { s ->
        new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())
    }
    md.digest().encodeHex().toString()
}

def pinnedFile(name,pin) {
    if(!safeAbsolute(name)||!(pin ==~ /[a-f0-9]{64}/)) error 'Safe absolute common-search pin required'
    def p=file(name,checkIfExists:true)
    if(fileHash(p)!=pin) error 'Common-search input changed before cache lookup'
    p
}

def readJson(p) { new groovy.json.JsonSlurper().parse(p.toFile()) }

process K03_COMMON_SEARCH_PREPARE {
    tag 'FIT_statistics_and_family'
    cpus params.fit_cpus
    memory "${params.statistics_memory_gb} GB"
    time params.statistics_time
    cache 'deep'
    publishDir "${params.output_dir}/plan", mode:'copy', overwrite:false
    input:
    tuple path(request), val(requestPin), path(admission), val(admissionPin),
          path(sourceInputs,stageAs:'source/??/*'), path(code)
    output:
    path 'search_plan', emit: plan
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_k03_common_search.py prepare \
      --request '${request}' --request-sha256 '${requestPin}' \
      --admission '${admission}' --admission-sha256 '${admissionPin}' --output search_plan
    """
}

process K03_COMMON_FINITE_MODEL {
    tag "${modelId}"
    cpus params.fit_cpus
    memory "${params.fit_memory_gb} GB"
    time params.fit_time
    maxForks params.max_forks
    cache 'deep'
    publishDir "${params.output_dir}/fit", mode:'copy', overwrite:false
    input:
    tuple val(modelId), path(request), val(requestPin), path(admission), val(admissionPin),
          path(sourceInputs,stageAs:'source/??/*'), path(code)
    output:
    path "fit_${modelId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_k03_fit.py --request '${request}' --request-sha256 '${requestPin}' \
      --admission '${admission}' --admission-sha256 '${admissionPin}' --output fit_${modelId}
    """
}

process K03_COMMON_FINITE_SELECT {
    tag 'complete_family_SELECT'
    cpus 1
    memory '1 GB'
    time '15m'
    cache 'deep'
    publishDir "${params.output_dir}/selection", mode:'copy', overwrite:false
    input:
    path plan
    path fits
    path code
    output:
    path 'selection'
    script:
    def fitArgs=fits.collect { "--fit-directory '${it}'" }.join(' ')
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_k03_common_search.py select --plan-directory '${plan}' \
      ${fitArgs} --output selection
    """
}

workflow {
    if(![params.request,params.admission,params.code_dir,params.output_dir].every{safeAbsolute(it)} ||
       ![params.statistics_memory_gb,params.fit_cpus,params.fit_memory_gb,params.max_forks].every{it?.toString()==~ /[1-9][0-9]*/} ||
       ![params.statistics_time,params.fit_time].every{it instanceof String && it==~ /[1-9][0-9]*[smh]/})
        error 'Explicit bounded common finite-search paths/resources required'
    def request=pinnedFile(params.request,params.request_sha256);def r=readJson(request)
    if(r.schema!='r02_k03_common_finite_search_request_v1'||
       r.resources.task_cpus.toString()!=params.fit_cpus.toString()||
       r.parameters.workers>params.fit_cpus.toString().toInteger())
        error 'Common finite-search request/resources differ'
    def admission=pinnedFile(params.admission,params.admission_sha256);def a=readJson(admission)
    if(a.schema!='r02_k03_common_finite_search_admission_v1'||a.status!='FIT_SELECT_FINITE_FAMILY_ADMITTED'||
       a.request_sha256!=params.request_sha256||a.scientific_spec_sha256!=r.scientific_spec_sha256||a.evaluate_allowed!=false)
        error 'Exact external finite-search admission required before cache'
    def code=r.implementation_sha256.collect { n,h ->
        if(!(n==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe common-search module name'
        pinnedFile(params.code_dir+'/'+n,h)
    }
    if(r.fit_implementation_sha256!=r.implementation_sha256.findAll{k,v -> k in r.fit_implementation_sha256.keySet()})
        error 'Fit implementation is not bound by the search implementation'
    def source=r.source;def manifest=pinnedFile(source.directory+'/manifest.json',source.manifest_sha256)
    def m=readJson(manifest)
    if(m.stage!='FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS'||m.evaluate_targets_computed!=false||
       m.outputs_sha256.keySet()!=['persons.tsv','arrays.private.npz'].toSet())
        error 'Exact frozen common FIT/SELECT bridge required'
    def sourceInputs=[manifest]+m.outputs_sha256.collect{n,h -> pinnedFile(source.directory+'/'+n,h)}
    K03_COMMON_SEARCH_PREPARE(Channel.of(tuple(request,params.request_sha256,admission,
        params.admission_sha256,sourceInputs,code)))
    def modelTasks=K03_COMMON_SEARCH_PREPARE.out.plan.flatMap { directory ->
        def b=readJson(file(directory.toString()+'/fit-batch.json',checkIfExists:true))
        b.models.collect { model -> tuple(model.id,
            pinnedFile(model.request,model.request_sha256),model.request_sha256,
            pinnedFile(model.admission,model.admission_sha256),model.admission_sha256,
            sourceInputs,code) }
    }
    K03_COMMON_FINITE_MODEL(modelTasks)
    K03_COMMON_FINITE_SELECT(K03_COMMON_SEARCH_PREPARE.out.plan,
        K03_COMMON_FINITE_MODEL.out.collect(),Channel.value(code))
}
