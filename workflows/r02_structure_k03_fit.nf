nextflow.enable.dsl=2

process K03_FIT_SELECT_MODEL {
    tag "${modelId}"
    cpus params.cpus
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(modelId), path(request), val(requestPin), path(admission), val(admissionPin), path(inputs,stageAs:'inputs/??/*')
    path code
    output:
    path "fit_${modelId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_k03_fit.py --request ${request} --request-sha256 '${requestPin}' \
      --admission ${admission} --admission-sha256 '${admissionPin}' --output fit_${modelId}
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
        if(!safe(name)||!(pin ==~ /[a-f0-9]{64}/)) error 'Safe absolute K03 input pin required'
        def p=file(name,checkIfExists:true)
        if(hash(p)!=pin) error 'K03 FIT/SELECT input changed before cache lookup'
        p
    }
    if(![params.batch,params.code_dir,params.output_dir].every{safe(it)}||
       ![params.cpus,params.memory_gb,params.max_forks].every{it?.toString() ==~ /[1-9][0-9]*/}||
       !(params.task_time instanceof String)||!(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit bounded K03 FIT/SELECT resources required'
    def batch=pinned(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(!(b.schema in ['r02_k03_fit_select_batch_v1','r02_k03_common_finite_fit_batch_v1'])||!b.models||b.models.collect{it.id}.toSet().size()!=b.models.size()) error 'Unique explicit K03 model requests required'
    def code=b.implementation_sha256.collect{n,h ->
        if(!(n ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe K03 module name'
        pinned(params.code_dir+'/'+n,h)
    }
    def tasks=b.models.collect{d ->
        if(!(d.id ==~ /[A-Za-z0-9_]+/)) error 'Unsafe model id'
        def request=pinned(d.request,d.request_sha256)
        def r=new groovy.json.JsonSlurper().parse(request.toFile())
        if(!(r.schema in ['r02_k03_fit_select_request_v1','r02_k03_fit_select_request_v2','r02_k03_fit_select_request_v3'])||r.implementation_sha256!=b.implementation_sha256||
           r.scientific_spec_sha256!=b.scientific_spec_sha256||r.resources.task_cpus!=params.cpus||
           !(r.parameters.workers instanceof Number)||r.parameters.workers<1||r.parameters.workers>params.cpus) error 'K03 model family/code differs'
        def admission=pinned(d.admission,d.admission_sha256)
        def a=new groovy.json.JsonSlurper().parse(admission.toFile())
        if(a.request_sha256!=d.request_sha256||a.scientific_spec_sha256!=b.scientific_spec_sha256||
           a.status!='FIT_SELECT_ONLY_ADMITTED'||a.evaluate_allowed!=false) error 'Exact FIT/SELECT admission required before cache'
        def inputs=[]
        if(r.schema in ['r02_k03_fit_select_request_v1','r02_k03_fit_select_request_v3']) {
            def manifest=pinned(r.inputs.directory+'/manifest.json',r.inputs.manifest_sha256)
            def m=new groovy.json.JsonSlurper().parse(manifest.toFile())
            if(m.stage!='FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS'||m.evaluate_targets_computed!=false||
               m.outputs_sha256.keySet()!=['persons.tsv','arrays.private.npz'].toSet()) error 'Exact prepared FIT/SELECT arrays required'
            inputs=[manifest]+m.outputs_sha256.collect{n,h -> pinned(r.inputs.directory+'/'+n,h)}
        } else {
            ['base','candidate'].each{k ->
                def desc=r.inputs[k]
                def manifest=pinned(desc.directory+'/manifest.json',desc.manifest_sha256)
                def m=new groovy.json.JsonSlurper().parse(manifest.toFile())
                def expected = k=='base' ? ['persons.tsv','arrays.private.npz','support.json'] : ['membership.private.npz']
                if(m.outputs_sha256.keySet()!=expected.toSet()||m.alpha_selection_performed!=false) error 'Exact factored K03 input inventory required'
                inputs.add(manifest);inputs.addAll(m.outputs_sha256.collect{n,h -> pinned(desc.directory+'/'+n,h)})
            }
        }
        tuple(d.id,request,d.request_sha256,admission,d.admission_sha256,inputs)
    }
    K03_FIT_SELECT_MODEL(Channel.fromList(tasks),code)
}
