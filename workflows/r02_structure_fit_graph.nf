nextflow.enable.dsl=2

def safeGraphPath(s) {
    s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') &&
        !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}
def graphFileHash(p) {
    def md=java.security.MessageDigest.getInstance('SHA-256')
    p.toFile().withInputStream {s -> new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())}
    md.digest().encodeHex().toString()
}
def pinnedGraphFile(name,pin) {
    if(!safeGraphPath(name) || !(pin ==~ /[0-9a-f]{64}/)) error 'Safe absolute graph source and SHA256 required'
    def p=file(name,checkIfExists:true)
    if(graphFileHash(p)!=pin) error 'FIT graph source changed before cache lookup'
    p
}

process FIT_GRAPH_TASK {
    tag "${mode}:${taskId}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    tuple val(taskId), val(mode), path(request, stageAs:'request.json'), val(pin), path(admission, stageAs:'admission.json'), val(admissionPin), path(inputs, stageAs:'inputs/??/*')
    path code
    output:
    path "${mode}_${taskId}"
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl
    python3 r02_structure_fit_graph.py --request ${request} --request-sha256 '${pin}' --admission ${admission} --admission-sha256 '${admissionPin}' --output ${mode}_${taskId}
    """
}

workflow {
    if(![params.batch,params.code_dir,params.output_dir].every{safeGraphPath(it)} ||
       !(params.batch_sha256 ==~ /[0-9a-f]{64}/) ||
       !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
       !(params.max_forks?.toString() ==~ /[1-9][0-9]*/) ||
       !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/))
        error 'Explicit graph paths and operational limits required'
    def batch=pinnedGraphFile(params.batch,params.batch_sha256)
    def b=new groovy.json.JsonSlurper().parse(batch.toFile())
    if(b.schema!='r02_fit_graph_task_batch_v1' || !b.tasks || b.tasks.collect{it.id}.toSet().size()!=b.tasks.size())
        error 'Explicit unique graph tasks required'
    def code=b.implementation_sha256.collect {name,pin ->
        if(!(name ==~ /(r02_[a-z0-9_]+|rare_allele_sharing_painter)\.py/)) error 'Unsafe graph implementation filename'
        pinnedGraphFile(params.code_dir+'/'+name,pin)
    }
    def tasks=b.tasks.collect {entry ->
        if(!(entry.id ==~ /[a-zA-Z0-9_-]+/)) error 'Safe graph task ID required'
        def request=pinnedGraphFile(entry.request,entry.sha256)
        def admission=pinnedGraphFile(entry.admission,entry.admission_sha256)
        def r=new groovy.json.JsonSlurper().parse(request.toFile())
        def a=new groovy.json.JsonSlurper().parse(admission.toFile())
        def schemas=['r02_fit_graph_request_v1','r02_fit_cached_candidate_request_v1',
            'r02_fit_family_graph_request_v2','r02_fit_family_cached_candidate_request_v2',
            'r02_fit_family_projection_request_v1']
        if(!(r.schema in schemas) ||
           r.implementation_sha256!=b.implementation_sha256) error 'Graph request/code differ'
        def mode=(r.schema in ['r02_fit_graph_request_v1','r02_fit_family_graph_request_v2']) ? 'GRAPH' :
            (r.schema in ['r02_fit_cached_candidate_request_v1','r02_fit_family_cached_candidate_request_v2'] ? 'CANDIDATE' : 'PROJECTION')
        if(a.schema!='r02_fit_graph_task_admission_v1' || a.request_sha256!=entry.sha256 ||
           a.scientific_spec_sha256!=r.scientific_spec_sha256 || a.evidence_scope!=r.evidence_scope ||
           a.mode!=mode || a.status!='FIT_ONLY_ADMITTED' || a.evaluate_allowed!=false)
            error 'FIT graph task admission differs before cache lookup'
        if(r.resources.max_memory_bytes>params.memory_gb.toBigInteger()*1024L*1024L*1024L)
            error 'Graph byte budget exceeds allocated task memory'
        def inputs=[]
        if(r.schema=='r02_fit_graph_request_v1') {
            inputs=r.inputs.collect {name,record -> pinnedGraphFile(record.path,record.sha256)}
        } else if(mode in ['GRAPH','PROJECTION']) {
            inputs=r.source_files.collect {name,record -> pinnedGraphFile(record.path,record.sha256)}
            if(mode=='PROJECTION') {
                def mp=pinnedGraphFile(r.graph.directory+'/graph_manifest.json',r.graph.manifest_sha256)
                def m=new groovy.json.JsonSlurper().parse(mp.toFile())
                if(m.schema!='r02_fit_family_graph_frozen_v2' || m.implementation_sha256!=b.implementation_sha256)
                    error 'Cached family graph producer differs'
                inputs.add(mp)
                m.outputs_sha256.each {name,pin -> inputs.add(pinnedGraphFile(r.graph.directory+'/'+name,pin)) }
            }
        } else {
            def mp=pinnedGraphFile(r.graph.directory+'/graph_manifest.json',r.graph.manifest_sha256)
            def m=new groovy.json.JsonSlurper().parse(mp.toFile())
            def expected=r.schema=='r02_fit_family_cached_candidate_request_v2' ?
                'r02_fit_family_graph_frozen_v2' : 'r02_fit_graph_frozen_v1'
            if(m.schema!=expected || m.implementation_sha256!=b.implementation_sha256)
                error 'Cached FIT graph producer differs'
            inputs=[mp,pinnedGraphFile(r.episode.path,r.episode.sha256)]
            m.outputs_sha256.each {name,pin ->
                if(!(name in ['persons.tsv','spectrum.private.npz','state.json'])) error 'Unexpected graph artifact'
                inputs.add(pinnedGraphFile(r.graph.directory+'/'+name,pin))
            }
        }
        tuple(entry.id,mode,request,entry.sha256,admission,entry.admission_sha256,inputs)
    }
    FIT_GRAPH_TASK(Channel.fromList(tasks),code)
}
