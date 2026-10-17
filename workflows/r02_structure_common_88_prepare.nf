nextflow.enable.dsl=2

def safeAbsolute(s) {
    s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') &&
        !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

def fileHash(p) {
    def md=java.security.MessageDigest.getInstance('SHA-256')
    p.toFile().withInputStream{s ->
        new java.security.DigestInputStream(s,md).transferTo(java.io.OutputStream.nullOutputStream())
    }
    md.digest().encodeHex().toString()
}

def pinnedFile(name,pin) {
    if(!safeAbsolute(name)||!(pin ==~ /[a-f0-9]{64}/)) error 'Safe absolute common target source required'
    def p=file(name,checkIfExists:true)
    if(fileHash(p)!=pin) error 'Common target input changed before cache lookup'
    p
}

def readJson(p) {
    new groovy.json.JsonSlurper().parse(p.toFile())
}

process K03_COMMON_TARGET_BRIDGE {
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
    tuple val(viewId), path("common_${viewId}")
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_common_target_bridge.py --request '${request}' --request-sha256 '${requestPin}' \
      --admission '${admission}' --admission-sha256 '${admissionPin}' --output common_${viewId}
    """
}


process COMMON_NEURAL_REQUEST {
    tag "${viewId}"
    cpus 1
    memory '1 GB'
    time '2m'
    cache 'deep'
    input:
    tuple val(viewId), path(source)
    path targetContract
    output:
    tuple val(viewId), path(source), path("adapter_${viewId}.json")
    script:
    """
    python3 - '${viewId}' '${source}' '${targetContract}' 'adapter_${viewId}.json' \
      '${params.adapter_max_metadata_bytes}' '${params.adapter_max_input_bytes}' \
      '${params.adapter_max_array_bytes}' '${params.adapter_max_output_bytes}' \
      '${params.adapter_min_free_bytes}' <<'PY'
import hashlib,json,pathlib,sys
view,source,contract,out,*limits=sys.argv[1:]
manifest_path=pathlib.Path(source)/'manifest.json'
manifest=json.loads(manifest_path.read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
block,scope=view.rsplit('__',1)
request={'schema':'r02_k03_neural_target_adapter_request_v1',
 'status':'PREPARE_TARGETS_NOT_TRAINING_ADMISSION',
 'source':{'directory':str(pathlib.Path(source).resolve()),'manifest_sha256':sha(manifest_path)},
 'target_contract':{'path':str(pathlib.Path(contract).resolve()),'sha256':sha(contract)},
 'expected':{'evidence_scope':'DNABR_R02','scientific_spec_sha256':manifest['scientific_spec_sha256'],
  'block_id':block,'exclusion_scope':scope,
  'person_counts':{'FIT':1460,'SELECT':476,'EVALUATE':683}},
 'resources':dict(zip(('max_metadata_bytes','max_input_bytes','max_array_bytes','max_output_bytes','min_free_bytes'),map(int,limits)))}
pathlib.Path(out).write_text(json.dumps(request,indent=2,sort_keys=True)+'\n')
PY
    """
}

process COMMON_NEURAL_ADAPTER {
    tag "${viewId}"
    cpus 1
    memory "${params.adapter_memory_gb} GB"
    time params.adapter_time
    maxForks params.adapter_max_forks
    cache 'deep'
    publishDir params.target_output_dir, mode:'copy', overwrite:false
    input:
    tuple val(viewId), path(source), path(request)
    path targetContract
    path adapterCode
    output:
    tuple val(viewId), path("neural_${viewId}")
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    request_sha=\$(sha256sum '${request}' | cut -d' ' -f1)
    python3 '${adapterCode}' --request '${request}' --request-sha256 "\$request_sha" \
      --source-directory '${source}' --target-contract '${targetContract}' --output neural_${viewId}
    """
}

workflow {
    if(![params.batch,params.code_dir,params.output_dir,params.target_output_dir,params.target_contract,params.adapter_code].every{safeAbsolute(it)}||
       ![params.memory_gb,params.max_forks].every{it?.toString() ==~ /[1-9][0-9]*/}||
       !(params.task_time instanceof String)||!(params.task_time ==~ /[1-9][0-9]*[smh]/)||
       ![params.adapter_memory_gb,params.adapter_max_forks,params.adapter_max_metadata_bytes,params.adapter_max_input_bytes,params.adapter_max_array_bytes,params.adapter_max_output_bytes,params.adapter_min_free_bytes].every{it?.toString() ==~ /[1-9][0-9]*/}||
       !(params.adapter_time instanceof String)||!(params.adapter_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit bounded common target and adapter resources required'
    def b=readJson(pinnedFile(params.batch,params.batch_sha256))
    if(b.schema!='r02_k03_common_target_bridge_batch_v1'||!b.views||b.views.collect{it.id}.toSet().size()!=b.views.size()) error 'Unique admitted common target views required'
    if(!(b.implementation_sha256 instanceof Map)) error 'Pinned common target solver implementation required'
    def code=(b.implementation_sha256).collect{n,h ->
        if(!(n ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe common target module'
        pinnedFile(params.code_dir+'/'+n,h)
    }
    def jobs=b.views.collect{d ->
        if(!(d.id ==~ /[A-Za-z0-9_]+/)) error 'Unsafe common target view id'
        def request=pinnedFile(d.request,d.request_sha256);def r=readJson(request)
        if(r.schema!='r02_k03_common_target_bridge_request_v1'||r.implementation_sha256!=b.implementation_sha256||
           r.scientific_spec_sha256!=b.scientific_spec_sha256) error 'Common target request/code differs'
        def admission=pinnedFile(d.admission,d.admission_sha256);def a=readJson(admission)
        if(a.schema!='r02_k03_common_target_bridge_admission_v1'||a.status!='COMMON_FIT_SELECT_TARGET_INPUT_ADMITTED'||a.evaluate_allowed!=false||
           a.evidence_scope!=r.evidence_scope||a.request_sha256!=d.request_sha256||a.scientific_spec_sha256!=r.scientific_spec_sha256) error 'Exact common target admission required before cache'
        def inputs=[]
        def common=r.sources.common;def cm=pinnedFile(common.directory+'/manifest.json',common.manifest_sha256);inputs.add(cm)
        readJson(cm).outputs_sha256.each{n,h ->
            if(n.startsWith('/')||n.tokenize('/').contains('..')) error 'Unsafe common representation artifact'
            inputs.add(pinnedFile(common.directory+'/'+n,h))
        }
        def pd=r.sources.prepared;def contract=pinnedFile(pd.directory+'/contract.json',pd.contract_sha256);inputs.add(contract)
        def c=readJson(contract);c.sha256.each{n,h ->
            if(!(n in ['persons.tsv','blocks.tsv','block_index.json','protocol.md'])) error 'Unexpected target catalogue file'
            inputs.add(pinnedFile(pd.directory+'/'+n,h))
        }
        def index=readJson(file(pd.directory+'/block_index.json'));def matches=index.blocks.findAll{it.block_id==r.block_id}
        if(matches.size()!=1) error 'Exact common target target block required'
        def shard=matches[0]
        if(shard.path.startsWith('/')||shard.path.tokenize('/').contains('..')) error 'Unsafe common target target shard'
        def dir=pd.directory+'/'+shard.path
        inputs.add(pinnedFile(dir+'/contract.json',shard.contract_sha256))
        def sm=pinnedFile(dir+'/manifest.json',shard.manifest_sha256);inputs.add(sm)
        readJson(sm).outputs_sha256.each{n,h ->
            if(n.contains('/')||n=='..') error 'Unsafe target shard product'
            inputs.add(pinnedFile(dir+'/'+n,h))
        }
        inputs.add(pinnedFile(r.sources.evaluation.path,r.sources.evaluation.sha256))
        tuple(d.id,request,d.request_sha256,admission,d.admission_sha256,inputs)
    }
    def bridges=K03_COMMON_TARGET_BRIDGE(Channel.fromList(jobs),code)
    def targetContract=pinnedFile(params.target_contract,params.target_contract_sha256)
    def adapterCode=pinnedFile(params.adapter_code,params.adapter_code_sha256)
    def adapterRequests=COMMON_NEURAL_REQUEST(bridges,targetContract)
    COMMON_NEURAL_ADAPTER(adapterRequests,targetContract,adapterCode)
}
