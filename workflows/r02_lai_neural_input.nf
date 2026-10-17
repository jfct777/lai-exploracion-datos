nextflow.enable.dsl=2

params.request_manifest = null
params.output_dir = null
params.source_bin = "${projectDir}/../bin"
params.container = null
params.max_forks = 1

process ASSEMBLE_LAI_NEURAL_INPUT {
    tag "$task_id"
    cpus 1
    memory '3 GB'
    time '45 min'
    maxForks params.max_forks
    cache 'deep'
    container params.container
    publishDir { params.output_dir + '/' + task_id }, mode:'copy', overwrite:false
    input:
    tuple val(task_id),path(request,stageAs:'request.json'),val(digest),path(inputs,stageAs:'inputs/input????????')
    path code, stageAs:'code/*'
    output:
    tuple val(task_id),path('assembled'),emit: batches
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_lai_neural_input.py --request request.json --request-sha256 '$digest' --staged-inputs inputs --output-dir assembled
    """
}

workflow {
    if(!params.request_manifest || !params.output_dir || !params.container) error 'Required: --request_manifest --output_dir --container'
    if(!(params.max_forks instanceof Integer) || params.max_forks<1 || params.max_forks>4) error 'max_forks must be 1..4'
    def safe={p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)}
    if(![params.request_manifest,params.output_dir,params.source_bin].every{safe(it)}) error 'safe absolute paths required'
    def hash={p -> def d=java.security.MessageDigest.getInstance('SHA-256');p.toFile().withInputStream{s -> byte[] b=new byte[1048576];int n;while((n=s.read(b))>0)d.update(b,0,n)};d.digest().encodeHex().toString()}
    def manifestFile=file(params.request_manifest,checkIfExists:true)
    def manifest=new groovy.json.JsonSlurper().parse(manifestFile)
    if(manifest.schema!='r02_lai_neural_input_request_manifest_v1' || manifest.status!='READY_FOR_ASSEMBLY' || !manifest.tasks) error 'manifest is not admitted for assembly'
    if(!(manifest.protocol_id in ['FOLD_A_TO_FOLD_B','FOLD_B_TO_FOLD_A'])) error 'one explicit LAI protocol required; reciprocal pooling is forbidden'
    def rows=manifest.tasks.collect{row ->
        def request=file(row.request.path,checkIfExists:true)
        if(hash(request)!=row.request.sha256) error 'request differs: '+row.task_id
        def value=new groovy.json.JsonSlurper().parse(request)
        if(value.schema!='r02_lai_neural_input_request_v1') error 'request schema differs'
        if(value.protocol_id!=manifest.protocol_id || value.direction!=manifest.protocol_id) error 'request crosses the frozen protocol'
        def deps=value.inputs.values().findAll{it!=null}.collect{d ->
            def p=file(d.path,checkIfExists:true);if(hash(p)!=d.sha256)error 'input differs: '+d.path;p
        }
        tuple(row.task_id,request,row.request.sha256,deps.unique())
    }
    def code=['r02_lai_neural_input.py','r02_lai_separation.py','r02_lai_target_dependencies.py'].collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    ASSEMBLE_LAI_NEURAL_INPUT(Channel.fromList(rows),code)
}
