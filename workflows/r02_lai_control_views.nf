nextflow.enable.dsl=2

params.plan = null
params.output_dir = null
params.source_bin = "${projectDir}/../bin"
params.container = null
params.max_forks = 2

def safePath(p) {
    p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') &&
        !p.find(/[\n\r'"\x60$\\;|&<>*?\[\]{}]/)
}

def fileHash(p) {
    java.security.MessageDigest.getInstance('SHA-256').digest(p.toFile().bytes).encodeHex().toString()
}

process PREPARE_FOUR_LAI_CONTROL_VIEWS {
    tag "$task_id"
    cpus 1
    memory '2 GB'
    time '30 min'
    maxForks params.max_forks
    cache 'deep'
    container params.container
    publishDir { params.output_dir + '/' + task_id }, mode:'copy', overwrite:false
    input:
    tuple val(task_id),path(request,stageAs:'request.json'),val(request_sha256)
    path code, stageAs:'code/*'
    output:
    tuple val(task_id),path('prepared'),emit: products
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_lai_control_views.py prepare --request request.json \
      --request-sha256 '$request_sha256' --output-dir prepared
    """
}

workflow {
    if(!params.plan || !params.output_dir || !params.container) error 'Required: --plan --output_dir --container'
    if(!(params.max_forks instanceof Integer) || params.max_forks < 1 || params.max_forks > 2) error 'max_forks must be 1..2'
    if(![params.plan,params.output_dir,params.source_bin].every{safePath(it)}) error 'safe absolute paths required'
    def planFile=file(params.plan,checkIfExists:true)
    def plan=new groovy.json.JsonSlurper().parse(planFile)
    if(plan.schema!='r02_lai_four_control_plan_v1' || plan.status!='TWELVE_CASES_READY_FOR_FOUR_CONTROL_PREPARE' || plan.tasks.size()!=12) error '12-case four-control plan differs'
    def operations=[
      'NONE':'FULL_ADMITTED_COMMON_EVENT_AND_FIXED_REVISED_BASE',
      'NO_RARE':'NO_INDIVIDUAL_EVENT_CHANNEL_COMMON_AND_FIXED_REVISED_BASE_RETAINED',
      'COUNT_MATCHED':'EVENT_DOSAGE_PERMUTED_WITHIN_PERSON_OBSERVED_LOCUS_MASK_FIXED',
      'IDENTITY_SHAM':'EVENT_DOSAGE_AND_OBSERVATION_PERMUTED_JOINTLY_BETWEEN_PERSONS_PER_LOCUS']
    if(plan.conditions!=['NONE','NO_RARE','COUNT_MATCHED','IDENTITY_SHAM'] || plan.condition_operations!=operations || plan.controls_materialized || plan.training_authorized || plan.selection_authorized || plan.evaluate_opened || plan.truth_loaded) error 'partial plan semantics or forbidden-operation flags differ'
    def rows=plan.tasks.collect{row ->
        def request=file(row.request.path,checkIfExists:true)
        if(fileHash(request)!=row.request.sha256) error 'request differs: '+row.task_id
        tuple(row.task_id,request,row.request.sha256)
    }
    def code=['r02_lai_control_views.py','r02_lai_factorized_inputs.py','r02_lai_event_annotations.py'].collect{file(params.source_bin+'/'+it,checkIfExists:true)}
    PREPARE_FOUR_LAI_CONTROL_VIEWS(Channel.fromList(rows),code)
}
