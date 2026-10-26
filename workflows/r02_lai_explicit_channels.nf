nextflow.enable.dsl=2

params.request_manifest = null
params.output_dir = null
params.source_bin = "${projectDir}/../bin"
params.container = null
params.max_forks = 1

process PREPARE_EXACT_LAI_CHANNELS {
    tag "$task_id"
    cpus 1
    memory '4 GB'
    time '60 min'
    maxForks params.max_forks
    cache 'deep'
    container params.container
    publishDir { params.output_dir + '/' + task_id }, mode: 'copy', overwrite: false

    input:
    tuple val(task_id), path(regional, stageAs: 'regional.npz'),
          path(decision, stageAs: 'decision.json'), val(decision_sha),
          path(inputs, stageAs: 'inputs/input????????')
    path code, stageAs: 'code/*'

    output:
    tuple val(task_id), path('explicit.npz'), path('explicit.npz.json'), emit: prepared

    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_lai_explicit_channels.py \
      --regional-batch regional.npz --decision decision.json \
      --decision-sha256 '${decision_sha}' --staged-inputs inputs --output explicit.npz
    """
}

workflow {
    if (!params.request_manifest || !params.output_dir || !params.container)
        error 'request_manifest, output_dir and immutable container are required'
    if (!(params.max_forks instanceof Integer) || params.max_forks < 1 || params.max_forks > 4)
        error 'max_forks must be 1..4'
    def manifest = new groovy.json.JsonSlurper().parse(file(params.request_manifest, checkIfExists:true))
    if (manifest.schema != 'r02_lai_explicit_channels_prepare_batch_v1' ||
        manifest.status != 'READY_PREPARE_ONLY' || manifest.training_authorized != false ||
        manifest.controls_materialized != false || !manifest.tasks)
        error 'exact PREPARE-only batch manifest required'
    def rows = manifest.tasks.collect { task ->
        def regional=file(task.regional_batch.path,checkIfExists:true)
        def decision=file(task.decision.path,checkIfExists:true)
        def spec=new groovy.json.JsonSlurper().parse(decision)
        if(spec.task_id!=task.task_id || spec.status!='PREPARE_ADMITTED' ||
           spec.training_allowed!=false || spec.controls_materialized!=false ||
           spec.regional_batch_sha256!=task.regional_batch.sha256)
            error 'task decision differs: '+task.task_id
        def dependencies=[spec.event_catalogue,spec.reference_linkage_receipt].collect { item ->
            file(item.path,checkIfExists:true)
        }
        tuple(task.task_id,regional,decision,task.decision.sha256,dependencies)
    }
    def code=['r02_lai_explicit_channels.py','r02_lai_event_annotations.py'].collect {
        file(params.source_bin+'/'+it,checkIfExists:true)
    }
    PREPARE_EXACT_LAI_CHANNELS(Channel.fromList(rows),code)
}
