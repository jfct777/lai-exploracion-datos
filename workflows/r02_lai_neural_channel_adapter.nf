nextflow.enable.dsl=2

process BIND_ADMITTED_CHANNELS_AND_FLARE {
    tag "$task_id"
    cpus 1
    memory '4 GB'
    time '60 min'
    maxForks params.max_forks
    cache 'deep'
    container params.container
    publishDir { params.output_dir + '/' + task_id }, mode: 'copy', overwrite: false
    input:
    tuple val(task_id), path(request, stageAs:'request.json'), val(digest),
          path(inputs, stageAs:'inputs/input????????')
    path code, stageAs:'code/*'
    output:
    tuple val(task_id), path('bound/channel-batch.npz'), path('bound/receipt.json')
    script:
    """
    python3 code/r02_lai_neural_channel_adapter.py --request request.json --request-sha256 '${digest}' \
      --staged-inputs inputs --output-dir bound
    """
}

workflow {
    if (!params.request_manifest || !params.output_dir || !params.source_bin || !params.container)
        error 'request_manifest, output_dir, source_bin and immutable container are required'
    def manifest = new groovy.json.JsonSlurper().parse(file(params.request_manifest, checkIfExists:true))
    if (manifest.schema != 'r02_lai_neural_channel_adapter_batch_v1' || manifest.status != 'READY')
        error 'READY channel adapter batch required'
    def hash={p -> def d=java.security.MessageDigest.getInstance('SHA-256');p.toFile().withInputStream{s -> byte[] b=new byte[1048576];int n;while((n=s.read(b))>0)d.update(b,0,n)};d.digest().encodeHex().toString()}
    def rows = manifest.tasks.collect { task ->
        def request=file(task.request.path,checkIfExists:true)
        if(hash(request)!=task.request.sha256) error 'request differs: '+task.task_id
        def spec=new groovy.json.JsonSlurper().parse(request)
        def staged=spec.inputs.values().collect{file(it.path,checkIfExists:true)}.unique()
        tuple(task.task_id,request,task.request.sha256,staged)
    }
    def code = ['r02_lai_neural_channel_adapter.py', 'r02_lai_reference_features.py',
                'r02_lai_event_annotations.py'].collect {
        file(params.source_bin+'/'+it,checkIfExists:true)
    }
    BIND_ADMITTED_CHANNELS_AND_FLARE(Channel.fromList(rows), code)
}
