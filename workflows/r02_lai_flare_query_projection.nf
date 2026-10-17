nextflow.enable.dsl=2

process PROJECT_EXISTING_FLARE_TO_QUERY_GRID {
    tag { task_id }
    cpus 1
    memory { limits.memory_bytes + ' B' }
    time { limits.timeout_seconds + ' sec' }
    maxForks params.max_forks
    cache 'deep'
    container params.container
    publishDir { params.output_dir + '/' + task_id }, mode: 'copy', overwrite: false

    input:
    tuple val(task_id), path(request, stageAs: 'request.json'), val(request_sha), val(limits),
          path(inputs, stageAs: 'inputs/input????????')
    path source_script, stageAs: 'r02_lai_flare_query_projection.py'

    output:
    tuple val(task_id), path('projection/baseline-probabilities.npy'), path('projection/receipt.json')

    script:
    """
    python3 r02_lai_flare_query_projection.py \
      --request request.json --request-sha256 '${request_sha}' \
      --staged-inputs inputs --output-dir projection
    """
}

workflow {
    def workerUid = System.getenv('R02_LAI_UID')
    def workerGid = System.getenv('R02_LAI_GID')
    if (!params.request_manifest || !params.output_dir || !params.source_bin || !params.container)
        error 'request_manifest, output_dir, source_bin and immutable container are required'
    if (!(workerUid ==~ /[0-9]+/) || !(workerGid ==~ /[0-9]+/))
        error 'R02_LAI_UID and R02_LAI_GID are required numeric execution identities'
    def manifest = new groovy.json.JsonSlurper().parse(file(params.request_manifest, checkIfExists: true))
    if (manifest.schema != 'r02_lai_flare_query_projection_batch_v1' || manifest.status != 'READY')
        error 'READY authenticated FLARE projection batch required'
    def rows = manifest.tasks.collect { task ->
        def request = file(task.request.path, checkIfExists: true)
        def spec = new groovy.json.JsonSlurper().parse(request)
        def staged = spec.inputs.values().collect { file(it.path, checkIfExists: true) }.unique()
        tuple(task.task_id, request, task.request.sha256,
              [memory_bytes: task.resources.memory_bytes, timeout_seconds: task.resources.timeout_seconds], staged)
    }
    PROJECT_EXISTING_FLARE_TO_QUERY_GRID(Channel.fromList(rows),
        file(params.source_bin + '/r02_lai_flare_query_projection.py', checkIfExists: true))
}
