nextflow.enable.dsl = 2

params.batch = null
params.batch_sha256 = null
params.code_dir = null
params.output_dir = null
params.max_forks = 1

process PREPARE_LAI_REGIONAL_LAYOUT {
    tag task_id
    container 'sha256:f8c26d3769b82ef68b07ca5b67c9a6acff862bcde397addc61dc014ce8dd2896'
    cpus 1
    memory '2 GB'
    time '30m'
    maxForks params.max_forks
    cache 'deep'
    publishDir { params.output_dir + '/' + task_id }, mode: 'copy', overwrite: false

    input:
    tuple val(task_id), val(index_dir), val(scales_cm), val(core_chunks),
          val(depth), val(query_max_gap_cm), val(cnn_query_configs_json)
    path code
    path annotations
    path cnn_config_code

    output:
    tuple val(task_id), path('prepared/layout.json'), path('prepared/receipt.json')

    script:
    """
    python3 '${code}' --index-dir '${index_dir}' --output-dir prepared \
      --scales-cm '${scales_cm}' --core-chunks '${core_chunks}' \
      --depth '${depth}' --query-max-gap-cm '${query_max_gap_cm}' \
      --cnn-query-configs-by-scale '${cnn_query_configs_json}'
    """
}

workflow {
    if (!params.batch || !params.batch_sha256 || !params.code_dir || !params.output_dir)
        error 'batch, batch_sha256, code_dir and output_dir are required'
    def batchFile = file(params.batch, checkIfExists: true)
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    if (digest.digest(batchFile.bytes).encodeHex().toString() != params.batch_sha256)
        error 'batch SHA-256 differs'
    def batch = new groovy.json.JsonSlurper().parse(batchFile.toFile())
    if (batch.schema != 'r02_lai_regional_layout_batch_v2' ||
        batch.status != 'PREPARE_ONLY_NO_TRAINING_NO_SELECTION' ||
        batch.tasks?.size() != 12 || batch.tasks.collect { it.task_id }.toSet().size() != 12 ||
        batch.evaluate_opened != false || batch.training_authorized != false)
        error 'exact twelve-case preparation-only batch required'
    def rows = batch.tasks.collect { task ->
        if (!(task.role in ['FIT', 'SELECT']) || task.index_dir.contains('..') ||
            !(task.scales_cm instanceof List) || task.scales_cm.isEmpty() ||
            task.core_chunks != 16 || task.depth != 2 || task.query_max_gap_cm != 0.01)
            error 'regional task fields differ: ' + task.task_id
        def configs = task.cnn_query_configs_by_scale
        if (!(configs instanceof Map) || configs.keySet().collect { it as String }.toSet() !=
                task.scales_cm.collect { it.toString() }.toSet())
            error 'CNN configuration must cover every scale: ' + task.task_id
        configs.each { scale, cnn ->
            def cnnSteps = cnn.query_step_cm_by_chromosome instanceof Map
                ? cnn.query_step_cm_by_chromosome.values() : [cnn.query_step_cm]
            if (!(cnn instanceof Map) || !(cnn.keySet() in [
                    ['schema','kernel_sizes','query_step_cm'].toSet(),
                    ['schema','kernel_sizes','query_step_cm_by_chromosome'].toSet()]) ||
                cnn.schema != 'lai_query_multiscale_cnn_v1' || cnnSteps.isEmpty() ||
                cnnSteps.any { !(it instanceof Number) || it <= 0 || it > task.query_max_gap_cm } ||
                !(cnn.kernel_sizes instanceof List) || cnn.kernel_sizes.size() < 2 ||
                cnn.kernel_sizes.any { !(it instanceof Integer) || it < 3 || it % 2 != 1 } ||
                cnn.kernel_sizes != cnn.kernel_sizes.unique(false).sort(false))
                error 'invalid scale-specific multiscale convolution: ' + task.task_id + ':' + scale
        }
        tuple(task.task_id, task.index_dir, task.scales_cm.join(','), task.core_chunks,
              task.depth, task.query_max_gap_cm, groovy.json.JsonOutput.toJson(configs))
    }
    PREPARE_LAI_REGIONAL_LAYOUT(Channel.fromList(rows),
        file(params.code_dir + '/r02_lai_regional_inputs.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_event_annotations.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_cnn_config.py', checkIfExists: true))
    PREPARE_LAI_REGIONAL_LAYOUT.out.map { task, layout, receipt -> tuple(task, layout, receipt) }
        .set { prepared }
    prepared.view { task, layout, receipt -> "prepared ${task}" }
}
