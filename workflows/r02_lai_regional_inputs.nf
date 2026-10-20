nextflow.enable.dsl = 2

params.batch = null
params.batch_sha256 = null
params.code_dir = null
params.output_dir = null
params.max_forks = 1

process PREPARE_LAI_REGIONAL_LAYOUT {
    tag task_id
    cpus 1
    memory '2 GB'
    time '30m'
    maxForks params.max_forks
    cache 'deep'
    publishDir { params.output_dir + '/' + task_id }, mode: 'copy', overwrite: false

    input:
    tuple val(task_id), val(index_dir), val(scales_cm), val(core_chunks),
          val(depth), val(query_max_gap_cm)
    path code
    path annotations

    output:
    tuple val(task_id), path('prepared/layout.json'), path('prepared/receipt.json')

    script:
    """
    python3 '${code}' --index-dir '${index_dir}' --output-dir prepared \
      --scales-cm '${scales_cm}' --core-chunks '${core_chunks}' \
      --depth '${depth}' --query-max-gap-cm '${query_max_gap_cm}'
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
    if (batch.schema != 'r02_lai_regional_layout_batch_v1' ||
        batch.status != 'PREPARE_ONLY_NO_TRAINING_NO_SELECTION' ||
        batch.tasks?.size() != 12 || batch.tasks.collect { it.task_id }.toSet().size() != 12 ||
        batch.evaluate_opened != false || batch.training_authorized != false)
        error 'exact twelve-case preparation-only batch required'
    def rows = batch.tasks.collect { task ->
        if (!(task.role in ['FIT', 'SELECT']) || task.index_dir.contains('..') ||
            !(task.scales_cm instanceof List) || task.scales_cm.isEmpty() ||
            task.core_chunks != 16 || task.depth != 2 || task.query_max_gap_cm != 0.01)
            error 'regional task fields differ: ' + task.task_id
        tuple(task.task_id, task.index_dir, task.scales_cm.join(','), task.core_chunks,
              task.depth, task.query_max_gap_cm)
    }
    PREPARE_LAI_REGIONAL_LAYOUT(Channel.fromList(rows),
        file(params.code_dir + '/r02_lai_regional_inputs.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_event_annotations.py', checkIfExists: true))
    PREPARE_LAI_REGIONAL_LAYOUT.out.map { task, layout, receipt -> tuple(task, layout, receipt) }
        .set { prepared }
    prepared.view { task, layout, receipt -> "prepared ${task}" }
}
