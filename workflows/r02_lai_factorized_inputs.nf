nextflow.enable.dsl = 2
params.plan = null
params.code = null
params.output = null

process LAI_PREDICTOR_INDEX {
    tag task_id
    cpus 1
    memory '2 GB'
    time '30m'
    maxForks 2
    input:
    val task_id
    path code
    path annotations
    output:
    tuple val(task_id), path('done.json')
    script:
    """
    python3 '${code}' \
      --plan '${params.plan}' --task-id '${task_id}' \
      --output '${params.output}/${task_id}'
    cp '${params.output}/${task_id}/receipt.json' done.json
    """
}

workflow {
    if (!params.plan || !params.code || !params.output) error 'plan, code, output required'
    def plan = new groovy.json.JsonSlurper().parse(file(params.plan).toFile())
    LAI_PREDICTOR_INDEX(Channel.fromList(plan.tasks.collect{it.task_id}),
        file(params.code + '/r02_lai_factorized_inputs.py', checkIfExists: true),
        file(params.code + '/r02_lai_event_annotations.py', checkIfExists: true))
}
