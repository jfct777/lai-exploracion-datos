nextflow.enable.dsl = 2

params.plan = null
params.plan_sha256 = null
params.code_dir = null
params.output_dir = null
params.max_forks = 2

process PREPARE_LAI_DENSITY_VIEW {
    tag density_task_id
    container 'sha256:f8c26d3769b82ef68b07ca5b67c9a6acff862bcde397addc61dc014ce8dd2896'
    cpus 1
    memory '2 GB'
    time '30 min'
    maxForks params.max_forks
    cache 'deep'
    publishDir { params.output_dir + '/' + density_task_id }, mode: 'copy', overwrite: false

    input:
    tuple val(density_task_id), path(request), val(request_sha256)
    path density_code
    path control_code
    path factorized_code

    output:
    tuple val(density_task_id), path('prepared/control-views.npz'), path('prepared/receipt.json')

    script:
    """
    python3 '${density_code}' --request '${request}' --request-sha256 '${request_sha256}' \
      --output-dir prepared
    """
}

workflow {
    if (!params.plan || !params.plan_sha256 || !params.code_dir || !params.output_dir)
        error 'plan, plan_sha256, code_dir and output_dir are required'
    def planFile = file(params.plan, checkIfExists: true)
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    if (digest.digest(planFile.bytes).encodeHex().toString() != params.plan_sha256)
        error 'density plan SHA-256 differs'
    def plan = new groovy.json.JsonSlurper().parse(planFile.toFile())
    if (plan.schema != 'r02_lai_density_plan_v1' ||
        plan.status != 'THIRTY_SIX_RADIUS_SPECIFIC_DENSITY_PREPARE_REQUESTS' ||
        plan.tasks?.size() != 36 || plan.tasks.collect { it.density_task_id }.toSet().size() != 36 ||
        plan.training_authorized != false || plan.selection_authorized != false ||
        plan.evaluate_opened != false || plan.truth_loaded != false)
        error 'exact PREPARE-only 12-case by 3-radius density plan required'
    def rows = plan.tasks.collect { task ->
        def request = file(task.request.path, checkIfExists: true)
        def requestDigest = digest.digest(request.bytes).encodeHex().toString()
        if (requestDigest != task.request.sha256) error 'density request SHA-256 differs: ' + task.density_task_id
        tuple(task.density_task_id, request, task.request.sha256)
    }
    PREPARE_LAI_DENSITY_VIEW(Channel.fromList(rows),
        file(params.code_dir + '/r02_lai_density_control.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_control_views.py', checkIfExists: true),
        file(params.code_dir + '/r02_lai_factorized_inputs.py', checkIfExists: true))
}
