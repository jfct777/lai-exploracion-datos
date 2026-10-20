nextflow.enable.dsl=2

params.plan = null
params.plan_sha256 = null
params.outdir = null
params.python = 'python3'
params.script = null
params.pending = null

process PREPARE_EXPANDED_LAI_CASE {
    tag { task_id }
    cpus 1
    memory '1536 MB'
    time '30m'
    maxForks 1
    // `path 'case-output/*'` includes the staging-directory prefix in the
    // publish name.  Strip it so a future run publishes the receipt and the
    // three canonical entries directly below the task directory.
    publishDir params.outdir, mode: 'copy', overwrite: false,
        saveAs: { name -> "${task_id}/${name.replaceFirst(/^case-output\//, '')}" }

    input:
    val task_id

    output:
    path 'case-output/*', emit: prepared

    script:
    """
    ${params.python} ${params.script} prepare-case \\
      --plan ${params.plan} \\
      --plan-sha256 ${params.plan_sha256} \\
      --task-id '${task_id}' \\
      --output case-output
    """
}

workflow {
    if (!params.plan || !params.plan_sha256 || !params.outdir || !params.script || !params.pending) {
        error 'Required: --plan --plan_sha256 --outdir --script --pending'
    }
    def plan = new groovy.json.JsonSlurper().parse(new File(params.plan))
    def admission = new groovy.json.JsonSlurper().parse(new File(params.pending))
    if (admission.plan_sha256 != params.plan_sha256) error 'Pending-case admission plan hash differs'
    def allIds = plan.cases.collect { it.task_id } as Set
    def pending = admission.pending_task_ids
    if (pending.size() != pending.toSet().size() || !allIds.containsAll(pending)) error 'Pending-case axis invalid'
    Channel.fromList(pending).set { tasks }
    PREPARE_EXPANDED_LAI_CASE(tasks)
}
