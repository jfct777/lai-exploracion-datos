#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// One frozen recipe. The driver may request the next only after this completes.
// Absolute authenticated tensor paths require a local/shared admitted filesystem.
params.plan = null
params.binding = null
params.binding_sha256 = null
params.objective = null
params.state_dir = null
params.outdir = null
params.code_dir = "${projectDir}/../bin"
params.source_manifest = "${projectDir}/../conf/r02_context_sources.txt"
params.python = 'python3'
params.task_cpus = 1
params.task_memory = '2 GB'
params.max_forks = 1
params.trial_number = null
// Technical durability only. The default confirms every completed Adam step.
// This local/shared directory remains stable across Nextflow retry work hashes.
params.checkpoint_dir = null
params.checkpoint_every_steps = 1
// 'auto' reads the precision frozen in a prospective resolved recipe. Historical
// plans retain their original fp32 behavior.
params.adam_state_precision = 'auto'
params.task_retries = 1

def absoluteCheckpointRoot(value) {
    def candidate = java.nio.file.Paths.get(value.toString())
    def root = java.nio.file.Paths.get(launchDir.toString())
    return (candidate.isAbsolute() ? candidate : root.resolve(candidate)).normalize().toString()
}

process PREFLIGHT_CONTEXT {
    cache false
    cpus 1
    memory params.task_memory
    input:
    path plan
    path binding
    val binding_sha
    output:
    path 'admission.json', emit: gate
    script:
    """
    ${params.python} '${params.code_dir}/r02_context_search_bridge.py' preflight \\
      --plan '${plan}' --binding '${binding}' --binding-sha256 '${binding_sha}' \\
      --objective '${params.objective}' --output admission.json
    """
}

process PREPARE_RECIPE {
    cache 'deep'
    cpus 1
    memory '1 GB'
    input:
    path plan
    path binding
    val binding_sha
    val trial_number
    path admission
    path code_fingerprint
    output:
    path 'proposal', emit: proposal
    script:
    """
    ${params.python} '${params.code_dir}/r02_context_search_bridge.py' prepare \
      --plan '${plan}' --binding '${binding}' --binding-sha256 '${binding_sha}' \
      --objective '${params.objective}' --state-dir '${params.state_dir}' --trial-number ${trial_number} --output proposal
    """
}

process TRAIN_CONTEXT_TASK {
    cache 'deep'
    tag "${taskfile.simpleName}"
    cpus params.task_cpus
    memory params.task_memory
    maxForks params.max_forks
    maxRetries params.task_retries
    errorStrategy { task.exitStatus in [130, 137, 143] ? 'retry' : 'terminate' }
    publishDir "${params.outdir}/trial-${params.trial_number}/tasks", mode:'copy', overwrite:false,
      saveAs: { filename -> "${taskfile.simpleName}/${filename}" }
    input:
    tuple path(taskfile), val(tasksha), path(resolved)
    path code_fingerprint
    val checkpoint_root
    output:
    path 'trained', emit: results
    script:
    """
    export OMP_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=${task.cpus}
    ${params.python} '${params.code_dir}/r02_context_search_bridge.py' task \
      --task '${taskfile}' --task-sha256 '${tasksha}' --resolved '${resolved}' --output trained \
      --checkpoint-root '${checkpoint_root}' \
      --checkpoint-every-steps ${params.checkpoint_every_steps} \
      --adam-state-precision '${params.adam_state_precision}'
    """
}

process COLLECT_RECIPE {
    cpus 1
    memory '1 GB'
    cache false
    publishDir "${params.outdir}/trial-${params.trial_number}", mode:'copy', overwrite:false
    input:
    path plan
    path proposal
    val result_paths
    output:
    path 'completed'
    script:
    def inputs = result_paths.collect { "'${it}/search-record.json'" }.join(' ')
    """
    ${params.python} '${params.code_dir}/r02_context_search_bridge.py' collect \
      --plan '${plan}' --state-dir '${params.state_dir}' --recipe '${proposal}/recipe.json' \
      --task-index '${proposal}/tasks.json' --results ${inputs} --output completed
    """
}

workflow {
    ['plan','binding','binding_sha256','objective','state_dir','outdir'].each {
        if (!params[it]) error "Missing --${it}"
    }
    if (!(params.objective in ['structure','lai'])) error 'Invalid objective'
    if (params.trial_number == null) error 'Missing --trial_number (zero based)'
    if (params.state_dir.toString().contains('://')) error 'SQLite state must be local'
    def checkpointSetting = (params.checkpoint_dir ?: params.outdir + "/internal-checkpoints").toString()
    if (checkpointSetting.contains('://')) error 'Checkpoint storage must be local/shared filesystem'
    def checkpointRoot = absoluteCheckpointRoot(checkpointSetting)
    if (!java.nio.file.Paths.get(checkpointRoot).isAbsolute()) error 'Resolved checkpoint storage must be absolute'
    if (!(params.checkpoint_every_steps instanceof Number) || params.checkpoint_every_steps < 1) error 'Invalid checkpoint step frequency'
    if (!(params.adam_state_precision in ['auto','fp32','fp64'])) error 'Invalid Adam state precision'
    if (!(params.task_retries instanceof Number) || params.task_retries < 0) error 'Invalid retry count'
    def plan = Channel.value(file(params.plan, checkIfExists:true))
    def binding = Channel.value(file(params.binding, checkIfExists:true))
    def names = file(params.source_manifest ?: "${projectDir}/../conf/r02_context_sources.txt",
                     checkIfExists:true).readLines().findAll { it }
    if (names.isEmpty() || names.toSet().size() != names.size() ||
        names.any { !(it ==~ /[A-Za-z0-9_]+\.py/) }) error 'Invalid Python source manifest'
    def fingerprint = Channel.value(names.collect {
        file("${params.code_dir}/${it}", checkIfExists:true) })
    PREFLIGHT_CONTEXT(plan, binding, params.binding_sha256)
    PREPARE_RECIPE(plan, binding, params.binding_sha256, params.trial_number as int,
                   PREFLIGHT_CONTEXT.out.gate, fingerprint)
    def jobs = PREPARE_RECIPE.out.proposal.flatMap { folder ->
        def entries = new groovy.json.JsonSlurper().parse(folder.resolve('tasks.json').toFile())
        entries.collect { entry ->
            tuple(file(entry.path, checkIfExists:true), entry.sha256,
                  folder.resolve('resolved.json'))
        }
    }
    TRAIN_CONTEXT_TASK(jobs, fingerprint, Channel.value(checkpointRoot))
    COLLECT_RECIPE(plan, PREPARE_RECIPE.out.proposal,
                   TRAIN_CONTEXT_TASK.out.results.map { it.toAbsolutePath().toString() }.collect())
}
