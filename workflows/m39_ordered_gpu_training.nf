nextflow.enable.dsl = 2

include { M39_ORDERED_GPU_TRAINING } from '../modules/39_ORDERED_GPU_TRAINING'

workflow {
    ['m39_train_store', 'm39_select_store', 'm39_development', 'm39_training_plan',
     'm39_source_seal', 'm39_source_commit', 'm39_plan_sha256', 'm39_output_dir'].each { key ->
        if (!params[key]) error "--${key} is required"
    }
    if (!(params.m39_gpu_run_id ==~ /m39-gpu-[a-z0-9-]{3,45}/)) error 'Use a unique GPU run ID'
    if (!(params.m39_gpu_image ==~ /us-central1-docker\.pkg\.dev\/uspbr-242713\/dnabr-lai\/[a-z0-9-]+@sha256:[0-9a-f]{64}/))
        error 'GPU image must be pinned by digest'
    if (params.m39_output_dir != "gs://teams-usp/frank/lai-exploracion-datos/runs/${params.m39_gpu_run_id}/outputs" && !workflow.stubRun)
        error 'Outputs must stay in the project-owned run prefix'
    if (workflow.resume) error 'Use a new immutable training run'
    def planPath = file(params.m39_training_plan, checkIfExists: true)
    def digest = java.security.MessageDigest.getInstance('SHA-256').digest(planPath.bytes).encodeHex().toString()
    if (digest != params.m39_plan_sha256) error 'Training plan hash differs'
    def plan = new groovy.json.JsonSlurper().parseText(planPath.text)
    if (plan.schema_version != 'm39-ordered-gpu-training-plan-v1' || plan.scope != 'exploratory_chr22_R0_development_anchors_only')
        error 'Training plan scope differs'
    if (plan.resources.max_workers != params.m39_max_workers || plan.resources.task_seconds != params.m39_task_seconds)
        error 'Worker limits differ from frozen plan'
    def seal = file(params.m39_source_seal, checkIfExists: true)
    def binding = new groovy.json.JsonSlurper().parseText(seal.text)
    if (binding.source_commit != params.m39_source_commit || binding.profile_sha256 != digest)
        error 'Source binding differs'
    def repoDir = projectDir.resolve('..')
    def sources = binding.source_sha256.keySet().sort().collect { name ->
        if (!(name ==~ /m[0-9]+_[a-z0-9_]+\.py/)) error 'Unsafe source member'
        file("${repoDir}/bin/${name}", checkIfExists: true)
    }
    def configs = plan.groups.collectMany { group -> group.configs.collect { it.file } }.collect { name ->
        if (!(name ==~ /[a-z0-9][a-z0-9-]*\.json/)) error 'Unsafe config filename'
        file(planPath.parent.resolve(name), checkIfExists: true)
    }
    def jobs = channel.fromList(plan.groups).map { group ->
        if (!(group.id ==~ /[a-z0-9][a-z0-9-]{2,79}/)) error 'Unsafe group identifier'
        tuple(group.id, [planPath] + configs)
    }
    M39_ORDERED_GPU_TRAINING(jobs,
        channel.value(file(params.m39_train_store, checkIfExists: true)),
        channel.value(file(params.m39_select_store, checkIfExists: true)),
        channel.value(file(params.m39_development, checkIfExists: true)),
        channel.value(seal), channel.value(sources))
}
