nextflow.enable.dsl = 2

def requireCompletedGroup(resultDir, plan, sourceCommit, planHash, sealHash, stubRun) {
    def receiptPath = resultDir.resolve('group.completion.json')
    if (!receiptPath.exists()) error "Missing group completion receipt; preserved work: ${resultDir}"
    def receipt = new groovy.json.JsonSlurper().parseText(receiptPath.text)
    def groupId = resultDir.name.toString().replaceFirst(/^training-/, '')
    def group = plan.groups.find { it.id == groupId }
    if (!group || receipt.group_id != groupId)
        error "Group receipt identifier differs; preserved work: ${resultDir}"
    if (stubRun && receipt.status == 'STUB_ONLY_NOT_TRAINING' && receipt.SCORE_opened == false)
        return resultDir
    if (receipt.status != 'COMPLETED_DECLARED_PAIRED_ARMS_NEEDS_SCIENTIFIC_POST' || receipt.exit_code != 0)
        error "Scientific group ${groupId} failed (${receipt.status}); preserved work: ${resultDir}"
    def stageArms = [exploratory_screen: ['common', 'real'],
        controlled_followup: ['common', 'pooled', 'real', 'sham'],
        technical_e2e: ['common', 'pooled', 'real', 'sham'],
        multichannel_screen: ['none', 'both'],
        multichannel_followup: ['none', 'summary', 'detail', 'both', 'sham'],
        multichannel_technical: ['none', 'summary', 'detail', 'both', 'sham']]
    def expectedArms = stageArms[plan.stage] ?: []
    if (receipt.schema_version != 'm39-ordered-gpu-group-completion-v1' ||
        receipt.stage != plan.stage || receipt.source_commit != sourceCommit ||
        receipt.plan_sha256 != planHash || receipt.source_seal_sha256 != sealHash ||
        !expectedArms || receipt.SCORE_opened != false || receipt.completed_arms != expectedArms ||
        receipt.case_receipt_sha256?.keySet() != expectedArms.toSet() ||
        !receipt.case_receipt_sha256.values().every { it ==~ /[0-9a-f]{64}/ })
        error "Incomplete or unbound scientific group ${groupId}; preserved work: ${resultDir}"
    return resultDir
}

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
    def sealHash = java.security.MessageDigest.getInstance('SHA-256').digest(seal.bytes).encodeHex().toString()
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
    // Stage-out to work has completed here; publishDir copies are asynchronous.
    // Fail immediately on a recorded scientific failure without another GPU task.
    M39_ORDERED_GPU_TRAINING.out.results.map { resultDir ->
        requireCompletedGroup(resultDir, plan, params.m39_source_commit, digest, sealHash, workflow.stubRun)
    }.subscribe { resultDir -> log.info "Group receipt accepted: ${resultDir.name}" }
}
