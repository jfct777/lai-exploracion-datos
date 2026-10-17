nextflow.enable.dsl=2

params.admission = null
params.admission_sha256 = null
params.review = null
params.review_sha256 = null
params.code_dir = null
params.output_dir = null
params.checkpoint_dir = null
params.container = null
params.fit_cpus = null
params.fit_memory_gb = null
params.fit_time = null
params.fit_max_forks = null
params.allow_compatible_budget_extension = false

process K04_FIT_SELECT_PUBLISHED {
    tag "K04-published-fit-select:${task_id}"
    cpus params.fit_cpus
    memory "${params.fit_memory_gb} GB"
    time params.fit_time
    maxForks params.fit_max_forks
    cache 'deep'
    container params.container
    containerOptions { "-v ${matrix_dir}:${matrix_dir}:ro -v ${fit_plan}:${fit_plan}:ro -v ${w_plan}:${w_plan}:ro -v ${state_root}:${state_root}" }
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple val(task_id), val(matrix_dir), val(matrix_manifest_sha256), val(fit_plan),
          val(fit_plan_sha256), val(w_plan), val(w_plan_sha256), val(state_root), val(extension),
          path(code_files, stageAs: 'code/*')

    output:
    tuple val(task_id), val(fit_plan_sha256), path("fit_select_${task_id}_${fit_plan_sha256.substring(0,12)}")

    script:
    def extensionFlag = extension ? '--allow-compatible-budget-extension' : ''
    def outputName = "fit_select_${task_id}_${fit_plan_sha256.substring(0,12)}"
    """
    set -euo pipefail
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code
    export OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    python3 code/r02_k04_fit_select.py --model-inputs '${matrix_dir}' \
      --model-inputs-manifest-sha256 '${matrix_manifest_sha256}' \
      --fit-plan '${fit_plan}' --fit-plan-sha256 '${fit_plan_sha256}' \
      --output '${outputName}' --state-root '${state_root}' ${extensionFlag}
    """
}

def safeAbsolute(value) {
    value instanceof String && value.startsWith('/') && !value.tokenize('/').contains('..') &&
        !value.find(/[\n\r'"`\$\\;|&<>*?\[\]{}]/)
}

def digest(path) {
    def md = java.security.MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream { input ->
        new java.security.DigestInputStream(input, md).transferTo(java.io.OutputStream.nullOutputStream())
    }
    md.digest().encodeHex().toString()
}

workflow {
    if (![params.admission, params.review, params.code_dir, params.output_dir, params.checkpoint_dir].every { safeAbsolute(it) } ||
        ![params.admission_sha256, params.review_sha256].every { it ==~ /[0-9a-f]{64}/ } ||
        !(params.container instanceof String) ||
        !params.container.contains('@sha256:') || !(params.allow_compatible_budget_extension instanceof Boolean) ||
        ![params.fit_cpus, params.fit_memory_gb, params.fit_max_forks].every { it?.toString() ==~ /[1-9][0-9]*/ } ||
        !(params.fit_time instanceof String && params.fit_time ==~ /[1-9][0-9]*[smh]/))
        error 'Pinned admission/container and measured positive FIT resources required'
    def admissionFile = file(params.admission, checkIfExists: true)
    if (digest(admissionFile) != params.admission_sha256) error 'Admission SHA-256 differs'
    def admission = new groovy.json.JsonSlurper().parse(admissionFile.toFile())
    if (admission.schema != 'r02_k04_published_fit_consumer_admission_v1' ||
        admission.status != 'READY_FOR_ROOT_REVIEW_NOT_LAUNCHED' || admission.launch_allowed != false ||
        admission.evaluate_allowed != false || admission.matrix_rebuild_allowed != false ||
        admission.scope != 'FIT_SELECT_ONLY_NOT_EVALUATE_OR_BIOLOGICAL_VALIDATION' ||
        admission.deep_payload_authentication != true || admission.tasks?.size() != 66 || admission.blockers)
        error 'Complete deeply authenticated published-matrix admission required'
    def reviewFile = file(params.review, checkIfExists: true)
    if (digest(reviewFile) != params.review_sha256) error 'Root review SHA-256 differs'
    def review = new groovy.json.JsonSlurper().parse(reviewFile.toFile())
    def reviewFields = ['schema', 'decision', 'admission_sha256', 'scope',
                        'evaluate_allowed', 'matrix_rebuild_allowed'] as Set
    if (review.keySet() != reviewFields ||
        review.schema != 'r02_k04_published_fit_select_root_review_v1' ||
        review.decision != 'ADMIT_FIT_SELECT' || review.admission_sha256 != params.admission_sha256 ||
        review.scope != 'FIT_SELECT_ONLY_NOT_EVALUATE_OR_BIOLOGICAL_VALIDATION' ||
        review.evaluate_allowed != false || review.matrix_rebuild_allowed != false)
        error 'Exact independent root review for this admission and FIT/SELECT-only scope required'
    def checkpointRoot = java.nio.file.Paths.get(params.checkpoint_dir).normalize()
    if (!checkpointRoot.isAbsolute() || !java.nio.file.Files.isDirectory(checkpointRoot) ||
        java.nio.file.Files.isSymbolicLink(checkpointRoot)) error 'Real persistent checkpoint root required'
    def codeFiles = admission.implementation_sha256.collect { name, expected ->
        def candidate = file(params.code_dir + '/' + name, checkIfExists: true)
        if (digest(candidate) != expected) error "Consumer code differs: ${name}"
        candidate
    }
    def jobs = admission.tasks.collect { task ->
        if (!safeAbsolute(task.w_plan) || !(task.w_plan_sha256 ==~ /[0-9a-f]{64}/))
            error "Safe pinned W plan required: ${task.id}"
        def wPlan = file(task.w_plan, checkIfExists: true)
        if (digest(wPlan) != task.w_plan_sha256) error "W plan differs: ${task.id}"
        def stateRoot = checkpointRoot.resolve(task.id)
        java.nio.file.Files.createDirectories(stateRoot)
        tuple(task.id, task.matrix_dir, task.matrix_manifest_sha256, task.fit_plan,
              task.fit_plan_sha256, task.w_plan, task.w_plan_sha256, stateRoot.toString(),
              params.allow_compatible_budget_extension, codeFiles)
    }
    K04_FIT_SELECT_PUBLISHED(channel.fromList(jobs))
}
