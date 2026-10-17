nextflow.enable.dsl=2

params.batch = null
params.batch_sha256 = null
params.code_dir = null
params.output_dir = null
params.container = null
params.build_cpus = null
params.build_memory_gb = null
params.build_time = null
params.build_max_forks = null
params.fit_cpus = null
params.fit_memory_gb = null
params.fit_time = null
params.fit_max_forks = null
params.checkpoint_dir = null
params.resume_code_sha256 = null
params.allow_compatible_budget_extension = false

include { K04_BUILD_DESIGN } from '../modules/r02_k04_build_design'

process K04_FIT_SELECT_W {
    tag "K04-fit-select:${task_id}"
    cpus params.fit_cpus
    memory "${params.fit_memory_gb} GB"
    time params.fit_time
    maxForks params.fit_max_forks
    cache 'deep'
    container params.container
    // state_root is passed as a value so Nextflow does not stage or copy the
    // persistent candidate store.  Mount the already validated root at the
    // identical absolute path for atomic candidate receipts across retries.
    containerOptions { "-v ${params.checkpoint_dir}:${params.checkpoint_dir}" }
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple val(task_id), path(model_inputs), path(fit_plan), val(fit_plan_sha256),
          val(output_name), val(state_root), val(allow_budget_extension),
          path(bridge_code, stageAs: 'code/*'), path(fit_code, stageAs: 'code/r02_k04_fit_select.py'),
          path(resume_code, stageAs: 'code/r02_k04_candidate_store.py')

    output:
    tuple val(task_id), val(fit_plan_sha256), path("${output_name}")

    script:
    def extensionFlag = allow_budget_extension ? '--allow-compatible-budget-extension' : ''
    """
    set -euo pipefail
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code
    export OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    manifest_sha="\$(sha256sum '${model_inputs}/manifest.json' | cut -d' ' -f1)"
    python3 code/r02_k04_fit_select.py --model-inputs '${model_inputs}' \
      --model-inputs-manifest-sha256 "\${manifest_sha}" --fit-plan '${fit_plan}' \
      --fit-plan-sha256 '${fit_plan_sha256}' --output '${output_name}' \
      --state-root '${state_root}' ${extensionFlag}
    """

    stub:
    """
    mkdir '${output_name}'
    printf '%s\n' '{"schema":"r02_k04_fit_select_stub_v1","scientific_result":false}' > '${output_name}/manifest.json'
    """
}

def safePath(value) {
    value instanceof String && value.startsWith('/') && !value.tokenize('/').contains('..') &&
        !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

def sha256(path) {
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream { stream ->
        new java.security.DigestInputStream(stream, digest).transferTo(java.io.OutputStream.nullOutputStream())
    }
    digest.digest().encodeHex().toString()
}

def pinned(path, expected) {
    if (!safePath(path) || !(expected ==~ /[0-9a-f]{64}/)) error 'Safe absolute path and SHA256 required'
    def value = file(path, checkIfExists: true)
    if (!value.toFile().isFile() || sha256(value) != expected) error "Pinned K04 file differs: ${path}"
    value
}

workflow {
    if (![params.batch, params.code_dir, params.output_dir, params.checkpoint_dir].every { safePath(it) } ||
        !(params.batch_sha256 ==~ /[0-9a-f]{64}/) || !(params.container instanceof String) ||
        !(params.resume_code_sha256 ==~ /[0-9a-f]{64}/) ||
        !(params.allow_compatible_budget_extension instanceof Boolean) ||
        !params.container.contains('@sha256:') ||
        ![params.build_cpus, params.build_memory_gb, params.build_max_forks,
          params.fit_cpus, params.fit_memory_gb, params.fit_max_forks].every { it?.toString() ==~ /[1-9][0-9]*/ } ||
        ![params.build_time, params.fit_time].every { it instanceof String && it ==~ /[1-9][0-9]*[smh]/ })
        error 'Pinned batch/container and measured positive process limits required'
    def checkpointRoot = java.nio.file.Paths.get(params.checkpoint_dir).normalize()
    if (!checkpointRoot.isAbsolute() || !java.nio.file.Files.isDirectory(checkpointRoot) ||
        java.nio.file.Files.isSymbolicLink(checkpointRoot))
        error 'Checkpoint root must be an existing real absolute local/shared directory'
    def batch = pinned(params.batch, params.batch_sha256)
    def value = new groovy.json.JsonSlurper().parse(batch.toFile())
    def codeNames = ['r02_k04_model_bridge.py', 'r02_k04_predictor_inputs.py',
                     'r02_pair_operator.py', 'r02_interval_ridge.py', 'r02_k04_fit_select.py',
                     'r02_k04_precision_policy.py'] as Set
    if (value.schema != 'r02_k04_production_pipeline_batch_v1' || value.tasks?.size() != 66 ||
        value.implementation_sha256.keySet() as Set != codeNames || value.evaluate_allowed != false)
        error 'Exact complete FIT/SELECT production batch required'
    def allCode = value.implementation_sha256.collectEntries { name, digest ->
        [(name): pinned(params.code_dir + '/' + name, digest)]
    }
    def resumeCode = pinned(params.code_dir + '/r02_k04_candidate_store.py', params.resume_code_sha256)
    def bridgeNames = codeNames - ['r02_k04_fit_select.py']
    def seen = [] as Set
    def buildJobs = []
    def fitJobs = []
    value.tasks.each { task ->
        if (!(task.id ==~ /chr(0[1-9]|1[0-9]|2[0-2])_(AFR|EUR|NAM)/) || !seen.add(task.id))
            error 'Unique 22x3 task identity required'
        def request = pinned(task.request, task.request_sha256)
        def plan = pinned(task.fit_plan, task.fit_plan_sha256)
        def staged = task.inputs.collect { pinned(it.path, it.sha256) }
        buildJobs << tuple(task.id, request, task.request_sha256, staged,
                           bridgeNames.collect { allCode[it] })
        def outputName = "fit_select_${task.id}_${task.fit_plan_sha256.substring(0, 12)}"
        def stateRoot = checkpointRoot.resolve(task.id).toString()
        fitJobs << tuple(task.id, plan, task.fit_plan_sha256, outputName, stateRoot,
                         params.allow_compatible_budget_extension,
                         bridgeNames.collect { allCode[it] }, allCode['r02_k04_fit_select.py'], resumeCode)
    }
    if (seen.size() != 66) error 'Complete 22 chromosome x 3 ancestry task set required'
    K04_BUILD_DESIGN(channel.fromList(buildJobs))
    K04_FIT_SELECT_W(K04_BUILD_DESIGN.out.join(channel.fromList(fitJobs), by: 0))
}
