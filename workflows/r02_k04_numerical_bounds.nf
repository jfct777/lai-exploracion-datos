nextflow.enable.dsl=2

/*
 * Future finite operation for 66 K04 numerical-bound tasks.  The workflow
 * consumes only complete local BUILD products pinned by manifest SHA-256.  It
 * does not construct model inputs, choose plan parameters, fit, SELECT, read
 * EVALUATE, publish to GCS or manage cloud resources.
 */

params.catalog = null
params.catalog_sha256 = null
params.bounds_code = null
params.bounds_code_sha256 = null
params.pair_operator_code = null
params.pair_operator_sha256 = null
params.output_dir = null
params.container = null
params.task_cpus = null
params.task_memory_gb = null
params.task_time = null
params.max_forks = null
params.rows_per_chunk = null
params.max_working_mib = null
params.workers = null


process K04_NUMERICAL_BOUNDS {
    tag "K04-bounds:${task_id}"
    cpus params.task_cpus
    memory "${params.task_memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    container params.container
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple val(task_id), path(model_inputs), val(manifest_sha256),
          path(bounds_code, stageAs: 'code/r02_k04_numerical_bounds.py'),
          path(pair_operator, stageAs: 'code/r02_pair_operator.py')

    output:
    tuple val(task_id), path("bounds_${task_id}.json")

    script:
    """
    set -euo pipefail
    umask 077
    python3 code/r02_k04_numerical_bounds.py derive \
      --model-inputs '${model_inputs}' \
      --model-inputs-manifest-sha256 '${manifest_sha256}' \
      --bounds-code-sha256 '${params.bounds_code_sha256}' \
      --pair-operator-code code/r02_pair_operator.py \
      --pair-operator-sha256 '${params.pair_operator_sha256}' \
      --rows-per-chunk ${params.rows_per_chunk} \
      --max-working-mib ${params.max_working_mib} \
      --workers ${params.workers} \
      --output 'bounds_${task_id}.json'
    """
}


process K04_NUMERICAL_BOUNDS_GATE {
    tag 'K04-bounds-gate:66'
    cpus 1
    memory '1 GB'
    time '15m'
    maxForks 1
    cache 'deep'
    container params.container
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    path(bound_files, stageAs: 'bounds/*.json')
    path(bounds_code, stageAs: 'code/r02_k04_numerical_bounds.py')

    output:
    path('COMPLETE.json')

    script:
    """
    set -euo pipefail
    umask 077
    python3 code/r02_k04_numerical_bounds.py gate \
      --bounds-dir bounds --output COMPLETE.json
    """
}


def safeAbsolute(value) {
    value instanceof String && value.startsWith('/') &&
        !value.tokenize('/').contains('..') && !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}


def sha256(path) {
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream { stream ->
        new java.security.DigestInputStream(stream, digest).transferTo(java.io.OutputStream.nullOutputStream())
    }
    digest.digest().encodeHex().toString()
}


def pinnedFile(path, expected) {
    if (!safeAbsolute(path) || !(expected ==~ /[0-9a-f]{64}/))
        error 'Safe absolute file and SHA-256 required'
    def value = file(path, checkIfExists: true)
    if (!value.toFile().isFile() || java.nio.file.Files.isSymbolicLink(value.toFile().toPath()) ||
        sha256(value) != expected) error "Pinned file differs: ${path}"
    value
}


workflow {
    if (![params.catalog, params.bounds_code, params.pair_operator_code,
          params.output_dir].every { safeAbsolute(it) } ||
        ![params.catalog_sha256, params.bounds_code_sha256,
          params.pair_operator_sha256].every { it ==~ /[0-9a-f]{64}/ } ||
        !(params.container instanceof String) || !params.container.contains('@sha256:') ||
        ![params.task_cpus, params.task_memory_gb, params.max_forks,
          params.rows_per_chunk, params.max_working_mib, params.workers].every {
              it?.toString() ==~ /[1-9][0-9]*/ } ||
        !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/))
        error 'Pinned inputs and explicit positive finite resources required'

    def output = java.nio.file.Paths.get(params.output_dir).normalize()
    if (!output.isAbsolute() || !java.nio.file.Files.isDirectory(output) ||
        java.nio.file.Files.isSymbolicLink(output)) error 'Existing real output directory required'

    def catalog = pinnedFile(params.catalog, params.catalog_sha256)
    def boundsCode = pinnedFile(params.bounds_code, params.bounds_code_sha256)
    def pairOperator = pinnedFile(params.pair_operator_code, params.pair_operator_sha256)
    def specification = new groovy.json.JsonSlurper().parse(catalog.toFile())
    if (specification.schema != 'r02_k04_numerical_bounds_catalog_v1' ||
        specification.evaluate_allowed != false || specification.tasks?.size() != 66)
        error 'Exact 66-task FIT/SELECT-only numerical-bound catalog required'

    def expected = (1..22).collectMany { chromosome ->
        ['AFR', 'EUR', 'NAM'].collect { ancestry -> String.format('chr%02d_%s', chromosome, ancestry) }
    } as Set
    def seen = [] as Set
    def rows = []
    specification.tasks.each { task ->
        if (task.keySet() as Set != (['id', 'model_inputs', 'manifest_sha256'] as Set) ||
            !expected.contains(task.id) || !seen.add(task.id) ||
            !safeAbsolute(task.model_inputs) || !(task.manifest_sha256 ==~ /[0-9a-f]{64}/))
            error 'Unique canonical task and pinned model-input manifest required'
        def directory = file(task.model_inputs, checkIfExists: true)
        if (!directory.toFile().isDirectory() ||
            java.nio.file.Files.isSymbolicLink(directory.toFile().toPath()) ||
            sha256(file(task.model_inputs + '/manifest.json', checkIfExists: true)) != task.manifest_sha256)
            error "Model-input directory/manifest differs: ${task.id}"
        rows << tuple(task.id, directory, task.manifest_sha256, boundsCode, pairOperator)
    }
    if (seen != expected) error 'Complete 22x3 task set required'

    K04_NUMERICAL_BOUNDS(channel.fromList(rows))
    K04_NUMERICAL_BOUNDS_GATE(K04_NUMERICAL_BOUNDS.out.map { id, artifact -> artifact }.collect(),
                              boundsCode)
}
