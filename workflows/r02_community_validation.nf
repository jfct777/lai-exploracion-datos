nextflow.enable.dsl=2
params.partition_id = null

// Conditional fixed-community comparisons. No genotyping, IBD inference or clustering.
process R02_COMMUNITY_VALIDATION {
    tag "${partition_id}"
    cache 'deep'
    cpus 1
    maxForks 1
    input:
    path source_bin
    path contract
    path dependencies, stageAs: 'verified_inputs/input????/*'
    val partition_id
    output:
    path 'community_validation*', emit: results
    script:
    def partitionOption = partition_id == 'ALL_LEGACY' ? '' : "--partition-id '${partition_id}'"
    def outputName = partition_id == 'ALL_LEGACY' ? 'community_validation' : "community_validation_${partition_id}"
    """
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 '${source_bin}/r02_community_validation.py' \
      --inputs '${contract}' --expected-inputs-sha256 '${params.input_contract_sha256}' \
      --max-memory-mb ${params.max_memory_mb} --min-free-mib ${params.min_free_mib} \
      --scratch-dir "\$PWD" --output-dir '${outputName}' ${partitionOption}
    """
}

workflow {
    ['source_bin', 'input_contract', 'input_contract_sha256', 'input_dependency_manifest',
     'max_memory_mb', 'min_free_mib'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    if (!(params.input_contract_sha256.toString() ==~ /[0-9a-f]{64}/)) error 'Invalid contract SHA256'
    ['max_memory_mb', 'min_free_mib'].each { key ->
        if (!(params[key].toString() ==~ /[1-9][0-9]*/)) error "Invalid positive resource limit: ${key}"
    }
    def names = new groovy.json.JsonSlurper().parse(file(params.input_dependency_manifest, checkIfExists: true).toFile())
    if (!(names instanceof List) || names.isEmpty() || names.toSet().size() != names.size()) error 'Unique dependency inventory required'
    def deps = names.collect { name ->
        if (!(name instanceof String) || name.find(/[\n\r'"`$\\;|&<>]/)) error 'Unsafe dependency path'
        file(name, checkIfExists: true)
    }
    def contractFile = file(params.input_contract, checkIfExists: true)
    def contract = new groovy.json.JsonSlurper().parse(contractFile.toFile())
    def partitionMode = contract.settings.analysis_mode == 'K05_PARTITION_ASSIGNED'
    def ids = partitionMode ? contract.settings.partition_ids : ['ALL_LEGACY']
    if (!(ids instanceof List) || ids.isEmpty() || ids.toSet().size() != ids.size() ||
        !ids.every { it instanceof String && it ==~ /[A-Za-z0-9][A-Za-z0-9_-]*/ } ||
        (partitionMode && ids.contains('ALL_LEGACY'))) error 'Unique safe partition IDs required'
    def taskIds = ids
    if (params.partition_id != null) {
        if (!partitionMode || !(params.partition_id instanceof String) ||
            !(params.partition_id ==~ /[A-Za-z0-9][A-Za-z0-9_-]*/) || !ids.contains(params.partition_id))
            error 'Operational partition ID must belong to the declared K05 family'
        taskIds = [params.partition_id]
    }
    // One input value is a separate cached task. IDs are scientific contract
    // hypotheses, not an unpinned command-line subset selected by results.
    R02_COMMUNITY_VALIDATION(file(params.source_bin, checkIfExists: true),
        contractFile, deps, Channel.fromList(taskIds))
}
