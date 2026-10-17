nextflow.enable.dsl=2
params.existing_evidence = null

// Fixed saved partitions: interpretation only, never re-cluster the cohort.
process R02_COMMUNITY_LEVELS {
    tag '57_saved_partitions'
    cache 'deep'
    cpus 1
    input:
    path source_bin
    path contract
    path dependencies, stageAs: 'verified_inputs/input????/*'
    output:
    path 'community_levels', emit: evidence
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_community_levels.py' \
      --inputs '${contract}' --expected-inputs-sha256 '${params.input_contract_sha256}' \
      --max-memory-mb ${params.max_memory_mb} --expected-samples ${params.expected_samples} \
      --expected-partitions ${params.expected_partitions} --min-free-mib ${params.min_free_mib} \
      --output-dir community_levels
    """
}

process R02_COMMUNITY_LEVEL_FIGURES {
    tag 'levels_figures_and_interpretation'
    cache 'deep'
    cpus 1
    input:
    path source_bin
    path evidence
    output:
    path 'presentation', emit: presentation
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export MPLCONFIGDIR="\$PWD/.mpl"
    python3 '${source_bin}/r02_community_level_figures.py' --evidence '${evidence}' --output-dir presentation
    """
}

workflow {
  if (params.existing_evidence != null) {
    ['source_bin', 'existing_evidence'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    R02_COMMUNITY_LEVEL_FIGURES(file(params.source_bin, checkIfExists:true),
        file(params.existing_evidence, checkIfExists:true))
  } else {
    ['source_bin', 'input_contract', 'input_contract_sha256', 'input_dependency_manifest', 'min_free_gib', 'max_memory_mb', 'expected_samples', 'expected_partitions', 'min_free_mib'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    if (!(params.input_contract_sha256.toString() ==~ /[0-9a-f]{64}/)) error 'Invalid input contract SHA256'
    if (!(params.max_memory_mb.toString() ==~ /[1-9][0-9]*/)) error 'Invalid memory limit'
    if (!(params.min_free_gib.toString() ==~ /[0-9]+(\.[0-9]+)?/)) error 'Invalid disk reserve'
    ['expected_samples', 'expected_partitions', 'min_free_mib'].each { key ->
        if (!(params[key].toString() ==~ /[1-9][0-9]*/)) error "Invalid positive integer: ${key}"
    }
    if (params.min_free_mib.toBigDecimal() != params.min_free_gib.toBigDecimal() * 1024) error 'Disk reserve units differ'
    def names = new groovy.json.JsonSlurper().parse(file(params.input_dependency_manifest, checkIfExists:true).toFile())
    if (!(names instanceof List) || names.isEmpty() || names.toSet().size() != names.size()) error 'Unique input inventory required'
    def deps = names.collect { name ->
        if (!(name instanceof String) || name.find(/[\n\r'"`$\\;|&<>]/)) error 'Unsafe input path'
        file(name, checkIfExists:true)
    }
    def code = file(params.source_bin, checkIfExists:true)
    R02_COMMUNITY_LEVELS(code, file(params.input_contract, checkIfExists:true), deps)
    R02_COMMUNITY_LEVEL_FIGURES(code, R02_COMMUNITY_LEVELS.out.evidence)
  }
}
