nextflow.enable.dsl=2

// Integrate saved communities only. Do not estimate kinship or cluster again.
// The dependency list stages the authenticated small tables for deep caching.
// Nextflow mounts the original paths needed by the absolute-path contract.
// The program reads inputs only and verifies their hashes again before closing.
process R02_COMMUNITY_EVIDENCE {
    tag 'saved_autosomal_communities'
    cache 'deep'
    cpus 1
    input:
    path source_bin
    path contract
    path dependencies, stageAs: 'verified_inputs/input????/*'
    output:
    path 'community_evidence', emit: evidence
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_community_evidence.py' \
      --inputs '${contract}' --expected-inputs-sha256 '${params.input_contract_sha256}' \
      --max-memory-mb ${params.max_memory_mb} \
      --output-dir community_evidence
    """
}

workflow {
    ['source_bin', 'input_contract', 'input_contract_sha256', 'input_dependency_manifest', 'min_free_gib', 'max_memory_mb'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    if (!(params.max_memory_mb.toString() ==~ /[1-9][0-9]*/)) error 'max_memory_mb must be a positive integer'
    if (!(params.input_contract_sha256.toString() ==~ /[0-9a-f]{64}/)) error 'Invalid contract SHA256'
    if (!(params.min_free_gib.toString() ==~ /[0-9]+(\.[0-9]+)?/)) error 'min_free_gib must be nonnegative'
    def names = new groovy.json.JsonSlurper().parse(file(params.input_dependency_manifest, checkIfExists:true).toFile())
    if (!(names instanceof List) || names.isEmpty() || names.toSet().size() != names.size())
        error 'Explicit unique dependency list required'
    def deps = names.collect { name ->
        if (!(name instanceof String) || name.find(/[\n\r'"`$\\;|&<>]/)) error 'Unsafe dependency path'
        file(name, checkIfExists:true)
    }
    R02_COMMUNITY_EVIDENCE(file(params.source_bin, checkIfExists:true),
        file(params.input_contract, checkIfExists:true), deps)
}
