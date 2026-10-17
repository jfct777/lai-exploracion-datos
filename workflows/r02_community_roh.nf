nextflow.enable.dsl=2

// Reuse completed ROH and fixed communities; never invoke a genotype caller.
process R02_COMMUNITY_ROH {
    tag 'saved_roh_by_community'
    cache 'deep'
    cpus 1
    input:
    path source_bin
    path contract
    path dependencies, stageAs: 'verified_inputs/input????/*'
    output:
    path 'community_roh', emit: evidence
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 '${source_bin}/r02_community_roh.py' \
      --contract '${contract}' --contract-sha256 '${params.contract_sha256}' \
      --max-memory-mb ${params.max_memory_mb} --min-free-mb ${params.min_free_mb} \
      --output-dir community_roh
    """
}

workflow {
    ['source_bin', 'contract', 'contract_sha256', 'dependency_manifest', 'max_memory_mb', 'min_free_mb'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    if (!(params.contract_sha256.toString() ==~ /[0-9a-f]{64}/)) error 'Invalid contract SHA256'
    if (!(params.max_memory_mb.toString() ==~ /[1-9][0-9]*/) || params.max_memory_mb.toInteger()>2048)
        error 'Memory must be positive and at most 2048 MiB'
    if (!(params.min_free_mb.toString() ==~ /[0-9]+/)) error 'Invalid disk reserve'
    def names = new groovy.json.JsonSlurper().parse(file(params.dependency_manifest, checkIfExists:true).toFile())
    if (!(names instanceof List) || names.isEmpty() || names.toSet().size()!=names.size()) error 'Unique dependency inventory required'
    def inputs = names.collect { name ->
        if (!(name instanceof String) || name.find(/[\n\r'"`$\\;|&<>]/)) error 'Unsafe dependency path'
        file(name, checkIfExists:true)
    }
    R02_COMMUNITY_ROH(file(params.source_bin, checkIfExists:true), file(params.contract, checkIfExists:true), inputs)
}
