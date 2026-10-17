nextflow.enable.dsl=2
params.selected_chromosomes = null

def safeFile(name) {
    if (!(name instanceof String) || name.find(/[\n\r'"`$\\;|&<>]/)) error 'Unsafe input path'
    return file(name, checkIfExists:true)
}

process R02_COMMUNITY_IBD_ANCESTRY_CHROMOSOME {
    tag "chr${chromosome}"
    cpus 1
    cache 'deep'
    maxForks params.chromosome_concurrency as int
    time { "${params.max_wall_seconds}s" }
    input:
    path source_bin
    path contract, stageAs: 'analysis_contract.json'
    path dependencies, stageAs: 'community_inputs/input????/*'
    tuple val(chromosome), path(annotation_manifest), path(annotation_contract), path(parents), path(children)
    output:
    tuple val(chromosome), path('ancestry_chr'), emit: partial
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 '${source_bin}/r02_community_ibd_ancestry.py' chromosome \
      --contract '${contract}' --contract-sha256 '${params.contract_sha256}' \
      --chromosome '${chromosome}' --output-dir ancestry_chr --scratch "\$PWD" \
      --max-memory-mb ${params.max_memory_mb} --min-free-mb ${params.min_free_mb} \
      --sort-memory-mb ${params.sort_memory_mb} --max-temp-mb ${params.max_temp_mb} \
      --block-size ${params.pair_block_size} --sort-timeout ${params.max_wall_seconds}
    """
}

process R02_COMMUNITY_IBD_ANCESTRY_AGGREGATE {
    tag 'complete_autosomes'
    cpus 1
    cache 'deep'
    time { "${params.max_wall_seconds}s" }
    input:
    path source_bin
    path contract, stageAs: 'analysis_contract.json'
    path dependencies, stageAs: 'community_inputs/input????/*'
    path parts, stageAs: 'partials/chr??'
    output:
    path 'community_ibd_ancestry', emit: evidence
    script:
    def manifests = parts.collect { "'${it}/manifest.json'" }.join(' ')
    """
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 '${source_bin}/r02_community_ibd_ancestry.py' aggregate \
      --contract '${contract}' --contract-sha256 '${params.contract_sha256}' \
      --parts ${manifests} --output-dir community_ibd_ancestry --scratch "\$PWD" \
      --max-memory-mb ${params.max_memory_mb} --min-free-mb ${params.min_free_mb} \
      --sort-timeout ${params.max_wall_seconds}
    """
}

workflow {
    ['source_bin','contract','contract_sha256','dependency_manifest','max_memory_mb','min_free_mb',
     'sort_memory_mb','max_temp_mb','pair_block_size','max_wall_seconds','chromosome_concurrency'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    if (!(params.contract_sha256.toString() ==~ /[0-9a-f]{64}/)) error 'Invalid contract hash'
    ['max_memory_mb','sort_memory_mb','max_temp_mb','pair_block_size','max_wall_seconds','chromosome_concurrency'].each { key ->
        if (!(params[key].toString() ==~ /[1-9][0-9]*/)) error "Invalid positive integer: ${key}"
    }
    if (!(params.min_free_mb.toString() ==~ /[0-9]+/)) error 'Invalid disk reserve'
    if (params.max_memory_mb.toInteger()>2048 || params.sort_memory_mb.toInteger()>256 ||
        params.max_temp_mb.toInteger()>2048 || params.pair_block_size.toInteger()>10000 ||
        params.max_wall_seconds.toInteger()>7200 || params.chromosome_concurrency.toInteger()>2)
        error 'Resource contract exceeds accepted bounds'
    def input = file(params.contract, checkIfExists:true)
    def specification = new groovy.json.JsonSlurper().parse(input.toFile())
    def names = new groovy.json.JsonSlurper().parse(file(params.dependency_manifest, checkIfExists:true).toFile())
    if (!(names instanceof List) || names.isEmpty() || names.toSet().size()!=names.size()) error 'Unique dependencies required'
    def dependencies = names.collect { safeFile(it) }
    def selected = params.selected_chromosomes == null ? specification.chromosomes : params.selected_chromosomes
    if (!(selected instanceof List) || selected.isEmpty() || selected.toSet().size()!=selected.size() ||
        !selected.every { specification.chromosomes.contains(it) }) error 'Invalid selected chromosome subset'
    def chromosomes = specification.annotations.findAll { selected.contains(it.chromosome) }.collect { row ->
        if (!(row.chromosome ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Invalid chromosome'
        def manifest=safeFile(row.manifest.path)
        tuple(row.chromosome, manifest, safeFile(row.contract.path),
              safeFile(manifest.parent.resolve('parents.tsv.gz').toString()),
              safeFile(manifest.parent.resolve('children.tsv.gz').toString()))
    }
    def code = file(params.source_bin, checkIfExists:true)
    R02_COMMUNITY_IBD_ANCESTRY_CHROMOSOME(code,input,dependencies,Channel.fromList(chromosomes))
    if (selected.toSet()==specification.chromosomes.toSet()) {
        R02_COMMUNITY_IBD_ANCESTRY_AGGREGATE(code,input,dependencies,
            R02_COMMUNITY_IBD_ANCESTRY_CHROMOSOME.out.partial.map { chr, data -> data }.collect())
    }
}
