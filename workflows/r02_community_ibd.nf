nextflow.enable.dsl=2

process R02_COMMUNITY_IBD_CHROMOSOME {
    tag "chr${chrom}"
    cache 'deep'
    cpus 1
    maxForks params.chromosome_concurrency
    publishDir params.outdir, mode: 'copy', overwrite: false
    input:
    path source_bin
    path bundle
    val chrom
    output:
    tuple val(chrom), path("chr${chrom}"), emit: parts
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_community_ibd.py' chromosome \
      --contract '${bundle}/contract.json' --contract-sha256 '${params.contract_sha256}' \
      --chromosome '${chrom}' --output-dir 'chr${chrom}' --scratch-dir "\$PWD" \
      --max-memory-mb ${params.max_memory_mb} --max-db-mb ${params.max_db_mb} \
      --min-free-mb ${params.min_free_mb} --max-pair-records ${params.max_pair_records}
    """
}

process R02_COMMUNITY_IBD_AGGREGATE {
    tag 'all_saved_communities'
    cache 'deep'
    cpus 1
    publishDir params.outdir, mode: 'copy', overwrite: false
    input:
    path source_bin
    path bundle
    path parts
    output:
    path 'community_ibd', emit: evidence
    script:
    def manifests = parts.collect { "'${it}/manifest.json'" }.join(' ')
    """
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_community_ibd.py' aggregate \
      --contract '${bundle}/contract.json' --contract-sha256 '${params.contract_sha256}' \
      --output-dir community_ibd --scratch-dir "\$PWD" --parts ${manifests} \
      --max-memory-mb ${params.max_memory_mb} --max-db-mb ${params.max_db_mb} \
      --min-free-mb ${params.min_free_mb} --pair-block-size ${params.pair_block_size}
    """
}

workflow {
    ['source_bin','bundle','outdir','contract_sha256','max_memory_mb','max_db_mb',
     'min_free_mb','min_free_gib','max_pair_records','pair_block_size','chromosome_concurrency'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    ['max_memory_mb','max_db_mb','min_free_mb','max_pair_records','pair_block_size','chromosome_concurrency'].each { key ->
        if (!(params[key].toString() ==~ /[1-9][0-9]*/)) error "Invalid positive integer: ${key}"
    }
    if (!(params.min_free_gib.toString() ==~ /[0-9]+(\.[0-9]+)?/) ||
        params.min_free_mb.toBigDecimal() != params.min_free_gib.toBigDecimal()*1024) error 'Disk reserve units differ'
    if (!(params.contract_sha256.toString() ==~ /[0-9a-f]{64}/)) error 'Invalid contract SHA256'
    def bundle = file(params.bundle, checkIfExists:true)
    def contract = new groovy.json.JsonSlurper().parse(file("${params.bundle}/contract.json", checkIfExists:true).toFile())
    if (!(contract.chromosomes instanceof List) || contract.chromosomes.isEmpty() ||
        contract.chromosomes.toSet().size() != contract.chromosomes.size() ||
        contract.chromosomes.any { !(it.toString() ==~ /([1-9]|1[0-9]|2[0-2])/) }) error 'Invalid chromosome inventory'
    def code = file(params.source_bin, checkIfExists:true)
    R02_COMMUNITY_IBD_CHROMOSOME(code, bundle, Channel.fromList(contract.chromosomes))
    def completed = R02_COMMUNITY_IBD_CHROMOSOME.out.parts.map { chrom, path -> path }.collect()
    R02_COMMUNITY_IBD_AGGREGATE(code, bundle, completed)
}
