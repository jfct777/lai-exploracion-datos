nextflow.enable.dsl=2
params.genetic_map = null
params.map_contract = null
params.max_map_knots = 1000000

// Bounded one-chromosome entry. Does not recompute study design or aggregate all
// chromosomes. A geometry-only preflight and scientific annotation are distinct
// processes with separate cache keys and completion markers.
process R02_SEGMENT_GEOMETRY {
    tag "chr${params.chromosome}_geometry_only"
    cache 'deep'
    cpus 1
    input:
    path source_bin
    path chains
    path configurations
    path rare_vcf
    path rare_index
    path samples
    output:
    path 'geometry', emit: geometry
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_segment_evidence.py' \
      --chains '${chains}' --configurations '${configurations}' --rare-vcf '${rare_vcf}' \
      --samples '${samples}' --chromosome '${params.chromosome}' \
      --expected-source-samples ${params.expected_source_samples} --genome-build '${params.genome_build}' \
      --block-bp ${params.block_bp} --chunk-sites ${params.chunk_sites} \
      --max-active-intervals ${params.max_active_intervals} \
      --preflight-only --max-preflight-memory-mb ${params.preflight_memory_mb} \
      --max-preflight-blocks ${params.max_preflight_blocks} \
      --min-free-disk-mb ${params.min_free_disk_mb} --resource-check-rows ${params.resource_check_rows} \
      --output-dir geometry
    """
}

process R02_SEGMENT_ANNOTATION {
    tag "chr${params.chromosome}_rare_interval_evidence"
    cache 'deep'
    cpus 1
    input:
    path source_bin
    path chains
    path configurations
    path rare_vcf
    path rare_index
    path samples
    path genetic_map
    path map_contract
    output:
    path 'segment_evidence', emit: evidence
    script:
    def mapArg = genetic_map ? "--genetic-map '${genetic_map}' --map-contract '${map_contract}' --max-map-knots ${params.max_map_knots ?: 1000000}" : ''
    """
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_segment_evidence.py' \
      --chains '${chains}' --configurations '${configurations}' --rare-vcf '${rare_vcf}' \
      --samples '${samples}' --chromosome '${params.chromosome}' \
      --expected-source-samples ${params.expected_source_samples} --genome-build '${params.genome_build}' \
      --block-bp ${params.block_bp} --chunk-sites ${params.chunk_sites} \
      --max-active-intervals ${params.max_active_intervals} --max-db-mb ${params.max_db_mb} \
      --min-free-disk-mb ${params.min_free_disk_mb} --resource-check-rows ${params.resource_check_rows} \
      ${mapArg} \
      --output-dir segment_evidence
    """
}

workflow {
    ['source_bin','chains','configurations','rare_vcf','rare_index','samples','chromosome',
     'expected_source_samples','genome_build','block_bp','chunk_sites','max_active_intervals',
     'min_free_gib','min_free_disk_mb','max_db_mb','preflight_memory_mb','max_preflight_blocks',
     'resource_check_rows','mode'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    if (!(params.mode in ['geometry','annotation'])) error 'Explicit mode geometry or annotation required'
    // Optional evidence is accepted only as a paired, authenticated file and
    // contract. Geometry preflight remains genotype/map-independent.
    ['genetic_map','map_contract'].each { key ->
        if (params[key] != null && params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Unsafe optional parameter: ${key}"
    }
    if (!!params.genetic_map != !!params.map_contract) error 'Map and contract must be supplied together'
    if (params.max_map_knots != null && !(params.max_map_knots.toString() ==~ /[1-9][0-9]*/))
        error 'Positive integer required: max_map_knots'
    if (!(params.chromosome.toString() ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Autosome 1-22 required'
    ['expected_source_samples','block_bp','chunk_sites','max_active_intervals','max_db_mb',
     'preflight_memory_mb','max_preflight_blocks','resource_check_rows'].each { key ->
        if (!(params[key].toString() ==~ /[1-9][0-9]*/)) error "Positive integer required: ${key}"
    }
    ['min_free_gib','min_free_disk_mb'].each { key ->
        if (!(params[key].toString() ==~ /[0-9]+(\.[0-9]+)?/)) error "Nonnegative limit required: ${key}"
    }
    def source = file(params.source_bin, checkIfExists:true)
    def chains = file(params.chains, checkIfExists:true)
    def configurations = file(params.configurations, checkIfExists:true)
    def rare = file(params.rare_vcf, checkIfExists:true)
    def index = file(params.rare_index, checkIfExists:true)
    def samples = file(params.samples, checkIfExists:true)
    if (!(index.name in [rare.name + '.tbi',rare.name + '.csi'])) error 'VCF/index basename mismatch'
    if (params.mode == 'geometry')
        R02_SEGMENT_GEOMETRY(source, chains, configurations, rare, index, samples)
    else
        R02_SEGMENT_ANNOTATION(source, chains, configurations, rare, index, samples,
            params.genetic_map ? file(params.genetic_map, checkIfExists:true) : [],
            params.map_contract ? file(params.map_contract, checkIfExists:true) : [])
}
