nextflow.enable.dsl=2

include { R02_BIOLOGICAL_EVALUATION } from '../modules/r02_biological_evaluation'

// Downstream-only workflow. Does not run preprocessing, communities or training.
// Each task is independently cached. This legacy entry stages source/bundle
// directories: cache 'deep' alone does NOT certify changes inside directories.
// The import-only entry declares individual content inputs and tests byte-level
// invalidation explicitly. Do not mutate completed bundles or frozen sources.
process R02_STUDY_DESIGN {
    tag 'cohort_metadata_dependence'
    cache 'deep'
    cpus 1
    input:
    path source_bin
    path samples
    path metadata
    path kinship
    path roles
    output:
    path 'study', emit: contract
    script:
    def roleArg = roles ? "--roles '${roles}'" : ''
    def metadataArgs = (params.metadata_columns ?: []).collect { "--metadata-column '${it}'" }.join(' ')
    """
    export PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_study_design.py' \
      --sample-ids '${samples}' --expected-samples ${params.expected_samples} \
      --metadata '${metadata}' --expected-metadata-sha256 '${params.metadata_sha256}' \
      --id-column '${params.metadata_id_column}' \
      --pcrelate-file '${kinship}' --expected-pcrelate-sha256 '${params.kinship_sha256}' \
      --phi ${params.kinship_thresholds.tokenize(',').join(' ')} ${roleArg} ${metadataArgs} \
      --max-matrix-mb ${params.max_matrix_mb} --output-dir study
    """
}

process R02_SEGMENT_EVIDENCE {
    tag "chr${chrom}"
    cache 'deep'
    cpus 1
    input:
    tuple val(chrom), path(chains), path(configurations), path(rare_vcf), path(rare_index), path(common_vcf), path(common_index), path(common_contract), path(genetic_map), path(map_contract), path(ibd_bundle)
    path source_bin
    path samples
    output:
    tuple val(chrom), path("segments_chr${chrom}"), emit: evidence
    script:
    def commonArg = common_vcf ? "--common-vcf '${common_vcf}' --common-contract '${common_contract}'" : ''
    def mapArg = genetic_map ? "--genetic-map '${genetic_map}' --map-contract '${map_contract}'" : ''
    def ibdArg = ibd_bundle ? "--ibd-contract '${ibd_bundle}/ibd_contract.json'" : ''
    """
    export PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_segment_evidence.py' \
      --chains '${chains}' --configurations '${configurations}' \
      --rare-vcf '${rare_vcf}' --samples '${samples}' --chromosome '${chrom}' \
      --expected-source-samples ${params.expected_source_samples} --genome-build '${params.genome_build}' \
      --block-bp ${params.block_bp} --chunk-sites ${params.chunk_sites} \
      --max-active-intervals ${params.max_active_intervals} \
      ${commonArg} ${mapArg} ${ibdArg} --output-dir segments_chr${chrom}
    """
}

workflow {
    def required = ['source_bin', 'sample_ids', 'metadata', 'metadata_sha256', 'metadata_id_column',
                    'pcrelate_file', 'kinship_sha256', 'chromosome_inputs', 'chromosomes',
                    'expected_samples', 'expected_source_samples', 'expected_configurations', 'genome_build',
                    'block_bp', 'chunk_sites', 'max_active_intervals', 'min_free_gib', 'max_matrix_mb',
                    'max_database_mb', 'min_free_disk_mb', 'kinship_thresholds', 'edge_thresholds_bp']
    required.each { if (params[it] == null) error "Missing explicit parameter: ${it}" }
    def wanted = params.chromosomes.toString().tokenize(',')
    if (wanted.isEmpty() || wanted.toSet().size() != wanted.size() || wanted.any { !(it ==~ /([1-9]|1[0-9]|2[0-2])/) }) error 'Invalid chromosome list'
    // No arbitrary shell expressions accepted in command fields or paths.
    (required + ['roles']).each { if (params[it] != null && params[it].toString().find(/[\n\r'"`$\\;|&<>]/)) error "Unsafe command parameter: ${it}" }
    ['expected_samples','expected_source_samples','expected_configurations','block_bp','chunk_sites',
     'max_active_intervals','max_matrix_mb','max_database_mb'].each {
        if (!(params[it].toString() ==~ /[1-9][0-9]*/)) error "Expected positive integer: ${it}"
    }
    ['min_free_gib','min_free_disk_mb'].each {
        if (!(params[it].toString() ==~ /[0-9]+(\.[0-9]+)?/)) error "Expected nonnegative resource limit: ${it}"
    }
    if (!(params.kinship_thresholds.toString() ==~ /0\.[0-9]+(,0\.[0-9]+)*/)) error 'Invalid kinship threshold list'
    if (!(params.edge_thresholds_bp.toString() ==~ /[0-9]+(,[0-9]+)*/)) error 'Invalid edge threshold list'
    ['metadata_sha256','kinship_sha256'].each { if (!(params[it].toString() ==~ /[0-9a-f]{64}/)) error "Invalid hash: ${it}" }
    if (params.metadata_columns != null && (!(params.metadata_columns instanceof List) || params.metadata_columns.any { !(it.toString() ==~ /[a-zA-Z0-9_.-]+/) })) error 'metadata_columns must be a list of column names'
    def source = file(params.source_bin, checkIfExists: true)
    def samples = file(params.sample_ids, checkIfExists: true)
    def kinship = file(params.pcrelate_file, checkIfExists: true)
    def roles = params.roles ? file(params.roles, checkIfExists: true) : []
    R02_STUDY_DESIGN(source, samples, file(params.metadata, checkIfExists: true), kinship, roles)

    def rows = new groovy.json.JsonSlurper().parse(file(params.chromosome_inputs, checkIfExists: true).toFile())
    if (!(rows instanceof List) || rows.collect { it.chrom.toString() }.sort() != wanted.sort()) error 'Input chromosomes differ from explicit scope or are duplicated'
    def chromosomeChannel = Channel.fromList(rows).map { row ->
        def paths = [:]
        ['chains','configurations','rare_vcf','rare_index','common_vcf','common_index','common_contract','genetic_map','map_contract','ibd_bundle'].each { name ->
            def value = row[name]
            if (value && value.toString().find(/[\n\r'"`$\\;|&<>]/)) error "Unsafe input path ${name}"
            paths[name] = value ? file(value.toString(), checkIfExists: true) : []
        }
        ['chains','configurations','rare_vcf','rare_index'].each { if (!paths[it]) error "Missing ${it}" }
        if (!([paths.common_vcf,paths.common_index,paths.common_contract].count { !!it } in [0,3])) error 'Common VCF, index and contract must be supplied together'
        if (!([paths.genetic_map,paths.map_contract].count { !!it } in [0,2])) error 'Map and contract must be supplied together'
        if (paths.rare_index.name != paths.rare_vcf.name + '.tbi' && paths.rare_index.name != paths.rare_vcf.name + '.csi') error 'Rare index must be named for its VCF'
        if (paths.common_vcf && paths.common_index.name != paths.common_vcf.name + '.tbi' && paths.common_index.name != paths.common_vcf.name + '.csi') error 'Common index must be named for its VCF/BCF'
        if (paths.ibd_bundle) {
            def ibd = new groovy.json.JsonSlurper().parse(paths.ibd_bundle.resolve('ibd_contract.json').toFile())
            ['intervals','callability'].each { name ->
                def relative = ibd[name]?.path
                if (!relative || relative.toString().startsWith('/') || relative.toString().tokenize('/').contains('..')) error 'IBD table paths must remain within the staged bundle'
                file(paths.ibd_bundle.resolve(relative.toString()), checkIfExists: true)
            }
        }
        tuple(row.chrom.toString(), paths.chains, paths.configurations, paths.rare_vcf, paths.rare_index,
              paths.common_vcf, paths.common_index, paths.common_contract, paths.genetic_map, paths.map_contract, paths.ibd_bundle)
    }
    R02_SEGMENT_EVIDENCE(chromosomeChannel, source, samples)
    def bundles = R02_SEGMENT_EVIDENCE.out.evidence.map { chrom, path -> path }.toSortedList { a, b -> a.name <=> b.name }
    R02_BIOLOGICAL_EVALUATION(bundles, R02_STUDY_DESIGN.out.contract, source, samples, kinship, [], [:], [])
}
