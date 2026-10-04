nextflow.enable.dsl=2

include { R02_BIOLOGICAL_EVALUATION } from '../modules/r02_biological_evaluation'

// Import authenticated, previously completed M14.2 + R02_ESTUDIO bundles.
// The only analytical process in this DAG is R02_BIOLOGICAL_EVALUATION.
// Use an isolated -C resource configuration, a frozen source_bin and local disk.
// Publication is deliberately separate: completion is not biological validation.
workflow {
    def required = ['source_bin', 'sample_ids', 'pcrelate_file', 'kinship_sha256',
                    'import_contract', 'import_contract_sha256', 'chromosomes',
                    'expected_samples', 'expected_source_samples', 'expected_configurations',
                    'genome_build', 'min_free_gib', 'max_database_mb', 'min_free_disk_mb',
                    'kinship_thresholds', 'edge_thresholds_bp']
    required.each {
        if (params[it] == null) error "Missing explicit parameter: ${it}"
        if (params[it].toString().find(/[\n\r'"`$\\;|&<>]/)) error "Unsafe command parameter: ${it}"
    }
    ['expected_samples', 'expected_source_samples', 'expected_configurations', 'max_database_mb'].each {
        if (!(params[it].toString() ==~ /[1-9][0-9]*/)) error "Expected positive integer: ${it}"
    }
    ['min_free_gib', 'min_free_disk_mb'].each {
        if (!(params[it].toString() ==~ /[0-9]+(\.[0-9]+)?/)) error "Expected nonnegative resource limit: ${it}"
    }
    ['kinship_sha256', 'import_contract_sha256'].each {
        if (!(params[it].toString() ==~ /[0-9a-f]{64}/)) error "Invalid hash: ${it}"
    }
    ['source_bin', 'sample_ids', 'pcrelate_file', 'import_contract'].each {
        if (params[it].toString().find(/[*?\[\]{}]/)) error "Explicit input path required; no glob: ${it}"
    }
    if (!(params.kinship_thresholds.toString() ==~ /0\.[0-9]+(,0\.[0-9]+)*/)) error 'Invalid kinship threshold list'
    if (!(params.edge_thresholds_bp.toString() ==~ /[0-9]+(,[0-9]+)*/)) error 'Invalid edge threshold list'
    def wanted = params.chromosomes.toString().split(',', -1).toList()
    if (wanted.isEmpty() || wanted.toSet().size() != wanted.size() || wanted.any { !(it ==~ /([1-9]|1[0-9]|2[0-2])/) }) error 'Invalid chromosome list'

    def contractPath = file(params.import_contract, checkIfExists: true)
    def contractBytes = contractPath.bytes
    def digest = java.security.MessageDigest.getInstance('SHA-256').digest(contractBytes).encodeHex().toString()
    if (digest != params.import_contract_sha256) error 'Import contract SHA256 mismatch'
    def contract = new groovy.json.JsonSlurper().parse(contractBytes)
    if (!(contract instanceof Map) || contract.schema != 'r02_biological_import_v1') error 'Unsupported import contract'
    if (!(contract.chromosomes instanceof List) || contract.chromosomes.sort() != wanted.sort()) error 'Contract chromosome scope mismatch'
    if (!(contract.segments instanceof List) || contract.segments.any { !(it instanceof Map) } || contract.segments.collect { it.chrom }.sort() != wanted.sort()) error 'Segment chromosomes missing, duplicated or outside explicit scope'
    ['expected_samples', 'expected_source_samples', 'expected_configurations', 'genome_build', 'kinship_sha256'].each {
        if (contract[it]?.toString() != params[it].toString()) error "Import contract/parameter mismatch: ${it}"
    }
    def manifestFile = { entry, basename ->
        if (!(entry instanceof Map) || !(entry.path instanceof String) || !entry.path.startsWith('/') || entry.path.tokenize('/').contains('..') ||
            entry.path.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)) error 'Import manifest paths must be explicit safe absolute local paths'
        if (!(entry.sha256 instanceof String) || !(entry.sha256 ==~ /[0-9a-f]{64}/)) error 'Import manifest SHA256 missing or invalid'
        def path = file(entry.path, checkIfExists: true)
        if (path.name != basename || !path.isFile()) error "Expected manifest file named ${basename}"
        path
    }
    def study = manifestFile(contract.study, 'study_contract.json').parent
    def ordered = contract.segments.sort { a, b -> a.chrom.toInteger() <=> b.chrom.toInteger() }
    def bundles = ordered.collect { manifestFile(it, 'manifest.json').parent }
    if (bundles.toSet().size() != bundles.size()) error 'Duplicate segment directory'
    def options = [sha256: params.import_contract_sha256, chromosomes: params.chromosomes,
                   expected_source_samples: params.expected_source_samples, genome_build: params.genome_build]
    def source = file(params.source_bin, checkIfExists: true)
    def contentInputs = bundles.collectMany { bundle ->
        def names = ['manifest.json', 'segment_evidence.tsv.gz', 'segment_configuration_links.tsv.gz', 'configurations.tsv']
        if (bundle.resolve('ibd_pair_territory.tsv.gz').exists()) names.add('ibd_pair_territory.tsv.gz')
        names.collect { file(bundle.resolve(it), checkIfExists: true) }
    }
    ['study_contract.json', 'persons.private.tsv'].each { contentInputs.add(file(study.resolve(it), checkIfExists: true)) }
    // Python imports remain a frozen directory for execution, but every source
    // byte is also an explicit file dependency of the deep task cache.
    java.nio.file.Files.newDirectoryStream(source, '*.py').withCloseable { stream ->
        stream.each { contentInputs.add(it) }
    }
    contentInputs = contentInputs.sort { a, b -> a.toString() <=> b.toString() }
    R02_BIOLOGICAL_EVALUATION(bundles, study, source,
        file(params.sample_ids, checkIfExists: true), file(params.pcrelate_file, checkIfExists: true),
        contractPath, options, contentInputs)
}
