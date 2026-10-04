nextflow.enable.dsl=2

// Technical, immutable checkpoints only. A changed list of chromosomes does
// not invalidate completed tasks for unchanged per-chromosome inputs.
process R02_PREPARE_CHROMOSOME {
    tag "chr${chrom}"
    cache 'deep'
    cpus 1
    input:
    tuple val(chrom), path(bundle, stageAs: 'bundle'), val(manifest_sha), path(content, stageAs: 'bundle_content/*')
    path source, stageAs: 'source_bin'
    path source_files, stageAs: 'source_content/*'
    path samples, stageAs: 'samples.ids'
    output:
    tuple val(chrom), path('checkpoint'), emit: checkpoints
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    python3 '${source}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source}/r02_biological_checkpoint.py' prepare \
      --manifest-path '${bundle}/manifest.json' --expected-manifest-sha256 '${manifest_sha}' --expected-chromosome '${chrom}' \
      --sample-ids '${samples}' --expected-samples ${params.expected_samples} \
      --expected-configurations ${params.expected_configurations} \
      --scratch-dir "\$PWD" --max-database-mb ${params.max_database_mb} \
      --min-free-disk-mb ${params.min_free_disk_mb} --output-dir checkpoint
    """
}

workflow {
    def required = ['source_bin','sample_ids','expected_samples','expected_configurations',
        'prepare_contract','prepare_contract_sha256','chromosomes','max_database_mb','min_free_disk_mb','min_free_gib']
    required.each {
        if (params[it] == null || params[it].toString().find(/[\n\r'"`$\\;|&<>]/)) error "Missing or unsafe parameter: ${it}"
    }
    ['expected_samples','expected_configurations','max_database_mb'].each {
        if (!(params[it].toString() ==~ /[1-9][0-9]*/)) error "Invalid positive integer: ${it}"
    }
    ['min_free_disk_mb','min_free_gib'].each {
        if (!(params[it].toString() ==~ /[0-9]+(\.[0-9]+)?/)) error "Invalid reserve: ${it}"
    }
    def contractPath = file(params.prepare_contract, checkIfExists:true)
    def bytes = contractPath.bytes
    def digest = java.security.MessageDigest.getInstance('SHA-256').digest(bytes).encodeHex().toString()
    if (digest != params.prepare_contract_sha256) error 'Prepare contract hash mismatch'
    def contract = new groovy.json.JsonSlurper().parse(bytes)
    if (contract.schema != 'r02_biological_checkpoint_prepare_v1') error 'Unsupported prepare contract'
    def wanted = params.chromosomes.toString().split(',').toList()
    if (wanted.toSet().size()!=wanted.size() || wanted.any { !(it ==~ /([1-9]|1[0-9]|2[0-2])/) }) error 'Invalid chromosome scope'
    if (contract.segments.collect { it.chrom }.sort() != wanted.sort()) error 'Prepare chromosome scope mismatch'
    ['expected_samples','expected_configurations'].each {
        if (contract[it].toString()!=params[it].toString()) error "Prepare contract mismatch: ${it}"
    }
    def samples = file(params.sample_ids, checkIfExists:true)
    def sampleDigest = java.security.MessageDigest.getInstance('SHA-256').digest(samples.bytes).encodeHex().toString()
    if (sampleDigest != contract.sample_ids_sha256) error 'Prepare ordered sample hash mismatch'
    def source = file(params.source_bin,checkIfExists:true)
    def sourceFiles = []
    java.nio.file.Files.newDirectoryStream(source,'*.py').withCloseable { stream -> stream.each { sourceFiles.add(it) } }
    def rows = contract.segments.collect { entry ->
        if (!(entry.path instanceof String) || !entry.path.startsWith('/') || entry.path.tokenize('/').contains('..') || entry.path.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)) error 'Unsafe manifest path'
        if (!(entry.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid manifest hash'
        def manifest = file(entry.path,checkIfExists:true)
        if (manifest.name != 'manifest.json') error 'Manifest basename mismatch'
        def bundle = manifest.parent
        def names = ['manifest.json','segment_evidence.tsv.gz','segment_configuration_links.tsv.gz','configurations.tsv']
        if (bundle.resolve('ibd_pair_territory.tsv.gz').exists()) names.add('ibd_pair_territory.tsv.gz')
        tuple(entry.chrom,bundle,entry.sha256,names.collect { file(bundle.resolve(it),checkIfExists:true) })
    }
    R02_PREPARE_CHROMOSOME(Channel.fromList(rows),source,sourceFiles.sort(),samples)
}
