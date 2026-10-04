nextflow.enable.dsl=2

// Import-only global reducer. Checkpoints are never deleted by this consumer.
process R02_AGGREGATE_CHECKPOINTS {
    tag 'all_requested_chromosomes'
    cache 'deep'
    cpus 1
    input:
    path checkpoints, stageAs: 'checkpoint??'
    val hashes
    path checkpoint_files, stageAs: 'checkpoint_content/file????'
    path study, stageAs: 'study'
    path study_files, stageAs: 'study_content/*'
    path source, stageAs: 'source_bin'
    path source_files, stageAs: 'source_content/*'
    path samples, stageAs: 'samples.ids'
    path kinship, stageAs: 'kinship.tsv'
    output:
    path 'evaluation', emit: evaluation
    script:
    def args = checkpoints.withIndex().collect { p, i -> "--checkpoint-manifest '${p}/checkpoint.json' --expected-checkpoint-sha256 '${hashes[i]}'" }.join(' ')
    """
    export PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    python3 '${source}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source}/r02_biological_checkpoint.py' aggregate ${args} \
      --chromosomes '${params.chromosomes}' --sample-ids '${samples}' \
      --expected-samples ${params.expected_samples} --expected-configurations ${params.expected_configurations} \
      --pcrelate-file '${kinship}' --expected-pcrelate-sha256 '${params.kinship_sha256}' \
      --study-contract '${study}/study_contract.json' --thresholds '${params.kinship_thresholds}' \
      --edge-thresholds-bp '${params.edge_thresholds_bp}' --expected-genome-build '${params.genome_build}' \
      --expected-source-samples ${params.expected_source_samples} \
      --expected-source-cohort-sha256 '${params.source_cohort_sha256}' \
      --scratch-dir "\$PWD" --min-free-disk-mb ${params.min_free_disk_mb} --output-dir evaluation
    """
}

workflow {
    def required = ['source_bin','sample_ids','pcrelate_file','expected_samples','expected_source_samples',
        'expected_configurations','aggregate_contract','aggregate_contract_sha256','chromosomes',
        'min_free_disk_mb','min_free_gib','kinship_sha256','kinship_thresholds','edge_thresholds_bp',
        'genome_build','source_cohort_sha256']
    required.each {
        if (params[it] == null || params[it].toString().find(/[\n\r'"`$\\;|&<>]/)) error "Missing or unsafe parameter: ${it}"
    }
    ['expected_samples','expected_source_samples','expected_configurations'].each {
        if (!(params[it].toString() ==~ /[1-9][0-9]*/)) error "Invalid positive integer: ${it}"
    }
    ['min_free_disk_mb','min_free_gib'].each {
        if (!(params[it].toString() ==~ /[0-9]+(\.[0-9]+)?/)) error "Invalid reserve: ${it}"
    }
    def contractPath=file(params.aggregate_contract,checkIfExists:true)
    def bytes=contractPath.bytes
    def digest=java.security.MessageDigest.getInstance('SHA-256').digest(bytes).encodeHex().toString()
    if (digest!=params.aggregate_contract_sha256) error 'Aggregate contract hash mismatch'
    def contract=new groovy.json.JsonSlurper().parse(bytes)
    if (contract.schema!='r02_biological_checkpoint_aggregate_v1') error 'Unsupported aggregate contract'
    def wanted=params.chromosomes.toString().split(',').toList()
    if (wanted.toSet().size()!=wanted.size() || wanted.any { !(it ==~ /([1-9]|1[0-9]|2[0-2])/) }) error 'Invalid chromosome scope'
    if (contract.checkpoints.collect { it.chrom }.sort()!=wanted.sort()) error 'Aggregate chromosome scope mismatch'
    ['expected_samples','expected_source_samples','expected_configurations','genome_build','source_cohort_sha256','kinship_sha256','kinship_thresholds','edge_thresholds_bp'].each {
        if (contract[it].toString()!=params[it].toString()) error "Aggregate contract mismatch: ${it}"
    }
    def safeManifest = { entry, basename ->
        if (!(entry.path instanceof String) || !entry.path.startsWith('/') || entry.path.tokenize('/').contains('..') || entry.path.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)) error 'Unsafe checkpoint/study path'
        if (!(entry.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid input hash'
        def p=file(entry.path,checkIfExists:true)
        if (p.name!=basename || java.security.MessageDigest.getInstance('SHA-256').digest(p.bytes).encodeHex().toString()!=entry.sha256) error 'Input manifest hash/name mismatch'
        p
    }
    def ordered=contract.checkpoints.sort { a,b -> a.chrom.toInteger()<=>b.chrom.toInteger() }
    def checkpoints=ordered.collect { safeManifest(it,'checkpoint.json').parent }
    def content=checkpoints.collectMany { p ->
        def rec=new groovy.json.JsonSlurper().parse(p.resolve('checkpoint.json').toFile())
        (['checkpoint.json'] + rec.streams.values().toList()).collect { name ->
            if (name.contains('/') || name.contains('..')) error 'Invalid checkpoint filename'
            file(p.resolve(name),checkIfExists:true)
        }
    }
    def study=safeManifest(contract.study,'study_contract.json').parent
    def source=file(params.source_bin,checkIfExists:true)
    def sourceFiles=[]
    java.nio.file.Files.newDirectoryStream(source,'*.py').withCloseable { stream -> stream.each { sourceFiles.add(it) } }
    def samples=file(params.sample_ids,checkIfExists:true)
    if (java.security.MessageDigest.getInstance('SHA-256').digest(samples.bytes).encodeHex().toString()!=contract.sample_ids_sha256) error 'Aggregate ordered sample hash mismatch'
    R02_AGGREGATE_CHECKPOINTS(checkpoints,ordered.collect { it.sha256 },content,
        study,['study_contract.json','persons.private.tsv'].collect { file(study.resolve(it),checkIfExists:true) },
        source,sourceFiles.sort(),samples,file(params.pcrelate_file,checkIfExists:true))
}
