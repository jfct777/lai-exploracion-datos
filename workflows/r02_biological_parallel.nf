nextflow.enable.dsl=2

// The original checkpoint streams stay immutable. Only small receipts and
// disjoint configuration results are created by these processes.
def digest(path) {
    java.security.MessageDigest.getInstance('SHA-256').digest(path.bytes).encodeHex().toString()
}

process R02_VALIDATE_CHECKPOINT {
    tag "chr${chrom}"
    cache 'deep'
    cpus 1
    input:
    tuple val(chrom), path(checkpoint, stageAs:'checkpoint'), val(checkpoint_hash), path(content, stageAs:'checkpoint_content/*')
    path samples
    path source, stageAs:'source_bin'
    path source_files, stageAs:'source_content/*'
    output:
    tuple val(chrom), path('validation'), emit: validated
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    umask 077
    python3 '${source}/r02_biological_parallel.py' validate \
      --checkpoint-manifest '${checkpoint}/checkpoint.json' --expected-checkpoint-sha256 '${checkpoint_hash}' \
      --sample-ids '${samples}' --expected-samples ${params.expected_samples} \
      --expected-configurations ${params.expected_configurations} --expected-chromosome '${chrom}' \
      --min-free-disk-mb ${params.min_free_disk_mb} --output-dir validation
    """
}

process R02_SEAL_CHECKPOINTS {
    tag 'all_autosomes_before_any_threshold'
    cache 'deep'
    cpus 1
    input:
    path validations, stageAs:'validation??'
    val validation_hashes
    path contract
    path checkpoint_files, stageAs:'checkpoint_content/file????'
    path samples
    path kinship
    path study_files, stageAs:'study_content/*'
    path source, stageAs:'source_bin'
    path source_files, stageAs:'source_content/*'
    output:
    path 'sealed', emit: sealed
    script:
    def args = validations.withIndex().collect { p,i -> "--validation-manifest '${p}/validation.json' --expected-validation-sha256 '${validation_hashes[i]}'" }.join(' ')
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    umask 077
    python3 '${source}/r02_biological_parallel.py' seal ${args} \
      --aggregate-contract '${contract}' --expected-aggregate-contract-sha256 '${params.aggregate_contract_sha256}' \
      --sample-ids '${samples}' --pcrelate-file '${kinship}' --chromosomes '${params.chromosomes}' \
      --min-free-disk-mb ${params.min_free_disk_mb} --output-dir sealed
    """
}

process R02_REDUCE_CONFIGURATIONS {
    tag "gap_${gap}bp_all_autosomes"
    cache 'deep'
    cpus 1
    input:
    tuple val(gap), path(sealed), val(seal_hash)
    path checkpoint_files, stageAs:'checkpoint_content/file????'
    path samples
    path kinship
    path study_files, stageAs:'study_content/*'
    path source, stageAs:'source_bin'
    path source_files, stageAs:'source_content/*'
    output:
    tuple val(gap), path('reduction'), emit: reductions
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    umask 077
    python3 '${source}/r02_biological_parallel.py' reduce \
      --seal-manifest '${sealed}/seal.json' --expected-seal-sha256 '${seal_hash}' --gap-bp ${gap} \
      --min-free-disk-mb ${params.min_free_disk_mb} --output-dir reduction
    """
}

process R02_COLLECT_CONFIGURATIONS {
    tag 'complete_disjoint_configuration_set'
    cache 'deep'
    cpus 1
    input:
    path sealed
    val seal_hash
    path reductions, stageAs:'reduction??'
    val reduction_hashes
    path checkpoint_files, stageAs:'checkpoint_content/file????'
    path samples
    path kinship
    path study_files, stageAs:'study_content/*'
    path source, stageAs:'source_bin'
    path source_files, stageAs:'source_content/*'
    output:
    path 'evaluation', emit: evaluation
    script:
    def args = reductions.withIndex().collect { p,i -> "--reducer-manifest '${p}/manifest.json' --expected-reducer-sha256 '${reduction_hashes[i]}'" }.join(' ')
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    umask 077
    python3 '${source}/r02_biological_parallel.py' collect ${args} \
      --seal-manifest '${sealed}/seal.json' --expected-seal-sha256 '${seal_hash}' \
      --min-free-disk-mb ${params.min_free_disk_mb} --output-dir evaluation
    """
}

workflow {
    def required = ['source_bin','sample_ids','pcrelate_file','expected_samples','expected_source_samples',
        'expected_configurations','aggregate_contract','aggregate_contract_sha256','chromosomes',
        'min_free_disk_mb','kinship_sha256','kinship_thresholds','edge_thresholds_bp',
        'genome_build','source_cohort_sha256']
    required.each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/)) error "Missing or unsafe parameter: ${key}"
    }
    ['expected_samples','expected_source_samples','expected_configurations'].each { key ->
        if (!(params[key].toString() ==~ /[1-9][0-9]*/)) error "Invalid integer: ${key}"
    }
    if (!(params.min_free_disk_mb.toString() ==~ /[0-9]+/)) error 'Invalid integer MiB disk reserve'
    def cp = file(params.aggregate_contract, checkIfExists:true)
    if (digest(cp) != params.aggregate_contract_sha256) error 'Aggregate contract hash mismatch'
    def contract = new groovy.json.JsonSlurper().parse(cp.toFile())
    if (contract.schema != 'r02_biological_checkpoint_aggregate_v1') error 'Unsupported aggregate contract'
    def wanted = params.chromosomes.toString().split(',').toList()
    if (wanted.toSet().size()!=wanted.size() || wanted.any { !(it ==~ /([1-9]|1[0-9]|2[0-2])/) }) error 'Invalid chromosome scope'
    if (contract.checkpoints.collect { it.chrom }.sort()!=wanted.sort()) error 'Aggregate chromosome scope mismatch'
    ['expected_samples','expected_source_samples','expected_configurations','genome_build','source_cohort_sha256','kinship_sha256','kinship_thresholds','edge_thresholds_bp'].each { key ->
        if (contract[key].toString()!=params[key].toString()) error "Aggregate contract mismatch: ${key}"
    }
    def manifestPath = { entry, basename ->
        if (!(entry.path instanceof String) || !entry.path.startsWith('/') || entry.path.tokenize('/').contains('..') || entry.path.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)) error 'Unsafe manifest path'
        if (!(entry.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid manifest hash'
        def p = file(entry.path, checkIfExists:true)
        if (p.name!=basename || digest(p)!=entry.sha256) error 'Input manifest hash/name mismatch'
        p
    }
    def entries = contract.checkpoints.sort { a,b -> a.chrom.toInteger()<=>b.chrom.toInteger() }
    def tasks = entries.collect { entry ->
        def p = manifestPath(entry, 'checkpoint.json')
        def record = new groovy.json.JsonSlurper().parse(p.toFile())
        def content = (['checkpoint.json'] + record.streams.values().toList()).collect { name ->
            if (name.contains('/') || name.contains('..')) error 'Invalid checkpoint filename'
            file(p.parent.resolve(name), checkIfExists:true)
        }
        tuple(entry.chrom, p.parent, entry.sha256, content)
    }
    def allFiles = tasks.collectMany { it[3] }
    def first = new groovy.json.JsonSlurper().parse(manifestPath(entries[0], 'checkpoint.json').toFile())
    def gaps = first.registry.configs.values().collect { it.max_gap_bp }.unique().sort()
    def study = manifestPath(contract.study, 'study_contract.json').parent
    def studyFiles = ['study_contract.json','persons.private.tsv'].collect { file(study.resolve(it), checkIfExists:true) }
    def source = file(params.source_bin, checkIfExists:true)
    def sourceFiles = []
    java.nio.file.Files.newDirectoryStream(source, '*.py').withCloseable { stream -> stream.each { sourceFiles.add(it) } }
    def samples = file(params.sample_ids, checkIfExists:true)
    if (digest(samples)!=contract.sample_ids_sha256) error 'Aggregate ordered sample hash mismatch'
    def kinship = file(params.pcrelate_file, checkIfExists:true)
    R02_VALIDATE_CHECKPOINT(Channel.fromList(tasks), samples, source, sourceFiles.sort())
    def validated = R02_VALIDATE_CHECKPOINT.out.validated.collect(flat:false).map { rows ->
        def ordered = rows.sort { a,b -> a[0].toInteger()<=>b[0].toInteger() }
        tuple(ordered.collect { it[1] }, ordered.collect { digest(it[1].resolve('validation.json')) })
    }
    R02_SEAL_CHECKPOINTS(validated.map { it[0] }, validated.map { it[1] }, cp, allFiles,
        samples, kinship, studyFiles, source, sourceFiles.sort())
    def sealed = R02_SEAL_CHECKPOINTS.out.sealed.map { p -> tuple(p, digest(p.resolve('seal.json'))) }
    def partitions = sealed.flatMap { p,h -> gaps.collect { g -> tuple(g,p,h) } }
    R02_REDUCE_CONFIGURATIONS(partitions, allFiles, samples, kinship, studyFiles, source, sourceFiles.sort())
    def reduced = R02_REDUCE_CONFIGURATIONS.out.reductions.collect(flat:false).map { rows ->
        def ordered = rows.sort { a,b -> a[0]<=>b[0] }
        tuple(ordered.collect { it[1] }, ordered.collect { digest(it[1].resolve('manifest.json')) })
    }
    R02_COLLECT_CONFIGURATIONS(sealed.map { it[0] }, sealed.map { it[1] },
        reduced.map { it[0] }, reduced.map { it[1] }, allFiles, samples, kinship,
        studyFiles, source, sourceFiles.sort())
}
