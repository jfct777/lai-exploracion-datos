nextflow.enable.dsl=2

// A supplementary view; does not alter M14.2 checkpoints or infer new IBD.
process R02_CHECKPOINT_REPORTED_IBD {
    tag "chr${chrom}"
    cache 'deep'
    cpus 1
    publishDir "${params.output_dir}/chr${chrom}", mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(contract), val(contract_sha), path(source_inputs, stageAs:'authenticated/*')
    path source_scripts, stageAs:'source_bin/*'
    output:
    tuple val(chrom), path('reported_geometry'), emit: geometry
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 source_bin/r02_checkpoint_ibd.py chromosome \
      --contract '${contract}' --contract-sha256 '${contract_sha}' \
      --scratch-dir "\$PWD" --output-dir reported_geometry \
      --min-free-mb ${params.min_free_mb} --max-db-mb ${params.max_db_mb} \
      --max-pair-intervals ${params.max_pair_intervals}
    """
}

process R02_REPORTED_IBD_AUTOSOMAL {
    cache 'deep'
    cpus 1
    publishDir "${params.output_dir}/autosomal", mode:'copy', overwrite:false
    input:
    path bundles, stageAs:'chromosomes??/*'
    path source_scripts, stageAs:'source_bin/*'
    output:
    path 'evaluation', emit: evaluation
    script:
    // Nextflow supplies a Path, not a List, when the collection has one item.
    // Path.collect would iterate its components rather than the one directory.
    def directories = bundles instanceof List ? bundles : [bundles]
    def manifests = directories.collect { "'${it}/manifest.json'" }.join(' ')
    def chromosomes = params.jobs.collect { it.chromosome.toString() }.join(' ')
    def thresholds = params.thresholds_bp.join(' ')
    """
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 source_bin/r02_checkpoint_ibd.py collect --manifests ${manifests} \
      --chromosomes ${chromosomes} --thresholds-bp ${thresholds} \
      --output-dir evaluation --min-free-mb ${params.min_free_mb}
    """
}

workflow {
    def safePath = { v -> v instanceof String && v.startsWith('/') &&
        !v.tokenize('/').contains('..') && !v.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (!safePath(params.output_dir)) error 'Absolute safe output_dir required'
    ['max_db_mb','max_pair_intervals'].each { key ->
        if (!(params[key]?.toString() ==~ /[1-9][0-9]*/)) error 'Positive resource limit required'
    }
    if (!(params.min_free_mb?.toString() ==~ /[0-9]+/)) error 'Nonnegative disk reserve required'
    if (!(params.thresholds_bp instanceof List) || params.thresholds_bp.isEmpty() ||
        params.thresholds_bp.any { !(it.toString() ==~ /[0-9]+/) } ||
        !params.thresholds_bp.contains(0) || params.thresholds_bp.unique(false).size()!=params.thresholds_bp.size())
        error 'Distinct nonnegative thresholds including zero required'
    def names=['r02_checkpoint_ibd.py','r02_segment_evidence.py','r02_genomic_pair_evidence.py']
    if (!(params.source_scripts instanceof List) || params.source_scripts.size()!=names.size() ||
        params.source_scripts.any { !safePath(it) }) error 'Three safe source scripts required'
    def scripts=params.source_scripts.collect { file(it,checkIfExists:true) }
    if (scripts.collect { it.name }.sort()!=names.sort()) error 'Source dependency closure mismatch'
    if (!(params.jobs instanceof List) || params.jobs.isEmpty()) error 'Explicit chromosome jobs required'
    def seen=[]
    def jobs=params.jobs.collect { j ->
        def c=j.chromosome.toString()
        if (!(c ==~ /([1-9]|1[0-9]|2[0-2])/) || seen.contains(c)) error 'Invalid/duplicate chromosome'
        seen.add(c)
        if (!safePath(j.contract) || !(j.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid path/hash'
        def f=file(j.contract,checkIfExists:true)
        if (java.security.MessageDigest.getInstance('SHA-256').digest(f.bytes).encodeHex().toString()!=j.contract_sha256)
            error 'Input contract SHA256 mismatch'
        def record=new groovy.json.JsonSlurper().parse(f.toFile())
        if (record.schema!='r02_checkpoint_ibd_contract_v1' || record.chromosome.toString()!=c)
            error 'Contract schema/chromosome mismatch'
        // Each absolute input is hash-pinned by the contract; the launcher must
        // mount these explicit source directories read-only in the container.
        ['samples','checkpoint','ibd_contract'].each { k ->
            if (!safePath(record[k]?.path)) error 'Absolute authenticated input required'
            file(record[k].path,checkIfExists:true)
        }
        def cp=file(record.checkpoint.path,checkIfExists:true)
        def ibd=file(record.ibd_contract.path,checkIfExists:true)
        def cpRecord=new groovy.json.JsonSlurper().parse(cp.toFile())
        def ibdRecord=new groovy.json.JsonSlurper().parse(ibd.toFile())
        def stream=cp.parent.resolve(cpRecord.streams.segments).toAbsolutePath().toString()
        def intervals=ibd.parent.resolve(ibdRecord.intervals.path).toAbsolutePath().toString()
        def refs=[record.samples,record.checkpoint,record.ibd_contract,
            [path:stream,sha256:cpRecord.outputs[cpRecord.streams.segments].sha256],
            [path:intervals,sha256:ibdRecord.intervals.sha256]]
        def inputs=refs.collect { ref ->
            if (!safePath(ref.path) || !(ref.sha256 ==~ /[0-9a-f]{64}/)) error 'Invalid source dependency'
            def p=file(ref.path,checkIfExists:true)
            def digest=java.security.MessageDigest.getInstance('SHA-256')
            p.toFile().withInputStream { s ->
                byte[] buffer=new byte[1048576];int count
                while ((count=s.read(buffer))>0) digest.update(buffer,0,count)
            }
            if (digest.digest().encodeHex().toString()!=ref.sha256) error 'Source dependency SHA256 mismatch'
            p
        }
        tuple(c,f,j.contract_sha256,inputs)
    }
    R02_CHECKPOINT_REPORTED_IBD(Channel.fromList(jobs),scripts)
    R02_REPORTED_IBD_AUTOSOMAL(R02_CHECKPOINT_REPORTED_IBD.out.geometry.map { c,p -> p }.collect(),scripts)
}
