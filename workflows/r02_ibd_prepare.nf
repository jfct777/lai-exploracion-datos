nextflow.enable.dsl=2

// Input preparation only; no rare genotype scan or IBD/LAI inference.
process R02_PREPARE_REPORTED_IBD {
    tag "chr${chrom}"
    cache 'deep'
    cpus 1
    publishDir "${params.output_dir}/chr${chrom}", mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(bundle, stageAs:'input_bundle'), val(contract_hash)
    path source_scripts, stageAs:'source_bin/*'
    output:
    tuple val(chrom), path('prepared'), emit: prepared
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 source_bin/r02_ibd_prepare.py \
      --contract '${bundle}/contract.json' --contract-sha256 '${contract_hash}' \
      --scratch-dir "\$PWD" --max-db-mb ${params.max_db_mb} \
      --min-free-mb ${params.min_free_mb} --max-pair-records ${params.max_pair_records} \
      --output-dir prepared
    """
}

workflow {
    def safePath = { value ->
        value instanceof String && value.startsWith('/') && !value.tokenize('/').contains('..') &&
          !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
    }
    if (!safePath(params.output_dir)) error 'Absolute safe output_dir required'
    ['max_db_mb','max_pair_records'].each { key ->
        if (!(params[key]?.toString() ==~ /[1-9][0-9]*/)) error "Positive integer required: ${key}"
    }
    if (!(params.min_free_mb?.toString() ==~ /[0-9]+/)) error 'Nonnegative min_free_mb required'
    def names=['r02_ibd_prepare.py','r02_ibd_compatibility.py','r02_genomic_pair_evidence.py','r02_segment_evidence.py']
    if (!(params.source_scripts instanceof List) || params.source_scripts.size()!=names.size() ||
        params.source_scripts.any { !safePath(it) }) error 'Four source scripts required'
    def scripts=params.source_scripts.collect { file(it,checkIfExists:true) }
    if (scripts.collect { it.name }.sort()!=names.sort()) error 'Source dependency closure mismatch'
    if (!(params.jobs instanceof List) || params.jobs.isEmpty()) error 'Explicit nonempty chromosome jobs required'
    def seen=[]
    def jobs=params.jobs.collect { job ->
        if (job.keySet()!=['chromosome','bundle','contract_sha256'].toSet()) error 'Unexpected job fields'
        def chrom=job.chromosome.toString()
        if (!(chrom ==~ /([1-9]|1[0-9]|2[0-2])/) || seen.contains(chrom)) error 'Invalid or duplicate chromosome'
        seen.add(chrom)
        if (!safePath(job.bundle) || !(job.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Unsafe bundle/hash'
        def bundle=file(job.bundle,checkIfExists:true)
        def contractFile=bundle.resolve('contract.json')
        def bytes=contractFile.bytes
        if (java.security.MessageDigest.getInstance('SHA-256').digest(bytes).encodeHex().toString()!=job.contract_sha256)
            error 'Preparation contract hash mismatch'
        def contract=new groovy.json.JsonSlurper().parse(bytes)
        if (contract.schema!='r02_native_ibd_prepare_v1' || contract.chromosome.toString()!=chrom)
            error 'Preparation schema/chromosome mismatch'
        ['native','audit','crosswalk','samples','log'].each { key ->
            def name=contract[key]?.path
            if (!(name instanceof String) || !(name ==~ /[a-zA-Z0-9_.-]+/) || name.contains('..'))
                error 'Bundle input must be a safe relative basename'
            file(bundle.resolve(name),checkIfExists:true)
        }
        tuple(chrom,bundle,job.contract_sha256)
    }
    R02_PREPARE_REPORTED_IBD(Channel.fromList(jobs),scripts)
}
