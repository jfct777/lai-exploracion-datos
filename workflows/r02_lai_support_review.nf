nextflow.enable.dsl=2

process R02_REVIEW_LAI_SUPPORT {
    tag "chr${chrom}_${role}"
    cpus 1
    memory '1 GB'
    maxForks 1
    cache 'deep'
    publishDir "${params.output_dir}/chr${chrom}/${role}", mode:'copy', overwrite:false
    input:
    tuple val(chrom), val(role), path(bundle, stageAs:'input_bundle'), val(contract_hash)
    path code, stageAs:'review_code.py'
    output:
    tuple val(chrom), val(role), path('review'), emit: reviewed
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    test "\$(sha256sum input_bundle/contract.json | cut -d ' ' -f1)" = '${contract_hash}'
    python3 review_code.py --contract input_bundle/contract.json --bundle input_bundle \
      --outdir review --min-free-gib ${params.min_free_gib} --timeout-seconds ${params.timeout_seconds}
    """
}

workflow {
    def safe = { p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (!safe(params.output_dir) || !safe(params.code)) error 'Absolute safe paths required'
    if (!(params.min_free_gib?.toString() ==~ /[0-9]+/) || !(params.timeout_seconds?.toString() ==~ /[1-9][0-9]*/)) error 'Resource parameters invalid'
    if (!(params.jobs instanceof List) || params.jobs.isEmpty()) error 'Explicit jobs required'
    def seen=[]
    def jobs=params.jobs.collect { j ->
        if (j.keySet()!=['chromosome','role','bundle','contract_sha256'].toSet()) error 'Unexpected job fields'
        def chrom=j.chromosome.toString(), role=j.role
        if (!(chrom ==~ /([1-9]|1[0-9]|2[0-2])/) || !(role in ['REF_TRAIN','SOURCE_VALID'])) error 'Forbidden chromosome/role'
        if (seen.contains([chrom,role])) error 'Duplicate chromosome/role'
        seen.add([chrom,role])
        if (!safe(j.bundle) || !(j.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Unsafe input bundle/hash'
        def bundle=file(j.bundle,checkIfExists:true)
        def c=new groovy.json.JsonSlurper().parse(bundle.resolve('contract.json').toFile())
        if (c.schema!='r02_lai_support_review_input_v1' || c.chromosome!=chrom || c.role!=role) error 'Input contract differs from job'
        tuple(chrom,role,bundle,j.contract_sha256)
    }
    R02_REVIEW_LAI_SUPPORT(Channel.fromList(jobs),file(params.code,checkIfExists:true))
}
