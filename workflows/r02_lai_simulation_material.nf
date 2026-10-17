nextflow.enable.dsl=2

process R02_LAI_SIMULATION_MATERIAL {
    tag "chr${chrom}"
    cpus 1
    memory '1 GB'
    maxForks 1
    cache 'deep'
    publishDir "${params.output_dir}/chr${chrom}", mode:'copy', overwrite:false
    input:
    tuple val(chrom), path(bundle, stageAs:'input_bundle'), val(contract_hash)
    path code, stageAs:'r02_lai_simulation_material.py'
    path support_code, stageAs:'r02_lai_support_review.py'
    output:
    tuple val(chrom), path('material'), emit: material
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    test "\$(sha256sum input_bundle/contract.json | cut -d ' ' -f1)" = '${contract_hash}'
    python3 r02_lai_simulation_material.py --contract input_bundle/contract.json --bundle input_bundle \
      --outdir material --min-free-gib ${params.min_free_gib} --timeout-seconds ${params.timeout_seconds} \
      --max-output-bytes ${params.max_output_bytes}
    """
}

workflow {
    def safe = { p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.output_dir,params.code,params.support_code].every { safe(it) }) error 'Absolute safe paths required'
    if (!(params.min_free_gib?.toString() ==~ /[0-9]+/) ||
        !(params.timeout_seconds?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.max_output_bytes?.toString() ==~ /[1-9][0-9]*/)) error 'Resource parameters invalid'
    if (!(params.jobs instanceof List) || params.jobs.isEmpty()) error 'Explicit jobs required'
    def seen=[]
    def jobs=params.jobs.collect { j ->
        if (j.keySet()!=['chromosome','bundle','contract_sha256'].toSet()) error 'Unexpected job fields'
        def chrom=j.chromosome.toString()
        if (!(chrom ==~ /([1-9]|1[0-9]|2[0-2])/) || seen.contains(chrom)) error 'Invalid/duplicate chromosome'
        seen.add(chrom)
        if (!safe(j.bundle) || !(j.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Unsafe input bundle/hash'
        def bundle=file(j.bundle,checkIfExists:true)
        def c=new groovy.json.JsonSlurper().parse(bundle.resolve('contract.json').toFile())
        if (c.schema!='r02_lai_simulation_material_input_v1' || c.chromosome!=chrom ||
            c.roles.keySet()!=['REF_TRAIN','SOURCE_VALID'].toSet()) error 'Input contract differs from job'
        tuple(chrom,bundle,j.contract_sha256)
    }
    R02_LAI_SIMULATION_MATERIAL(Channel.fromList(jobs),file(params.code,checkIfExists:true),file(params.support_code,checkIfExists:true))
}
