nextflow.enable.dsl=2

process H_CHROMOSOME {
    tag "H:chr${chromosome}:74-geometries"
    cpus params.chromosome_cpus
    memory "${params.chromosome_memory_gb} GB"
    time params.chromosome_time
    maxForks params.max_chromosome_forks
    cache 'deep'
    publishDir "${params.output_dir}/chromosomes", mode:'copy', overwrite:false

    input:
    tuple val(chromosome), path(request, stageAs:'request.json'), val(requestPin), path(code)

    output:
    tuple val(chromosome), path("chrom_${chromosome}")

    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_h_affinities.py \
      --stage chromosome --request ${request} --request-sha256 '${requestPin}' \
      --chromosome '${chromosome}' --output-dir chrom_${chromosome}
    """
}

process H_REDUCE {
    tag 'H:88-views:compact-reduce'
    cpus params.reduce_cpus
    memory "${params.reduce_memory_gb} GB"
    time params.reduce_time
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false

    input:
    path request
    val requestPin
    path code
    path chromosomeProducts

    output:
    path 'h_views'

    script:
    def inputs = chromosomeProducts.collect { "--chromosome-dir '${it}'" }.join(' ')
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_structure_h_affinities.py \
      --stage reduce --request ${request} --request-sha256 '${requestPin}' \
      ${inputs} --output-dir h_views
    """
}

workflow {
    def safe = { value -> value instanceof String && value.startsWith('/') &&
        !value.tokenize('/').contains('..') && !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    def digest = { path ->
        def md = java.security.MessageDigest.getInstance('SHA-256')
        path.toFile().withInputStream { stream ->
            new java.security.DigestInputStream(stream, md).transferTo(java.io.OutputStream.nullOutputStream())
        }
        md.digest().encodeHex().toString()
    }
    if (![params.request, params.code_dir, params.output_dir].every(safe) ||
        !(params.request_sha256 ==~ /[0-9a-f]{64}/) ||
        !(params.chromosome_cpus?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.reduce_cpus?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.chromosome_memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.reduce_memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.max_chromosome_forks?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.chromosome_time instanceof String && params.chromosome_time ==~ /[1-9][0-9]*[smh]/) ||
        !(params.reduce_time instanceof String && params.reduce_time ==~ /[1-9][0-9]*[smh]/))
        error 'Explicit safe H paths, SHA256 and positive runtime limits required'

    def request = file(params.request, checkIfExists:true)
    if (digest.call(request) != params.request_sha256) error 'H request changed before cache lookup'
    def parsed = new groovy.json.JsonSlurper().parse(request.toFile())
    if (parsed.schema != 'r02_structure_h_affinity_request_v1' ||
        parsed.preadmission?.status != 'PRE_ADMITTED_FINITE_H_FAMILY' ||
        parsed.preadmission?.finite_hypothesis_count != parsed.geometries?.size() ||
        parsed.preadmission?.view_count != parsed.territories?.size() ||
        parsed.preadmission?.evaluate_opened != false)
        error 'Self-consistent pre-admitted finite H request required'
    if (parsed.evidence_scope == 'DNABR_R02' &&
        (parsed.geometries?.size() != 74 || parsed.territories?.size() != 88 ||
         parsed.sources?.keySet() != (1..22).collect{it.toString()}.toSet()))
        error 'Human R02 requires the exact 74-geometry/88-view/autosome H family'
    if (parsed.resources.max_memory_bytes > params.chromosome_memory_gb.toBigInteger()*1024L*1024L*1024L ||
        parsed.resources.max_memory_bytes > params.reduce_memory_gb.toBigInteger()*1024L*1024L*1024L)
        error 'H byte envelope exceeds allocated process memory'

    def codeNames = ['r02_structure_h_affinities.py','r02_structure_h_core.py',
                     'r02_m14_chains.py','rare_allele_sharing_painter.py']
    def code = codeNames.collect { name ->
        def path = file(params.code_dir + '/' + name, checkIfExists:true)
        if (digest.call(path) != parsed.implementation_sha256[name]) error "H code pin differs: ${name}"
        path
    }
    def allChromosomes = parsed.sources.keySet().sort { a,b -> a.toInteger() <=> b.toInteger() }
    def selected = params.chromosomes == 'ALL' ? allChromosomes :
        params.chromosomes.toString().split(',').collect{it.trim()}
    if (!selected || selected.toSet().size() != selected.size() ||
        !allChromosomes.containsAll(selected)) error 'Selected H chromosomes must be unique request members'
    def tasks = selected.collect { chrom -> tuple(chrom, request, params.request_sha256, code) }
    H_CHROMOSOME(Channel.fromList(tasks))
    if (selected.toSet() == allChromosomes.toSet()) {
        def products = H_CHROMOSOME.out.map { chrom, directory -> directory }.collect()
        H_REDUCE(request, params.request_sha256, code, products)
    }
}
