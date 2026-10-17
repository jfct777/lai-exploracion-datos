nextflow.enable.dsl=2

def safeAbsolute(s) {
    s instanceof String && s.startsWith('/') && !s.tokenize('/').contains('..') &&
        !s.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

def fileHash(p) {
    def md = java.security.MessageDigest.getInstance('SHA-256')
    p.toFile().withInputStream { stream ->
        new java.security.DigestInputStream(stream, md).transferTo(java.io.OutputStream.nullOutputStream())
    }
    md.digest().encodeHex().toString()
}

def pinnedFile(name, pin) {
    if (!safeAbsolute(name) || !(pin ==~ /[a-f0-9]{64}/))
        error 'Safe absolute path and SHA-256 required'
    def value = file(name, checkIfExists: true)
    if (!value.toFile().isFile() || fileHash(value) != pin)
        error 'Pinned K03 event-to-ALT input is absent or changed'
    value
}

process K03_EVENT_ALT_CHROMOSOME {
    tag "chr${chromosome}"
    cpus 1
    memory "${params.converter_memory_gb} GB"
    time params.converter_time
    maxForks params.converter_max_forks
    cache 'deep'
    container params.container

    input:
    val chromosome
    path sourceBundle, stageAs: 'source'
    path code

    output:
    path "chr${chromosome}", emit: chromosome_outputs

    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_k03_event_alt_chunks.py chromosome \
      --campaign source/campaign.json --chromosome ${chromosome} --output chr${chromosome}
    """

    stub:
    """
    mkdir chr${chromosome}
    touch chr${chromosome}/STUB_NOT_SCIENTIFIC_OUTPUT
    """
}

process K03_ASSEMBLE_SHARED_ALT {
    tag 'all-22-autosomes'
    cpus 1
    memory "${params.assembler_memory_gb} GB"
    time params.assembler_time
    cache 'deep'
    container params.container
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    path sourceBundle, stageAs: 'source'
    path code
    path chromosomeOutputs

    output:
    path 'shared_outer_fit_alt'

    script:
    def args = chromosomeOutputs.sort { a, b ->
        a.baseName.replace('chr','').toInteger() <=> b.baseName.replace('chr','').toInteger()
    }.collect { "--chromosome-output ${it}" }.join(' ')
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_k03_event_alt_chunks.py assemble \
      --campaign source/campaign.json ${args} --output shared_outer_fit_alt
    """

    stub:
    """
    mkdir shared_outer_fit_alt
    touch shared_outer_fit_alt/STUB_NOT_SCIENTIFIC_OUTPUT
    """
}

workflow {
    if (![params.source_bundle, params.code_dir, params.output_dir].every { safeAbsolute(it) } ||
        !(params.container instanceof String) ||
        !(params.container ==~ /[^\s]+@sha256:[a-f0-9]{64}/) ||
        !(params.converter_memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.assembler_memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.converter_max_forks?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.converter_time ==~ /[1-9][0-9]*[smh]/) ||
        !(params.assembler_time ==~ /[1-9][0-9]*[smh]/))
        error 'Pinned source/code/container and explicit positive ALT-converter resources are required'
    def campaign = pinnedFile(params.source_bundle + '/campaign.json', params.campaign_sha256)
    def cfg = new groovy.json.JsonSlurper().parse(campaign.toFile())
    if (cfg.schema != 'r02_k03_event_alt_campaign_v1' ||
        cfg.chromosomes.collect { it.chromosome } != (1..22).toList() ||
        !(cfg.consumer_implementation_sha256 instanceof Map) ||
        !cfg.consumer_implementation_sha256)
        error 'Exact ordered 22-autosome K03 ALT campaign required'
    def sourceBundle = file(params.source_bundle, checkIfExists: true)
    if (!sourceBundle.toFile().isDirectory()) error 'K03 source bundle must be a directory'
    if (!(cfg.converter_implementation_sha256 instanceof Map) ||
        cfg.converter_implementation_sha256['r02_k03_event_alt_chunks.py'] != params.converter_sha256)
        error 'Exact converter implementation map required'
    def implementation = cfg.consumer_implementation_sha256 + cfg.converter_implementation_sha256
    def code = implementation.collect { name, pin ->
        if (!(name ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe K03 code dependency name'
        pinnedFile(params.code_dir + '/' + name, pin)
    }
    def chromosomes = Channel.fromList((1..22).toList())
    K03_EVENT_ALT_CHROMOSOME(chromosomes, sourceBundle, code)
    K03_ASSEMBLE_SHARED_ALT(sourceBundle, code,
        K03_EVENT_ALT_CHROMOSOME.out.chromosome_outputs.collect())
}
