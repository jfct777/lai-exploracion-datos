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
        error 'Safe absolute path and SHA-256 are required for every K03 INNER_FIT input'
    def value = file(name, checkIfExists: true)
    if (!value.toFile().isFile() || fileHash(value) != pin)
        error 'A K03 INNER_FIT input is absent, not a regular file, or changed'
    value
}

def readJson(p) { new groovy.json.JsonSlurper().parse(p.toFile()) }

process K03_INNER_FIT_AFFINITIES {
    tag "${contextId}"
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    maxForks params.max_forks
    cache 'deep'
    container params.container
    publishDir params.output_dir, mode: 'copy', overwrite: false

    input:
    tuple val(contextId), path(sharedBundle, stageAs: 'shared'), path(plan, stageAs: 'plan.json')
    path code

    output:
    tuple val(contextId), path("inner_${contextId}")

    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_k03_inner_fit_affinities.py \
      --manifest shared/manifest.json \
      --roster shared/persons.tsv \
      --fold-plan plan.json \
      --output inner_${contextId}
    """

    stub:
    """
    mkdir inner_${contextId}
    touch inner_${contextId}/STUB_NOT_SCIENTIFIC_OUTPUT
    """
}

workflow {
    if (![params.batch, params.code_dir, params.output_dir].every { safeAbsolute(it) } ||
        !(params.container instanceof String) ||
        !(params.container ==~ /[^\s]+@sha256:[a-f0-9]{64}/) ||
        !(params.memory_gb?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.max_forks?.toString() ==~ /[1-9][0-9]*/) ||
        !(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/))
        error 'Pinned batch/container and explicit positive K03 INNER_FIT resources are required'

    def batchPath = pinnedFile(params.batch, params.batch_sha256)
    def batch = readJson(batchPath)
    if (batch.keySet() != (['schema', 'implementation_sha256', 'container', 'resources',
                            'shared_bundle', 'shared_manifest_sha256', 'contexts'] as Set) ||
        batch.schema != 'r02_k03_inner_fit_affinity_batch_v2' || !batch.contexts ||
        batch.contexts.collect { it.id }.toSet().size() != batch.contexts.size() ||
        !(batch.implementation_sha256 instanceof Map) || !batch.implementation_sha256 ||
        batch.container != params.container || !safeAbsolute(batch.shared_bundle) ||
        batch.resources != [cpus: 1, memory_gb: params.memory_gb.toString().toInteger(),
                            max_forks: params.max_forks.toString().toInteger(), task_time: params.task_time])
        error 'One shared ALT bundle, unique contexts, pinned container and exact resources are required'

    def code = batch.implementation_sha256.collect { name, pin ->
        if (!(name ==~ /r02_[a-z0-9_]+\.py/)) error 'Unsafe K03 producer dependency name'
        pinnedFile(params.code_dir + '/' + name, pin)
    }
    def shared = file(batch.shared_bundle, checkIfExists: true)
    if (!shared.toFile().isDirectory()) error 'Shared K03 ALT bundle must be a directory'
    def manifestPath = pinnedFile(batch.shared_bundle + '/manifest.json', batch.shared_manifest_sha256)
    def manifest = readJson(manifestPath)
    def fixedCatalogue = manifest.learning_scope == 'FIXED_SOURCE_MINOR_DOSAGE_NO_INNER_TRANSFORMATIONS' &&
        manifest.dosage_allele == 'SOURCE_MINOR' &&
        manifest.genotype_encoding == 'DIPLOID_SOURCE_MINOR_DOSAGE_UINT8_0_1_2_MISSING_255' &&
        manifest.candidate_ascertainment == [
            scope: 'FIXED_SOURCE_COHORT_2723_MAC_GE_2_MAF_LE_0_01',
            source_cohort_people: 2723, source_minor_redefinition: false,
            inner_fit_reselection: false, inner_project_used_for_catalog: false,
            protocol_admission: 'TECHNICAL_FIXED_CATALOGUE_MODE_SCIENTIFIC_ADMISSION_REQUIRED']
    def newCatalogue = manifest.learning_scope == 'RAW_GENOTYPES_NO_LEARNED_TRANSFORMATIONS' &&
        manifest.dosage_allele == 'ALT' &&
        manifest.genotype_encoding == 'DIPLOID_GT_CODE_V1_PRESERVES_PARTIAL_CALLED_ALLELES' &&
        manifest.candidate_ascertainment == [
            scope: 'INNER_FIT_ONLY_FROM_PRE_FREQUENCY_ORIGINALLY_BIALLELIC_PASS',
            min_mac_inclusive: 2, max_maf_inclusive: '0.01',
            frequency_denominator: 'CALLED_ALLELES_INCLUDING_KNOWN_ALLELE_OF_PARTIAL_DIPLOID_GT',
            inner_fit_reselection: true, inner_project_used_for_eligibility: false,
            select_evaluate_used_for_eligibility: false,
            protocol_admission: 'TECHNICAL_NEW_CATALOGUE_MODE_SCIENTIFIC_ADMISSION_REQUIRED']
    if (manifest.schema != 'r02_k03_outer_fit_alt_chunks_v1' ||
        manifest.consumer_implementation_sha256 != batch.implementation_sha256 ||
        manifest.sample_scope != 'OUTER_FIT_GENOTYPES_ONLY' ||
        manifest.outer_fit_learned_inputs != false || manifest.roster?.path != 'persons.tsv' ||
        fixedCatalogue == newCatalogue ||
        manifest.producer?.all_autosomes_complete != true || !manifest.chunks ||
        manifest.chunks.collect { it.path }.toSet().size() != manifest.chunks.size())
        error 'Shared K03 manifest must declare exactly one fixed or INNER_FIT-learned catalogue mode'
    error 'K03 human execution is not scientifically admitted for either catalogue question; technical manifests alone cannot launch it'
    def descriptors = [manifest.roster] + manifest.chunks
    descriptors.each { descriptor ->
        if (!(descriptor instanceof Map) || descriptor.keySet() != (['path', 'sha256'] as Set) ||
            !(descriptor.path ==~ /(?:persons\.tsv|chunks\/chr(?:0[1-9]|1[0-9]|2[0-2])\/[0-9]{6}\.npz)/))
            error 'Only exact shared roster and chromosome ALT chunks are admitted'
        pinnedFile(batch.shared_bundle + '/' + descriptor.path, descriptor.sha256)
    }

    long scheduledMemory = params.memory_gb.toString().toLong() * 1024L * 1024L * 1024L
    if (!(manifest.resources.max_memory_bytes instanceof Number) ||
        manifest.resources.max_memory_bytes > scheduledMemory)
        error 'Shared manifest memory bound exceeds scheduled process memory'
    def jobs = batch.contexts.collect { job ->
        if (job.keySet() != (['id', 'plan', 'plan_sha256'] as Set) ||
            !(job.id ==~ /[A-Za-z0-9_]+/) || !safeAbsolute(job.plan))
            error 'Safe K03 context identifier and pinned fold plan are required'
        tuple(job.id, shared, pinnedFile(job.plan, job.plan_sha256))
    }
    K03_INNER_FIT_AFFINITIES(Channel.fromList(jobs), code)
}
