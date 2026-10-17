nextflow.enable.dsl=2

// Authenticated phase-candidate audit, not simulation or phase validation.
// Source REF audits are cached once per chromosome; roles remain disjoint.
process R02_PHASED_SOURCE_AUDIT {
    tag "chr${meta.chromosome}_source_REF"
    cache 'deep'
    cpus 1
    maxForks 2
    input:
    path code, stageAs: 'l1_code/*'
    tuple val(meta), path(job, stageAs:'job.json'), path(inputs, stageAs:'inputs/*')
    output:
    tuple val(meta.chromosome), path('source_audit/source_build_verification.json'), emit: proof
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD"
    python3 l1_code/r02_lai_phased_audit.py source --job '${job}' --input-dir inputs --outdir source_audit
    """
}

process R02_PHASED_ROLE_PREPARE {
    tag "chr${meta.chromosome}_${meta.role}_projection_REF"
    cache 'deep'
    cpus 1
    maxForks 2
    input:
    path code, stageAs: 'l1_code/*'
    tuple val(meta), path(job, stageAs:'job.json'), path(inputs, stageAs:'inputs/*')
    output:
    tuple val(meta), path('role_prepared'), emit: prepared
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD"
    python3 l1_code/r02_lai_phased_audit.py role --job '${job}' --input-dir inputs --outdir role_prepared
    """
}

process R02_PHASED_GENOTYPE_SUPPORT {
    tag "chr${meta.chromosome}_${meta.role}_support"
    cache 'deep'
    cpus 1
    maxForks 2
    input:
    path code, stageAs: 'l1_code/*'
    tuple val(meta), path(job, stageAs:'job.json'), path(inputs, stageAs:'inputs/*'),
          path(prepared, stageAs:'prepared'), path(source_proof, stageAs:'source_build_verification.json')
    output:
    tuple val(meta), path('phased_support'), emit: support
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD"
    python3 l1_code/r02_lai_phased_audit.py support --job '${job}' --input-dir inputs \\
      --prepared '${prepared}' --source-proof '${source_proof}' --outdir phased_support
    """
}

workflow {
    if (!params.jobs || !(params.jobs instanceof List)) error 'Explicit nonempty phased jobs required'
    ['source_bin','receipt_dir'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/)) error "Unsafe ${key}"
    }
    def keys = []
    def jobs = params.jobs.collect { item ->
        if (!(item.chromosome.toString() ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Autosome 1-22 required'
        if (!(item.role in ['REF_TRAIN','SOURCE_VALID'])) error 'Forbidden role; TEST is closed'
        def id = "chr${item.chromosome}_${item.role}"
        keys << id
        ([item.job] + item.inputs).each { value ->
            if (value == null || value.toString().find(/[\n\r'"`$\\;|&<>]/)) error 'Unsafe input path'
        }
        if (item.inputs.size() != 9 || item.inputs.collect { file(it).name }.unique().size() != 9)
            error 'Exactly nine distinct input basenames required'
        def contract = new groovy.json.JsonSlurper().parse(file(item.job,checkIfExists:true).toFile())
        if (contract.chromosome != item.chromosome.toString() || contract.panel_role != item.role)
            error 'Job contract chromosome/role differs from queue metadata'
        def meta = [id:id, chromosome:item.chromosome.toString(), role:item.role, job:item.job, inputs:item.inputs]
        tuple(meta, file(item.job,checkIfExists:true), item.inputs.collect { file(it,checkIfExists:true) })
    }
    if (keys.unique(false).size() != keys.size()) error 'Duplicate chromosome-role job'
    jobs.groupBy { it[0].chromosome }.each { chromosome, group ->
        if (group.collect { it[0].role }.sort() != ['REF_TRAIN','SOURCE_VALID']) error 'Both roles required per chromosome'
        if (group[0][0].inputs != group[1][0].inputs) error 'Both roles must use identical genomic inputs'
    }
    def sourceFiles = ['r02_lai_phased_audit.py','r02_lai_role_subset.py','r02_lai_allele_support.py',
        'r02_lai_genotype_support.py','r02_lai_build_verification.py','r02_genomic_pair_evidence.py',
        'preprocess_storage_guard.py'].collect { file("${params.source_bin}/${it}",checkIfExists:true) }
    R02_PHASED_SOURCE_AUDIT(sourceFiles, Channel.fromList(jobs.findAll { it[0].role == 'REF_TRAIN' }))
    R02_PHASED_ROLE_PREPARE(sourceFiles, Channel.fromList(jobs))
    def ready = R02_PHASED_ROLE_PREPARE.out.prepared
        .map { meta, prepared -> tuple(meta.chromosome,meta,prepared) }
        .combine(R02_PHASED_SOURCE_AUDIT.out.proof,by:0)
        .map { chromosome, meta, prepared, proof ->
            tuple(meta,file(meta.job,checkIfExists:true),meta.inputs.collect { file(it,checkIfExists:true) },prepared,proof)
        }
    R02_PHASED_GENOTYPE_SUPPORT(sourceFiles,ready)
    R02_PHASED_SOURCE_AUDIT.out.proof.map { chromosome, proof -> "${chromosome}\t${proof}\n" }
        .collectFile(name:'source_audits.tsv',storeDir:params.receipt_dir,sort:true)
    R02_PHASED_GENOTYPE_SUPPORT.out.support.map { meta, output -> "${meta.id}\t${meta.chromosome}\t${meta.role}\t${output}\n" }
        .collectFile(name:'genotype_tasks.tsv',storeDir:params.receipt_dir,sort:true)
}
