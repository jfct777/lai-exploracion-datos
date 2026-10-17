nextflow.enable.dsl=2

params.plan = null
params.plan_sha256 = null
params.source_bin = null
params.output_dir = null
params.task_time = '3h'
params.plink_memory_mib = 4096
params.plink_task_memory = '6 GB'

process HISTORICAL_PANEL_MISSING {
    cpus 1
    memory params.plink_task_memory
    time params.task_time
    publishDir "${params.output_dir}/missing", mode: 'copy', overwrite: false
    input:
    tuple path(bed, stageAs:'input.bed'), path(bim, stageAs:'input.bim'), path(fam, stageAs:'input.fam')
    output:
    path 'missing.smiss', emit: samples
    path 'missing.vmiss', emit: variants
    path 'missing.log'
    script:
    """
    umask 077
    plink2 --bfile input --missing scols=fid,nmiss,nobs,fmiss vcols=chrom,pos,nmiss,nobs,fmiss \\
      --threads 1 --memory '${params.plink_memory_mib}' require --out missing
    """
}

process HISTORICAL_PANEL_AUDIT {
    cpus 1
    memory '1 GB'
    time params.task_time
    publishDir "${params.output_dir}/audit", mode:'copy', overwrite:false
    input:
    path bim
    path fam
    path sizes
    path sample_report
    path variant_report
    val people
    val sites
    val index_bytes
    path sources
    output:
    path 'audit'
    script:
    """
    umask 077
    python3 -B r02_roh_panel_audit.py --bim '${bim}' --fam '${fam}' \\
      --smiss '${sample_report}' --vmiss '${variant_report}' --chromosome-sizes '${sizes}' \\
      --expected-people '${people}' --expected-sites '${sites}' --max-index-bytes '${index_bytes}' --output audit
    """
}

def shaFile(path) {
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    path.withInputStream { stream ->
        byte[] buffer = new byte[1048576]
        int length
        while ((length = stream.read(buffer)) > 0) digest.update(buffer, 0, length)
    }
    digest.digest().encodeHex().toString()
}

workflow {
    if (!params.plan || !params.source_bin || !params.output_dir) error('Explicit plan, code and output required')
    def planFile = file(params.plan, checkIfExists:true)
    if (shaFile(planFile) != params.plan_sha256) error('Plan hash differs')
    def plan = new groovy.json.JsonSlurper().parse(planFile.toFile())
    if (plan.schema != 'r02_historical_panel_audit_plan_v1' || plan.filters_applied != false) error('Invalid unfiltered panel scope')
    if (plan.people <= 0 || plan.sites <= 0 || plan.max_index_bytes <= 0) error('Positive dimensions required')
    if (plan.inputs.keySet() != ['bed','bim','fam','chromosome_sizes'] as Set) error('Unexpected input set')
    def inputs = [:]
    plan.inputs.each { name, descriptor ->
        def path = file(descriptor.path, checkIfExists:true)
        if (shaFile(path) != descriptor.sha256) error("Input hash differs: ${name}")
        inputs[name] = path
    }
    def required = ['r02_roh_panel_audit.py', 'r02_lai_support_review.py']
    if (plan.code_sha256.keySet() != required as Set) error('Unexpected code set')
    def code = required.collect { name ->
        def path = file("${params.source_bin}/${name}", checkIfExists:true)
        if (shaFile(path) != plan.code_sha256[name]) error("Code hash differs: ${name}")
        path
    }
    HISTORICAL_PANEL_MISSING(Channel.value(tuple(inputs.bed,inputs.bim,inputs.fam)))
    HISTORICAL_PANEL_AUDIT(inputs.bim,inputs.fam,inputs.chromosome_sizes,
        HISTORICAL_PANEL_MISSING.out.samples,HISTORICAL_PANEL_MISSING.out.variants,
        plan.people,plan.sites,plan.max_index_bytes,code)
}
