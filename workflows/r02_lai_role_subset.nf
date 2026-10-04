nextflow.enable.dsl=2

process R02_LAI_ROLE_SUBSET {
    tag 'chr22_SOURCE_VALID_mechanical_subset'
    cache 'deep'
    cpus 1
    input:
    path subset_script
    path source_vcf
    path roles
    path ref_receipt, stageAs: 'reference_receipt.json'
    path contract, stageAs: 'subset_contract.json'
    output:
    path 'role_subset', emit: subset
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD"
    python3 '${subset_script}' --source-vcf '${source_vcf}' --roles '${roles}' \\
      --ref-receipt '${ref_receipt}' --contract '${contract}' --outdir role_subset \\
      --min-free-gib ${params.min_free_gib} --max-line-bytes ${params.max_line_bytes} \\
      --timeout-seconds ${params.timeout_seconds}
    """
}

workflow {
    ['subset_script','source_vcf','roles','ref_receipt','contract','min_free_gib','max_line_bytes','timeout_seconds'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/)) error "Missing or unsafe parameter: ${key}"
    }
    ['max_line_bytes','timeout_seconds'].each { key ->
        if (!(params[key].toString() ==~ /[1-9][0-9]*/)) error "Positive integer required: ${key}"
    }
    if (!(params.min_free_gib.toString() ==~ /[0-9]+(\.[0-9]+)?/)) error 'Invalid disk reserve'
    R02_LAI_ROLE_SUBSET(file(params.subset_script,checkIfExists:true),file(params.source_vcf,checkIfExists:true),
        file(params.roles,checkIfExists:true),file(params.ref_receipt,checkIfExists:true),file(params.contract,checkIfExists:true))
}
