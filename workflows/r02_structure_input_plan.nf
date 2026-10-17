nextflow.enable.dsl=2

process R02_STRUCTURE_INPUT_PLAN {
    tag 'cohort_support_and_exclusion'
    cpus 1
    time '10m'
    memory "${params.memory_gb} GB"
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path bundle, stageAs:'input_bundle'
    path code, stageAs:'r02_structure_input_plan.py'
    path roleCode, stageAs:'r02_structure_increment.py'
    val contractHash
    output:
    path 'preparation', emit: prepared
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    test "\$(sha256sum input_bundle/contract.json | cut -d ' ' -f1)" = '${contractHash}'
    python3 r02_structure_input_plan.py --bundle input_bundle \
      --contract input_bundle/contract.json --output-dir preparation
    """
}

workflow {
    def safe = { p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.bundle, params.code, params.role_code, params.output_dir].every(safe)) error 'Absolute safe paths required'
    if (!(params.contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Contract SHA256 required'
    if (!(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)) error 'Positive memory_gb required'
    R02_STRUCTURE_INPUT_PLAN(file(params.bundle,checkIfExists:true),file(params.code,checkIfExists:true),
      file(params.role_code,checkIfExists:true),params.contract_sha256)
}
