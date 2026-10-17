nextflow.enable.dsl=2

// Metadata admission only. No fit, projection, clustering or evaluation entry.
process R02_STRUCTURE_DISCRETE_ADMISSION {
    tag 'K03_metadata_not_training'
    cpus 1
    memory "${params.memory_gb} GB"
    time params.task_time
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path bundle, stageAs:'bundle'
    path code, stageAs:'code'
    path admissionContract, stageAs:'admission_contract.json'
    val admissionHash
    output:
    path 'admission', emit: admitted
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 code/r02_structure_discrete.py --bundle bundle \
      --admission-contract admission_contract.json --admission-contract-sha256 '${admissionHash}' \
      --output-dir admission
    """
}

workflow {
    def safe = { p -> p instanceof String && p.startsWith('/') && !p.tokenize('/').contains('..') && !p.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/) }
    if (![params.bundle, params.code_dir, params.admission_contract, params.output_dir].every(safe)) error 'Absolute safe paths required'
    if (!(params.admission_contract_sha256 ==~ /[0-9a-f]{64}/)) error 'Admission SHA256 required'
    if (!(params.memory_gb?.toString() ==~ /[1-9][0-9]*/)) error 'Positive memory_gb required'
    if (!(params.task_time instanceof String) || !(params.task_time ==~ /[1-9][0-9]*[smh]/)) error 'Explicit bounded task_time required'
    R02_STRUCTURE_DISCRETE_ADMISSION(file(params.bundle,checkIfExists:true),
        file(params.code_dir,checkIfExists:true),file(params.admission_contract,checkIfExists:true),params.admission_contract_sha256)
}
