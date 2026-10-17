nextflow.enable.dsl=2
params.contract=null
params.contract_sha256=null
params.runtime=null
params.sources=[]
params.outdir=null
process EXTRACT_EXISTING_EXTERNAL_IBD_PAIRS {
    cpus 1
    memory '2 GB'
    time '45m'
    cache 'deep'
    publishDir params.outdir, mode:'copy', overwrite:false
    input:
    path contract
    path runtime
    path sources, stageAs:'source??/*'
    val digest
    output:
    path 'extraction'
    script:
    """
    python3 '${runtime}/r02_lai_external_ibd_pairs.py' --contract '${contract}' --sha256 '${digest}' --outdir extraction
    """
}
workflow {
    EXTRACT_EXISTING_EXTERNAL_IBD_PAIRS(file(params.contract,checkIfExists:true),file(params.runtime,checkIfExists:true),params.sources.collect{file(it,checkIfExists:true)},params.contract_sha256)
}
