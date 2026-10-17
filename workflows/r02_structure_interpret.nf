nextflow.enable.dsl=2

def safePath(value) {
    value instanceof String && value.startsWith('/') && !value.tokenize('/').contains('..') &&
        !value.find(/[\n\r'"`$\\;|&<>*?\[\]{}]/)
}

process R02_STRUCTURE_INTERPRET {
    cpus 1
    memory "${params.memory_mb ?: 512} MB"
    time '20m'
    cache 'deep'
    publishDir params.output_dir, mode:'copy', overwrite:false
    input:
    path packageFile, stageAs:'package.json'
    path packagePublication, stageAs:'package.publication.json'
    path integrationReceipt, stageAs:'integration.receipt.json'
    path ibdReceipt, stageAs:'ibd.receipt.json'
    path integrationFiles, stageAs:'integration/*'
    path ibdFiles, stageAs:'ibd/*'
    path code, stageAs:'r02_structure_interpret.py'
    val packageHash
    val maxInputMiB
    val maxTableRows
    output:
    path 'interpretation', emit: report
    script:
    """
    umask 077
    python3 r02_structure_interpret.py \
      --package package.json --package-sha256 '${packageHash}' \
      --package-publication package.publication.json \
      --integration-receipt integration.receipt.json --ibd-receipt ibd.receipt.json \
      --integration-dir integration --ibd-dir ibd --output interpretation \
      --max-input-mib '${maxInputMiB}' --max-table-rows '${maxTableRows}'
    """
}

workflow {
    def required=[params.package,params.package_publication,params.integration_receipt,
                  params.ibd_receipt,params.code,params.output_dir]
    if (!required.every {safePath(it)}) error 'Absolute safe input/output paths required'
    if (!(params.package_sha256 ==~ /[a-f0-9]{64}/)) error 'Authenticated package SHA256 required'
    def integration=new groovy.json.JsonSlurper().parse(new File(params.integration_receipt))
    def ibd=new groovy.json.JsonSlurper().parse(new File(params.ibd_receipt))
    def resolveFiles={receipt ->
        if (!receipt.files || !receipt.files.every {safePath(it.path)}) error 'Explicit publication file paths required'
        receipt.files.collect {file(it.path,checkIfExists:true)}
    }
    def inputMiB=params.max_input_mib ?: 64
    def tableRows=params.max_table_rows ?: 50000
    if (!(inputMiB instanceof Number) || inputMiB<=0 || !(tableRows instanceof Number) || tableRows<=0) error 'Positive resource bounds required'
    R02_STRUCTURE_INTERPRET(file(params.package,checkIfExists:true),file(params.package_publication,checkIfExists:true),
        file(params.integration_receipt,checkIfExists:true),file(params.ibd_receipt,checkIfExists:true),
        resolveFiles(integration),resolveFiles(ibd),file(params.code,checkIfExists:true),params.package_sha256,inputMiB,tableRows)
}
