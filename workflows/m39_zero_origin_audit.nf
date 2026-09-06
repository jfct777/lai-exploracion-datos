nextflow.enable.dsl = 2

process M39_ZERO_ORIGIN_AUDIT {
    container params.m39zero_image
    publishDir params.m39zero_output_dir, mode: 'copy', overwrite: false

    input:
    path input_bundle, stageAs: 'inputs'
    path manifest, stageAs: 'manifest.json'
    path audit_source, stageAs: 'm39_zero_origin_audit.py'

    output:
    path 'zero-origin.receipt.json', emit: receipt

    script:
    """
    python3 '${audit_source}' --manifest '${manifest}' \\
      --input-root '${input_bundle}' --chunk-people ${params.m39zero_chunk_people} \\
      --output zero-origin.receipt.json
    """
}

workflow {
    ['m39zero_inputs', 'm39zero_manifest', 'm39zero_output_dir', 'm39zero_image'].each { name ->
        if (!params[name]) error "--${name} is required"
    }
    if (!(params.m39zero_image.toString() ==~ /.+@sha256:[a-f0-9]{64}/))
        error 'Use a container pinned by SHA256 digest'
    if (!(params.m39zero_chunk_people instanceof Integer) || params.m39zero_chunk_people < 1)
        error 'Person chunk size must be a positive integer'
    if (!(params.m39zero_container_user.toString() ==~ /[0-9]+:[0-9]+/))
        error 'Container user must be an explicit uid:gid pair'
    if (file(params.m39zero_output_dir).exists() && !workflow.resume)
        error 'Use a fresh output directory'
    def bundle = file(params.m39zero_inputs, checkIfExists: true)
    if (!bundle.isDirectory()) error '--m39zero_inputs must be a directory'
    M39_ZERO_ORIGIN_AUDIT(
        channel.value(bundle),
        channel.value(file(params.m39zero_manifest, checkIfExists: true)),
        channel.value(file("${projectDir.resolve('..')}/bin/m39_zero_origin_audit.py", checkIfExists: true))
    )
}
