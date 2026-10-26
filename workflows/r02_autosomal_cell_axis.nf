nextflow.enable.dsl = 2

params.batch = null
params.batch_sha256 = null
params.implementation = null
params.grid_implementation = null
params.output_dir = null
params.max_forks = 2

process PREPARE_AUTOSOMAL_CELLS {
    tag "chr${chromosome}"
    cpus 1
    memory '512 MB'
    time '5m'
    maxForks params.max_forks
    cache 'deep'
    errorStrategy 'finish'
    publishDir { "${params.output_dir}/chr${chromosome}" }, mode: 'copy', overwrite: false

    input:
    tuple val(chromosome), path(request), path(genetic_map), path(reference_grid)
    path implementation
    path grid_implementation

    output:
    tuple val(chromosome), path('prepared/cells.tsv'), path('prepared/receipt.json')

    script:
    def referenceArg = reference_grid ? "--reference-grid '${reference_grid}'" : ''
    """
    python3 '${implementation}' --request '${request}' --map '${genetic_map}' \
      --grid-implementation '${grid_implementation}' --output-dir prepared ${referenceArg}
    """
}

workflow {
    if (!params.batch || !params.batch_sha256 || !params.implementation ||
        !params.grid_implementation || !params.output_dir)
        error 'batch, batch_sha256, implementation, grid_implementation, output_dir required'
    def batchFile = file(params.batch, checkIfExists: true)
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    if (digest.digest(batchFile.bytes).encodeHex().toString() != params.batch_sha256)
        error 'batch SHA-256 differs'
    def batch = new groovy.json.JsonSlurper().parse(batchFile.toFile())
    if (batch.schema != 'r02_all22_cell_axis_batch_v1' ||
        batch.scope != 'GEOMETRY_ONLY_NO_GENOTYPES_NO_TRAINING' ||
        batch.tasks.collect { it.chromosome }.sort() != (1..22).collect { it.toString() }.sort())
        error 'exact 22 distinct autosomes and geometry-only scope required'
    if (params.max_forks < 1 || params.max_forks > 2)
        error 'this measured preparation allows one or two simultaneous tasks'
    def rows = batch.tasks.collect { row ->
        tuple(row.chromosome, file(row.request, checkIfExists: true),
              file(row.genetic_map, checkIfExists: true),
              row.reference_grid ? file(row.reference_grid, checkIfExists: true) : [])
    }
    PREPARE_AUTOSOMAL_CELLS(Channel.fromList(rows),
        file(params.implementation, checkIfExists: true),
        file(params.grid_implementation, checkIfExists: true))
}
