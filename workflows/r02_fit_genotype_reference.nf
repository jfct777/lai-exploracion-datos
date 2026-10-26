nextflow.enable.dsl = 2

params.batch = null
params.batch_sha256 = null
params.code_dir = null
params.image_id = null
params.restored_index = null
params.output_dir = null
params.reuse_summary_dir = null
params.max_forks = 2
params.max_seconds = 5400

process FIT_GENOTYPE_COUNTS {
    tag "chr${chromosome}_${genotype_channel}"
    cpus 1
    memory '1 GB'
    time '100m'
    maxForks params.max_forks
    errorStrategy 'ignore'
    cache 'deep'

    input:
    tuple val(chromosome), val(genotype_channel), val(store)
    path code
    path restored_index

    output:
    tuple val(chromosome), val(genotype_channel), path('summary.json')

    script:
    def reuseArg = chromosome == '22' && genotype_channel == 'rare' ? '--reuse-summary /reuse/summary.json' : ''
    def target = "${params.output_dir}/chr${chromosome}_${genotype_channel}"
    """
    mkdir -p '${target}'
    python3 -c "import shutil; assert shutil.disk_usage('${target}').free >= 5*1024**3, '5 GiB free reserve required'"
    docker run --rm --name 'r02-fit-ea-${chromosome}-${genotype_channel}' --pull never \\
      --label dnabr_campaign=r02-fit-genotype-reference-20261021ea \\
      --network none --cpus 1 --memory 1g --memory-swap 1g --pids-limit 64 \\
      --user 1017:1020 -e PYTHONDONTWRITEBYTECODE=1 \\
      -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \\
      -v '${params.code_dir}:/code:ro' -v '${store}:/store:ro' \\
      -v '${params.restored_index}:/restored-index.json:ro' \\
      -v '${params.reuse_summary_dir}:/reuse:ro' -v '${target}:/output:rw' \\
      -w /code -e PYTHONPATH=/code/bin '${params.image_id}' \\
      python bin/r02_fit_genotype_reference.py --store /store --restored-index /restored-index.json \\
      --chromosome '${chromosome}' --channel '${genotype_channel}' --output /output \\
      --max-seconds '${params.max_seconds}' --max-rss-mib 768 ${reuseArg}
    cp '${target}/summary.json' summary.json
    """
}

workflow {
    if (!params.batch || !params.batch_sha256 || !params.code_dir || !params.image_id ||
        !params.restored_index || !params.output_dir || !params.reuse_summary_dir)
        error 'All paths and identities are required'
    def source = file(params.batch, checkIfExists: true)
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    if (digest.digest(source.bytes).encodeHex().toString() != params.batch_sha256)
        error 'Batch identity differs'
    def batch = new groovy.json.JsonSlurper().parse(source.toFile())
    def required = (1..22).collectMany { c -> ['common','rare'].collect { k -> "${c}_${k}".toString() } }.sort()
    if (batch.schema != 'r02_fit_genotype_reference_batch_v1' ||
        batch.tasks.collect { "${it.chromosome}_${it.channel}".toString() }.sort() != required ||
        params.max_forks != 2 || params.max_seconds != 5400)
        error 'Exact 44 tasks and reviewed resource bounds required'
    def rows = batch.tasks.collect { tuple(it.chromosome, it.channel, it.store) }
    FIT_GENOTYPE_COUNTS(Channel.fromList(rows), file(params.code_dir, checkIfExists: true),
        file(params.restored_index, checkIfExists: true))
}
