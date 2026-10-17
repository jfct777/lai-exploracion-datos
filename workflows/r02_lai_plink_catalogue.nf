nextflow.enable.dsl=2

process R02_LAI_PLINK_CATALOGUE {
    tag "chr${job.chromosome}_BIM_catalogue_NO_GT"
    cache 'deep'
    cpus 1
    input:
    path source_scripts
    tuple val(job), path(bim), path(fam), path(roles), path(crosswalk), path(rare_vcf), path(rare_contract), path(fasta), path(fai), path(reference_contract), path(contig_map), path(input_hashes)
    output:
    path 'catalogue', emit: evidence
    tuple val(job.chromosome), path('catalogue'), emit: per_chromosome
    script:
    """
    umask 077
    export PYTHONDONTWRITEBYTECODE=1 TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 r02_lai_plink_catalogue.py --bim '${bim}' --fam '${fam}' --roles '${roles}' \\
      --crosswalk '${crosswalk}' --rare-vcf '${rare_vcf}' --rare-contract '${rare_contract}' \\
      --fasta '${fasta}' --fai '${fai}' --reference-contract '${reference_contract}' \\
      --contig-map '${contig_map}' --input-hashes '${input_hashes}' --outdir catalogue \\
      --chromosome '${job.chromosome}' --reference-contig '${job.reference_contig}' \\
      --expected-source-samples ${job.expected_source_samples} --expected-rare-sites ${job.expected_rare_sites} \\
      --max-bim-bytes ${job.max_bim_bytes} --max-bim-records ${job.max_bim_records} \\
      --max-line-bytes ${job.max_line_bytes} --max-sqlite-bytes ${job.max_sqlite_bytes} \\
      --sqlite-cache-mib ${job.sqlite_cache_mib} --batch-rows ${job.batch_rows} \\
      --max-rows-per-position ${job.max_rows_per_position} --timeout-seconds ${job.timeout_seconds} \\
      --min-free-gib ${job.min_free_gib}
    """
}

workflow {
    def paths = ['bim','fam','roles','crosswalk','rare_vcf','rare_contract','fasta','fai','reference_contract','contig_map','input_hashes']
    def ints = ['expected_source_samples','expected_rare_sites','max_bim_bytes','max_bim_records','max_line_bytes',
                'max_sqlite_bytes','sqlite_cache_mib','batch_rows','max_rows_per_position','timeout_seconds']
    def names = ['r02_lai_plink_catalogue.py','r02_lai_allele_support.py','r02_lai_build_verification.py','r02_genomic_pair_evidence.py']
    if (!(params.source_scripts instanceof List) || params.source_scripts.size() != names.size()) error 'Explicit four-file source closure required'
    def scripts = params.source_scripts.collect { item ->
        if (item.toString().find(/[\n\r'"`$\\;|&<>]/)) error 'Unsafe source path'
        file(item, checkIfExists:true)
    }
    if (scripts.collect { it.name }.sort() != names.sort()) error 'Unexpected source closure'
    def keys = paths + ints + ['chromosome','reference_contig','min_free_gib']
    def entries = params.containsKey('jobs') ? params.jobs : [keys.collectEntries { key -> [(key):params[key]] }]
    if (!(entries instanceof List) || entries.isEmpty() || entries.size()>22) error 'One to 22 explicit chromosome jobs required'
    def seen = []
    def jobs = entries.collect { job ->
        if (!(job instanceof Map) || job.keySet().sort() != keys.sort()) error 'Every job must contain exactly the documented parameters'
        keys.each { key ->
            if (job[key] == null || job[key].toString().find(/[\n\r'"`$\\;|&<>]/)) error "Missing or unsafe parameter: ${key}"
        }
        ints.each { key -> if (!(job[key].toString() ==~ /[1-9][0-9]*/)) error "Positive integer required: ${key}" }
        if (!(job.chromosome.toString() ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Canonical autosome number 1-22 required'
        if (seen.contains(job.chromosome.toString())) error 'Duplicate chromosome job'
        seen << job.chromosome.toString()
        if (!(job.min_free_gib.toString() ==~ /[0-9]+(\.[0-9]+)?/)) error 'Invalid disk reserve'
        def files = paths.collect { key -> file(job[key],checkIfExists:true) }
        if ((scripts + files).collect { it.name }.unique().size() != scripts.size()+files.size()) error 'Duplicate staged input names'
        tuple(job,*files)
    }
    R02_LAI_PLINK_CATALOGUE(scripts,Channel.fromList(jobs))
}
