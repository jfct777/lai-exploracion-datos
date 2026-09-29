nextflow.enable.dsl=2

process PRESENT_RARE_SEGMENT_CANDIDATE {
    cpus 1
    memory '4 GB'
    time '30m'
    publishDir "${params.m14_followup_results_dir}", mode: 'copy', overwrite: false
    input:
    tuple path(chains), path(configurations), path(kinship), path(samples), path(renderer)
    output:
    path 'presentation', emit: figures
    script:
    def flags = [length: '--length', gap: '--gap', min_shared: '--min-shared', detail_pairs: '--detail-pairs']
    def numeric = flags.collect { name, flag ->
        def value = params["m14_presentation_${name}"]
        if (!(value.toString() ==~ /[0-9]+/)) error "Invalid m14_presentation_${name}"
        "${flag} ${value}"
    }.join(' ')
    def prefix = params.m14_presentation_prefix.toString()
    def chrom = params.m14_followup_chromosome.toString()
    if (!(prefix ==~ /[A-Za-z0-9_-]+/) || !(chrom ==~ /[0-9XYMT]+/)) error 'Invalid presentation prefix/chromosome'
    """
    python3 '${renderer}' --chains '${chains}' --configuration-summary '${configurations}' \
      --kinship-summary '${kinship}' --sample-ids '${samples}' --output-dir presentation \
      --prefix '${prefix}' --chr '${chrom}' ${numeric} \
      --kinship-threshold '${params.m14_presentation_kinship_threshold}' --dpi '${params.m14_followup_dpi}'
    """
}
