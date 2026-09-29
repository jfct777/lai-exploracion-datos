nextflow.enable.dsl=2

def rareShellQuote(value) {
    return "'" + value.toString().replace("'", "'\"'\"'") + "'"
}

process LAI_RARE_BIALELIC_ONLY {
    tag "chr${chr}"

    publishDir "${params.outdir}/lai_rare", mode: 'copy', overwrite: false

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi)
    path keep_samples
    path select_rare_minor_py

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.rare.minor.vcf.gz"), path("dnabr.hg38.2723.chr${chr}.rare.minor.vcf.gz.tbi"), emit: rare_vcfs
    path "dnabr.hg38.2723.chr${chr}.rare.minor.counts.tsv", emit: rare_counts
    path "dnabr.hg38.2723.chr${chr}.rare.minor.contract.json", emit: rare_contract

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}.rare.minor"
    def threads = params.resources?.lai_rare_bialelic_only?.threads ?: 2

    def options = [
        'max-maf': params.lai_rare_max_maf == null ? 'none' : params.lai_rare_max_maf,
        'min-mac': params.lai_rare_min_mac,
        'keep-format': params.lai_rare_keep_format ?: '',
    ].collect { name, value -> "--${name} ${rareShellQuote(value)}" }
    if (keep_samples) options.add("--samples ${rareShellQuote(keep_samples)}")
    if (params.lai_rare_remove_info) options.add('--remove-info')
    """
    set -euo pipefail

    python3 ${rareShellQuote(select_rare_minor_py)} \
        --input ${rareShellQuote(vcf_gz)} --output ${sample_id}.vcf.gz \
        --report ${sample_id}.contract.json --counts ${sample_id}.counts.tsv \
        --chrom ${rareShellQuote(chr)} ${options.join(' ')}
    bcftools index --threads ${threads} -t ${sample_id}.vcf.gz
    """
}
