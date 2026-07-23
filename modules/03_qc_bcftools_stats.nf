nextflow.enable.dsl=2

process QC_BCFTOOLS_STATS {
    tag "chr${chr}"

    publishDir "${params.outdir}/03_bcftools_stats", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(counts_tsv), path(vcf_in), path(vcf_tbi), path(parse_bcftools_stats_py)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.bcftools.stats.parsed.tsv")
    path "dnabr.hg38.2723.chr${chr}.bcftools.stats.txt"

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def threads = params.resources?.qc_bcftools_stats?.threads ?: 2
    """
    set -euo pipefail

    bcftools stats --threads ${threads} -s - ${vcf_in} > ${sample_id}.bcftools.stats.txt

    python3 ${parse_bcftools_stats_py} --chr ${chr} --in ${sample_id}.bcftools.stats.txt --out ${sample_id}.bcftools.stats.parsed.tsv
    """
}
