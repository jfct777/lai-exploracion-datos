nextflow.enable.dsl=2

process PREPROCESS_NORM_LEFTALIGN {
    tag "chr${chr}"

    publishDir "${params.outdir}/01_norm", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(ref_fasta)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.norm.vcf.gz"), path("dnabr.hg38.2723.chr${chr}.norm.vcf.gz.tbi")
    path "dnabr.hg38.2723.chr${chr}.norm.log"

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def threads = params.resources?.preprocess_norm_leftalign?.threads ?: 2
    """
    set -euo pipefail

    test -f ${ref_fasta}.fai || samtools faidx ${ref_fasta}

    bcftools norm -m -any -f ${ref_fasta} --threads ${threads} -Oz -o ${sample_id}.norm.vcf.gz ${vcf_gz} 2> ${sample_id}.norm.log

    bcftools index --threads ${threads} -t ${sample_id}.norm.vcf.gz
    """
}
