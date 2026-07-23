nextflow.enable.dsl=2

process PREPROCESS_FILTER_SNV_BIALLELIC_PASS {
    tag "chr${chr}"

    publishDir "${params.outdir}/02_filter", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(norm_vcf), path(norm_tbi)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.counts.tsv"), path("dnabr.hg38.2723.chr${chr}.snv.bi.pass.vcf.gz"), path("dnabr.hg38.2723.chr${chr}.snv.bi.pass.vcf.gz.tbi")

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def threads = params.resources?.preprocess_filter_snv_biallelic_pass?.threads ?: 2
    def pass_flag_cmd = params.keep_pass ? "bcftools view -f PASS --threads ${threads} -Oz -o ${sample_id}.snv.bi.pass.vcf.gz ${sample_id}.snv.bi.vcf.gz" : "cp ${sample_id}.snv.bi.vcf.gz ${sample_id}.snv.bi.pass.vcf.gz"
    """
    set -euo pipefail

    bcftools view -m2 -M2 -v snps --threads ${threads} -Oz -o ${sample_id}.snv.bi.vcf.gz ${norm_vcf}

    ${pass_flag_cmd}

    bcftools index --threads ${threads} -t ${sample_id}.snv.bi.pass.vcf.gz

    raw_total=\$(bcftools view -H ${vcf_gz} | wc -l)
    norm_total=\$(bcftools view -H ${norm_vcf} | wc -l)
    snv_bi_total=\$(bcftools view -H ${sample_id}.snv.bi.vcf.gz | wc -l)
    snv_bi_pass_total=\$(bcftools view -H ${sample_id}.snv.bi.pass.vcf.gz | wc -l)

    printf "chr\tstep\tn_variants\n" > ${sample_id}.counts.tsv
    printf "%s\traw\t%s\n" "${chr}" "\$raw_total" >> ${sample_id}.counts.tsv
    printf "%s\tnorm\t%s\n" "${chr}" "\$norm_total" >> ${sample_id}.counts.tsv
    printf "%s\tsnv_bi\t%s\n" "${chr}" "\$snv_bi_total" >> ${sample_id}.counts.tsv
    printf "%s\tsnv_bi_pass\t%s\n" "${chr}" "\$snv_bi_pass_total" >> ${sample_id}.counts.tsv
    """
}
