nextflow.enable.dsl=2

process PREPROCESS_FILTER_SNV_BIALLELIC_PASS {
    tag "chr${chr}"

    publishDir "${params.outdir}/02_filter", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(norm_vcf), path(norm_tbi)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.counts.tsv"), path("dnabr.hg38.2723.chr${chr}.snv.bi.pass.vcf.gz"), path("dnabr.hg38.2723.chr${chr}.snv.bi.pass.vcf.gz.tbi")

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def threads = params.resources?.preprocess_filter_snv_biallelic_pass?.threads ?: 2
    def min_alleles = params.bcftools_min_alleles ?: 2
    def max_alleles = params.plink_max_alleles ?: 2
    def variant_types = params.plink_snps_only ? "-v snps" : ""
    def alleles_filter = "-m${min_alleles} -M${max_alleles}"
    def maf_filter = params.max_maf ? "-Q ${params.max_maf}:minor" : ""
    def pass_flag_cmd = params.keep_pass ? "bcftools view -f PASS --threads ${threads} -Oz -o ${sample_id}.snv.bi.pass.vcf.gz ${sample_id}.snv.bi.vcf.gz" : "cp ${sample_id}.snv.bi.vcf.gz ${sample_id}.snv.bi.pass.vcf.gz"
    """
    set -euo pipefail

    bcftools view ${alleles_filter} ${variant_types} ${maf_filter} --threads ${threads} -Oz -o ${sample_id}.snv.bi.vcf.gz ${norm_vcf}
    bcftools index --threads ${threads} -t ${sample_id}.snv.bi.vcf.gz

    ${pass_flag_cmd}

    bcftools index --threads ${threads} -t ${sample_id}.snv.bi.pass.vcf.gz

    raw_total=\$(bcftools index -n ${vcf_gz})
    norm_total=\$(bcftools index -n ${norm_vcf})
    snv_bi_total=\$(bcftools index -n ${sample_id}.snv.bi.vcf.gz)
    snv_bi_pass_total=\$(bcftools index -n ${sample_id}.snv.bi.pass.vcf.gz)

    printf "chr\tstep\tn_variants\n" > ${sample_id}.counts.tsv
    printf "%s\traw\t%s\n" "${chr}" "\$raw_total" >> ${sample_id}.counts.tsv
    printf "%s\tnorm\t%s\n" "${chr}" "\$norm_total" >> ${sample_id}.counts.tsv
    printf "%s\tsnv_bi\t%s\n" "${chr}" "\$snv_bi_total" >> ${sample_id}.counts.tsv
    printf "%s\tsnv_bi_pass\t%s\n" "${chr}" "\$snv_bi_pass_total" >> ${sample_id}.counts.tsv
    """
}
