nextflow.enable.dsl=2

process LAI_RARE_BIALELIC_ONLY {
    tag "chr${chr}"

    publishDir "${params.outdir}/lai_rare", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.rare.vcf.gz"), path("dnabr.hg38.2723.chr${chr}.rare.vcf.gz.tbi"), emit: rare_vcfs
    path "dnabr.hg38.2723.chr${chr}.rare.counts.tsv", emit: rare_counts

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def threads = params.resources?.lai_rare_bialelic_only?.threads ?: 2

    // Declarative flag-list for bcftools view. Order matters semantically:
    // samples first → MAF/MAC are recomputed sobre el subset retenido (no sobre el VCF original).
    // Manichaikul 2010 (KING) motiva el filtro de samples para separar demografía de pedigrés;
    // Anderson 2010 / Browning 2015 motivan MAC≥2 para excluir singletons (artefactos de calling).
    def view_flags = []
    if (params.lai_rare_keep_samples_file) {
        view_flags.add("-S ${file(params.lai_rare_keep_samples_file)} --force-samples")
    }
    if (params.lai_rare_max_maf != null) {
        view_flags.add("-Q ${params.lai_rare_max_maf}:minor")
    }
    if (params.lai_rare_min_mac != null && params.lai_rare_min_mac > 1) {
        view_flags.add("-c ${params.lai_rare_min_mac}:minor")
    }
    def view_flag_str = view_flags.join(' ')

    def remove_parts = []
    if (params.lai_rare_remove_info) remove_parts.add("INFO")
    if (params.lai_rare_keep_format) remove_parts.add("^FORMAT/${params.lai_rare_keep_format}")
    def remove_expr = remove_parts.join(',')
    def annotate_cmd = remove_expr ? "bcftools annotate -x '${remove_expr}' --threads ${threads} -Oz -o ${sample_id}.rare.vcf.gz ${sample_id}.rare.maf.vcf.gz && rm -f ${sample_id}.rare.maf.vcf.gz" : "mv ${sample_id}.rare.maf.vcf.gz ${sample_id}.rare.vcf.gz"
    """
    set -euo pipefail

    bcftools view ${view_flag_str} --threads ${threads} -Oz -o ${sample_id}.rare.maf.vcf.gz ${vcf_gz}

    ${annotate_cmd}

    bcftools index --threads ${threads} -t ${sample_id}.rare.vcf.gz

    input_total=\$(bcftools index -n ${vcf_gz})
    rare_total=\$(bcftools index -n ${sample_id}.rare.vcf.gz)

    printf "chr\\tstep\\tn_variants\\n" > ${sample_id}.rare.counts.tsv
    printf "%s\\tinput_filtered\\t%s\\n" "${chr}" "\$input_total" >> ${sample_id}.rare.counts.tsv
    printf "%s\\trare\\t%s\\n" "${chr}" "\$rare_total" >> ${sample_id}.rare.counts.tsv
    """
}
