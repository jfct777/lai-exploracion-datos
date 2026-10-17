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
    // Recovery can reuse M01 without downloading RAW again. bcftools index -n
    // reads the authentic tabix metadata even when its companion VCF is absent.
    // No placeholder VCF is created and no normalized count substitutes RAW.
    def rawIndexOnly = params.preprocess_raw_index_only == true
    // Only recovery may provide the pre-normalization input count, authenticated
    // by its admission step. A legacy TBI can be usable but omit optional counts.
    def rawCount = params.preprocess_authenticated_raw_count
    if (rawCount != null && (!rawIndexOnly || !(rawCount instanceof Number) || rawCount <= 0 || rawCount != rawCount.toLong())) {
        throw new IllegalArgumentException('Authenticated RAW count requires positive integer index-only recovery')
    }
    def rawCountCommand = rawCount == null ? 'raw_total=$(bcftools index -n "$raw_source")' : "raw_total=${rawCount.toLong()}"
    def rawCountDescription = rawCount == null ? 'tabix index only' : 'authenticated M01 normalization input log; see admission/count receipt'
    if (rawIndexOnly && (!(vcf_gz instanceof Collection) || !vcf_gz.isEmpty() || !vcf_tbi.toString().endsWith('.tbi'))) {
        throw new IllegalArgumentException('Index-only recovery requires an empty RAW VCF input and an authentic .tbi input')
    }
    def rawSourceSetup = rawIndexOnly ? """
    raw_index_source=\$(readlink -f '${vcf_tbi}')
    raw_source=\${raw_index_source%.tbi}
    printf 'RAW count source: ${rawCountDescription} (%s); RAW VCF not supplied\\n' "\$raw_index_source" >&2
    """ : "raw_source=\$(readlink -f '${vcf_gz}')"
    def largeRoot = params.preprocess_large_temp_dir ?: ''
    if (largeRoot && (!largeRoot.startsWith('/') || largeRoot.contains('\n') || largeRoot.contains('\r'))) {
        throw new IllegalArgumentException('preprocess_large_temp_dir must be an absolute filesystem directory')
    }
    def quotedRoot = "'" + largeRoot.replace("'", "'\"'\"'") + "'"
    def bulkStart = largeRoot ? """
    local_task_dir=\$PWD
    ${rawSourceSetup}
    norm_source=\$(readlink -f '${norm_vcf}')
    test -d ${quotedRoot}
    bulk_dir=\$(mktemp -d ${quotedRoot}/m02_chr${chr}.XXXXXXXX)
    printf '%s\\n' "\$bulk_dir" > '${sample_id}.m02.large_temp_dir.txt'
    cd "\$bulk_dir"
    """ : """
    ${rawIndexOnly ? rawSourceSetup : "raw_source='${vcf_gz}'"}
    norm_source='${norm_vcf}'
    """
    def bulkFinish = largeRoot ? """
    cd "\$local_task_dir"
    ln -s "\$bulk_dir/${sample_id}.snv.bi.pass.vcf.gz" '${sample_id}.snv.bi.pass.vcf.gz'
    ln -s "\$bulk_dir/${sample_id}.snv.bi.pass.vcf.gz.tbi" '${sample_id}.snv.bi.pass.vcf.gz.tbi'
    cp "\$bulk_dir/${sample_id}.counts.tsv" '${sample_id}.counts.tsv'
    """ : ''
    """
    set -euo pipefail

    ${bulkStart}
    bcftools view ${alleles_filter} ${variant_types} ${maf_filter} --threads ${threads} -Oz -o ${sample_id}.snv.bi.vcf.gz "\$norm_source"
    bcftools index --threads ${threads} -t ${sample_id}.snv.bi.vcf.gz

    ${pass_flag_cmd}

    bcftools index --threads ${threads} -t ${sample_id}.snv.bi.pass.vcf.gz

    ${rawCountCommand}
    norm_total=\$(bcftools index -n "\$norm_source")
    snv_bi_total=\$(bcftools index -n ${sample_id}.snv.bi.vcf.gz)
    snv_bi_pass_total=\$(bcftools index -n ${sample_id}.snv.bi.pass.vcf.gz)

    printf "chr\tstep\tn_variants\n" > ${sample_id}.counts.tsv
    printf "%s\traw\t%s\n" "${chr}" "\$raw_total" >> ${sample_id}.counts.tsv
    printf "%s\tnorm\t%s\n" "${chr}" "\$norm_total" >> ${sample_id}.counts.tsv
    printf "%s\tsnv_bi\t%s\n" "${chr}" "\$snv_bi_total" >> ${sample_id}.counts.tsv
    printf "%s\tsnv_bi_pass\t%s\n" "${chr}" "\$snv_bi_pass_total" >> ${sample_id}.counts.tsv
    ${bulkFinish}
    """
}
