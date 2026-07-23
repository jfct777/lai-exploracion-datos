nextflow.enable.dsl=2

process TAG_SNPS_FROM_PGEN {
    tag "chr${chr}"

    publishDir "${params.outdir}/11_tag_snps", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(pgen), path(pvar), path(psam), path(tag_summary_py)
    path(keep_samples, stageAs: 'keep_samples.txt')
    path(exclude_bed, stageAs: 'exclude_regions.bed')

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.prune.in"), path("dnabr.hg38.2723.chr${chr}.prune.in.strict"), path("dnabr.hg38.2723.chr${chr}.tag.summary.json")

    script:
    def sample_id = pgen.baseName
    def keep_arg = keep_samples.size() > 0 ? "--keep ${keep_samples}" : ""
    def excl_arg = exclude_bed.size() > 0 ? "--exclude range ${exclude_bed}" : ""
    """
    set -euo pipefail

    plink2 --pfile ${sample_id} \
      ${keep_arg} \
      ${excl_arg} \
      --maf ${params.tag_maf_min} \
      --geno ${params.tag_site_missing_max} \
      --set-all-var-ids '@:#:\$r:\$a' \
      --indep-pairwise ${params.tag_window_kb}kb ${params.tag_step} ${params.tag_r2} \
      --threads ${params.resources?.tag_snps_from_pgen?.threads ?: params.cpus} \
      --out ${sample_id}.tag

    plink2 --pfile ${sample_id} \
      ${keep_arg} \
      ${excl_arg} \
      --maf ${params.tag_maf_min} \
      --geno ${params.tag_site_missing_max} \
      --set-all-var-ids '@:#:\$r:\$a' \
      --indep-pairwise ${params.tag_window_kb}kb ${params.tag_step} ${params.tag_r2_strict} \
      --threads ${params.resources?.tag_snps_from_pgen?.threads ?: params.cpus} \
      --out ${sample_id}.tag_strict

    cp ${sample_id}.tag.prune.in dnabr.hg38.2723.chr${chr}.prune.in
    cp ${sample_id}.tag_strict.prune.in dnabr.hg38.2723.chr${chr}.prune.in.strict

    python3 ${tag_summary_py} \
      --chr ${chr} \
      --prune_in dnabr.hg38.2723.chr${chr}.prune.in \
      --prune_in_strict dnabr.hg38.2723.chr${chr}.prune.in.strict \
      --out_json dnabr.hg38.2723.chr${chr}.tag.summary.json
    """
}
