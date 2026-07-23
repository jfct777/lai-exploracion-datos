nextflow.enable.dsl=2

process DAF_DSFS_FROM_ANCESTRAL_TSV {
    tag "chr${chr}"

    publishDir "${params.outdir}/09_daf_dsfs", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(ancestral_tsv_gz), path(ancestral_summary_json), path(daf_dsfs_py)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.dsfs.tsv"), path("dnabr.hg38.2723.chr${chr}.dsfs_rare_tail_dac.tsv"), path("dnabr.hg38.2723.chr${chr}.daf_per_site.tsv.gz"), path("dnabr.hg38.2723.chr${chr}.daf.summary.json")

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    """
    set -euo pipefail

    python3 ${daf_dsfs_py} \
      --vcf ${vcf_gz} \
      --chr ${chr} \
      --ancestral_tsv_gz ${ancestral_tsv_gz} \
      --out_prefix ${sample_id} \
      --rare_tail_max_ac ${params.rare_tail_max_ac} \
      --sfs_bins_af '${params.sfs_bins_af}'

    mv ${sample_id}.summary.json ${sample_id}.daf.summary.json
    """
}
