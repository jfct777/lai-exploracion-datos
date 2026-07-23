nextflow.enable.dsl=2

process SFS_FROM_FILTERED_VCF {
    tag "chr${chr}"

    publishDir "${params.outdir}/07_sfs", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(sfs_report_py)
    path keep_samples

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.sfs.tsv"), path("dnabr.hg38.2723.chr${chr}.rare_tail_ac.tsv"), path("dnabr.hg38.2723.chr${chr}.sfs.summary.json")

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def keep_arg = keep_samples.size() > 0 ? "--keep_samples ${keep_samples}" : ""
    // Folded → bins MAF [0,0.5]; unfolded → bins ALT-AF [0,1]. Una sola fuente de verdad por modo.
    def sfs_bins = params.sfs_fold ? params.sfs_bins_maf : params.sfs_bins_af
    """
    set -euo pipefail

    python3 ${sfs_report_py} \
      --vcf ${vcf_gz} \
      --chr ${chr} \
      --out_prefix ${sample_id} \
      ${keep_arg} \
      --min_an_frac ${params.min_an_frac} \
      --rare_tail_max_ac ${params.rare_tail_max_ac} \
      --fold ${params.sfs_fold} \
      --sfs_bins_af '${sfs_bins}'

    mv ${sample_id}.summary.json ${sample_id}.sfs.summary.json
    """
}
