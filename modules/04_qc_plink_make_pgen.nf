nextflow.enable.dsl=2

process QC_PLINK_MAKE_PGEN {
    tag "chr${chr}"

    publishDir "${params.outdir}/04_plink_pgen", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(counts_tsv), path(vcf_in), path(vcf_tbi)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.pgen"), path("dnabr.hg38.2723.chr${chr}.pvar"), path("dnabr.hg38.2723.chr${chr}.psam")
    path "dnabr.hg38.2723.chr${chr}.plink.makepgen.log"

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def snps_only = params.plink_snps_only ? "--snps-only" : ""
    def psam_arg = (chr == "X") ? "--psam ${sample_id}.tmp.psam" : ""
    def split_par = (chr == "X") ? "--split-par hg38" : ""
    """
    set -euo pipefail

    if [ "${chr}" == "X" ]; then
      # For chrX: create .psam with ambiguous sex (0) to enable processing without sex metadata
      bcftools query -l ${vcf_in} | awk 'BEGIN {print "#IID\\tSEX"} {print \$1"\\t0"}' > ${sample_id}.tmp.psam
    fi

    plink2 --vcf ${vcf_in} --make-pgen --out ${sample_id} \
      ${snps_only} \
      ${psam_arg} \
      ${split_par} \
      --max-alleles ${params.plink_max_alleles} \
      --mac ${params.plink_mac_min} \
      > ${sample_id}.plink.makepgen.log 2>&1
    """
}
