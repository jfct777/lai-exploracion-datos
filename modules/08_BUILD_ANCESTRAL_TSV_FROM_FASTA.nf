nextflow.enable.dsl=2

process BUILD_ANCESTRAL_TSV_FROM_FASTA {
    tag "chr${chr}"

    publishDir "${params.outdir}/08_ancestral_polarization", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(ancestral_fasta), path(ancestral_fai), path(build_ancestral_tsv_py)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.ancestral.tsv.gz"), path("dnabr.hg38.2723.chr${chr}.ancestral.summary.json")

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def flip = params.allow_flip_if_ancestral_is_alt ? "--allow_flip_if_ancestral_is_alt" : ""
    def amb = params.ancestral_accept_ambiguous ? "--ancestral_accept_ambiguous" : ""
    """
    set -euo pipefail

    python3 ${build_ancestral_tsv_py} \
      --vcf ${vcf_gz} \
      --chr ${chr} \
      --ancestral_fasta ${ancestral_fasta} \
      --out_tsv_gz ${sample_id}.ancestral.tsv.gz \
      --out_summary_json ${sample_id}.ancestral.summary.json \
      ${flip} \
      ${amb}
    """
}
