nextflow.enable.dsl=2

process QC_PLINK_MISSING_HET {
    tag "chr${chr}"

    publishDir "${params.outdir}/05_plink_qc", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(pgen), path(pvar), path(psam), path(aggregate_qc_report_py)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.qc_per_sample.tsv")
    path "dnabr.hg38.2723.chr${chr}.smiss"
    path "dnabr.hg38.2723.chr${chr}.het"
    path "dnabr.hg38.2723.chr${chr}.genoFilt.log"
    path "dnabr.hg38.2723.chr${chr}.missing.log"
    path "dnabr.hg38.2723.chr${chr}.het.log"

    script:
    def sample_id = "dnabr.hg38.2723.chr${chr}"
    def threads = params.resources?.qc_plink_missing_het?.threads ?: 4
    def is_autosome = chr ==~ /^[0-9]+$/  // true if chr is 1-22
    """
    set -euo pipefail

    plink2 --pfile ${sample_id} --geno ${params.max_missing_site} --make-pgen --threads ${threads} --out ${sample_id}.genoFilt > ${sample_id}.genoFilt.log 2>&1

    plink2 --pfile ${sample_id} --missing --threads ${threads} --out ${sample_id} > ${sample_id}.missing.log 2>&1

    # Heterozygosity calculation only works for autosomes (1-22)
    if [[ "${chr}" =~ ^[0-9]+\$ ]]; then
      # Autosomal chromosome: calculate heterozygosity
      plink2 --pfile ${sample_id}.genoFilt --het --threads ${threads} --out ${sample_id} > ${sample_id}.het.log 2>&1
    else
      # Non-autosomal chromosome (X, Y, MT): skip --het and create empty files
      echo "Skipping heterozygosity calculation for chr${chr} (non-autosomal)" > ${sample_id}.het.log
      # Header compatible with aggregate_qc_report.py::read_plink_het()
      echo "#IID\tO(HOM)\tE(HOM)\tOBS_CT\tF" > ${sample_id}.het
    fi

    python3 ${aggregate_qc_report_py} --mode per_chr --chr ${chr} --imiss ${sample_id}.smiss --het ${sample_id}.het --out ${sample_id}.qc_per_sample.tsv
    """
}
