nextflow.enable.dsl=2

process LD_DECAY_FROM_PGEN {
    tag "chr${chr}"

    publishDir "${params.outdir}/10_ld_decay", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(pgen), path(pvar), path(psam), path(ld_decay_py)
    path(keep_samples, stageAs: 'keep_samples.txt')
    path(exclude_bed, stageAs: 'exclude_regions.bed')

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.ld_decay.tsv"), path("dnabr.hg38.2723.chr${chr}.ld_decay.summary.json")
    path "dnabr.hg38.2723.chr${chr}.ld_pairs.tsv.gz"

    script:
    def sample_id = pgen.baseName
    def keep_arg = keep_samples.size() > 0 ? "--keep ${keep_samples}" : ""
    def excl_arg = exclude_bed.size() > 0 ? "--exclude range ${exclude_bed}" : ""
    """
    set -euo pipefail

    # Compute pairwise LD (PLINK2). Output as gz TSV.
    plink2 --pfile ${sample_id} \
      ${keep_arg} \
      ${excl_arg} \
      --maf ${params.ld_maf_min} \
      --geno ${params.ld_site_missing_max} \
      --r2-unphased \
      --ld-window-kb ${params.ld_window_kb} \
      --ld-window ${params.ld_window} \
      --ld-window-r2 ${params.ld_r2_min} \
      --threads ${params.resources?.ld_decay_from_pgen?.threads ?: params.cpus} \
      --out ${sample_id}.ld

    # PLINK2 output filename varies across versions; pick the first output for this prefix (excluding logs)
    LD_IN=""
    for f in ${sample_id}.ld*; do
      if [ ! -f "\$f" ]; then
        continue
      fi
      case "\$f" in
        *.log) continue ;;
      esac
      LD_IN="\$f"
      break
    done
    if [ -z "\$LD_IN" ]; then
      echo "Missing expected LD output (${sample_id}.ld*)" >&2
      ls -lah
      exit 1
    fi

    # Standardize to gz TSV expected by downstream parser (use streaming to avoid memory issues)
    if [[ "\$LD_IN" == *.gz ]]; then
      cp "\$LD_IN" dnabr.hg38.2723.chr${chr}.ld_pairs.tsv.gz
    else
      cat "\$LD_IN" | gzip -1 > dnabr.hg38.2723.chr${chr}.ld_pairs.tsv.gz
      rm -f "\$LD_IN"
    fi

    python3 ${ld_decay_py} \
      --pairs dnabr.hg38.2723.chr${chr}.ld_pairs.tsv.gz \
      --chr ${chr} \
      --bin_size_bp ${params.ld_bin_size_bp} \
      --max_dist_bp ${params.ld_max_dist_bp} \
      --out_tsv dnabr.hg38.2723.chr${chr}.ld_decay.tsv \
      --out_summary_json dnabr.hg38.2723.chr${chr}.ld_decay.summary.json
    """
}
