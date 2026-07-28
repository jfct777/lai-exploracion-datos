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
    // Thinning de sitios (plink2 --bp-space): mantiene ~1 sitio cada N bp para acotar el
    // nº de pares cuando r2_min=0; no sesga la forma del decay (propiedad de la distancia).
    def thin_arg = params.ld_thin_bp_space ? "--bp-space ${params.ld_thin_bp_space}" : ""
    """
    set -euo pipefail

    # Calcula el LD por pares con PLINK2 y guarda la salida como TSV comprimido.
    plink2 --pfile ${sample_id} \
      ${keep_arg} \
      ${excl_arg} \
      ${thin_arg} \
      --maf ${params.ld_maf_min} \
      --geno ${params.ld_site_missing_max} \
      --set-all-var-ids '@:#:\$r:\$a' \
      --r2-unphased \
      --ld-window-kb ${params.ld_window_kb} \
      --ld-window ${params.ld_window} \
      --ld-window-r2 ${params.ld_r2_min} \
      --threads ${params.resources?.ld_decay_from_pgen?.threads ?: params.cpus} \
      --out ${sample_id}.ld

    # El nombre de salida cambia entre versiones de PLINK2; se usa el primer archivo del prefijo,
    # sin considerar el log.
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

    # Normaliza el nombre y el formato que espera el parser posterior.
    if [[ "\$LD_IN" == *.gz ]]; then
      cp "\$LD_IN" dnabr.hg38.2723.chr${chr}.ld_pairs.tsv.gz
    else
      gzip -c "\$LD_IN" > dnabr.hg38.2723.chr${chr}.ld_pairs.tsv.gz
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
