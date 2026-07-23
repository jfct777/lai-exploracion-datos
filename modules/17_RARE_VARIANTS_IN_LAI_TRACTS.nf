nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Module 17 — Rare variants in LAI tracts (Gnomix local-ancestry painting)
// ---------------------------------------------------------------------------
// Mapea cada copia de alelo raro (MAF<1%, upstream lai_rare) sobre la ancestría
// local del painting Gnomix (.msp por cromosoma), y mide el enriquecimiento de
// raras por ancestría local (African/European/Native_American) contra un baseline
// posicional de la composición de la cohorte. Atribución EXACTA sin fase (primaria)
// + fraccional 0.5/0.5 para los het-on-het ambiguos (sensibilidad).
//
// Es DESCRIPTIVO/control: el baseline posicional NO remueve la tautología
// burden-raras ∝ NAM a nivel-individuo (eso requiere residualización + null
// condicional, Paso 2 del plan). Códigos .msp: African=0 European=1 Native-American=2.
//
// Per-chr: ANALYZE_RARE_IN_LAI  (1 rare VCF + 1 .msp + metadata -> summary.json + by_ancestry.tsv)
// Genome-wide: AGGREGATE_RARE_IN_LAI (suma conteos crudos, recomputa enriquecimiento)

process ANALYZE_RARE_IN_LAI {
    tag "chr${chr}"

    publishDir "${params.rare_in_lai_results_dir}/per_chr", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(msp), path(metadata), path(analyze_py)

    output:
    tuple val(chr), path("dnabr.hg38.2723.chr${chr}.rare_in_lai.summary.json"), emit: summaries
    path "dnabr.hg38.2723.chr${chr}.rare_in_lai.by_ancestry.tsv", emit: by_ancestry

    script:
    def out_prefix = "dnabr.hg38.2723.chr${chr}.rare_in_lai"
    """
    set -euo pipefail

    python3 ${analyze_py} \
      --msp ${msp} \
      --rare_vcf ${vcf_gz} \
      --metadata ${metadata} \
      --vcf_chrom chr${chr} \
      --out_prefix ${out_prefix} \
      --bcftools bcftools
    """
}

process AGGREGATE_RARE_IN_LAI {
    tag "aggregate"

    publishDir "${params.rare_in_lai_results_dir}", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple path(summary_files), path(aggregate_py)

    output:
    path "rare_in_lai.genomewide.json", emit: genomewide
    path "rare_in_lai.per_chr_enrichment.tsv", emit: per_chr

    script:
    """
    set -euo pipefail

    python3 ${aggregate_py} \
      --glob '*.rare_in_lai.summary.json' \
      --out_prefix rare_in_lai
    """
}
