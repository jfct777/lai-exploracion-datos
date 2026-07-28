nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Module 19 — Rare variants on LAI painting (per-chromosome publication figure)
// ---------------------------------------------------------------------------
// Por cromosoma genera una figura de tres paneles que muestra dónde caen las
// variantes raras (MAF<1%, upstream lai_rare) respecto al painting de ancestria
// local de Gnomix (.msp por cromosoma):
//   (a) karyograma por-haplotipo (todos los haplotipos pintados);
//   (b) densidad de copias de alelo raro por Mb estratificada por ancestria local;
//   (c) zoom (region o cromosoma completo) con individuos seleccionados.
//
// Reutiliza dos scripts compartidos (DRY): la densidad estratificada por ventana
// la produce `bin/rare_variants_in_lai_tracts.py --emit_windows_bp` (misma logica
// de atribucion exacta que M17); la figura la produce `scripts/plot_rare_karyogram_lai.py`.
// Codigos del .msp: African=0 European=1 Native-American=2 (painting 3-way; sin EAS).
//
// Seleccion del panel c (parametrizable):
//   rare_on_lai_individuals  -> lista CSV de IDs explicitos (mayor prioridad)
//   rare_on_lai_panel_c_select = extremes | mosaic   (si no hay lista)

process RARE_ON_LAI_PAINTING {
    tag "chr${chr}"

    publishDir "${params.rare_on_lai_results_dir}/per_chr", mode: 'copy'

    cpus params.cpus
    memory params.memory
    time params.time

    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi), path(msp), path(metadata), path(analyze_py), path(plot_py)

    output:
    path "fig_rare_on_lai_chr${chr}.png",        emit: figures
    path "fig_rare_on_lai_chr${chr}.pdf",        optional: true
    path "rare_in_lai_chr${chr}.windows.tsv",    emit: windows

    script:
    def indiv = params.rare_on_lai_individuals?.trim()
    def indiv_arg = indiv ? "--individuals '${indiv}'" : ""
    """
    set -euo pipefail

    # (1) densidad estratificada por ventana (copias exactas + territorio por ancestria)
    python3 ${analyze_py} \
      --msp ${msp} \
      --rare_vcf ${vcf_gz} \
      --metadata ${metadata} \
      --vcf_chrom chr${chr} \
      --emit_windows_bp ${params.rare_on_lai_window_bp} \
      --out_prefix rare_in_lai_chr${chr} \
      --bcftools bcftools

    # (2) figura de 3 paneles
    python3 ${plot_py} \
      --msp ${msp} \
      --windows_tsv rare_in_lai_chr${chr}.windows.tsv \
      --metadata ${metadata} \
      --rare_vcf ${vcf_gz} \
      --vcf_chrom chr${chr} \
      --chrom ${chr} \
      --n_grid ${params.rare_on_lai_n_grid} \
      --panel_c ${params.rare_on_lai_panel_c} \
      --panel_c_select ${params.rare_on_lai_panel_c_select} \
      --zoom_mb ${params.rare_on_lai_zoom_mb} \
      --n_zoom_per_group ${params.rare_on_lai_n_zoom_per_group} \
      --dpi ${params.rare_on_lai_dpi} \
      ${indiv_arg} \
      --out_prefix fig_rare_on_lai_chr${chr} \
      --outdir .
    """
}
