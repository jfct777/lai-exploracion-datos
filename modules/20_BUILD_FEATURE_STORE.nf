nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Módulo 20: feature store V.02-A con Q, Σℓ y densidad por individuo
// ---------------------------------------------------------------------------
// Tabla de características por individuo (una fila por persona). Reúne Q, presencia
// NAMBR, anotaciones y el espectro {ℓ_i} en un formato reproducible.
//
// Columnas: Q en orden NAM/EUR/EAS/AFR, Σℓ (total_shared_bp de M14; aislados→0),
// densidad de raras (3 lecturas: dosis ALT, sitios-portador, sitios-no-missing), flags.
//
// Topología (mismo molde que M17/M21):
//   DEFINE_COHORT          1 vez  -> cohorte certificada N (rare ∩ metadata, dedup no-silencioso)
//   COUNT_RARE_DENSITY     × cromosoma -> bloque PSC de `bcftools stats -s -` (DRY con M03)
//   AGGREGATE_FEATURE_STORE 1 vez  -> join + feature_store.{tsv,parquet} + manifest.json
//
// La densidad es intrínseca al individuo (no depende de la cohorte) → se cuenta sobre
// todas las muestras del VCF y el subconjunto a N se aplica en el AGGREGATE.

process DEFINE_COHORT {
    tag "cohort"

    publishDir "${params.feature_build_results_dir}/_cohort", mode: 'copy'

    cpus   params.resources.feature_define_cohort.cpus
    memory params.resources.feature_define_cohort.memory
    time   params.time

    input:
    tuple path(rare_vcf), path(rare_tbi), path(metadata), path(build_py)

    output:
    path "cohort_${params.feature_build_expected_n}.tsv", emit: cohort
    path "cohort_define.report.json"

    script:
    """
    set -euo pipefail
    bcftools query -l ${rare_vcf} > rare_samples.txt
    python3 ${build_py} define-cohort \
      --rare-samples rare_samples.txt \
      --metadata ${metadata} \
      --dedup-id ${params.feature_build_dedup_id} \
      --expected-n ${params.feature_build_expected_n} \
      --out cohort_${params.feature_build_expected_n}.tsv \
      --report cohort_define.report.json
    """
}

process COUNT_RARE_DENSITY {
    tag "chr${chr}"

    publishDir "${params.feature_build_results_dir}/per_chr", mode: 'copy'

    cpus   params.resources.feature_count_density.cpus
    memory params.resources.feature_count_density.memory
    time   params.time

    input:
    tuple val(chr), path(rare_vcf), path(rare_tbi)

    output:
    path "chr${chr}.psc.tsv", emit: psc

    script:
    """
    set -euo pipefail
    # bcftools stats per-sample (PSC). Columnas PSC: \$3=sample \$4=nRefHom \$5=nNonRefHom
    # \$6=nHets \$14=nMissing (formato estable de bcftools). DRY: mismo tool que M03.
    bcftools stats -s - ${rare_vcf} > chr${chr}.stats.txt
    awk -F'\\t' 'BEGIN{OFS="\\t"; print "sample_id","chr","nRefHom","nNonRefHom","nHets","nMissing"}
                 \$1=="PSC"{print \$3,"${chr}",\$4,\$5,\$6,\$14}' chr${chr}.stats.txt > chr${chr}.psc.tsv
    if [ \$(wc -l < chr${chr}.psc.tsv) -le 1 ]; then
      echo "ERROR: bloque PSC vacío en chr${chr}" >&2; exit 1
    fi
    """
}

process AGGREGATE_FEATURE_STORE {
    tag "feature_store"

    publishDir "${params.feature_build_results_dir}", mode: 'copy'

    cpus   params.resources.feature_aggregate.cpus
    memory params.resources.feature_aggregate.memory
    time   params.time

    input:
    tuple path(cohort), path(sigma_summary), path(metadata), path(psc_files), path(build_py)

    output:
    path "feature_store.tsv",     emit: store
    path "feature_store.parquet", optional: true, emit: parquet
    path "manifest.json",         emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${build_py} aggregate \
      --cohort ${cohort} \
      --sigma-summary ${sigma_summary} \
      --metadata ${metadata} \
      --psc-glob 'chr*.psc.tsv' \
      --expected-n ${params.feature_build_expected_n} \
      --build-date ${params.feature_build_date} \
      --out-prefix feature_store \
      --manifest manifest.json
    """
}
