nextflow.enable.dsl=2

import groovy.json.JsonOutput

// ---------------------------------------------------------------------------
// Module 23 — Benchmark de la matriz rara individuo x variante (recuperabilidad tecnica)
// ---------------------------------------------------------------------------
// El diseño mide si la matriz rara reducida, añadida
// al burden (E = Q+sexo+burden+matriz) frente a C = Q+sexo+burden, RECUPERA fuera de muestra mas de
// la etiqueta Leiden (interna a M14). Como la matriz y la etiqueta comparten el mismo substrato de
// raras mac2, un E-C>0 es CONCORDANCIA/recuperabilidad, no descubrimiento biologico; y por DPI-en-
// muestra-finita el resultado PUEDE ser <=0. Fuente exacta del label = results_modtest_mac2/lai_rare.
//
// Este archivo implementa las ETAPAS del benchmark:
//   1. EXTRACT_RARE_MATRIX_CHR  extraccion sparse por cromosoma (+ RARE_BENCH_SMOKE tecnico chr22)
//   2. CONCAT_RARE_MATRIX       concatenacion genoma-completa de las 22 matrices (nunca densifica)
//   3. RARE_BENCH_CV            CV anidada agrupada dentro de TRAIN: sets A-E, contraste E-C
// Cada etapa se activa con su enable_* y es independiente (descubre las salidas publicadas si la
// etapa previa esta desactivada). El fold 3 (TEST) no entra en ninguna etapa.
//
// Validaciones:
//  - EL FOLD 3 (TEST) NUNCA SE DECODIFICA: extract_rare_matrix_chr.py pasa a `bcftools query -S`
//    unicamente los IDs de TRAIN -> el genotipo de TEST no se lee del VCF. Ademas el script aborta
//    (fail-closed) si algun ID esta a la vez en TRAIN y TEST.
//  - El SMOKE es tecnico: etiqueta permutada, sin CV, sin grilla, sin metricas cientificas. Solo
//    reporta dimensiones, missingness, memoria, tiempo, sparse-safety y el bloqueo de fold 3.
//  - Matriz siempre sparse (CSC/CSR); nunca se densifica.
//  - Manifiesto sha256 por etapa via bin/write_stage_manifest.py (bin/ en el PATH del worker).

process EXTRACT_RARE_MATRIX_CHR {
    tag "extract_chr${chrom}"

    publishDir "${params.rare_bench_results_dir}/extract", mode: 'copy'

    cpus   params.resources.rare_bench_extract.cpus
    memory params.resources.rare_bench_extract.memory
    time   params.time

    input:
    tuple val(chrom), path(vcf), path(vcf_tbi), path(split_manifest), path(extract_py)
    val prov_b64

    output:
    tuple val(chrom),
          path("${chrom}.rare_matrix.npz"),
          path("${chrom}.variants.tsv"),
          path("${chrom}.samples.tsv"), emit: matrix
    path "${chrom}.extract_summary.json", emit: summary
    path "${chrom}.manifest.json",        emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${extract_py} \
      --vcf ${vcf} \
      --split-manifest ${split_manifest} \
      --chrom ${chrom} \
      --sample-id-col ${params.rare_bench_sample_id_col} \
      --split-col ${params.rare_bench_split_col} \
      --train-label ${params.rare_bench_train_label} \
      --test-label ${params.rare_bench_test_label} \
      --outdir .

    write_stage_manifest.py --stage EXTRACT_RARE_MATRIX_CHR \
      --input ${vcf} --input ${split_manifest} \
      --output ${chrom}.rare_matrix.npz --output ${chrom}.variants.tsv \
      --output ${chrom}.samples.tsv --output ${chrom}.extract_summary.json \
      --provenance-b64 ${prov_b64} \
      --params-json '{"chrom":"${chrom}","train_label":"${params.rare_bench_train_label}","test_label":"${params.rare_bench_test_label}"}' \
      --stamp "\$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M %Z')" \
      --out ${chrom}.manifest.json
    """
}

process RARE_BENCH_SMOKE {
    tag "smoke_chr${chrom}"

    publishDir "${params.rare_bench_results_dir}/smoke", mode: 'copy'

    cpus   params.resources.rare_bench_smoke.cpus
    memory params.resources.rare_bench_smoke.memory
    time   params.time

    input:
    tuple val(chrom), path(matrix), path(variants), path(samples), path(split_manifest), path(smoke_py)
    val prov_b64

    output:
    path "${chrom}.rare_bench_smoke_report.json", emit: report
    path "${chrom}.manifest.json",                emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${smoke_py} \
      --matrix-npz ${matrix} \
      --variants-tsv ${variants} \
      --samples-tsv ${samples} \
      --split-manifest ${split_manifest} \
      --chrom ${chrom} \
      --sample-id-col ${params.rare_bench_sample_id_col} \
      --split-col ${params.rare_bench_split_col} \
      --label-col ${params.rare_bench_label_col} \
      --train-label ${params.rare_bench_train_label} \
      --test-label ${params.rare_bench_test_label} \
      --min-mac-train ${params.rare_bench_min_mac_train} \
      --min-alt-carriers-train ${params.rare_bench_min_alt_carriers_train} \
      --max-missing-train ${params.rare_bench_max_missing_train} \
      --min-variance-train ${params.rare_bench_min_variance_train} \
      --smoke-c ${params.rare_bench_smoke_c} \
      --smoke-l1-ratio ${params.rare_bench_smoke_l1_ratio} \
      --smoke-max-iter ${params.rare_bench_smoke_max_iter} \
      --smoke-tol ${params.rare_bench_smoke_tol} \
      --seed ${params.rare_bench_seed} \
      --outdir .

    write_stage_manifest.py --stage RARE_BENCH_SMOKE \
      --input ${matrix} --input ${variants} --input ${samples} --input ${split_manifest} \
      --output ${chrom}.rare_bench_smoke_report.json \
      --provenance-b64 ${prov_b64} \
      --params-json '{"chrom":"${chrom}","min_mac_train":${params.rare_bench_min_mac_train},"min_alt_carriers_train":${params.rare_bench_min_alt_carriers_train},"max_missing_train":${params.rare_bench_max_missing_train},"smoke_c":${params.rare_bench_smoke_c},"smoke_l1_ratio":${params.rare_bench_smoke_l1_ratio}}' \
      --stamp "\$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M %Z')" \
      --out ${chrom}.manifest.json
    """
}

// ---------------------------------------------------------------------------
// ETAPA 2 — Concatenación genoma-completa de las 22 matrices por-cromosoma
// ---------------------------------------------------------------------------
// hstack de las 22 CSC en una matriz individuo×variante genoma-completa (nunca densifica). El orden de
// filas se re-asegura idéntico en los 22 y == orden TRAIN del split (fail-closed). El pre-filtro de
// missingness (call-rate) es GLOBAL-TRAIN aquí (la extracción imputó faltantes a 0 → la máscara por
// individuo no es recuperable; es casi inerte y label-independiente). MAC/portadores/varianza se
// re-derivan POR-FOLD en la CV, no aquí.
process CONCAT_RARE_MATRIX {
    tag "concat_genome"

    publishDir "${params.rare_bench_results_dir}/concat", mode: 'copy'

    cpus   params.resources.rare_bench_concat.cpus
    memory params.resources.rare_bench_concat.memory
    time   params.rare_bench_concat_time

    input:
    path matrices
    path variants
    path samples
    path split_manifest
    path concat_py
    val  chroms_csv
    val  prov_b64

    output:
    tuple path("genome.rare_matrix.npz"), path("genome.samples.tsv"), emit: matrix
    path "genome.variant_index.npy",   emit: index
    path "genome.concat_summary.json", emit: summary
    path "concat.manifest.json",       emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${concat_py} \
      --extract-dir . \
      --chromosomes "${chroms_csv}" \
      --split-manifest ${split_manifest} \
      --sample-id-col ${params.rare_bench_sample_id_col} \
      --split-col ${params.rare_bench_split_col} \
      --train-label ${params.rare_bench_train_label} \
      --max-missing-train ${params.rare_bench_max_missing_train} \
      --outdir .

    write_stage_manifest.py --stage CONCAT_RARE_MATRIX \
      --input ${split_manifest} \
      --output genome.rare_matrix.npz --output genome.samples.tsv \
      --output genome.variant_index.npy --output genome.concat_summary.json \
      --provenance-b64 ${prov_b64} \
      --params-json '{"chromosomes":"${chroms_csv}","max_missing_train":${params.rare_bench_max_missing_train}}' \
      --stamp "\$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M %Z')" \
      --out concat.manifest.json
    """
}

// ---------------------------------------------------------------------------
// ETAPA 3 — Benchmark científico PARTICIONADO por (set,fold): reanudabilidad Nextflow-first
// ---------------------------------------------------------------------------
// El CV monolítico se divide en cuatro procesos para reanudar después de una interrupción:
//   PREFLIGHT  → valida cohortes + fold-3 + huella fail-closed (única lectura que HASHEA la matriz).
//   CV_ABC     → sets densos A,B,C (ms) en una tarea; per_fold por los 4 folds.
//   CV_FOLD    → una tarea por (set∈{D,E}, fold∈{0,1,2,4}) = 8 tareas paralelas (maxForks acotado).
//   AGGREGATE  → reensambla en orden, arma contrastes → rare_bench_cv_results.json (esquema idéntico).
// Diseño científico INTACTO (grilla/modelos/umbral/fold-3/SVD-off): probado bit-equivalente al monolítico
// (tests/test_partition_equivalence.py). Fold 3 sellado antes de honrar cualquier resultado. Los flags
// científicos (${sci_flags}) son IDÉNTICOS en los 4 procesos (fuente única en el subworkflow) y matriz/
// split/samples/modeling_master viajan por los MISMOS canales `path` inmutables → la herencia del hash y
// el contraste E−C coinciden. bin/_common.py viaja como `path` a cada proceso (import del helper).
process RARE_BENCH_PREFLIGHT {
    tag "preflight"
    publishDir "${params.rare_bench_results_dir}/cv", mode: 'copy'
    cpus   params.resources.rare_bench_preflight.cpus
    memory params.resources.rare_bench_preflight.memory
    time   params.rare_bench_cv_light_time

    input:
    tuple path(matrix), path(samples)
    path split_manifest
    path modeling_master
    path cv_py
    path common_py
    val  sci_flags
    val  container_sha
    val  prov_b64

    output:
    path "preflight.json", emit: preflight

    script:
    """
    set -euo pipefail
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ${cv_py} \
      --mode preflight ${sci_flags} \
      --matrix-npz ${matrix} --samples-tsv ${samples} \
      --split-manifest ${split_manifest} --modeling-master ${modeling_master} \
      --container-sha256 ${container_sha} \
      --outdir .
    """
}

process RARE_BENCH_CV_ABC {
    tag "abc"
    publishDir "${params.rare_bench_results_dir}/cv", mode: 'copy'
    cpus   params.resources.rare_bench_cv_abc.cpus
    memory params.resources.rare_bench_cv_abc.memory
    time   params.rare_bench_cv_light_time

    input:
    tuple path(matrix), path(samples)
    path split_manifest
    path modeling_master
    path cv_py
    path common_py
    path preflight_json
    val  sci_flags
    val  container_sha
    val  prov_b64

    output:
    path "abc_results.json", emit: results

    script:
    """
    set -euo pipefail
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ${cv_py} \
      --mode abc ${sci_flags} \
      --matrix-npz ${matrix} --samples-tsv ${samples} \
      --split-manifest ${split_manifest} --modeling-master ${modeling_master} \
      --container-sha256 ${container_sha} --preflight-json ${preflight_json} \
      --n-jobs ${params.rare_bench_cv_n_jobs} --pre-dispatch ${params.rare_bench_cv_pre_dispatch} --outdir .
    """
}

process RARE_BENCH_CV_FOLD {
    tag "fold_${set_name}_${fold}"
    publishDir "${params.rare_bench_results_dir}/cv", mode: 'copy'
    cpus   params.resources.rare_bench_cv_fold.cpus
    // Valores base para una ejecución directa; conf/auto_resources.config aplica la política
    // de reintento durante la ejecución completa.
    memory params.resources.rare_bench_cv_fold.memory
    time   params.rare_bench_cv_fold_time

    input:
    tuple val(set_name), val(fold)
    tuple path(matrix), path(samples)
    path split_manifest
    path modeling_master
    path cv_py
    path common_py
    path preflight_json
    val  sci_flags
    val  container_sha
    val  prov_b64

    output:
    path "${set_name}.fold${fold}.json", emit: results

    script:
    """
    set -euo pipefail
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ${cv_py} \
      --mode fold --set ${set_name} --fold ${fold} ${sci_flags} \
      --matrix-npz ${matrix} --samples-tsv ${samples} \
      --split-manifest ${split_manifest} --modeling-master ${modeling_master} \
      --container-sha256 ${container_sha} --preflight-json ${preflight_json} \
      --n-jobs ${params.rare_bench_cv_n_jobs} --pre-dispatch ${params.rare_bench_cv_pre_dispatch} --outdir .
    """
}

process RARE_BENCH_AGGREGATE {
    tag "aggregate"
    publishDir "${params.rare_bench_results_dir}/cv", mode: 'copy'
    cpus   params.resources.rare_bench_aggregate.cpus
    memory params.resources.rare_bench_aggregate.memory
    time   params.rare_bench_cv_light_time

    input:
    path abc_json
    path fold_jsons
    path cv_py
    path common_py
    val  sci_flags
    val  prov_b64

    output:
    path "rare_bench_cv_results.json", emit: results
    path "cv.manifest.json",           emit: manifest

    script:
    """
    set -euo pipefail
    FOLD_ARGS=""
    for f in ${fold_jsons}; do FOLD_ARGS="\$FOLD_ARGS --fold-json \$f"; done
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ${cv_py} \
      --mode aggregate ${sci_flags} \
      --abc-json ${abc_json} \$FOLD_ARGS --outdir .

    write_stage_manifest.py --stage RARE_BENCH_AGGREGATE \
      --input ${abc_json} \
      --output rare_bench_cv_results.json \
      --provenance-b64 ${prov_b64} \
      --params-json '{"mode":"aggregate","heavy_sets":"${params.rare_bench_cv_heavy_sets}","folds":"${params.rare_bench_cv_folds}","test_fold":${params.rare_bench_test_fold}}' \
      --stamp "\$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M %Z')" \
      --out cv.manifest.json
    """
}



// ---------------------------------------------------------------------------
// SUBWORKFLOW — orquestacion de M23 (extraido de main.nf: el workflow{} unico rozaba el
// limite de 64KB de Groovy; ademas modulariza la logica en su propio modulo).
// ---------------------------------------------------------------------------
workflow RARE_MATRIX_BENCHMARK {
    main:
    // =========================================================================
    // Módulo 23 — Benchmark matriz rara individuo×variante: extracción + SMOKE técnico chr22
    // =========================================================================
    // Diseño reducido. La fuente de entrada es results_modtest_mac2/lai_rare.
    // El fold 3 (TEST) NUNCA se decodifica: extract pasa solo IDs de TRAIN a `bcftools query -S`.
    // El smoke es técnico (etiqueta permutada, sin CV/grilla/métricas). Procedencia calcada de M22.
    def do_rb_extract = params.enable_rare_bench_extract && params.run_rare_bench
    def do_rb_smoke   = params.enable_rare_bench_smoke   && params.run_rare_bench
    def do_rb_concat  = params.enable_rare_bench_concat  && params.run_rare_bench
    def do_rb_cv      = params.enable_rare_bench_cv       && params.run_rare_bench

    if( do_rb_extract || do_rb_smoke || do_rb_concat || do_rb_cv ) {
        def extract_rare_matrix_chr_py = file("${projectDir}/bin/extract_rare_matrix_chr.py")
        def rare_bench_smoke_py        = file("${projectDir}/bin/rare_bench_smoke.py")
        def concat_rare_matrix_py      = file("${projectDir}/bin/concat_rare_matrix.py")
        def rare_bench_cv_py           = file("${projectDir}/bin/rare_bench_cv.py")
        def reqRB = { val, name -> if( val == null ) throw new IllegalStateException("M23: falta --${name}"); return val }

        // Procedencia de ARTEFACTO (estable, entra al cache-key vía val) — calcada de M22: commit por
        // I/O de .git (git puede no estar en el PATH del nodo), sha256 del contenedor por coreutils.
        def shOutRB = { cmd -> try { def p = ['bash','-c',cmd].execute(); p.waitFor(); return p.exitValue()==0 ? p.text.trim() : '' } catch( ignored ) { return '' } }
        def resolveGitCommitRB = { dir ->
            try {
                def head = new File("${dir}/.git/HEAD").text.trim()
                if( !head.startsWith('ref:') ) return head
                def ref = head.substring(4).trim()
                def refFile = new File("${dir}/.git/${ref}")
                if( refFile.exists() ) return refFile.text.trim()
                def packed = new File("${dir}/.git/packed-refs")
                if( packed.exists() )
                    for( line in packed.readLines() )
                        if( line.endsWith(" ${ref}") ) return line.split(' ')[0]
                return 'unknown'
            } catch( ignored ) { return 'unknown' }
        }
        def rb_git_commit    = resolveGitCommitRB(projectDir.toString()) ?: 'unknown'
        def rb_container_sha = shOutRB("sha256sum '${params.container_image}' 2>/dev/null | cut -d' ' -f1") ?: 'unavailable'
        if( !rb_container_sha ) rb_container_sha = 'unavailable'
        def rb_prov_map = [
            git_commit       : rb_git_commit,
            nextflow_version : workflow.nextflow.version.toString(),
            container_path   : params.container_image,
            container_sha256 : rb_container_sha,
        ]
        def rb_prov_b64 = JsonOutput.toJson(rb_prov_map).bytes.encodeBase64().toString()
        def ch_rb_prov  = channel.value(rb_prov_b64)
        // Comando Nextflow literal (volátil con -resume) → run_provenance.json, fuera del cache-key.
        def rb_run_prov = [
            git_commit       : rb_git_commit,
            nextflow_command : workflow.commandLine,
            nextflow_version : workflow.nextflow.version.toString(),
            container_path   : params.container_image,
            container_sha256 : rb_container_sha,
            launch_dir       : workflow.launchDir.toString(),
            project_dir      : projectDir.toString(),
        ]
        def rb_rp_dir = new File("${params.rare_bench_results_dir}")
        rb_rp_dir.mkdirs()
        new File(rb_rp_dir, 'run_provenance.json').text = JsonOutput.prettyPrint(JsonOutput.toJson(rb_run_prov))

        // split_manifest congelado (M22): entrada obligatoria (sin inferencias no sustentadas), fuente única del fold 3.
        def rb_split = file(reqRB(params.rare_bench_split_manifest, 'rare_bench_split_manifest'))
        if( !rb_split.exists() ) throw new IllegalStateException("M23: split_manifest no encontrado en ${rb_split}.")

        def rb_chroms = params.rare_bench_chromosomes.toString().split(',').collect { it.trim() }.findAll { it }
        if( rb_chroms.isEmpty() ) throw new IllegalStateException("M23: rare_bench_chromosomes vacío.")

        def ch_rb_matrix
        if( do_rb_extract ) {
            def rb_vcf_dir = reqRB(params.rare_bench_rare_vcf_dir, 'rare_bench_rare_vcf_dir')
            def extract_tuples = rb_chroms.collect { chrom ->
                def vcf = file("${rb_vcf_dir}/${params.rare_bench_vcf_prefix}.chr${chrom}.rare.vcf.gz")
                def tbi = file("${rb_vcf_dir}/${params.rare_bench_vcf_prefix}.chr${chrom}.rare.vcf.gz.tbi")
                if( !vcf.exists() ) throw new IllegalStateException("M23: VCF no encontrado ${vcf}.")
                if( !tbi.exists() ) throw new IllegalStateException("M23: índice .tbi no encontrado ${tbi}.")
                tuple(chrom, vcf, tbi, rb_split, extract_rare_matrix_chr_py)
            }
            ch_rb_matrix = EXTRACT_RARE_MATRIX_CHR(Channel.fromList(extract_tuples), ch_rb_prov).matrix
        } else if( do_rb_smoke || do_rb_concat ) {  // sin extract: descubre las matrices per-cromosoma publicadas
            def disc_tuples = rb_chroms.collect { chrom ->
                def ex  = "${params.rare_bench_results_dir}/extract"
                def npz = file("${ex}/${chrom}.rare_matrix.npz")
                def var = file("${ex}/${chrom}.variants.tsv")
                def sam = file("${ex}/${chrom}.samples.tsv")
                for( f in [npz, var, sam] ) if( !f.exists() ) throw new IllegalStateException("M23: falta ${f} (corre extract primero).")
                tuple(chrom, npz, var, sam)
            }
            ch_rb_matrix = Channel.fromList(disc_tuples)
        }

        if( do_rb_smoke ) {
            RARE_BENCH_SMOKE(
                ch_rb_matrix.map { chrom, npz, var, sam -> tuple(chrom, npz, var, sam, rb_split, rare_bench_smoke_py) },
                ch_rb_prov
            )
        }

        // --- ETAPA 2 (concat) + ETAPA 3 (CV científica) ------------------------------------------
        if( do_rb_concat || do_rb_cv ) {
            def ch_genome
            if( do_rb_concat ) {
                ch_rb_matrix.multiMap { chrom, npz, var, sam ->
                    matrices: npz
                    variants: var
                    samples:  sam
                }.set { rb_parts }
                ch_genome = CONCAT_RARE_MATRIX(
                    rb_parts.matrices.collect(),
                    rb_parts.variants.collect(),
                    rb_parts.samples.collect(),
                    rb_split,
                    concat_rare_matrix_py,
                    rb_chroms.join(','),
                    ch_rb_prov
                ).matrix
            } else {  // cv sin concat: descubre la matriz genoma-completa publicada
                def cc   = "${params.rare_bench_results_dir}/concat"
                def gnpz = file("${cc}/genome.rare_matrix.npz")
                def gsam = file("${cc}/genome.samples.tsv")
                for( f in [gnpz, gsam] ) if( !f.exists() ) throw new IllegalStateException("M23 cv: falta ${f} (corre concat primero).")
                ch_genome = Channel.of( tuple(gnpz, gsam) )
            }

            if( do_rb_cv ) {
                def rb_mm = file(reqRB(params.rare_bench_modeling_master, 'rare_bench_modeling_master'))
                if( !rb_mm.exists() ) throw new IllegalStateException("M23 cv: modeling_master no encontrado en ${rb_mm}.")
                def rb_common_py = file("${projectDir}/bin/_common.py")
                if( !rb_common_py.exists() ) throw new IllegalStateException("M23 cv: bin/_common.py no encontrado (helper compartido).")

                // Flags cientificos COMUNES: fuente unica -> identicos en preflight/abc/fold/aggregate,
                // de modo que la huella y el contraste E-C coincidan. Los valores no llevan espacios ->
                // sin comillas de shell. Cambiar cualquier param cientifico invalida el cache (correcto).
                def rb_sci = [
                    "--sample-id-col ${params.rare_bench_sample_id_col}",
                    "--split-col ${params.rare_bench_split_col}",
                    "--label-col ${params.rare_bench_label_col}",
                    "--fold-col ${params.rare_bench_fold_col}",
                    "--group-col ${params.rare_bench_group_col}",
                    "--train-label ${params.rare_bench_train_label}",
                    "--test-fold ${params.rare_bench_test_fold}",
                    "--min-mac-train ${params.rare_bench_min_mac_train}",
                    "--min-alt-carriers-train ${params.rare_bench_min_alt_carriers_train}",
                    "--min-variance-train ${params.rare_bench_min_variance_train}",
                    "--c-grid ${params.rare_bench_c_grid}",
                    "--l1-ratios ${params.rare_bench_l1_ratios}",
                    "--inner-splits ${params.rare_bench_inner_splits}",
                    "--inner-scorer ${params.rare_bench_inner_scorer}",
                    "--decision-threshold ${params.rare_bench_decision_threshold}",
                    "--class-weight ${params.rare_bench_class_weight}",
                    "--svd-components ${params.rare_bench_svd_components}",
                    "--max-iter ${params.rare_bench_cv_max_iter}",
                    "--tol ${params.rare_bench_cv_tol}",
                    "--seed ${params.rare_bench_seed}",
                    (params.rare_bench_run_svd ? "--run-svd" : ""),
                ].findAll { it }.join(' ')

                // Canales inmutables compartidos por preflight y todas las tareas; así se conserva
                // el mismo hash de entrada durante la ejecución.
                def ch_genome_v = ch_genome.first()

                def rb_pf = RARE_BENCH_PREFLIGHT(ch_genome_v, rb_split, rb_mm, rare_bench_cv_py,
                                                 rb_common_py, rb_sci, rb_container_sha, ch_rb_prov).preflight.first()

                def rb_abc = RARE_BENCH_CV_ABC(ch_genome_v, rb_split, rb_mm, rare_bench_cv_py, rb_common_py,
                                               rb_pf, rb_sci, rb_container_sha, ch_rb_prov).results

                // 8 tareas (set,fold) = {D,E} x {0,1,2,4}. Los sets/folds son parametrizables pero deben
                // coincidir con HEAVY_SETS del script y con los outer_folds del split (el agregador lo
                // exige fail-closed).
                def rb_sets  = params.rare_bench_cv_heavy_sets.toString().split(',').collect { it.trim() }.findAll { it }
                def rb_folds = params.rare_bench_cv_folds.toString().split(',').collect { it.trim() }.findAll { it }
                if( rb_sets.isEmpty() || rb_folds.isEmpty() ) throw new IllegalStateException("M23 cv: rare_bench_cv_heavy_sets/folds vacios.")
                def rb_sf = []
                rb_sets.each { s -> rb_folds.each { f -> rb_sf << tuple(s, f) } }

                def rb_fold_res = RARE_BENCH_CV_FOLD(Channel.fromList(rb_sf), ch_genome_v, rb_split, rb_mm,
                                                     rare_bench_cv_py, rb_common_py, rb_pf, rb_sci,
                                                     rb_container_sha, ch_rb_prov).results

                RARE_BENCH_AGGREGATE(rb_abc, rb_fold_res.collect(), rare_bench_cv_py, rb_common_py,
                                     rb_sci, ch_rb_prov)
            }
        }
    }
}
