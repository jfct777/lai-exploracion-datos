nextflow.enable.dsl=2

import groovy.json.JsonOutput

// ---------------------------------------------------------------------------
// Módulo 23: benchmark de la matriz rara individuo por variante
// ---------------------------------------------------------------------------
// Compara E = Q+sexo+burden+matriz con C = Q+sexo+burden para medir cuánto aporta la matriz rara
// fuera de muestra respecto a la etiqueta Leiden generada por M14. Como la matriz y la etiqueta se
// derivan de las mismas variantes mac2, un E-C positivo indica concordancia o recuperabilidad, no un
// descubrimiento biológico. La etiqueta se lee de results_modtest_mac2/lai_rare.
//
// Etapas del benchmark:
//   1. EXTRACT_RARE_MATRIX_CHR  extracción sparse por cromosoma y smoke técnico en chr22
//   2. CONCAT_RARE_MATRIX       concatenación de las 22 matrices sin densificarlas
//   3. RARE_BENCH_CV            CV anidada agrupada dentro de TRAIN: sets A-E, contraste E-C
// Cada etapa se activa mediante su parámetro enable_* y puede reutilizar las salidas publicadas por
// la etapa anterior. El fold 3, reservado para test, queda fuera de todo el módulo.
//
// Validaciones:
//  - extract_rare_matrix_chr.py entrega a `bcftools query -S` únicamente los ID de TRAIN. Los
//    genotipos del fold de test no se leen y el script termina si un ID aparece en ambos grupos.
//  - El smoke usa una etiqueta permutada y no ejecuta CV ni grilla. Solo informa dimensiones,
//    missingness, memoria, tiempo y conservación del formato sparse.
//  - Matriz siempre sparse (CSC/CSR); nunca se densifica.
//  - Cada etapa escribe un manifiesto sha256 mediante bin/write_stage_manifest.py.

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
// Etapa 2: concatenación de las 22 matrices por cromosoma
// ---------------------------------------------------------------------------
// Las 22 matrices CSC se unen con hstack sin densificarlas. El orden de las filas debe coincidir en
// todos los cromosomas y con TRAIN en el split. El prefiltro de missingness se calcula sobre TRAIN;
// MAC, portadores y varianza se vuelven a calcular dentro de cada fold durante la CV.
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
// Etapa 3: benchmark científico dividido por set y fold
// ---------------------------------------------------------------------------
// El CV monolítico se divide en cuatro procesos para reanudar después de una interrupción:
//   PREFLIGHT  → valida cohortes, fold 3 y huella; es la única tarea que calcula el hash de la matriz.
//   CV_ABC     → sets densos A,B,C (ms) en una tarea; per_fold por los 4 folds.
//   CV_FOLD    → una tarea por (set∈{D,E}, fold∈{0,1,2,4}) = 8 tareas paralelas (maxForks acotado).
//   AGGREGATE  → reensambla en orden, arma contrastes → rare_bench_cv_results.json (esquema idéntico).
// La grilla, los modelos, el umbral, la exclusión del fold 3 y SVD-off se mantienen sin cambios. La
// equivalencia con el modo monolítico se prueba en tests/test_partition_equivalence.py. Los cuatro
// procesos comparten los parámetros científicos y reciben matriz, split, samples y modeling_master
// por los mismos canales `path`, de modo que la herencia del hash y
// el contraste E−C coinciden. bin/_common.py viaja como `path` a cada proceso (import del helper).
process RARE_BENCH_PREFLIGHT {
    tag "preflight"
    publishDir "${params.rare_bench_results_dir}/${params.rare_bench_stage_subdir}", mode: 'copy'
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
    publishDir "${params.rare_bench_results_dir}/${params.rare_bench_stage_subdir}", mode: 'copy'
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
    publishDir "${params.rare_bench_results_dir}/${params.rare_bench_stage_subdir}", mode: 'copy'
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
    publishDir "${params.rare_bench_results_dir}/${params.rare_bench_stage_subdir}", mode: 'copy'
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
      --run-provenance-ref ./run_provenance.json \
      --out cv.manifest.json
    """
}


// ---------------------------------------------------------------------------
// Control acotado de convergencia (23e). Reajusta el modelo final de cada combinación de set y fold
// con los hiperparámetros elegidos en la corrida base y un techo de iteraciones más alto. Cada
// combinación se ejecuta como una tarea independiente que puede reanudarse con -resume.
// ---------------------------------------------------------------------------
process RARE_BENCH_REFIT_FOLD {
    tag "refit_${set_name}_${fold}"
    publishDir "${params.rare_bench_results_dir}/${params.rare_bench_stage_subdir}", mode: 'copy'
    cpus   params.resources.rare_bench_refit_fold.cpus
    memory params.resources.rare_bench_refit_fold.memory
    time   params.rare_bench_refit_time

    input:
    tuple val(set_name), val(fold), path(baseline_json)
    tuple path(matrix), path(samples)
    path split_manifest
    path modeling_master
    path cv_py
    path common_py
    path preflight_json
    // El preflight de la corrida base y el de esta etapa se llaman preflight.json. stageAs evita que
    // ambos archivos colisionen en el directorio de trabajo.
    path baseline_preflight_json, stageAs: 'baseline_preflight.json'
    val  sci_flags
    val  container_sha
    val  prov_b64

    output:
    path "${set_name}.fold${fold}.refit.json", emit: results

    script:
    """
    set -euo pipefail
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ${cv_py} \
      --mode refit --set ${set_name} --fold ${fold} ${sci_flags} \
      --matrix-npz ${matrix} --samples-tsv ${samples} \
      --split-manifest ${split_manifest} --modeling-master ${modeling_master} \
      --container-sha256 ${container_sha} --preflight-json ${preflight_json} \
      --baseline-fold-json ${baseline_json} \
      --baseline-preflight-json ${baseline_preflight_json} \
      --n-jobs 1 --pre-dispatch 1 --outdir .
    """
}

process RARE_BENCH_REFIT_AGGREGATE {
    tag "refit_aggregate"
    publishDir "${params.rare_bench_results_dir}/${params.rare_bench_stage_subdir}", mode: 'copy'
    cpus   params.resources.rare_bench_aggregate.cpus
    memory params.resources.rare_bench_aggregate.memory
    time   params.rare_bench_cv_light_time

    input:
    path abc_json
    path refit_jsons
    path cv_py
    path common_py
    val  sci_flags
    val  prov_b64

    output:
    path "rare_bench_refit_results.json", emit: results
    path "refit.manifest.json",           emit: manifest

    script:
    """
    set -euo pipefail
    REFIT_ARGS=""
    for f in ${refit_jsons}; do REFIT_ARGS="\$REFIT_ARGS --refit-json \$f"; done
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ${cv_py} \
      --mode refit_aggregate ${sci_flags} \
      --abc-json ${abc_json} \$REFIT_ARGS \
      --refit-material-delta ${params.rare_bench_refit_material_delta} --outdir .

    write_stage_manifest.py --stage RARE_BENCH_REFIT_AGGREGATE \
      --input ${abc_json} \
      --output rare_bench_refit_results.json \
      --provenance-b64 ${prov_b64} \
      --params-json '{"mode":"refit_aggregate","sets":"${params.rare_bench_refit_sets}","folds":"${params.rare_bench_refit_folds}","max_iter":${params.rare_bench_refit_max_iter},"baseline_max_iter":${params.rare_bench_cv_max_iter},"material_delta":${params.rare_bench_refit_material_delta},"test_fold":${params.rare_bench_test_fold}}' \
      --stamp "\$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M %Z')" \
      --run-provenance-ref ./run_provenance.json \
      --out refit.manifest.json
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
    def do_rb_refit   = params.enable_rare_bench_refit    && params.run_rare_bench

    // El benchmark y su control de convergencia comparten el proceso PREFLIGHT (misma maquinaria,
    // distinta huella por max_iter). Nextflow no permite invocar un proceso dos veces en el mismo
    // workflow, y ademas ambos publicarian en el mismo subdir -> exclusion explicita.
    if( do_rb_cv && do_rb_refit )
        throw new IllegalStateException("M23: enable_rare_bench_cv y enable_rare_bench_refit no pueden "
            + "estar activos en la misma corrida (comparten PREFLIGHT y subdir de publicación). "
            + "Corre el control de convergencia en una invocación aparte con "
            + "--rare_bench_stage_subdir distinto de '${params.rare_bench_stage_subdir}'.")

    if( do_rb_extract || do_rb_smoke || do_rb_concat || do_rb_cv || do_rb_refit ) {
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
        // Cada etapa guarda run_provenance.json en su propio subdirectorio. Así una segunda corrida no
        // reemplaza la procedencia de la anterior y el manifiesto puede usar una ruta relativa.
        def rb_rp_dir = new File("${params.rare_bench_results_dir}/${params.rare_bench_stage_subdir}")
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

        // --- ETAPA 2 (concat) + ETAPA 3 (CV científica) + ETAPA 3b (control de convergencia) -----
        if( do_rb_concat || do_rb_cv || do_rb_refit ) {
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

            if( do_rb_cv || do_rb_refit ) {
                def rb_mm = file(reqRB(params.rare_bench_modeling_master, 'rare_bench_modeling_master'))
                if( !rb_mm.exists() ) throw new IllegalStateException("M23 cv: modeling_master no encontrado en ${rb_mm}.")
                def rb_common_py = file("${projectDir}/bin/_common.py")
                if( !rb_common_py.exists() ) throw new IllegalStateException("M23 cv: bin/_common.py no encontrado (helper compartido).")

                // Los parámetros científicos se construyen en un solo lugar para que preflight, abc,
                // fold y aggregate compartan la misma huella y el mismo contraste E-C. Como los valores
                // no contienen espacios, pueden pasarse sin comillas adicionales. Un cambio científico
                // invalida la caché. En el control de convergencia solo cambia el techo de iteraciones.
                def rb_sci_with = { max_iter -> [
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
                    "--max-iter ${max_iter}",
                    "--tol ${params.rare_bench_cv_tol}",
                    "--seed ${params.rare_bench_seed}",
                    (params.rare_bench_run_svd ? "--run-svd" : ""),
                ].findAll { it }.join(' ') }

                // Canales inmutables compartidos por preflight y todas las tareas; así se conserva
                // el mismo hash de entrada durante la ejecución.
                def ch_genome_v = ch_genome.first()

                // Un solo par (set,fold) por tarea: fuente unica de la expansion, reusada por el
                // benchmark y por el control de convergencia.
                def rb_expand = { sets_csv, folds_csv, what ->
                    def ss = sets_csv.toString().split(',').collect { it.trim() }.findAll { it }
                    def ff = folds_csv.toString().split(',').collect { it.trim() }.findAll { it }
                    if( ss.isEmpty() || ff.isEmpty() ) throw new IllegalStateException("M23 ${what}: sets/folds vacios.")
                    def out = []
                    ss.each { s -> ff.each { f -> out << [s, f] } }
                    return out
                }

                // El preflight se invoca igual en ambas ramas y solo cambia el juego de flags (el `throw`
                // de arriba garantiza que solo UNA rama corre, asi que el proceso se usa una vez).
                def rb_do_preflight = { sci -> RARE_BENCH_PREFLIGHT(ch_genome_v, rb_split, rb_mm,
                                                                    rare_bench_cv_py, rb_common_py, sci,
                                                                    rb_container_sha, ch_rb_prov).preflight }

                if( do_rb_cv ) {
                    def rb_sci = rb_sci_with(params.rare_bench_cv_max_iter)

                    def rb_pf = rb_do_preflight(rb_sci)

                    def rb_abc = RARE_BENCH_CV_ABC(ch_genome_v, rb_split, rb_mm, rare_bench_cv_py, rb_common_py,
                                                   rb_pf, rb_sci, rb_container_sha, ch_rb_prov).results

                    // Se crean ocho tareas: los sets D y E para los folds 0, 1, 2 y 4. Los valores
                    // configurados deben coincidir con HEAVY_SETS y con los outer folds del split.
                    def rb_sf = rb_expand(params.rare_bench_cv_heavy_sets, params.rare_bench_cv_folds, 'cv')
                                  .collect { s, f -> tuple(s, f) }

                    def rb_fold_res = RARE_BENCH_CV_FOLD(Channel.fromList(rb_sf), ch_genome_v, rb_split, rb_mm,
                                                         rare_bench_cv_py, rb_common_py, rb_pf, rb_sci,
                                                         rb_container_sha, ch_rb_prov).results

                    RARE_BENCH_AGGREGATE(rb_abc, rb_fold_res.collect(), rare_bench_cv_py, rb_common_py,
                                         rb_sci, ch_rb_prov)
                }

                // --- ETAPA 3b: control acotado de convergencia del solver -------------------------
                // Reajusta los modelos finales con los hiperparámetros elegidos en la corrida base y
                // un techo de iteraciones más alto. A/B/C no se recalculan; las métricas de C se leen
                // de abc_results.json.
                if( do_rb_refit ) {
                    def rb_sci_refit = rb_sci_with(params.rare_bench_refit_max_iter)
                    if( params.rare_bench_refit_max_iter.toString().toInteger() <= params.rare_bench_cv_max_iter.toString().toInteger() )
                        throw new IllegalStateException("M23 refit: rare_bench_refit_max_iter "
                            + "(${params.rare_bench_refit_max_iter}) debe ser > rare_bench_cv_max_iter "
                            + "(${params.rare_bench_cv_max_iter}); si no, no es un control de convergencia.")

                    def rf_base_dir = reqRB(params.rare_bench_refit_baseline_dir, 'rare_bench_refit_baseline_dir')
                    def rf_abc = file("${rf_base_dir}/abc_results.json")
                    if( !rf_abc.exists() ) throw new IllegalStateException("M23 refit: falta ${rf_abc} "
                        + "(las métricas de C se leen de la corrida base y no se recalculan).")
                    // Se comprueba el contenido del destino y no solo su ruta. Así también se detecta
                    // una copia archivada de la corrida base y se evita reemplazar su preflight o su
                    // información de procedencia.
                    def rf_out_dir = "${params.rare_bench_results_dir}/${params.rare_bench_stage_subdir}"
                    for( guard in ['abc_results.json', 'rare_bench_cv_results.json'] )
                        if( file("${rf_out_dir}/${guard}").exists() )
                            throw new IllegalStateException("M23 refit: el subdir de publicación ${rf_out_dir} "
                                + "ya contiene ${guard} (artefactos de una etapa CV) -> publicar aqui pisaria "
                                + "su preflight.json y su run_provenance.json. Usa un "
                                + "--rare_bench_stage_subdir propio para el control (p.ej. refit_check).")

                    def rf_pf_base = file("${rf_base_dir}/preflight.json")
                    if( !rf_pf_base.exists() ) throw new IllegalStateException("M23 refit: falta ${rf_pf_base}; "
                        + "su input_sha256 es lo unico que prueba que la base uso estas mismas entradas.")

                    def rf_tuples = rb_expand(params.rare_bench_refit_sets, params.rare_bench_refit_folds, 'refit')
                        .collect { s, f ->
                            def bj = file("${rf_base_dir}/${s}.fold${f}.json")
                            if( !bj.exists() ) throw new IllegalStateException("M23 refit: falta el JSON base "
                                + "${bj}; de ahi se leen los hiperparametros ganadores de ese fold.")
                            tuple(s, f, bj)
                        }

                    def rf_pf = rb_do_preflight(rb_sci_refit)

                    def rf_res = RARE_BENCH_REFIT_FOLD(Channel.fromList(rf_tuples), ch_genome_v, rb_split,
                                                       rb_mm, rare_bench_cv_py, rb_common_py, rf_pf,
                                                       rf_pf_base, rb_sci_refit, rb_container_sha,
                                                       ch_rb_prov).results

                    RARE_BENCH_REFIT_AGGREGATE(rf_abc, rf_res.collect(), rare_bench_cv_py, rb_common_py,
                                               rb_sci_refit, ch_rb_prov)
                }
            }
        }
    }
}
