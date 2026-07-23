nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Module 22 — Pipeline de modelado (modeling_master → split → CV interna → TEST)
// ---------------------------------------------------------------------------
// Integra en Nextflow las cuatro etapas del experimento train/test.
//
// Topología: cadena lineal de procesos únicos (no hay paralelismo por cromosoma),
//   BUILD_MODELING_MASTER → BUILD_SPLIT_MANIFEST → MODEL_PRIMARY_CV → [EVALUATE_TEST]
// + VERIFY_TEST_HASH (independiente, solo integridad del TEST congelado).
//
// Validaciones:
//  - EVALUATE_TEST DESACTIVADO por defecto; doble llave (force + reason) en main.nf.
//    Publica en un subdirectorio del resultado, nunca sobre el archivo canónico congelado.
//  - VERIFY_TEST_HASH es SOLO sha256 del JSON congelado; nunca importa sklearn/pandas ni
//    carga TRAIN/TEST → no puede reabrir el fold 3.
//  - Los scripts se pasan como path() para stagearlos en el work dir. EVALUATE_TEST recibe
//    ADEMÁS model_primary_cv.py aunque no lo invoque: evaluate_test.py hace
//    `from model_primary_cv import ...` y el sibling debe estar en el cwd del task
//    (mismo precedente que M18 COMPARE_ASIBD_COMMON).
//  - Manifiesto sha256 por etapa vía bin/write_stage_manifest.py (bin/ está en el PATH del worker).

process BUILD_MODELING_MASTER {
    tag "modeling_master"

    publishDir "${params.model_pipeline_results_dir}/modeling_master", mode: 'copy'

    cpus   params.resources.model_build_master.cpus
    memory params.resources.model_build_master.memory
    time   params.time

    input:
    tuple path(feature_store), path(leiden), path(graph_nodes), path(pcrelate_kin),
          path(metadata), path(build_py)
    val prov_b64

    output:
    path "modeling_master.tsv",              emit: modeling_master
    path "modeling_master_dict.md"
    path "join_audit.json",                  emit: join_audit
    path "qc_sample_flags.tsv",              emit: qc_flags
    path "kinship_components_report.json"
    path "kinship_x_leiden_crosstab.tsv"
    path "duplicate_or_mz_report.json"
    path "manifest.json",                    emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${build_py} \
      --feature-store ${feature_store} \
      --leiden ${leiden} \
      --graph-nodes ${graph_nodes} \
      --pcrelate-kin ${pcrelate_kin} \
      --metadata ${metadata} \
      --red-samples ${params.model_pipeline_red_samples} \
      --kinship-thresholds ${params.model_pipeline_kinship_thresholds} \
      --kinship-group-columns ${params.model_pipeline_kinship_group_columns} \
      --replicate-threshold ${params.model_pipeline_replicate_threshold} \
      --leiden-col-for-crosstab ${params.model_pipeline_leiden_col_for_crosstab} \
      --outdir .

    write_stage_manifest.py --stage BUILD_MODELING_MASTER \
      --input ${feature_store} --input ${leiden} --input ${graph_nodes} \
      --input ${pcrelate_kin} --input ${metadata} \
      --output modeling_master.tsv --output modeling_master_dict.md \
      --output join_audit.json --output qc_sample_flags.tsv \
      --output kinship_components_report.json --output kinship_x_leiden_crosstab.tsv \
      --output duplicate_or_mz_report.json \
      --provenance-b64 ${prov_b64} \
      --params-json '{"red_samples":"${params.model_pipeline_red_samples}","kinship_group_columns":"${params.model_pipeline_kinship_group_columns}","replicate_threshold":${params.model_pipeline_replicate_threshold}}' \
      --stamp "\$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M %Z')" \
      --out manifest.json
    """
}

process BUILD_SPLIT_MANIFEST {
    tag "split_manifest"

    publishDir "${params.model_pipeline_results_dir}/split", mode: 'copy'

    cpus   params.resources.model_build_split.cpus
    memory params.resources.model_build_split.memory
    time   params.time

    input:
    tuple path(modeling_master), path(build_py)
    val prov_b64

    output:
    path "split_manifest.tsv",           emit: split_manifest
    path "split_manifest_audit.json",    emit: audit
    path "label_definition.md"
    path "manifest.json",                emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${build_py} \
      --master ${modeling_master} \
      --red-samples ${params.model_pipeline_red_samples} \
      --group-col ${params.model_pipeline_group_col} \
      --community-col ${params.model_pipeline_community_col} \
      --n-splits ${params.model_pipeline_n_splits} \
      --seed ${params.model_pipeline_seed} \
      --outdir .

    write_stage_manifest.py --stage BUILD_SPLIT_MANIFEST \
      --input ${modeling_master} \
      --output split_manifest.tsv --output split_manifest_audit.json --output label_definition.md \
      --provenance-b64 ${prov_b64} \
      --params-json '{"group_col":"${params.model_pipeline_group_col}","community_col":"${params.model_pipeline_community_col}","n_splits":${params.model_pipeline_n_splits},"seed":${params.model_pipeline_seed}}' \
      --stamp "\$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M %Z')" \
      --out manifest.json
    """
}

process MODEL_PRIMARY_CV {
    tag "model_primary_cv"

    publishDir "${params.model_pipeline_results_dir}/model_primary", mode: 'copy'

    cpus   params.resources.model_primary_cv.cpus
    memory params.resources.model_primary_cv.memory
    time   params.time

    input:
    tuple path(modeling_master), path(split_manifest), path(model_cv_py)
    val prov_b64

    output:
    path "model_primary_cv_results.json", emit: cv_results
    path "manifest.json",                 emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${model_cv_py} \
      --master ${modeling_master} \
      --split ${split_manifest} \
      --n-perm ${params.model_pipeline_n_perm} \
      --outdir .

    write_stage_manifest.py --stage MODEL_PRIMARY_CV \
      --input ${modeling_master} --input ${split_manifest} \
      --output model_primary_cv_results.json \
      --provenance-b64 ${prov_b64} \
      --params-json '{"n_perm":${params.model_pipeline_n_perm},"seed":${params.model_pipeline_seed}}' \
      --stamp "\$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M %Z')" \
      --out manifest.json
    """
}

// TEST CERRADO — desactivado por defecto (enable_evaluate_test=false + doble llave en main.nf).
// model_cv_py se stagea SOLO para el `from model_primary_cv import ...` de evaluate_test.py
// (no se invoca aquí); NO borrar ese input pensando que es muerto.
process EVALUATE_TEST {
    tag "evaluate_test"

    publishDir "${params.model_pipeline_results_dir}/test_eval", mode: 'copy'

    cpus   params.resources.model_evaluate_test.cpus
    memory params.resources.model_evaluate_test.memory
    time   params.time

    input:
    tuple path(modeling_master), path(split_manifest), path(split_audit),
          path(cv_results_json), path(model_cv_py), path(evaluate_test_py)
    val force_flag
    val prov_b64

    output:
    path "evaluate_test_results.json", emit: test_results
    path "manifest.json",              emit: manifest

    script:
    def force_arg = force_flag ? '--force' : ''
    """
    set -euo pipefail
    python3 ${evaluate_test_py} \
      --master ${modeling_master} \
      --split ${split_manifest} \
      --split-audit ${split_audit} \
      --cv-results ${cv_results_json} \
      --candidate-model ${params.model_pipeline_candidate_model} \
      --outdir . ${force_arg}

    write_stage_manifest.py --stage EVALUATE_TEST \
      --input ${modeling_master} --input ${split_manifest} \
      --input ${split_audit} --input ${cv_results_json} \
      --output evaluate_test_results.json \
      --provenance-b64 ${prov_b64} \
      --params-json '{"candidate_model":"${params.model_pipeline_candidate_model}","force":${force_flag}}' \
      --stamp "\$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M %Z')" \
      --out manifest.json
    """
}

// Integridad del TEST congelado — SOLO sha256, sin importar sklearn/pandas ni cargar TEST.
process VERIFY_TEST_HASH {
    tag "verify_test_hash"

    publishDir "${params.model_pipeline_results_dir}/test_verify", mode: 'copy'

    cpus   params.resources.model_verify_test_hash.cpus
    memory params.resources.model_verify_test_hash.memory
    time   params.time

    input:
    tuple path(frozen_json), path(frozen_sha256)

    output:
    path "verify_test_hash.report.json", emit: report

    script:
    """
    set -euo pipefail
    # sha256sum -c falla (exit != 0) si un solo byte del JSON congelado cambió.
    if sha256sum -c ${frozen_sha256} > sha_check.log 2>&1; then
      status=PASS
    else
      status=FAIL
    fi
    actual=\$(sha256sum ${frozen_json} | cut -d' ' -f1)
    # el reporte se escribe con python (sin heredoc: el delimitador indentado bajo el bloque
    # script: de Nextflow no cierra el heredoc -> EOF). python evita todo ese footgun de bash.
    python3 -c "import json,sys; json.dump({'status': sys.argv[1], 'checked_file': sys.argv[2], 'actual_sha256': sys.argv[3], 'expected_from': sys.argv[4]}, open('verify_test_hash.report.json','w'), indent=2)" \
      "\${status}" "${frozen_json}" "\${actual}" "${frozen_sha256}"
    if [ "\${status}" != "PASS" ]; then cat sha_check.log >&2; exit 1; fi
    """
}
