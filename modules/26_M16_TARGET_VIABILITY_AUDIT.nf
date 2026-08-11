nextflow.enable.dsl=2

process WRITE_M16_TARGET_AUDIT_RUN_PROVENANCE {
    tag "m16_target_audit_provenance"
    publishDir "${params.m16_target_audit_results_dir}", mode: 'copy', overwrite: false
    cpus 1
    memory '1 GB'
    time '10m'

    input:
    val run_provenance_b64

    output:
    path "run_provenance.json"

    script:
    """
    set -euo pipefail
    printf '%s' '${run_provenance_b64}' | base64 -d > run_provenance.json
    """
}

process AUDIT_M16_TARGET_VIABILITY {
    tag "m16_5_minor_target_train_validation"
    publishDir "${params.m16_target_audit_results_dir}/audit", mode: 'copy', overwrite: false
    cpus params.m16_target_audit_cpus
    memory params.m16_target_audit_memory
    time params.m16_target_audit_time

    input:
    path minor_assignments
    path modeling_master
    path split_manifest
    path preregistration
    path audit_py
    path manifest_py
    val provenance_b64

    output:
    path "m16_target_state_counts.tsv", emit: state_counts
    path "m16_target_fold_support.tsv", emit: fold_support
    path "m16_target_community_fold_support.tsv", emit: community_support
    path "m16_target_continuous_effects.tsv", emit: continuous_effects
    path "m16_target_categorical_composition.tsv", emit: categorical_composition
    path "m16_target_viability_summary.json", emit: summary
    path "m16_target_viability.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${audit_py} \
      --minor-assignments ${minor_assignments} \
      --modeling-master ${modeling_master} \
      --split-manifest ${split_manifest} \
      --preregistration ${preregistration} \
      --outdir .

    python3 ${manifest_py} \
      --stage M26_M16_TARGET_VIABILITY_AUDIT \
      --input ${minor_assignments} --input ${modeling_master} \
      --input ${split_manifest} --input ${preregistration} --input ${audit_py} \
      --output m16_target_state_counts.tsv \
      --output m16_target_fold_support.tsv \
      --output m16_target_community_fold_support.tsv \
      --output m16_target_continuous_effects.tsv \
      --output m16_target_categorical_composition.tsv \
      --output m16_target_viability_summary.json \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"single_pass_internal_target_closure_train_validation_only","outer_folds":"${params.m16_target_audit_outer_folds}","resolution":"community_res_1","test_fold":3,"expected_train_validation_samples":${params.m16_target_audit_expected_samples}}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m16_target_viability.manifest.json
    """
}
