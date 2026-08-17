nextflow.enable.dsl=2

process WRITE_M28B_V5_RUN_PROVENANCE {
    tag "m28b_v5_run_provenance"
    publishDir "${params.m28b_v5_results_dir}", mode: 'copy', overwrite: false
    container params.m28b_v5_container_image
    containerOptions params.m28b_v5_container_options
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    val provenance_b64

    output:
    path "run_provenance.json"

    script:
    """
    set -euo pipefail
    printf '%s' '${provenance_b64}' | base64 -d > run_provenance.json
    """
}

process RUN_M28B_V5_DEVELOPMENT {
    tag "m28b_v5_development"
    publishDir "${params.m28b_v5_results_dir}", mode: 'copy', overwrite: false
    container params.m28b_v5_container_image
    containerOptions params.m28b_v5_container_options
    cpus params.m28b_v5_cpus
    memory params.m28b_v5_memory
    time params.m28b_v5_time

    input:
    path dev_tree
    path dev_pools
    path dev_preflight_report
    path dev_preflight_manifest
    path preflight_reproducibility
    path genetic_map
    path baseline_template
    path m28_preregistration
    path preregistration
    path audit_v5_py
    path audit_v3_py
    path audit_v2_py
    path audit_v1_py
    path m28_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "m28b_v5_dev/m28b_v5_dev.public.json", emit: report
    path "m28b_v5_dev/m28b_v5_dev_screens.tsv", emit: screens
    path "m28b_v5_dev/m28b_v5_frozen_selection.json", emit: frozen
    path "m28b_v5_dev/m28b_v5_dev_B0.tsv.gz", emit: b0
    path "m28b_v5_dev/m28b_v5_dev_BR_additions.tsv.gz", emit: br
    path "m28b_v5_dev/m28b_v5_dev_BS_additions.tsv.gz", emit: bs
    path "m28b_v5_dev/m28b_v5_dev_BR_BS_pairs.tsv.gz", emit: pairs
    path "m28b_v5_dev/m28b_v5_dev_common_common_null.tsv", emit: nulls
    path "m28b_v5_dev/m28b_v5_dev.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${audit_v5_py} \
      --phase development --tree-sequence ${dev_tree} --pool-manifest ${dev_pools} \
      --preflight-report ${dev_preflight_report} \
      --preflight-manifest ${dev_preflight_manifest} \
      --preflight-reproducibility ${preflight_reproducibility} \
      --genetic-map ${genetic_map} --baseline-template ${baseline_template} \
      --m28-preregistration ${m28_preregistration} \
      --preregistration ${preregistration} --outdir m28b_v5_dev

    python3 ${manifest_py} \
      --stage M28B_V5_INDIVIDUAL_SAFE_MATCHING_DEV \
      --input ${dev_tree} --input ${dev_pools} --input ${dev_preflight_report} \
      --input ${dev_preflight_manifest} --input ${preflight_reproducibility} \
      --input ${genetic_map} --input ${baseline_template} \
      --input ${m28_preregistration} --input ${preregistration} \
      --input ${audit_v5_py} --input ${audit_v3_py} --input ${audit_v2_py} \
      --input ${audit_v1_py} --input ${m28_py} --input ${run_provenance} \
      --output m28b_v5_dev/m28b_v5_dev.public.json \
      --output m28b_v5_dev/m28b_v5_dev_screens.tsv \
      --output m28b_v5_dev/m28b_v5_frozen_selection.json \
      --output m28b_v5_dev/m28b_v5_dev_B0.tsv.gz \
      --output m28b_v5_dev/m28b_v5_dev_BR_additions.tsv.gz \
      --output m28b_v5_dev/m28b_v5_dev_BS_additions.tsv.gz \
      --output m28b_v5_dev/m28b_v5_dev_BR_BS_pairs.tsv.gz \
      --output m28b_v5_dev/m28b_v5_dev_common_common_null.tsv \
      --params-json '{"phase":"development","scope":"individual_safe_geometry_only_no_LAI"}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m28b_v5_dev/m28b_v5_dev.manifest.json
    """
}

process RUN_M28B_V5_VALIDATION {
    tag "m28b_v5_single_validation"
    publishDir "${params.m28b_v5_results_dir}", mode: 'copy', overwrite: false
    container params.m28b_v5_container_image
    containerOptions params.m28b_v5_container_options
    cpus params.m28b_v5_cpus
    memory params.m28b_v5_memory
    time params.m28b_v5_time

    input:
    path validation_tree
    path validation_pools
    path validation_preflight_report
    path validation_preflight_manifest
    path preflight_reproducibility
    path genetic_map
    path baseline_template
    path m28_preregistration
    path preregistration
    path frozen_selection
    path audit_v5_py
    path audit_v3_py
    path audit_v2_py
    path audit_v1_py
    path m28_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "m28b_v5_validation/m28b_v5_validation.public.json", emit: report
    path "m28b_v5_validation/m28b_v5_validation_B0.tsv.gz", emit: b0
    path "m28b_v5_validation/m28b_v5_validation_BR_additions.tsv.gz", emit: br
    path "m28b_v5_validation/m28b_v5_validation_BS_additions.tsv.gz", emit: bs
    path "m28b_v5_validation/m28b_v5_validation_BR_BS_pairs.tsv.gz", emit: pairs
    path "m28b_v5_validation/m28b_v5_validation_common_common_null.tsv", emit: nulls
    path "m28b_v5_validation/m28b_v5_validation.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${audit_v5_py} \
      --phase validation --tree-sequence ${validation_tree} \
      --pool-manifest ${validation_pools} \
      --preflight-report ${validation_preflight_report} \
      --preflight-manifest ${validation_preflight_manifest} \
      --preflight-reproducibility ${preflight_reproducibility} \
      --genetic-map ${genetic_map} --baseline-template ${baseline_template} \
      --m28-preregistration ${m28_preregistration} --preregistration ${preregistration} \
      --frozen-selection ${frozen_selection} --outdir m28b_v5_validation

    python3 ${manifest_py} \
      --stage M28B_V5_INDIVIDUAL_SAFE_MATCHING_VALIDATION \
      --input ${validation_tree} --input ${validation_pools} \
      --input ${validation_preflight_report} --input ${validation_preflight_manifest} \
      --input ${preflight_reproducibility} --input ${genetic_map} \
      --input ${baseline_template} --input ${m28_preregistration} \
      --input ${preregistration} --input ${frozen_selection} \
      --input ${audit_v5_py} --input ${audit_v3_py} --input ${audit_v2_py} \
      --input ${audit_v1_py} --input ${m28_py} --input ${run_provenance} \
      --output m28b_v5_validation/m28b_v5_validation.public.json \
      --output m28b_v5_validation/m28b_v5_validation_B0.tsv.gz \
      --output m28b_v5_validation/m28b_v5_validation_BR_additions.tsv.gz \
      --output m28b_v5_validation/m28b_v5_validation_BS_additions.tsv.gz \
      --output m28b_v5_validation/m28b_v5_validation_BR_BS_pairs.tsv.gz \
      --output m28b_v5_validation/m28b_v5_validation_common_common_null.tsv \
      --params-json '{"phase":"single_validation","scope":"individual_safe_geometry_only_no_LAI"}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m28b_v5_validation/m28b_v5_validation.manifest.json
    """
}
