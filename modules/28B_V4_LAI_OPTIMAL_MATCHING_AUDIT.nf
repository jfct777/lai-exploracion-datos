nextflow.enable.dsl=2

process WRITE_M28B_V4_RUN_PROVENANCE {
    tag "m28b_v4_run_provenance"
    publishDir "${params.m28b_v4_results_dir}", mode: 'copy', overwrite: false
    container params.m28b_v4_container_image
    containerOptions params.m28b_v4_container_options
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

process RUN_M28B_V4_DEVELOPMENT {
    tag "m28b_v4_development"
    publishDir "${params.m28b_v4_results_dir}", mode: 'copy', overwrite: false
    container params.m28b_v4_container_image
    containerOptions params.m28b_v4_container_options
    cpus params.m28b_v4_cpus
    memory params.m28b_v4_memory
    time params.m28b_v4_time

    input:
    path dev_tree
    path dev_pools
    path genetic_map
    path baseline_template
    path m28_preregistration
    path preregistration
    path audit_v4_py
    path audit_v3_py
    path audit_v2_py
    path audit_v1_py
    path m28_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "m28b_v4_dev/m28b_v4_dev.public.json", emit: report
    path "m28b_v4_dev/m28b_v4_dev_screens.tsv", emit: screens
    path "m28b_v4_dev/m28b_v4_frozen_selection.json", emit: frozen
    path "m28b_v4_dev/m28b_v4_dev_B0.tsv.gz", emit: b0
    path "m28b_v4_dev/m28b_v4_dev_BR_additions.tsv.gz", emit: br
    path "m28b_v4_dev/m28b_v4_dev_BS_additions.tsv.gz", emit: bs
    path "m28b_v4_dev/m28b_v4_dev_BR_BS_pairs.tsv.gz", emit: pairs
    path "m28b_v4_dev/m28b_v4_dev_common_common_null.tsv", emit: nulls
    path "m28b_v4_dev/m28b_v4_dev.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${audit_v4_py} \
      --phase development --tree-sequence ${dev_tree} --pool-manifest ${dev_pools} \
      --genetic-map ${genetic_map} --baseline-template ${baseline_template} \
      --m28-preregistration ${m28_preregistration} \
      --preregistration ${preregistration} --outdir m28b_v4_dev

    python3 ${manifest_py} \
      --stage M28B_V4_LAI_OPTIMAL_MATCHING_DEV \
      --input ${dev_tree} --input ${dev_pools} --input ${genetic_map} \
      --input ${baseline_template} --input ${m28_preregistration} \
      --input ${preregistration} --input ${audit_v4_py} --input ${audit_v3_py} \
      --input ${audit_v2_py} --input ${audit_v1_py} --input ${m28_py} \
      --input ${run_provenance} \
      --output m28b_v4_dev/m28b_v4_dev.public.json \
      --output m28b_v4_dev/m28b_v4_dev_screens.tsv \
      --output m28b_v4_dev/m28b_v4_frozen_selection.json \
      --output m28b_v4_dev/m28b_v4_dev_B0.tsv.gz \
      --output m28b_v4_dev/m28b_v4_dev_BR_additions.tsv.gz \
      --output m28b_v4_dev/m28b_v4_dev_BS_additions.tsv.gz \
      --output m28b_v4_dev/m28b_v4_dev_BR_BS_pairs.tsv.gz \
      --output m28b_v4_dev/m28b_v4_dev_common_common_null.tsv \
      --params-json '{"phase":"development","scope":"geometry_only_no_LAI"}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m28b_v4_dev/m28b_v4_dev.manifest.json
    """
}

process RUN_M28B_V4_VALIDATION {
    tag "m28b_v4_single_validation"
    publishDir "${params.m28b_v4_results_dir}", mode: 'copy', overwrite: false
    container params.m28b_v4_container_image
    containerOptions params.m28b_v4_container_options
    cpus params.m28b_v4_cpus
    memory params.m28b_v4_memory
    time params.m28b_v4_time

    input:
    path validation_tree
    path validation_pools
    path validation_preflight_manifest
    path genetic_map
    path baseline_template
    path m28_preregistration
    path preregistration
    path frozen_selection
    path audit_v4_py
    path audit_v3_py
    path audit_v2_py
    path audit_v1_py
    path m28_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "m28b_v4_validation/m28b_v4_validation.public.json", emit: report
    path "m28b_v4_validation/m28b_v4_validation_B0.tsv.gz", emit: b0
    path "m28b_v4_validation/m28b_v4_validation_BR_additions.tsv.gz", emit: br
    path "m28b_v4_validation/m28b_v4_validation_BS_additions.tsv.gz", emit: bs
    path "m28b_v4_validation/m28b_v4_validation_BR_BS_pairs.tsv.gz", emit: pairs
    path "m28b_v4_validation/m28b_v4_validation_common_common_null.tsv", emit: nulls
    path "m28b_v4_validation/m28b_v4_validation.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${audit_v4_py} \
      --phase validation --tree-sequence ${validation_tree} \
      --pool-manifest ${validation_pools} --preflight-manifest ${validation_preflight_manifest} \
      --genetic-map ${genetic_map} --baseline-template ${baseline_template} \
      --m28-preregistration ${m28_preregistration} --preregistration ${preregistration} \
      --frozen-selection ${frozen_selection} --outdir m28b_v4_validation

    python3 ${manifest_py} \
      --stage M28B_V4_LAI_OPTIMAL_MATCHING_VALIDATION \
      --input ${validation_tree} --input ${validation_pools} \
      --input ${validation_preflight_manifest} --input ${genetic_map} \
      --input ${baseline_template} --input ${m28_preregistration} \
      --input ${preregistration} --input ${frozen_selection} \
      --input ${audit_v4_py} --input ${audit_v3_py} --input ${audit_v2_py} \
      --input ${audit_v1_py} --input ${m28_py} --input ${run_provenance} \
      --output m28b_v4_validation/m28b_v4_validation.public.json \
      --output m28b_v4_validation/m28b_v4_validation_B0.tsv.gz \
      --output m28b_v4_validation/m28b_v4_validation_BR_additions.tsv.gz \
      --output m28b_v4_validation/m28b_v4_validation_BS_additions.tsv.gz \
      --output m28b_v4_validation/m28b_v4_validation_BR_BS_pairs.tsv.gz \
      --output m28b_v4_validation/m28b_v4_validation_common_common_null.tsv \
      --params-json '{"phase":"single_validation","scope":"geometry_only_no_LAI"}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m28b_v4_validation/m28b_v4_validation.manifest.json
    """
}
