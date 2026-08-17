nextflow.enable.dsl=2

process WRITE_M28B_V2_RUN_PROVENANCE {
    tag "m28b_v2_run_provenance"
    publishDir "${params.m28b_v2_results_dir}", mode: 'copy', overwrite: false
    container params.m28b_v2_container_image
    containerOptions params.m28b_v2_container_options
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

process RUN_M28B_V2_JOINT_CAPACITY_AUDIT {
    tag "m28b_v2_joint_capacity"
    publishDir "${params.m28b_v2_results_dir}", mode: 'copy', overwrite: false
    container params.m28b_v2_container_image
    containerOptions params.m28b_v2_container_options
    cpus params.m28b_v2_cpus
    memory params.m28b_v2_memory
    time params.m28b_v2_time

    input:
    path tree_sequence
    path pool_manifest
    path genetic_map
    path baseline_template
    path m28_preregistration
    path preregistration
    path audit_v2_py
    path audit_v1_py
    path m28_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "m28b_v2/m28b_v2_capacity.public.json", emit: report
    path "m28b_v2/m28b_v2_capacity_screens.tsv", emit: screens
    path "m28b_v2/m28b_v2_B0.tsv.gz", emit: b0
    path "m28b_v2/m28b_v2_BR_additions.tsv.gz", emit: br
    path "m28b_v2/m28b_v2_BS_additions.tsv.gz", emit: bs
    path "m28b_v2/m28b_v2_B0_mapping.tsv.gz", emit: b0_mapping
    path "m28b_v2/m28b_v2_BR_BS_pairs.tsv.gz", emit: pairs
    path "m28b_v2/m28b_v2_capacity.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${audit_v2_py} \
      --tree-sequence ${tree_sequence} --pool-manifest ${pool_manifest} \
      --genetic-map ${genetic_map} --baseline-template ${baseline_template} \
      --m28-preregistration ${m28_preregistration} \
      --preregistration ${preregistration} --outdir m28b_v2

    python3 ${manifest_py} \
      --stage M28B_V2_LAI_JOINT_MARKER_CAPACITY_AUDIT \
      --input ${tree_sequence} --input ${pool_manifest} --input ${genetic_map} \
      --input ${baseline_template} --input ${m28_preregistration} \
      --input ${preregistration} --input ${audit_v2_py} --input ${audit_v1_py} \
      --input ${m28_py} --input ${run_provenance} \
      --output m28b_v2/m28b_v2_capacity.public.json \
      --output m28b_v2/m28b_v2_capacity_screens.tsv \
      --output m28b_v2/m28b_v2_B0.tsv.gz --output m28b_v2/m28b_v2_BR_additions.tsv.gz \
      --output m28b_v2/m28b_v2_BS_additions.tsv.gz \
      --output m28b_v2/m28b_v2_B0_mapping.tsv.gz \
      --output m28b_v2/m28b_v2_BR_BS_pairs.tsv.gz \
      --params-json '{"scope":"joint_capacity_only_no_LAI"}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m28b_v2/m28b_v2_capacity.manifest.json
    """
}
