nextflow.enable.dsl=2

process WRITE_M27F_VALID_RUN_PROVENANCE {
    tag "m27f_valid_run_provenance"
    publishDir "${params.m27f_valid_results_dir}", mode: 'copy', overwrite: false
    container params.m27f_valid_container_image
    containerOptions params.m27f_valid_container_options
    cpus 1
    memory '1 GB'
    time '5m'

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

process PROJECT_M27F_VALID_PANEL {
    tag "m27f_source_valid_positive_allowlist_projection"
    publishDir "${params.m27f_valid_results_dir}", mode: 'copy', overwrite: false, pattern: 'm27f_valid*'
    container params.m27f_valid_container_image
    containerOptions params.m27f_valid_container_options
    cpus params.m27f_valid_cpus
    memory params.m27f_valid_memory
    time params.m27f_valid_time

    input:
    path source_panel_vcf
    path split_private
    path split_public
    path split_manifest
    path preregistration
    path projection_py
    path projection_common_py

    output:
    tuple path("m27f_valid.chr22.private.bcf"), path("m27f_valid.chr22.private.bcf.csi"), emit: valid_projection
    path "m27f_valid.samples.private.txt", emit: valid_allowlist
    path "m27f_valid_projection.public.json", emit: public_receipt

    script:
    """
    set -euo pipefail
    PYTHONPATH=. python3 ${projection_py} \
      --bcftools bcftools \
      --source-panel-vcf ${source_panel_vcf} \
      --split-private ${split_private} \
      --split-public ${split_public} \
      --split-manifest ${split_manifest} \
      --preregistration ${preregistration} \
      --outdir .
    """
}

process AUDIT_M27F_VALID_TRANSFER {
    tag "m27f_one_shot_valid_local_transfer"
    publishDir "${params.m27f_valid_results_dir}", mode: 'copy', overwrite: false
    container params.m27f_valid_container_image
    containerOptions params.m27f_valid_container_options
    cpus params.m27f_valid_cpus
    memory params.m27f_valid_memory
    time params.m27f_valid_time

    input:
    tuple path(valid_bcf), path(valid_bcf_index)
    path split_private
    path split_manifest
    path projection_public
    path ref_eligible_catalog
    path ref_support_public
    path ref_support_manifest
    path historical_baseline_vcf
    path genetic_map
    path preregistration
    path audit_py
    path ref_audit_py
    path bridge_py
    path manifest_py
    val provenance_b64

    output:
    path "m27f_valid_site_support.private.tsv", emit: private_site_support
    path "m27f_valid_block_support.private.tsv", emit: private_block_support
    path "m27f_valid_transfer.public.json", emit: public_support
    path "m27f_valid_transfer.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    PYTHONPATH=. python3 ${audit_py} \
      --bcftools bcftools \
      --valid-bcf ${valid_bcf} \
      --split-private ${split_private} \
      --split-manifest ${split_manifest} \
      --projection-public ${projection_public} \
      --ref-eligible-catalog ${ref_eligible_catalog} \
      --ref-support-public ${ref_support_public} \
      --ref-support-manifest ${ref_support_manifest} \
      --historical-baseline-vcf ${historical_baseline_vcf} \
      --genetic-map ${genetic_map} \
      --preregistration ${preregistration} \
      --outdir .

    chmod 600 m27f_valid_site_support.private.tsv m27f_valid_block_support.private.tsv

    python3 ${manifest_py} \
      --stage M27F_VALID_LOCAL_TRANSFER \
      --input ${valid_bcf} \
      --input ${split_private} \
      --input ${split_manifest} \
      --input ${projection_public} \
      --input ${ref_eligible_catalog} \
      --input ${ref_support_public} \
      --input ${ref_support_manifest} \
      --input ${historical_baseline_vcf} \
      --input ${genetic_map} \
      --input ${preregistration} \
      --input ${audit_py} \
      --input ${ref_audit_py} \
      --input ${bridge_py} \
      --output m27f_valid_site_support.private.tsv \
      --output m27f_valid_block_support.private.tsv \
      --output m27f_valid_transfer.public.json \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"one_shot_valid_local_transfer","source_valid_opened_once":true,"source_test_opened":false,"simulation":false,"lai":false,"training":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27f_valid_transfer.manifest.json
    """
}
