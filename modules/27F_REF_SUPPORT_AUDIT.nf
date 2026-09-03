nextflow.enable.dsl=2

process WRITE_M27F_SUPPORT_RUN_PROVENANCE {
    tag "m27f_support_run_provenance"
    publishDir "${params.m27f_support_results_dir}", mode: 'copy', overwrite: false
    container params.m27f_support_container_image
    containerOptions params.m27f_support_container_options
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

process PROJECT_M27F_SUPPORT_PANEL {
    tag "m27f_positive_allowlist_projection"
    publishDir "${params.m27f_support_results_dir}", mode: 'copy', overwrite: false, pattern: 'm27f_projection.*.json'
    container params.m27f_support_container_image
    containerOptions params.m27f_support_container_options
    cpus params.m27f_support_cpus
    memory params.m27f_support_memory
    time params.m27f_support_time

    input:
    path source_panel_vcf
    path split_private
    path split_public
    path split_manifest
    path preregistration
    path projection_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    tuple path("m27f_discovery_core.chr22.private.bcf"), path("m27f_discovery_core.chr22.private.bcf.csi"), emit: discovery_projection
    tuple path("m27f_ref.chr22.private.bcf"), path("m27f_ref.chr22.private.bcf.csi"), emit: ref_projection
    tuple path("m27f_valid.chr22.private.bcf"), path("m27f_valid.chr22.private.bcf.csi"), emit: valid_projection
    path "m27f_projection.public.json", emit: public_receipt
    path "m27f_projection.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${projection_py} \
      --bcftools bcftools \
      --source-panel-vcf ${source_panel_vcf} \
      --split-private ${split_private} \
      --split-public ${split_public} \
      --split-manifest ${split_manifest} \
      --preregistration ${preregistration} \
      --outdir .
    chmod 600 m27f_*.private.*

    python3 ${manifest_py} \
      --stage M27F_REF_VALID_MECHANICAL_PROJECTION \
      --input ${source_panel_vcf} \
      --input ${split_private} \
      --input ${split_public} \
      --input ${split_manifest} \
      --input ${preregistration} \
      --input ${projection_py} \
      --input ${run_provenance} \
      --output m27f_discovery_core.chr22.private.bcf \
      --output m27f_discovery_core.chr22.private.bcf.csi \
      --output m27f_ref.chr22.private.bcf \
      --output m27f_ref.chr22.private.bcf.csi \
      --output m27f_valid.chr22.private.bcf \
      --output m27f_valid.chr22.private.bcf.csi \
      --output m27f_projection.public.json \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"positive_allowlist_projection_no_test","source_info_fields_removed":true,"source_test_projection_created":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref run_provenance.json \
      --out m27f_projection.manifest.json
    """
}

process AUDIT_M27F_REF_SUPPORT {
    tag "m27f_ref_select_frozen_support"
    publishDir "${params.m27f_support_results_dir}", mode: 'copy', overwrite: false
    container params.m27f_support_container_image
    containerOptions params.m27f_support_container_options
    cpus params.m27f_support_cpus
    memory params.m27f_support_memory
    time params.m27f_support_time

    input:
    path raw_wgs_vcf
    path baseline_vcf
    tuple path(discovery_bcf), path(discovery_bcf_index)
    tuple path(ref_bcf), path(ref_bcf_index)
    path split_private
    path split_manifest
    path projection_public
    path projection_manifest
    path m27e_manifest
    path m27e_support
    path m27e_preregistration
    path preregistration
    path ref_audit_py
    path m27e_py
    path bridge_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "m27f_ref_site_support.private.tsv", emit: private_support
    path "m27f_ref_primary_catalog.private.tsv", emit: private_primary_catalog
    path "m27f_ref_support.public.json", emit: public_support
    path "m27f_ref_support.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    PYTHONPATH=. python3 ${ref_audit_py} \
      --bcftools bcftools \
      --raw-wgs-vcf ${raw_wgs_vcf} \
      --baseline-vcf ${baseline_vcf} \
      --discovery-bcf ${discovery_bcf} \
      --ref-bcf ${ref_bcf} \
      --split-private ${split_private} \
      --split-manifest ${split_manifest} \
      --projection-public ${projection_public} \
      --projection-manifest ${projection_manifest} \
      --m27e-manifest ${m27e_manifest} \
      --m27e-support ${m27e_support} \
      --m27e-preregistration ${m27e_preregistration} \
      --preregistration ${preregistration} \
      --outdir .

    chmod 600 m27f_ref_site_support.private.tsv m27f_ref_primary_catalog.private.tsv

    python3 ${manifest_py} \
      --stage M27F_REF_SUPPORT_SELECTION \
      --input ${raw_wgs_vcf} \
      --input ${baseline_vcf} \
      --input ${discovery_bcf} \
      --input ${ref_bcf} \
      --input ${split_private} \
      --input ${split_manifest} \
      --input ${projection_public} \
      --input ${projection_manifest} \
      --input ${m27e_manifest} \
      --input ${m27e_support} \
      --input ${m27e_preregistration} \
      --input ${preregistration} \
      --input ${ref_audit_py} \
      --input ${m27e_py} \
      --input ${bridge_py} \
      --input ${run_provenance} \
      --output m27f_ref_site_support.private.tsv \
      --output m27f_ref_primary_catalog.private.tsv \
      --output m27f_ref_support.public.json \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"discovery_reproduction_and_ref_only_selection","threshold_source":"preregistration","source_valid_analyzed":false,"source_test_opened":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref run_provenance.json \
      --out m27f_ref_support.manifest.json
    """
}

process CLAIM_M27F_VALIDATION_OPENING {
    tag "m27f_freeze_single_validation_opening"
    publishDir "${params.m27f_support_results_dir}", mode: 'copy', overwrite: false
    container false
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    path projection_public
    path projection_manifest
    path ref_private_support
    path ref_primary_catalog
    path ref_public
    path ref_manifest
    path preregistration
    path claim_py
    path validation_contract_py
    path valid_audit_py
    path ref_audit_py
    path m27e_py
    path bridge_py
    val container_image
    val container_digest
    val run_id
    val claim_registry_dir
    val claim_key

    output:
    path "m27f_validation_opening.receipt.json", emit: receipt

    script:
    """
    set -euo pipefail
    python3 ${claim_py} \
      --run-id '${run_id}' \
      --claim-registry-dir '${claim_registry_dir}' \
      --claim-key '${claim_key}' \
      --claim-uri '${params.m27f_support_validation_claim_uri}' \
      --claim-py ${claim_py} \
      --validation-contract-py ${validation_contract_py} \
      --valid-audit-py ${valid_audit_py} \
      --ref-audit-py ${ref_audit_py} \
      --m27e-py ${m27e_py} \
      --bridge-py ${bridge_py} \
      --container-image '${container_image}' \
      --container-digest '${container_digest}' \
      --projection-public ${projection_public} \
      --projection-manifest ${projection_manifest} \
      --ref-support-private ${ref_private_support} \
      --ref-primary-catalog ${ref_primary_catalog} \
      --ref-public ${ref_public} \
      --ref-manifest ${ref_manifest} \
      --preregistration ${preregistration} \
      --out m27f_validation_opening.receipt.json
    """
}

process AUDIT_M27F_VALID_SUPPORT {
    tag "m27f_valid_evaluate_frozen_catalog_once"
    publishDir "${params.m27f_support_results_dir}", mode: 'copy', overwrite: false
    container params.m27f_support_container_image
    containerOptions params.m27f_support_container_options
    cpus params.m27f_support_cpus
    memory params.m27f_support_memory
    time params.m27f_support_time

    input:
    tuple path(valid_bcf), path(valid_bcf_index)
    path split_private
    path split_manifest
    path projection_public
    path projection_manifest
    path ref_private_support
    path ref_primary_catalog
    path ref_public
    path ref_manifest
    path validation_opening
    path preregistration
    path valid_audit_py
    path ref_audit_py
    path m27e_py
    path bridge_py
    path claim_py
    path validation_contract_py
    val container_image
    val container_digest
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "m27f_ref_valid_support.private.tsv", emit: private_support
    path "m27f_ref_valid_support.public.json", emit: public_support
    path "m27f_ref_valid_support.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    PYTHONPATH=. python3 ${valid_audit_py} \
      --bcftools bcftools \
      --valid-bcf ${valid_bcf} \
      --split-private ${split_private} \
      --split-manifest ${split_manifest} \
      --projection-public ${projection_public} \
      --projection-manifest ${projection_manifest} \
      --ref-support-private ${ref_private_support} \
      --ref-primary-catalog ${ref_primary_catalog} \
      --ref-public ${ref_public} \
      --ref-manifest ${ref_manifest} \
      --validation-opening ${validation_opening} \
      --claim-py ${claim_py} \
      --validation-contract-py ${validation_contract_py} \
      --valid-audit-py ${valid_audit_py} \
      --ref-audit-py ${ref_audit_py} \
      --m27e-py ${m27e_py} \
      --bridge-py ${bridge_py} \
      --container-image '${container_image}' \
      --container-digest '${container_digest}' \
      --preregistration ${preregistration} \
      --outdir .

    chmod 600 m27f_ref_valid_support.private.tsv

    python3 ${manifest_py} \
      --stage M27F_REF_VALID_SUPPORT_AUDIT \
      --input ${valid_bcf} \
      --input ${split_private} \
      --input ${split_manifest} \
      --input ${projection_public} \
      --input ${projection_manifest} \
      --input ${ref_private_support} \
      --input ${ref_primary_catalog} \
      --input ${ref_public} \
      --input ${ref_manifest} \
      --input ${validation_opening} \
      --input ${preregistration} \
      --input ${valid_audit_py} \
      --input ${ref_audit_py} \
      --input ${m27e_py} \
      --input ${bridge_py} \
      --input ${claim_py} \
      --input ${validation_contract_py} \
      --input ${run_provenance} \
      --output m27f_ref_valid_support.private.tsv \
      --output m27f_ref_valid_support.public.json \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"single_validation_of_ref_frozen_catalog","threshold_source":"preregistration","source_test_opened":false,"simulation":false,"lai":false,"training":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref run_provenance.json \
      --out m27f_ref_valid_support.manifest.json
    """
}
