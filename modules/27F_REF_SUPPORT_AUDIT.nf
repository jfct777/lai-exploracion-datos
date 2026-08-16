nextflow.enable.dsl=2

process WRITE_M27F_REF_RUN_PROVENANCE {
    tag "m27f_ref_run_provenance"
    publishDir "${params.m27f_ref_results_dir}", mode: 'copy', overwrite: false
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

process PROJECT_M27F_REF_PANEL {
    tag "m27f_ref_positive_allowlist_projection"
    publishDir "${params.m27f_ref_results_dir}", mode: 'copy', overwrite: false, pattern: 'm27f_ref*'
    container params.m27f_ref_container_image
    cpus params.m27f_ref_cpus
    memory params.m27f_ref_memory
    time params.m27f_ref_time

    input:
    path source_panel_vcf
    path split_private
    path split_public
    path split_manifest
    path preregistration
    path projection_py

    output:
    tuple path("m27f_discovery_core.chr22.work.bcf"), path("m27f_discovery_core.chr22.work.bcf.csi"), emit: discovery_projection
    tuple path("m27f_ref.chr22.private.bcf"), path("m27f_ref.chr22.private.bcf.csi"), emit: ref_projection
    path "m27f_ref.samples.private.txt", emit: ref_allowlist
    path "m27f_ref_projection.public.json", emit: public_receipt

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
    """
}

process AUDIT_M27F_REF_SUPPORT {
    tag "m27f_ref_only_frozen_954_support"
    publishDir "${params.m27f_ref_results_dir}", mode: 'copy', overwrite: false
    container params.m27f_ref_container_image
    cpus params.m27f_ref_cpus
    memory params.m27f_ref_memory
    time params.m27f_ref_time

    input:
    path raw_wgs_vcf
    tuple path(discovery_bcf), path(discovery_bcf_index)
    tuple path(ref_bcf), path(ref_bcf_index)
    path split_private
    path split_manifest
    path projection_public
    path m27e_manifest
    path m27e_support
    path m27e_preregistration
    path preregistration
    path audit_py
    path m27e_py
    path bridge_py
    path manifest_py
    val provenance_b64

    output:
    path "m27f_ref_site_support.private.tsv.gz", emit: private_support
    path "m27f_ref_support.public.json", emit: public_support
    path "m27f_ref_support.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    PYTHONPATH=. python3 ${audit_py} \
      --bcftools bcftools \
      --raw-wgs-vcf ${raw_wgs_vcf} \
      --discovery-bcf ${discovery_bcf} \
      --ref-bcf ${ref_bcf} \
      --split-private ${split_private} \
      --split-manifest ${split_manifest} \
      --projection-public ${projection_public} \
      --m27e-manifest ${m27e_manifest} \
      --m27e-support ${m27e_support} \
      --m27e-preregistration ${m27e_preregistration} \
      --preregistration ${preregistration} \
      --outdir .

    chmod 600 m27f_ref_site_support.private.tsv.gz

    python3 ${manifest_py} \
      --stage M27F_REF_SUPPORT_AUDIT \
      --input ${raw_wgs_vcf} \
      --input ${discovery_bcf} \
      --input ${ref_bcf} \
      --input ${split_private} \
      --input ${split_manifest} \
      --input ${projection_public} \
      --input ${m27e_manifest} \
      --input ${m27e_support} \
      --input ${m27e_preregistration} \
      --input ${preregistration} \
      --input ${audit_py} \
      --input ${m27e_py} \
      --input ${bridge_py} \
      --output m27f_ref_site_support.private.tsv.gz \
      --output m27f_ref_support.public.json \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"discovery_catalog_reproduction_and_ref_only_support","source_valid_opened":false,"source_test_opened":false,"simulation":false,"lai":false,"training":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27f_ref_support.manifest.json
    """
}
