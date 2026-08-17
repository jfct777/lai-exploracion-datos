nextflow.enable.dsl=2

process WRITE_M28C_B0_RUN_PROVENANCE {
    tag "m28c_b0_run_provenance"
    publishDir "${params.m28c_b0_results_dir}", mode: 'copy', overwrite: false
    container params.m28c_b0_container_image
    containerOptions params.m28c_b0_container_options
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

process RUN_M28C_B0_INPUT_PREFLIGHT {
    tag "m28c_b0_seed_${params.m28c_b0_root_seed}"
    publishDir "${params.m28c_b0_results_dir}/seed-${params.m28c_b0_root_seed}", mode: 'copy', overwrite: false
    container params.m28c_b0_container_image
    containerOptions params.m28c_b0_container_options
    cpus params.m28c_b0_cpus
    memory params.m28c_b0_memory
    time params.m28c_b0_time

    input:
    path tree_sequence
    path pool_manifest
    path mosaic_events
    path b0_markers
    path preregistration
    path materialize_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "m28c_b0/m28c_b0_reference.vcf.gz", emit: reference_vcf
    path "m28c_b0/m28c_b0_target.vcf.gz", emit: target_vcf
    path "m28c_b0/m28c_b0_reference.sample_map.tsv", emit: sample_map
    path "m28c_b0/m28c_b0_reference_pairs.private.tsv", emit: private_pairs
    path "m28c_b0/m28c_b0_input_preflight.public.json", emit: report
    path "m28c_b0/m28c_b0_input_preflight.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${materialize_py} \
      --tree-sequence ${tree_sequence} \
      --pool-manifest ${pool_manifest} \
      --mosaic-events ${mosaic_events} \
      --b0-markers ${b0_markers} \
      --preregistration ${preregistration} \
      --outdir m28c_b0

    python3 ${manifest_py} \
      --stage M28C_B0_INPUT_PREFLIGHT \
      --input ${tree_sequence} \
      --input ${pool_manifest} \
      --input ${mosaic_events} \
      --input ${b0_markers} \
      --input ${preregistration} \
      --input ${materialize_py} \
      --input ${run_provenance} \
      --output m28c_b0/m28c_b0_reference.vcf.gz \
      --output m28c_b0/m28c_b0_target.vcf.gz \
      --output m28c_b0/m28c_b0_reference.sample_map.tsv \
      --output m28c_b0/m28c_b0_reference_pairs.private.tsv \
      --output m28c_b0/m28c_b0_input_preflight.public.json \
      --params-json '{"root_seed":${params.m28c_b0_root_seed},"scope":"technical_smoke_no_LAI"}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m28c_b0/m28c_b0_input_preflight.manifest.json
    chmod 600 m28c_b0/m28c_b0_reference_pairs.private.tsv
    """
}

process AUDIT_M28C_B0_GNOMIX_INGEST {
    tag "m28c_b0_gnomix_ingest_seed_${params.m28c_b0_root_seed}"
    publishDir "${params.m28c_b0_results_dir}/seed-${params.m28c_b0_root_seed}", mode: 'copy', overwrite: false
    container params.m28c_gnomix_container_image
    containerOptions params.m28c_b0_container_options
    cpus params.m28c_gnomix_ingest_cpus
    memory params.m28c_gnomix_ingest_memory
    time params.m28c_gnomix_ingest_time

    input:
    path reference_vcf
    path target_vcf
    path materialization_report
    path preregistration
    path ingest_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "m28c_ingest/m28c_b0_reference.vcf.gz", emit: reference_bgzf
    path "m28c_ingest/m28c_b0_reference.vcf.gz.tbi", emit: reference_tbi
    path "m28c_ingest/m28c_b0_target.vcf.gz", emit: target_bgzf
    path "m28c_ingest/m28c_b0_target.vcf.gz.tbi", emit: target_tbi
    path "m28c_ingest/m28c_b0_gnomix_ingest.public.json", emit: report
    path "m28c_ingest/m28c_b0_gnomix_ingest.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${ingest_py} \
      --reference-vcf ${reference_vcf} \
      --target-vcf ${target_vcf} \
      --materialization-report ${materialization_report} \
      --preregistration ${preregistration} \
      --gnomix-root /opt/gnomix \
      --root-seed ${params.m28c_b0_root_seed} \
      --outdir m28c_ingest

    python3 ${manifest_py} \
      --stage M28C_B0_GNOMIX_INGEST_AUDIT \
      --input ${reference_vcf} \
      --input ${target_vcf} \
      --input ${materialization_report} \
      --input ${preregistration} \
      --input ${ingest_py} \
      --input ${run_provenance} \
      --output m28c_ingest/m28c_b0_reference.vcf.gz \
      --output m28c_ingest/m28c_b0_reference.vcf.gz.tbi \
      --output m28c_ingest/m28c_b0_target.vcf.gz \
      --output m28c_ingest/m28c_b0_target.vcf.gz.tbi \
      --output m28c_ingest/m28c_b0_gnomix_ingest.public.json \
      --params-json '{"root_seed":${params.m28c_b0_root_seed},"scope":"technical_ingest_no_training"}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m28c_ingest/m28c_b0_gnomix_ingest.manifest.json
    """
}
