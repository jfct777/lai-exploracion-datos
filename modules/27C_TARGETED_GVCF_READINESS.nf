nextflow.enable.dsl=2

process WRITE_TARGETED_GVCF_RUN_PROVENANCE {
    tag "targeted_gvcf_provenance"
    publishDir "${params.targeted_gvcf_results_dir}", mode: 'copy', overwrite: false
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

process BENCHMARK_TARGETED_GVCF_ACCESS {
    tag "chr22_targeted_gvcf_resource_smoke"
    publishDir "${params.targeted_gvcf_results_dir}/resource_smoke", mode: 'copy', overwrite: false
    cpus 8
    memory '16 GB'
    time '1h'

    input:
    path smoke_gvcfs
    path smoke_indexes
    path gnomix_reference_vcf
    path benchmark_py

    output:
    path "m27c_resource_screen.json", emit: summary
    path "m27c_resource_screen.tsv", emit: table
    path "m27c_smoke_model_positions.bed", emit: regions

    script:
    """
    set -euo pipefail
    python3 ${benchmark_py} \
      --gvcfs ${smoke_gvcfs.join(' ')} \
      --gnomix-reference-vcf ${gnomix_reference_vcf} \
      --reader-grid '${params.targeted_gvcf_reader_grid}' \
      --full-sample-count ${params.targeted_gvcf_expected_samples} \
      --outdir .
    """
}

process AUDIT_TARGETED_GVCF_READINESS {
    tag "chr22_targeted_gvcf_readiness"
    publishDir "${params.targeted_gvcf_results_dir}/audit", mode: 'copy', overwrite: false
    cpus params.targeted_gvcf_cpus
    memory params.targeted_gvcf_memory
    time params.targeted_gvcf_time

    input:
    path gvcfs
    path gvcf_indexes
    path gcs_input_manifest
    path phased_scaffold_vcf
    path gnomix_reference_vcf
    path gnomix_config
    path metadata
    path reference_fasta
    path reference_fai
    path preregistration
    path audit_py
    path core_py
    path bridge_py
    path manifest_py
    val provenance_b64

    output:
    path "m27c_input_contract.json", emit: input_contract
    path "m27c_identity_control.json", emit: identity
    path "m27c_callability_summary.json", emit: callability
    path "m27c_ancestral_information.json", emit: information
    path "m27c_spatial_diagnostics.json", emit: spatial
    path "m27c_operational_metrics.json", emit: operations
    path "m27c_readiness_by_policy.tsv", emit: policies
    path "m27c_window_summary.tsv", emit: windows
    path "m27c_marker_summary.tsv.gz", emit: markers
    path "m27c_gates.tsv", emit: gates
    path "m27c_summary.json", emit: summary
    path "m27c_target_positions.bed", emit: regions
    path "m27c_targeted_gvcf_readiness.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${audit_py} \
      --gvcfs ${gvcfs.join(' ')} \
      --gvcf-indexes ${gvcf_indexes.join(' ')} \
      --gcs-input-manifest ${gcs_input_manifest} \
      --phased-scaffold-vcf ${phased_scaffold_vcf} \
      --gnomix-reference-vcf ${gnomix_reference_vcf} \
      --gnomix-config ${gnomix_config} \
      --metadata ${metadata} \
      --reference-fasta ${reference_fasta} \
      --reference-fai ${reference_fai} \
      --preregistration ${preregistration} \
      --readers ${params.targeted_gvcf_readers} \
      --outdir .

    python3 ${manifest_py} \
      --stage M27C_TARGETED_GVCF_READINESS \
      --input ${gcs_input_manifest} \
      --input ${phased_scaffold_vcf} \
      --input ${gnomix_reference_vcf} \
      --input ${gnomix_config} \
      --input ${metadata} \
      --input ${preregistration} \
      --input ${audit_py} \
      --input ${core_py} \
      --input ${bridge_py} \
      --output m27c_input_contract.json \
      --output m27c_identity_control.json \
      --output m27c_callability_summary.json \
      --output m27c_ancestral_information.json \
      --output m27c_spatial_diagnostics.json \
      --output m27c_operational_metrics.json \
      --output m27c_readiness_by_policy.tsv \
      --output m27c_window_summary.tsv \
      --output m27c_marker_summary.tsv.gz \
      --output m27c_gates.tsv \
      --output m27c_summary.json \
      --output m27c_target_positions.bed \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"chr22_targeted_gvcf_callability_compatibility_phase_and_information_audit","chromosome":"22","gvcf_inputs":"generation_size_checksum_from_input_manifest","whole_gvcf_hashing_or_staging":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27c_targeted_gvcf_readiness.manifest.json
    """
}
