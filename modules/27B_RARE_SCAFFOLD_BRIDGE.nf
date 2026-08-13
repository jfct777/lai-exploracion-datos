nextflow.enable.dsl=2

process WRITE_RARE_SCAFFOLD_BRIDGE_RUN_PROVENANCE {
    tag "rare_scaffold_bridge_provenance"
    publishDir "${params.rare_scaffold_bridge_results_dir}", mode: 'copy', overwrite: false
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

process AUDIT_RARE_SCAFFOLD_BRIDGE {
    tag "chr22_raw_wgs_to_phased_scaffold_bridge"
    publishDir "${params.rare_scaffold_bridge_results_dir}/audit", mode: 'copy', overwrite: false
    cpus params.rare_scaffold_bridge_cpus
    memory params.rare_scaffold_bridge_memory
    time params.rare_scaffold_bridge_time

    input:
    path raw_wgs_vcf
    path phased_scaffold_vcf
    path gnomix_reference_vcf
    path preregistration
    path audit_py
    path manifest_py
    val provenance_b64

    output:
    path "m27b_input_contract.json", emit: input_contract
    path "m27b_sample_identity.json", emit: sample_identity
    path "m27b_rare_support.json", emit: rare_support
    path "m27b_phase_bridge.json", emit: phase_bridge
    path "m27b_baseline_overlap.json", emit: baseline_overlap
    path "m27b_rare_scaffold_bridge_gates.tsv", emit: gates
    path "m27b_rare_scaffold_bridge_summary.json", emit: summary
    path "m27b_rare_scaffold_bridge.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${audit_py} \\
      --raw-wgs-vcf ${raw_wgs_vcf} \\
      --phased-scaffold-vcf ${phased_scaffold_vcf} \\
      --gnomix-reference-vcf ${gnomix_reference_vcf} \\
      --preregistration ${preregistration} \\
      --outdir .

    python3 ${manifest_py} \\
      --stage M27B_RARE_SCAFFOLD_BRIDGE \\
      --input ${raw_wgs_vcf} \\
      --input ${phased_scaffold_vcf} \\
      --input ${gnomix_reference_vcf} \\
      --input ${preregistration} \\
      --input ${audit_py} \\
      --output m27b_input_contract.json \\
      --output m27b_sample_identity.json \\
      --output m27b_rare_support.json \\
      --output m27b_phase_bridge.json \\
      --output m27b_baseline_overlap.json \\
      --output m27b_rare_scaffold_bridge_gates.tsv \\
      --output m27b_rare_scaffold_bridge_summary.json \\
      --provenance-b64 ${provenance_b64} \\
      --params-json '{"scope":"chr22_read_only_raw_wgs_to_phased_scaffold_bridge_audit","chromosome":"22","build":"hg38","rare_orientation":"minor_allele_from_GT","rare_mac_min":2,"rare_maf_lt":0.01,"identity_concordance_min":0.99,"frozen_baseline_marker_fraction_min":0.8}' \\
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \\
      --run-provenance-ref ../run_provenance.json \\
      --out m27b_rare_scaffold_bridge.manifest.json
    """
}
