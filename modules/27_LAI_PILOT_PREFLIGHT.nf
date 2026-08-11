nextflow.enable.dsl=2

process WRITE_LAI_PILOT_PREFLIGHT_RUN_PROVENANCE {
    tag "lai_pilot_preflight_provenance"
    publishDir "${params.lai_pilot_preflight_results_dir}", mode: 'copy', overwrite: false
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

process AUDIT_LAI_PILOT_PREFLIGHT {
    tag "chr22_lai_identifiability_preflight"
    publishDir "${params.lai_pilot_preflight_results_dir}/audit", mode: 'copy', overwrite: false
    cpus params.lai_pilot_preflight_cpus
    memory params.lai_pilot_preflight_memory
    time params.lai_pilot_preflight_time

    input:
    path gnomix_reference_vcf
    path external_panel_vcf
    path gnomix_model
    path gnomix_config
    path genetic_map
    path metadata
    path top95_nam
    path nam_unrelated_keep
    path preregistration
    path audit_py
    path manifest_py
    val provenance_b64

    output:
    path "g0_model_contract.json", emit: g0
    path "g1_donor_identity_and_parentals.json", emit: g1
    path "g2_marker_compatibility.json", emit: g2
    path "g3_wgs_rare_support.json", emit: g3
    path "g4_identifiability_power.json", emit: g4
    path "m27_preflight_gates.tsv", emit: gates
    path "m27_lai_pilot_preflight_summary.json", emit: summary
    path "m27_lai_pilot_preflight.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${audit_py} \
      --gnomix-reference-vcf ${gnomix_reference_vcf} \
      --external-panel-vcf ${external_panel_vcf} \
      --gnomix-model ${gnomix_model} \
      --gnomix-config ${gnomix_config} \
      --genetic-map ${genetic_map} \
      --metadata ${metadata} \
      --top95-nam ${top95_nam} \
      --nam-unrelated-keep ${nam_unrelated_keep} \
      --preregistration ${preregistration} \
      --outdir .

    python3 ${manifest_py} \
      --stage M27_LAI_PILOT_PREFLIGHT \
      --input ${gnomix_reference_vcf} --input ${external_panel_vcf} \
      --input ${gnomix_model} --input ${gnomix_config} --input ${genetic_map} \
      --input ${metadata} --input ${top95_nam} --input ${nam_unrelated_keep} \
      --input ${preregistration} --input ${audit_py} \
      --output g0_model_contract.json \
      --output g1_donor_identity_and_parentals.json \
      --output g2_marker_compatibility.json \
      --output g3_wgs_rare_support.json \
      --output g4_identifiability_power.json \
      --output m27_preflight_gates.tsv \
      --output m27_lai_pilot_preflight_summary.json \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"chr22_identifiability_preflight_no_simulation_no_training","chromosome":"22","build":"hg38","minimum_model_marker_fraction":0.8}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27_lai_pilot_preflight.manifest.json
    """
}
