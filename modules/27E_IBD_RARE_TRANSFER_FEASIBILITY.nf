nextflow.enable.dsl=2

process WRITE_IBD_RARE_TRANSFER_RUN_PROVENANCE {
    tag "m27e_ibd_rare_transfer_provenance"
    publishDir "${params.ibd_rare_transfer_results_dir}", mode: 'copy', overwrite: false
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

process AUDIT_IBD_RARE_TRANSFER_FEASIBILITY {
    tag "m27e_autosomal_ibd_chr22_rare_transfer"
    publishDir "${params.ibd_rare_transfer_results_dir}/audit", mode: 'copy', overwrite: false
    cpus params.ibd_rare_transfer_cpus
    memory params.ibd_rare_transfer_memory
    time params.ibd_rare_transfer_time

    input:
    path ibd_files
    path ibd_logs
    path genetic_maps
    path raw_wgs_vcf
    path phased_panel_vcf
    path gnomix_reference_vcf
    path resolved_strata
    path resolved_strata_manifest
    path preregistration
    path audit_py
    path bridge_py
    path manifest_py
    val provenance_b64

    output:
    path "m27e_input_contract.json", emit: input_contract
    path "m27e_ibd_relatedness_summary.json", emit: relatedness
    path "m27e_rare_transfer_support.json", emit: rare_support
    path "m27e_gates.tsv", emit: gates
    path "m27e_summary.json", emit: summary
    path "m27e_ibd_rare_transfer.manifest.json", emit: manifest

    script:
    def ibdArgs = ibd_files.collect { "--ibd-file ${it}" }.join(' \\\n+      ')
    def logArgs = ibd_logs.collect { "--ibd-log ${it}" }.join(' \\\n+      ')
    def mapArgs = genetic_maps.collect { "--genetic-map ${it}" }.join(' \\\n+      ')
    def manifestInputs = (ibd_files + ibd_logs + genetic_maps).collect { "--input ${it}" }.join(' \\\n+      ')
    """
    set -euo pipefail
    PYTHONPATH=. python3 ${audit_py} \
      ${ibdArgs} \
      ${logArgs} \
      ${mapArgs} \
      --raw-wgs-vcf ${raw_wgs_vcf} \
      --phased-panel-vcf ${phased_panel_vcf} \
      --gnomix-reference-vcf ${gnomix_reference_vcf} \
      --resolved-strata ${resolved_strata} \
      --resolved-strata-manifest ${resolved_strata_manifest} \
      --preregistration ${preregistration} \
      --outdir .

    python3 ${manifest_py} \
      --stage M27E_IBD_RARE_TRANSFER_FEASIBILITY \
      ${manifestInputs} \
      --input ${raw_wgs_vcf} \
      --input ${phased_panel_vcf} \
      --input ${gnomix_reference_vcf} \
      --input ${resolved_strata} \
      --input ${resolved_strata_manifest} \
      --input ${preregistration} \
      --input ${audit_py} \
      --input ${bridge_py} \
      --output m27e_input_contract.json \
      --output m27e_ibd_relatedness_summary.json \
      --output m27e_rare_transfer_support.json \
      --output m27e_gates.tsv \
      --output m27e_summary.json \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"autosomal_ibd_chr22_rare_transfer_feasibility","simulation":false,"lai":false,"training":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27e_ibd_rare_transfer.manifest.json
    """
}
