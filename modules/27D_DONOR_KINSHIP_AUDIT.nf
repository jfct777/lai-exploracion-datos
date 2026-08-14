nextflow.enable.dsl=2

process WRITE_DONOR_KINSHIP_RUN_PROVENANCE {
    tag "donor_kinship_provenance"
    publishDir "${params.donor_kinship_results_dir}", mode: 'copy', overwrite: false
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

process BENCHMARK_DONOR_KINSHIP_RESOURCES {
    tag "m27d_official_panel_resource_smoke"
    publishDir "${params.donor_kinship_results_dir}/resource_smoke", mode: 'copy', overwrite: false
    cpus params.donor_kinship_cpus
    memory params.donor_kinship_memory
    time params.donor_kinship_time

    input:
    path panel_vcfs
    path metadata
    path exclude_bed
    path preregistration
    path sample_strata_py
    path pcrelate_smoke_r
    path bridge_py
    path manifest_py
    val provenance_b64

    output:
    path "m27d_sample_strata_summary.json", emit: strata_summary
    path "m27d_resource_smoke.json", emit: summary
    path "m27d_resource_smoke.tsv", emit: benchmark
    path "m27d_marker_qc.tsv", emit: marker_qc
    path "private/m27d_sample_strata.private.tsv", emit: private_strata
    path "private/m27d_official_panel_autosomes.gds", emit: private_gds
    path "private/m27d_ld_pruned_anchor_snp_ids.rds", emit: private_anchor
    path "private/m27d_ld_pruned_strict_snp_ids.rds", emit: private_strict
    path "m27d_resource_smoke.manifest.json", emit: manifest

    script:
    def inputArgs = panel_vcfs.collect { "--input ${it}" }.join(' \\\n      ')
    """
    set -euo pipefail
    mkdir -p private

    PYTHONPATH=. python3 ${sample_strata_py} \
      --panel-vcf ${panel_vcfs[0]} \
      --metadata ${metadata} \
      --private-out m27d_sample_strata.private.tsv \
      --summary-out m27d_sample_strata_summary.json \
      --suppress-below 5

    Rscript ${pcrelate_smoke_r} \
      --panel-vcfs '${panel_vcfs.join(',')}' \
      --metadata-strata m27d_sample_strata.private.tsv \
      --exclude-bed ${exclude_bed} \
      --preregistration ${preregistration} \
      --thread-grid '${params.donor_kinship_thread_grid}' \
      --outdir .

    mv m27d_sample_strata.private.tsv private/
    mv m27d_official_panel_autosomes.gds private/
    mv m27d_ld_pruned_anchor_snp_ids.rds private/
    mv m27d_ld_pruned_strict_snp_ids.rds private/

    python3 ${manifest_py} \
      --stage M27D_DONOR_KINSHIP_RESOURCE_SMOKE \
      ${inputArgs} \
      --input ${metadata} \
      --input ${exclude_bed} \
      --input ${preregistration} \
      --input ${sample_strata_py} \
      --input ${pcrelate_smoke_r} \
      --input ${bridge_py} \
      --output m27d_sample_strata_summary.json \
      --output m27d_resource_smoke.json \
      --output m27d_resource_smoke.tsv \
      --output m27d_marker_qc.tsv \
      --output private/m27d_sample_strata.private.tsv \
      --output private/m27d_official_panel_autosomes.gds \
      --output private/m27d_ld_pruned_anchor_snp_ids.rds \
      --output private/m27d_ld_pruned_strict_snp_ids.rds \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"scope":"m27d_resource_smoke_only","pcrelate":"without_KING","scientific_result":false,"full_run_authorized":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27d_resource_smoke.manifest.json
    """
}
