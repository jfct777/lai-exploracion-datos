nextflow.enable.dsl=2

process WRITE_M28_RUN_PROVENANCE {
    tag "m28_run_provenance"
    publishDir "${params.m28_results_dir}", mode: 'copy', overwrite: false
    container params.m28_container_image
    containerOptions params.m28_container_options
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
process RUN_M28_SIMULATION_PREFLIGHT {
    tag { "m28_seed_${root_seed}" }
    publishDir { "${params.m28_results_dir}/seed-${root_seed}" }, mode: 'copy', overwrite: false
    container params.m28_container_image
    containerOptions params.m28_container_options
    cpus params.m28_cpus
    memory params.m28_memory
    time params.m28_time

    input:
    path genetic_map
    path preregistration
    path preflight_py
    path manifest_py
    path run_provenance
    val root_seed
    val provenance_b64

    output:
    path "m28/m28_sources.trees", emit: sources
    path "m28/m28_pools.private.tsv", emit: pools
    path "m28/m28_mosaic_events.private.tsv.gz", emit: mosaic_events
    path "m28/m28_lai_truth.tsv.gz", emit: truth
    path "m28/m28_rare_catalog.tsv.gz", emit: rare_catalog
    path "m28/m28_rare_haplotypes.tsv.gz", emit: rare_haplotypes
    path "m28/m28_preflight.public.json", emit: report
    path "m28/m28_preflight.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${preflight_py} \
      --genetic-map ${genetic_map} \
      --preregistration ${preregistration} \
      --root-seed ${root_seed} \
      --outdir m28

    python3 ${manifest_py} \
      --stage M28_LAI_SIMULATION_PREFLIGHT \
      --input ${genetic_map} \
      --input ${preregistration} \
      --input ${preflight_py} \
      --input ${run_provenance} \
      --output m28/m28_sources.trees \
      --output m28/m28_pools.private.tsv \
      --output m28/m28_mosaic_events.private.tsv.gz \
      --output m28/m28_lai_truth.tsv.gz \
      --output m28/m28_rare_catalog.tsv.gz \
      --output m28/m28_rare_haplotypes.tsv.gz \
      --output m28/m28_preflight.public.json \
      --params-json '{"root_seed":${root_seed},"scope":"technical_preflight_no_LAI"}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m28/m28_preflight.manifest.json
    chmod 600 m28/m28_pools.private.tsv m28/m28_mosaic_events.private.tsv.gz
    """
}
