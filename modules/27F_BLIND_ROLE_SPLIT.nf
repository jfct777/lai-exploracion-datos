nextflow.enable.dsl=2

process WRITE_M27F_SPLIT_RUN_PROVENANCE {
    tag "m27f_split_provenance"
    publishDir "${params.m27f_split_results_dir}", mode: 'copy', overwrite: false
    cpus params.m27f_split_cpus
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

process AUDIT_M27F_BLIND_ROLE_SPLIT {
    tag "m27f_blind_role_split"
    publishDir "${params.m27f_split_results_dir}", mode: 'copy', overwrite: false
    cpus 1
    memory params.m27f_split_memory
    time params.m27f_split_time

    input:
    path ibd_files
    path genetic_maps
    path resolved_strata
    path resolved_strata_manifest
    path upstream_m27e_manifest
    path preregistration
    path audit_py
    path m27e_py
    path manifest_py
    val provenance_b64

    output:
    path "m27f_split.private.tsv", emit: private_split
    path "m27f_split.public.json", emit: public_receipt
    path "m27f_split.manifest.json", emit: manifest

    script:
    def ibdArgs = ibd_files.collect { "--ibd-file ${it}" }.join(' ')
    def mapArgs = genetic_maps.collect { "--genetic-map ${it}" }.join(' ')
    def manifestInputs = (ibd_files + genetic_maps).collect { "--input ${it}" }.join(' ')
    """
    set -euo pipefail
    PYTHONPATH=. python3 ${audit_py} \
      ${ibdArgs} \
      ${mapArgs} \
      --resolved-strata ${resolved_strata} \
      --resolved-strata-manifest ${resolved_strata_manifest} \
      --upstream-m27e-manifest ${upstream_m27e_manifest} \
      --preregistration ${preregistration} \
      --outdir .

    chmod 600 m27f_split.private.tsv

    python3 ${manifest_py} \
      --stage M27F_BLIND_ROLE_SPLIT \
      ${manifestInputs} \
      --input ${resolved_strata} \
      --input ${resolved_strata_manifest} \
      --input ${upstream_m27e_manifest} \
      --input ${preregistration} \
      --input ${audit_py} \
      --input ${m27e_py} \
      --output m27f_split.private.tsv \
      --output m27f_split.public.json \
      --provenance-b64 ${provenance_b64} \
      --params-json '{"allocation":"balanced_atomic_units_with_exact_small_stratum_audit","vcf_inputs_declared":false,"genotypes_parsed":false,"rare_support_used":false,"source_test_opened":false}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../run_provenance.json \
      --out m27f_split.manifest.json
    """
}
