nextflow.enable.dsl=2

process WRITE_M29_ROOT_B0_PROVENANCE {
    tag "m29_root_b0_provenance"
    publishDir params.m29_b0_results_dir, mode: 'copy', overwrite: false
    container params.m29_b0_sim_container
    containerOptions params.m29_b0_container_options
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    val provenance_b64

    output:
    path "run_provenance.json", emit: provenance

    script:
    """
    set -euo pipefail
    printf '%s' '${provenance_b64}' | base64 -d > run_provenance.json
    """
}

process SELECT_M29_ROOT_B0 {
    tag "m29_b0_select_${root_label}"
    publishDir params.m29_b0_results_dir, mode: 'copy', overwrite: false, saveAs: { name -> name.startsWith('root') ? name : null }
    container params.m29_b0_sim_container
    containerOptions params.m29_b0_container_options
    cpus 1
    memory '4 GB'
    time '60m'

    input:
    tuple val(root_label), val(root_seed), path(tree), path(pools), path(preflight_report), path(preflight_manifest), path(mosaic_events)
    path reproducibility
    path genetic_map
    path baseline_template
    path m28_contract
    path m28b_contract
    path production_contract
    path selector_py
    path m28b_py
    path m28b_generic_py
    path m28b_joint_py
    path m28b_marker_py
    path m28_py
    path manifest_py
    path run_provenance

    output:
    tuple val(root_label), val(root_seed), path(tree), path(pools), path(mosaic_events), path("${root_label}/selection/m29_b0_markers.tsv.gz"), path("${root_label}/selection/m29_b0_selection.public.json"), emit: selected
    path "${root_label}/selection/m29_b0_selection.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    mkdir -p ${root_label}
    python3 ${selector_py} \
      --root-seed ${root_seed} --tree-sequence ${tree} --pool-manifest ${pools} \
      --preflight-report ${preflight_report} --preflight-manifest ${preflight_manifest} \
      --preflight-reproducibility ${reproducibility} --genetic-map ${genetic_map} \
      --baseline-template ${baseline_template} --m28-preregistration ${m28_contract} \
      --m28b-contract ${m28b_contract} --production-contract ${production_contract} \
      --outdir ${root_label}/selection
    python3 ${manifest_py} --stage M29_ROOT_B0_SELECTION \
      --input ${tree} --input ${pools} --input ${preflight_report} --input ${preflight_manifest} \
      --input ${reproducibility} --input ${genetic_map} --input ${baseline_template} \
      --input ${m28_contract} --input ${m28b_contract} --input ${production_contract} \
      --input ${selector_py} --input ${m28b_py} --input ${m28b_generic_py} \
      --input ${m28b_joint_py} --input ${m28b_marker_py} --input ${m28_py} \
      --input ${run_provenance} \
      --output ${root_label}/selection/m29_b0_markers.tsv.gz \
      --output ${root_label}/selection/m29_b0_selection.public.json \
      --params-json '{"root_seed":${root_seed},"truth_accessed":false,"BR_BS_evaluated":false}' \
      --out ${root_label}/selection/m29_b0_selection.manifest.json
    """
}

process MATERIALIZE_M29_ROOT_B0 {
    tag "m29_b0_materialize_${root_label}"
    publishDir params.m29_b0_results_dir, mode: 'copy', overwrite: false, saveAs: { name -> name.startsWith('root') ? name : null }
    container params.m29_b0_sim_container
    containerOptions params.m29_b0_container_options
    cpus 2
    memory '4 GB'
    time '45m'

    input:
    tuple val(root_label), val(root_seed), path(tree), path(pools), path(mosaic_events), path(b0_markers), path(selection_report)
    path production_contract
    path adapter_py
    path materializer_py
    path manifest_py
    path run_provenance

    output:
    tuple val(root_label), val(root_seed), path("${root_label}/materialized/m28c_b0_reference.vcf.gz"), path("${root_label}/materialized/m28c_b0_target.vcf.gz"), path("${root_label}/materialized/m28c_b0_input_preflight.public.json"), emit: materialized
    path "${root_label}/materialized/m28c_b0_reference.sample_map.tsv", emit: sample_map
    path "${root_label}/materialized/m28c_b0_reference_pairs.private.tsv", emit: private_pairs
    path "${root_label}/materialized/m29_b0_materialization.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    mkdir -p ${root_label}
    python3 ${adapter_py} --root-seed ${root_seed} --tree-sequence ${tree} \
      --pool-manifest ${pools} --mosaic-events ${mosaic_events} --b0-markers ${b0_markers} \
      --selection-report ${selection_report} --production-contract ${production_contract} \
      --outdir ${root_label}/materialized
    python3 ${manifest_py} --stage M29_ROOT_B0_MATERIALIZATION \
      --input ${tree} --input ${pools} --input ${mosaic_events} --input ${b0_markers} \
      --input ${selection_report} --input ${production_contract} --input ${adapter_py} \
      --input ${materializer_py} --input ${run_provenance} \
      --output ${root_label}/materialized/m28c_b0_reference.vcf.gz \
      --output ${root_label}/materialized/m28c_b0_target.vcf.gz \
      --output ${root_label}/materialized/m28c_b0_reference.sample_map.tsv \
      --output ${root_label}/materialized/m28c_b0_reference_pairs.private.tsv \
      --output ${root_label}/materialized/m28c_b0_input_preflight.public.json \
      --params-json '{"root_seed":${root_seed},"truth_accessed":false,"training":false}' \
      --out ${root_label}/materialized/m29_b0_materialization.manifest.json
    chmod 600 ${root_label}/materialized/m28c_b0_reference_pairs.private.tsv
    """
}

process INGEST_M29_ROOT_B0 {
    tag "m29_b0_ingest_${root_label}"
    publishDir params.m29_b0_results_dir, mode: 'copy', overwrite: false
    container params.m29_b0_gnomix_container
    containerOptions params.m29_b0_container_options
    cpus 2
    memory '4 GB'
    time '30m'

    input:
    tuple val(root_label), val(root_seed), path(reference_vcf), path(target_vcf), path(materialization_report)
    path production_contract
    path adapter_py
    path ingest_py
    path manifest_py
    path run_provenance

    output:
    tuple val(root_label), val(root_seed), path("${root_label}/ingest/m28c_b0_reference.vcf.gz"), path("${root_label}/ingest/m28c_b0_reference.vcf.gz.tbi"), path("${root_label}/ingest/m28c_b0_target.vcf.gz"), path("${root_label}/ingest/m28c_b0_target.vcf.gz.tbi"), path("${root_label}/ingest/m28c_b0_gnomix_ingest.public.json"), path("${root_label}/ingest/m29_b0_ingest.manifest.json"), emit: ready

    script:
    """
    set -euo pipefail
    mkdir -p ${root_label}
    python3 ${adapter_py} --root-seed ${root_seed} --reference-vcf ${reference_vcf} \
      --target-vcf ${target_vcf} --materialization-report ${materialization_report} \
      --production-contract ${production_contract} --gnomix-root /opt/gnomix \
      --outdir ${root_label}/ingest
    python3 ${manifest_py} --stage M29_ROOT_B0_GNOMIX_INGEST \
      --input ${reference_vcf} --input ${target_vcf} --input ${materialization_report} \
      --input ${production_contract} --input ${adapter_py} --input ${ingest_py} \
      --input ${run_provenance} \
      --output ${root_label}/ingest/m28c_b0_reference.vcf.gz \
      --output ${root_label}/ingest/m28c_b0_reference.vcf.gz.tbi \
      --output ${root_label}/ingest/m28c_b0_target.vcf.gz \
      --output ${root_label}/ingest/m28c_b0_target.vcf.gz.tbi \
      --output ${root_label}/ingest/m28c_b0_gnomix_ingest.public.json \
      --params-json '{"root_seed":${root_seed},"truth_accessed":false,"training":false,"inference":false}' \
      --out ${root_label}/ingest/m29_b0_ingest.manifest.json
    """
}
