nextflow.enable.dsl=2

process WRITE_M28D_B0_SCORING_PROVENANCE {
    tag "m28d_b0_scoring_provenance"
    publishDir params.m28d_results_dir, mode: 'copy', overwrite: false
    container params.m28d_container_image
    containerOptions params.m28d_container_options
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

process VALIDATE_M28D_B0_SCORER {
    tag "m28d_b0_known_answers"
    publishDir "${params.m28d_results_dir}/known_answers", mode: 'copy', overwrite: false
    container params.m28d_container_image
    containerOptions params.m28d_container_options
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    path scorer_py
    path known_answers_py
    path unit_test_py
    path preregistration
    path run_provenance

    output:
    path "m28d_b0_known_answers.public.json", emit: receipt

    script:
    """
    set -euo pipefail
    python3 ${known_answers_py} \
      --scorer ${scorer_py} \
      --contract ${preregistration} \
      --unit-test-file ${unit_test_py} \
      --output m28d_b0_known_answers.public.json
    """
}

process AUTHENTICATE_M28D_B0_PAIR {
    tag "m28d_b0_pair_authentication"
    publishDir "${params.m28d_results_dir}/authentication", mode: 'copy', overwrite: false
    container params.m28d_container_image
    containerOptions params.m28d_container_options
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    path known_answer_receipt
    path truth
    path b0_markers
    path genetic_map
    path fb_A, stageAs: 'replicate_A/query_results.fb'
    path msp_A, stageAs: 'replicate_A/query_results.msp'
    path fb_B, stageAs: 'replicate_B/query_results.fb'
    path msp_B, stageAs: 'replicate_B/query_results.msp'
    path m28c_comparison
    path simulation_manifest
    path b0_preflight_manifest
    path ingest_report
    path inference_manifest_A, stageAs: 'replicate_A/m28c_gnomix_full_b0_inference.manifest.json'
    path inference_manifest_B, stageAs: 'replicate_B/m28c_gnomix_full_b0_inference.manifest.json'
    path preregistration
    path scorer_py

    output:
    path "m28d_b0_pair_authentication.public.json", emit: receipt

    script:
    """
    set -euo pipefail
    python3 ${scorer_py} authenticate-pair \
      --contract ${preregistration} \
      --truth ${truth} \
      --b0-markers ${b0_markers} \
      --genetic-map ${genetic_map} \
      --fb-a ${fb_A} --msp-a ${msp_A} \
      --fb-b ${fb_B} --msp-b ${msp_B} \
      --m28c-comparison ${m28c_comparison} \
      --simulation-manifest ${simulation_manifest} \
      --b0-preflight-manifest ${b0_preflight_manifest} \
      --ingest-report ${ingest_report} \
      --inference-manifest-a ${inference_manifest_A} \
      --inference-manifest-b ${inference_manifest_B} \
      --known-answer-receipt ${known_answer_receipt} \
      --output m28d_b0_pair_authentication.public.json
    """
}

process SCORE_M28D_B0_PAIR {
    tag "m28d_b0_score_A_B"
    publishDir "${params.m28d_results_dir}/scoring", mode: 'copy', overwrite: false
    container params.m28d_container_image
    containerOptions params.m28d_container_options
    cpus params.m28d_score_cpus
    memory params.m28d_score_memory
    time params.m28d_score_time

    input:
    path known_answer_receipt
    path pair_auth_receipt
    path truth
    path b0_markers
    path genetic_map
    path fb_A, stageAs: 'replicate_A/query_results.fb'
    path msp_A, stageAs: 'replicate_A/query_results.msp'
    path fb_B, stageAs: 'replicate_B/query_results.fb'
    path msp_B, stageAs: 'replicate_B/query_results.msp'
    path m28c_comparison
    path simulation_manifest
    path b0_preflight_manifest
    path ingest_report
    path inference_manifest_A, stageAs: 'replicate_A/m28c_gnomix_full_b0_inference.manifest.json'
    path inference_manifest_B, stageAs: 'replicate_B/m28c_gnomix_full_b0_inference.manifest.json'
    path preregistration
    path scorer_py
    path run_provenance

    output:
    path "m28d_b0_score_A.public.json", emit: score_A
    path "m28d_b0_score_A.manifest.json", emit: manifest_A
    path "m28d_b0_score_B.public.json", emit: score_B
    path "m28d_b0_score_B.manifest.json", emit: manifest_B
    path "m28d_b0_score_compare.public.json", emit: comparison
    path "m28d_b0_score_compare.manifest.json", emit: comparison_manifest

    script:
    """
    set -euo pipefail
    python3 ${scorer_py} score \
      --contract ${preregistration} \
      --truth ${truth} \
      --b0-markers ${b0_markers} \
      --genetic-map ${genetic_map} \
      --fb ${fb_A} --msp ${msp_A} \
      --m28c-comparison ${m28c_comparison} \
      --simulation-manifest ${simulation_manifest} \
      --b0-preflight-manifest ${b0_preflight_manifest} \
      --ingest-report ${ingest_report} \
      --inference-manifest ${inference_manifest_A} \
      --known-answer-receipt ${known_answer_receipt} \
      --pair-auth-receipt ${pair_auth_receipt} \
      --run-provenance ${run_provenance} \
      --replicate A \
      --output m28d_b0_score_A.public.json \
      --manifest m28d_b0_score_A.manifest.json

    python3 ${scorer_py} score \
      --contract ${preregistration} \
      --truth ${truth} \
      --b0-markers ${b0_markers} \
      --genetic-map ${genetic_map} \
      --fb ${fb_B} --msp ${msp_B} \
      --m28c-comparison ${m28c_comparison} \
      --simulation-manifest ${simulation_manifest} \
      --b0-preflight-manifest ${b0_preflight_manifest} \
      --ingest-report ${ingest_report} \
      --inference-manifest ${inference_manifest_B} \
      --known-answer-receipt ${known_answer_receipt} \
      --pair-auth-receipt ${pair_auth_receipt} \
      --run-provenance ${run_provenance} \
      --replicate B \
      --output m28d_b0_score_B.public.json \
      --manifest m28d_b0_score_B.manifest.json

    python3 ${scorer_py} compare \
      --score-a m28d_b0_score_A.public.json \
      --score-b m28d_b0_score_B.public.json \
      --output m28d_b0_score_compare.public.json \
      --manifest m28d_b0_score_compare.manifest.json
    """
}
