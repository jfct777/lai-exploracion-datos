nextflow.enable.dsl=2

process WRITE_M30_FLARE_PROVENANCE {
    tag 'm30_flare_provenance'
    publishDir params.m30_results_dir, mode: 'copy', overwrite: false
    container params.m30_container_image
    cpus 1
    memory '1 GB'
    time '5m'

    input:
    val provenance_b64

    output:
    path 'run_provenance.json', emit: provenance

    script:
    """
    set -euo pipefail
    printf '%s' '${provenance_b64}' | base64 -d > run_provenance.json
    """
}

process M30_SCORER_KNOWN_ANSWERS {
    tag 'm30_scorer_known_answers_before_inference'
    publishDir params.m30_results_dir, mode: 'copy', overwrite: false
    container params.m30_container_image
    cpus 1
    memory '2 GB'
    time '10m'

    input:
    path scoring_contract
    path scorer_py
    path base_scorer_py
    path run_provenance

    output:
    path 'scoring/m30_scorer_known_answers.json', emit: receipt

    script:
    """
    set -euo pipefail
    mkdir -p scoring
    python3 ${scorer_py} known-answers --contract ${scoring_contract} --base-scorer ${base_scorer_py} \
      --output scoring/m30_scorer_known_answers.json
    """
}

process M30_PREFLIGHT_ROOT17 {
    tag 'm30_preflight_root17'
    publishDir params.m30_results_dir, mode: 'copy', overwrite: false,
        saveAs: { name -> name.startsWith('root17/') ? name : null }
    container params.m30_container_image
    cpus 1
    memory '4 GB'
    time params.m30_preflight_time

    input:
    tuple val(root_label), val(root_seed), path(reference_vcf), path(reference_tbi),
        path(target_vcf), path(target_tbi), path(sample_map), path(gnomix_binding),
        path(gnomix_fb), path(gnomix_msp)
    path genetic_map
    path preregistration
    path runner_py

    output:
    tuple path('root17/preflight/root17.m30.run_contract.json'),
        path('root17/preflight/root17.flare.ref-panel.tsv'),
        path('root17/preflight/root17.flare.map'),
        path('root17/preflight/root17.m30.preflight.json'), emit: prepared

    script:
    """
    set -euo pipefail
    python3 ${runner_py} preflight \
      --root-label ${root_label} --root-seed ${root_seed} \
      --preregistration ${preregistration} \
      --container-image '${params.m30_container_image}' --container-digest '${params.m30_container_digest}' \
      --flare-jar-sha256 '${params.m30_flare_jar_sha256}' \
      --reference-vcf ${reference_vcf} --reference-tbi ${reference_tbi} \
      --target-vcf ${target_vcf} --target-tbi ${target_tbi} \
      --sample-map ${sample_map} --genetic-map ${genetic_map} \
      --gnomix-binding ${gnomix_binding} --gnomix-fb ${gnomix_fb} --gnomix-msp ${gnomix_msp} \
      --outdir root17/preflight
    """
}

process M30_RUN_FLARE_ROOT17 {
    tag 'm30_flare_smoke_root17'
    publishDir params.m30_results_dir, mode: 'copy', overwrite: false,
        saveAs: { name -> name.startsWith('root17/') ? name : null }
    container params.m30_container_image
    cpus params.m30_cpus
    memory params.m30_memory
    time params.m30_inference_time

    input:
    tuple path(runtime_contract), path(panel_map), path(genetic_map), path(preflight_report)
    path reference_vcf
    path target_vcf
    path flare_jar
    path runner_py
    path run_provenance
    path scorer_receipt
    path scoring_contract
    path scorer_py

    output:
    path 'root17/flare/root17.m30.flare_audit.json', emit: audit
    tuple path('root17/flare/root17.flare.anc.vcf.gz'),
        path('root17/flare/root17.flare.global.anc.gz'),
        path('root17/flare/root17.flare.model'),
        path('root17/flare/root17.flare.log'), emit: predictions

    script:
    """
    set -euo pipefail
    python3 ${runner_py} run --root-label root17 \
      --runtime-contract ${runtime_contract} \
      --preflight-report ${preflight_report} --run-provenance ${run_provenance} \
      --scorer-receipt ${scorer_receipt} --scoring-contract ${scoring_contract} --scorer ${scorer_py} \
      --reference-vcf ${reference_vcf} --target-vcf ${target_vcf} \
      --panel-map ${panel_map} --genetic-map ${genetic_map} \
      --flare-jar ${flare_jar} --java java --ancestry-order AFR EUR ASIA \
      --outdir root17/flare
    """
}

process M30_PREFLIGHT_ROOT18 {
    tag 'm30_preflight_root18_after_root17_pass'
    publishDir params.m30_results_dir, mode: 'copy', overwrite: false,
        saveAs: { name -> name.startsWith('root18/') ? name : null }
    container params.m30_container_image
    cpus 1
    memory '4 GB'
    time params.m30_preflight_time

    input:
    tuple val(root_label), val(root_seed), path(reference_vcf), path(reference_tbi),
        path(target_vcf), path(target_tbi), path(sample_map), path(gnomix_binding),
        path(gnomix_fb), path(gnomix_msp)
    path genetic_map
    path preregistration
    path runner_py
    path root17_audit

    output:
    tuple path('root18/preflight/root18.m30.run_contract.json'),
        path('root18/preflight/root18.flare.ref-panel.tsv'),
        path('root18/preflight/root18.flare.map'),
        path('root18/preflight/root18.m30.preflight.json'), emit: prepared

    script:
    """
    set -euo pipefail
    python3 ${runner_py} preflight \
      --root-label ${root_label} --root-seed ${root_seed} \
      --preregistration ${preregistration} \
      --container-image '${params.m30_container_image}' --container-digest '${params.m30_container_digest}' \
      --flare-jar-sha256 '${params.m30_flare_jar_sha256}' \
      --reference-vcf ${reference_vcf} --reference-tbi ${reference_tbi} \
      --target-vcf ${target_vcf} --target-tbi ${target_tbi} \
      --sample-map ${sample_map} --genetic-map ${genetic_map} \
      --gnomix-binding ${gnomix_binding} --gnomix-fb ${gnomix_fb} --gnomix-msp ${gnomix_msp} \
      --prior-root-audit ${root17_audit} --outdir root18/preflight
    """
}

process M30_RUN_FLARE_ROOT18 {
    tag 'm30_flare_root18'
    publishDir params.m30_results_dir, mode: 'copy', overwrite: false,
        saveAs: { name -> name.startsWith('root18/') ? name : null }
    container params.m30_container_image
    cpus params.m30_cpus
    memory params.m30_memory
    time params.m30_inference_time

    input:
    tuple path(runtime_contract), path(panel_map), path(genetic_map), path(preflight_report)
    path reference_vcf
    path target_vcf
    path flare_jar
    path runner_py
    path run_provenance
    path scorer_receipt
    path scoring_contract
    path scorer_py

    output:
    path 'root18/flare/root18.m30.flare_audit.json', emit: audit
    tuple path('root18/flare/root18.flare.anc.vcf.gz'),
        path('root18/flare/root18.flare.global.anc.gz'),
        path('root18/flare/root18.flare.model'),
        path('root18/flare/root18.flare.log'), emit: predictions

    script:
    """
    set -euo pipefail
    python3 ${runner_py} run --root-label root18 \
      --runtime-contract ${runtime_contract} \
      --preflight-report ${preflight_report} --run-provenance ${run_provenance} \
      --scorer-receipt ${scorer_receipt} --scoring-contract ${scoring_contract} --scorer ${scorer_py} \
      --reference-vcf ${reference_vcf} --target-vcf ${target_vcf} \
      --panel-map ${panel_map} --genetic-map ${genetic_map} \
      --flare-jar ${flare_jar} --java java --ancestry-order AFR EUR ASIA \
      --outdir root18/flare
    """
}

process M30_SCORE_FLARE_VS_GNOMIX {
    tag 'm30_score_flare_vs_gnomix_dev'
    publishDir params.m30_results_dir, mode: 'copy', overwrite: false,
        saveAs: { name -> name.startsWith('scoring/') ? name : null }
    container params.m30_container_image
    cpus params.m30_scoring_cpus
    memory params.m30_scoring_memory
    time params.m30_scoring_time

    input:
    tuple path(root17_runtime_contract), path(root17_panel), path(root17_map), path(root17_preflight_report)
    tuple path(root17_flare_vcf), path(root17_global), path(root17_model), path(root17_log)
    path root17_flare_audit
    tuple path(root18_runtime_contract), path(root18_panel), path(root18_map), path(root18_preflight_report)
    tuple path(root18_flare_vcf), path(root18_global), path(root18_model), path(root18_log)
    path root18_flare_audit
    path root17_truth, stageAs: 'inputs/root17/truth.tsv.gz'
    path root17_target_vcf, stageAs: 'inputs/root17/target.vcf.gz'
    path root17_gnomix_binding, stageAs: 'inputs/root17/gnomix_binding.json'
    path root17_gnomix_fb, stageAs: 'inputs/root17/gnomix.fb'
    path root17_gnomix_msp, stageAs: 'inputs/root17/gnomix.msp'
    path root18_truth, stageAs: 'inputs/root18/truth.tsv.gz'
    path root18_target_vcf, stageAs: 'inputs/root18/target.vcf.gz'
    path root18_gnomix_binding, stageAs: 'inputs/root18/gnomix_binding.json'
    path root18_gnomix_fb, stageAs: 'inputs/root18/gnomix.fb'
    path root18_gnomix_msp, stageAs: 'inputs/root18/gnomix.msp'
    path genetic_map, stageAs: 'inputs/genetic.map.chr22'
    path scoring_contract, stageAs: 'inputs/m30_scoring_contract.json'
    path scorer_py
    path base_scorer_py
    path known_answer_receipt
    path run_provenance

    output:
    path 'scoring/m30_flare_vs_gnomix_dev.json', emit: score
    path 'scoring/m30_flare_vs_gnomix_dev.manifest.json', emit: manifest

    script:
    """
    set -euo pipefail
    mkdir -p scoring
    python3 ${scorer_py} score-compare \
      --contract ${scoring_contract} --base-scorer ${base_scorer_py} \
      --genetic-map ${genetic_map} --known-answer-receipt ${known_answer_receipt} \
      --run-provenance ${run_provenance} \
      --root17-truth ${root17_truth} --root17-target-vcf ${root17_target_vcf} \
      --root17-gnomix-binding ${root17_gnomix_binding} \
      --root17-gnomix-fb ${root17_gnomix_fb} --root17-gnomix-msp ${root17_gnomix_msp} \
      --root17-flare-vcf ${root17_flare_vcf} --root17-flare-audit ${root17_flare_audit} \
      --root17-runtime-contract ${root17_runtime_contract} \
      --root17-preflight-report ${root17_preflight_report} \
      --root18-truth ${root18_truth} --root18-target-vcf ${root18_target_vcf} \
      --root18-gnomix-binding ${root18_gnomix_binding} \
      --root18-gnomix-fb ${root18_gnomix_fb} --root18-gnomix-msp ${root18_gnomix_msp} \
      --root18-flare-vcf ${root18_flare_vcf} --root18-flare-audit ${root18_flare_audit} \
      --root18-runtime-contract ${root18_runtime_contract} \
      --root18-preflight-report ${root18_preflight_report} \
      --output scoring/m30_flare_vs_gnomix_dev.json \
      --manifest scoring/m30_flare_vs_gnomix_dev.manifest.json
    """
}
