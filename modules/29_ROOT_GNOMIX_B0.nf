nextflow.enable.dsl=2

process WRITE_M29_ROOT_GNOMIX_PROVENANCE {
    tag 'm29_root_gnomix_provenance'
    publishDir params.m29_gnomix_results_dir, mode: 'copy', overwrite: false
    container params.m29_gnomix_container
    containerOptions params.m29_gnomix_container_options
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

process BIND_M29_ROOT_GNOMIX_CONTRACT {
    tag "m29_gnomix_bind_${root_label}"
    publishDir params.m29_gnomix_results_dir, mode: 'copy', overwrite: false, saveAs: { name -> name.startsWith('root') ? name : null }
    container params.m29_gnomix_container
    containerOptions params.m29_gnomix_container_options
    cpus 1
    memory '4 GB'
    time '20m'
    maxForks params.m29_gnomix_max_parallel_roots

    input:
    tuple val(root_label), val(root_seed), path(reference_vcf), path(reference_tbi), path(target_vcf), path(target_tbi), path(sample_map), path(b0_markers), path(selection_report), path(selection_manifest), path(materialization_report), path(materialization_manifest), path(ingest_report), path(ingest_manifest)
    path genetic_map
    path gnomix_config
    path preregistration
    path production_contract
    path template_contract
    path builder_py
    path runner_py

    output:
    tuple val(root_label), val(root_seed), path(reference_vcf), path(reference_tbi), path(target_vcf), path(target_tbi), path(sample_map), path(b0_markers), path("${root_label}/contract/m29_root_gnomix_runtime.contract.json"), emit: bound

    script:
    """
    set -euo pipefail
    mkdir -p ${root_label}/contract
    python3 ${builder_py} --root-label ${root_label} --root-seed ${root_seed} \
      --preregistration ${preregistration} --production-contract ${production_contract} \
      --template-contract ${template_contract} --selection-report ${selection_report} \
      --selection-manifest ${selection_manifest} --materialization-report ${materialization_report} \
      --materialization-manifest ${materialization_manifest} --ingest-report ${ingest_report} \
      --ingest-manifest ${ingest_manifest} --reference-vcf ${reference_vcf} \
      --reference-tbi ${reference_tbi} --target-vcf ${target_vcf} --target-tbi ${target_tbi} \
      --sample-map ${sample_map} --b0-markers ${b0_markers} --genetic-map ${genetic_map} \
      --gnomix-config ${gnomix_config} --runner ${runner_py} --gnomix-root /opt/gnomix \
      --out ${root_label}/contract/m29_root_gnomix_runtime.contract.json
    """
}

process VALIDATE_M29_ROOT_GNOMIX_B0 {
    tag "m29_gnomix_validate_${root_label}"
    publishDir params.m29_gnomix_results_dir, mode: 'copy', overwrite: false, saveAs: { name -> name.startsWith('root') ? name : null }
    container params.m29_gnomix_container
    containerOptions params.m29_gnomix_container_options
    cpus 4
    memory params.m29_gnomix_memory
    time params.m29_gnomix_validate_time
    maxForks params.m29_gnomix_max_parallel_roots

    input:
    tuple val(root_label), val(root_seed), path(reference_vcf), path(reference_tbi), path(target_vcf), path(target_tbi), path(sample_map), path(b0_markers), path(runtime_contract)
    path genetic_map
    path gnomix_config
    path runner_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    tuple val(root_label), val(root_seed), path(reference_vcf), path(sample_map), path(runtime_contract), path("${root_label}/validation/m28c_gnomix_full_b0_validate.public.json"), emit: training_ready
    tuple val(root_label), path(target_vcf), emit: targets_ready
    path "${root_label}/validation/m29_root_gnomix_validate.manifest.json", emit: manifests

    script:
    """
    set -euo pipefail
    mkdir -p ${root_label}
    python3 ${runner_py} validate-full --reference-vcf ${reference_vcf} --reference-tbi ${reference_tbi} \
      --target-vcf ${target_vcf} --target-tbi ${target_tbi} --sample-map ${sample_map} \
      --b0-markers ${b0_markers} --genetic-map ${genetic_map} --gnomix-config ${gnomix_config} \
      --preregistration ${runtime_contract} --outdir ${root_label}/validation
    python3 ${manifest_py} --stage M29_ROOT_GNOMIX_VALIDATE_${root_label} \
      --input ${reference_vcf} --input ${reference_tbi} --input ${target_vcf} --input ${target_tbi} \
      --input ${sample_map} --input ${b0_markers} --input ${genetic_map} --input ${gnomix_config} \
      --input ${runtime_contract} --input ${runner_py} --input ${run_provenance} \
      --output ${root_label}/validation/m28c_gnomix_full_b0_validate.public.json \
      --params-json '{"root":"${root_label}","root_seed":${root_seed},"truth_accessed":false}' \
      --provenance-b64 '${provenance_b64}' --run-provenance-ref ../../run_provenance.json \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --out ${root_label}/validation/m29_root_gnomix_validate.manifest.json
    """
}

process TRAIN_M29_ROOT_GNOMIX_B0 {
    tag "m29_gnomix_train_${root_label}"
    publishDir params.m29_gnomix_results_dir, mode: 'copy', overwrite: false, saveAs: { name -> name.startsWith('root') ? name : null }
    container params.m29_gnomix_container
    containerOptions params.m29_gnomix_container_options
    cpus 4
    memory params.m29_gnomix_memory
    time params.m29_gnomix_train_time
    maxForks params.m29_gnomix_max_parallel_roots

    input:
    tuple val(root_label), val(root_seed), path(reference_vcf), path(sample_map), path(runtime_contract), path(validation_report)
    path genetic_map
    path gnomix_config
    path runner_py
    path rss_guard_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    tuple val(root_label), val(root_seed), path("${root_label}/training"), path(runtime_contract), path(validation_report), path("${root_label}/training_rss_gate.json"), emit: trained

    script:
    """
    set -euo pipefail
    mkdir -p ${root_label}
    python3 ${rss_guard_py} --report ${root_label}/training_rss_gate.json \
      --max-rss-gib ${params.m29_gnomix_peak_rss_stop_gib} --poll-seconds 0.1 -- \
      python3 ${runner_py} train --reference-vcf ${reference_vcf} --sample-map ${sample_map} \
      --genetic-map ${genetic_map} --gnomix-config ${gnomix_config} --prepare-report ${validation_report} \
      --gnomix-root /opt/gnomix --replicate ${root_label} --preregistration ${runtime_contract} \
      --outdir ${root_label}/training
    python3 ${manifest_py} --stage M29_ROOT_GNOMIX_TRAIN_${root_label} \
      --input ${reference_vcf} --input ${sample_map} --input ${genetic_map} --input ${gnomix_config} \
      --input ${validation_report} --input ${runtime_contract} --input ${runner_py} --input ${rss_guard_py} \
      --input ${root_label}/training_rss_gate.json --input ${run_provenance} \
      --output ${root_label}/training/models/m28c_b0_anchor_chm_22/m28c_b0_anchor_chm_22.pkl \
      --output ${root_label}/training/m28c_gnomix_full_b0_train.public.json \
      --output ${root_label}/training_rss_gate.json \
      --params-json '{"root":"${root_label}","root_seed":${root_seed},"truth_accessed":false,"target_input_present":false,"memory_gib":8,"peak_rss_stop_gib":6.4}' \
      --provenance-b64 '${provenance_b64}' --run-provenance-ref ../../run_provenance.json \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --out ${root_label}/training/m29_root_gnomix_train.manifest.json
    """
}

process INFER_M29_ROOT_GNOMIX_B0 {
    tag "m29_gnomix_infer_${root_label}"
    publishDir params.m29_gnomix_results_dir, mode: 'copy', overwrite: false, saveAs: { name -> name.startsWith('root') ? name : null }
    container params.m29_gnomix_container
    containerOptions params.m29_gnomix_container_options
    cpus 4
    memory params.m29_gnomix_memory
    time params.m29_gnomix_infer_time
    maxForks params.m29_gnomix_max_parallel_roots

    input:
    tuple val(root_label), val(root_seed), path(training_dir), path(target_vcf), path(runtime_contract), path(validation_report), path(training_rss_gate)
    path gnomix_config
    path runner_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    tuple val(root_label), val(root_seed), path("${root_label}/inference/results/query_results.fb"), path("${root_label}/inference/results/query_results.msp"), path("${root_label}/inference/m28c_gnomix_full_b0_inference.public.json"), path("${root_label}/inference/m29_root_gnomix_inference.manifest.json"), path(runtime_contract), path(training_rss_gate), emit: predictions

    script:
    """
    set -euo pipefail
    mkdir -p ${root_label}
    python3 ${runner_py} infer --training-dir ${training_dir} \
      --train-report ${training_dir}/m28c_gnomix_full_b0_train.public.json \
      --target-vcf ${target_vcf} --prepare-report ${validation_report} --gnomix-config ${gnomix_config} \
      --gnomix-root /opt/gnomix --replicate ${root_label} --preregistration ${runtime_contract} \
      --outdir ${root_label}/inference
    python3 ${manifest_py} --stage M29_ROOT_GNOMIX_INFER_${root_label} \
      --input ${training_dir}/m28c_gnomix_full_b0_train.public.json \
      --input ${training_dir}/models/m28c_b0_anchor_chm_22/m28c_b0_anchor_chm_22.pkl \
      --input ${target_vcf} --input ${validation_report} --input ${training_rss_gate} --input ${gnomix_config} \
      --input ${runtime_contract} --input ${runner_py} --input ${run_provenance} \
      --output ${root_label}/inference/results/query_results.fb \
      --output ${root_label}/inference/results/query_results.msp \
      --output ${root_label}/inference/m28c_gnomix_full_b0_inference.public.json \
      --params-json '{"root":"${root_label}","root_seed":${root_seed},"truth_accessed":false,"target_truth_accuracy_computed":false}' \
      --provenance-b64 '${provenance_b64}' --run-provenance-ref ../../run_provenance.json \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --out ${root_label}/inference/m29_root_gnomix_inference.manifest.json
    """
}

process BIND_M29_ROOT_PREDICTIONS {
    tag "m29_prediction_binding_${root_label}"
    publishDir params.m29_gnomix_results_dir, mode: 'copy', overwrite: false
    container params.m29_gnomix_container
    containerOptions params.m29_gnomix_container_options
    cpus 1
    memory '1 GB'
    time '5m'
    maxForks params.m29_gnomix_max_parallel_roots

    input:
    tuple val(root_label), val(root_seed), path(fb), path(msp), path(inference_report), path(inference_manifest), path(runtime_contract), path(training_rss_gate)
    path binder_py
    path run_provenance

    output:
    tuple val(root_label), val(root_seed), path("${root_label}/binding/m29_b0_binding.json"), emit: bindings

    script:
    """
    set -euo pipefail
    mkdir -p ${root_label}/binding
    python3 ${binder_py} --root-seed ${root_seed} --fb ${fb} --msp ${msp} \
      --inference-report ${inference_report} --inference-manifest ${inference_manifest} \
      --runtime-contract ${runtime_contract} --training-rss-gate ${training_rss_gate} \
      --run-provenance ${run_provenance} \
      --out ${root_label}/binding/m29_b0_binding.json
    """
}
