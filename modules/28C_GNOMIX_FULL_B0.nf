nextflow.enable.dsl=2

process WRITE_M28C_GNOMIX_FULL_B0_PROVENANCE {
    tag "m28c_gnomix_full_b0_provenance_${params.m28c_full_replicate}"
    publishDir params.m28c_full_results_dir, mode: 'copy', overwrite: false
    container params.m28c_full_container_image
    containerOptions params.m28c_full_container_options
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

process VALIDATE_M28C_GNOMIX_FULL_B0 {
    tag "m28c_gnomix_full_b0_validate_${params.m28c_full_replicate}"
    publishDir "${params.m28c_full_results_dir}/full_b0", mode: 'copy', overwrite: false
    container params.m28c_full_container_image
    containerOptions params.m28c_full_container_options
    cpus params.m28c_full_validate_cpus
    memory params.m28c_full_validate_memory
    time params.m28c_full_validate_time

    input:
    path reference_vcf
    path reference_tbi
    path target_vcf
    path target_tbi
    path sample_map
    path b0_markers
    path genetic_map
    path gnomix_config
    path preregistration
    path runner_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "validation/m28c_gnomix_full_b0_validate.public.json", emit: report
    path "validation/m28c_gnomix_full_b0_validate.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${runner_py} validate-full \
      --reference-vcf ${reference_vcf} --reference-tbi ${reference_tbi} \
      --target-vcf ${target_vcf} --target-tbi ${target_tbi} \
      --sample-map ${sample_map} --b0-markers ${b0_markers} \
      --genetic-map ${genetic_map} --gnomix-config ${gnomix_config} \
      --preregistration ${preregistration} --outdir validation

    python3 ${manifest_py} \
      --stage M28C_GNOMIX_FULL_B0_VALIDATE \
      --input ${reference_vcf} --input ${reference_tbi} \
      --input ${target_vcf} --input ${target_tbi} \
      --input ${sample_map} --input ${b0_markers} --input ${genetic_map} \
      --input ${gnomix_config} --input ${preregistration} --input ${runner_py} \
      --input ${run_provenance} \
      --output validation/m28c_gnomix_full_b0_validate.public.json \
      --params-json '{"markers":79791,"M":215,"W":371,"remainder":26,"truth_accessed":false}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../../run_provenance.json \
      --out validation/m28c_gnomix_full_b0_validate.manifest.json
    """
}

process TRAIN_M28C_GNOMIX_FULL_B0 {
    tag "m28c_gnomix_full_b0_train_${replicate}"
    publishDir "${params.m28c_full_results_dir}/full_b0", mode: 'copy', overwrite: false
    container params.m28c_full_container_image
    containerOptions params.m28c_full_container_options
    cpus params.m28c_full_train_cpus
    memory params.m28c_full_train_memory
    time params.m28c_full_train_time

    input:
    val replicate
    path reference_vcf
    path sample_map
    path genetic_map
    path gnomix_config
    path preregistration
    path validation_report
    path runner_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    tuple val(replicate), path("training_${replicate}"), emit: bundle

    script:
    """
    set -euo pipefail
    python3 ${runner_py} train \
      --reference-vcf ${reference_vcf} --sample-map ${sample_map} \
      --genetic-map ${genetic_map} --gnomix-config ${gnomix_config} \
      --prepare-report ${validation_report} --gnomix-root /opt/gnomix \
      --replicate ${replicate} --preregistration ${preregistration} \
      --outdir training_${replicate}

    python3 ${manifest_py} \
      --stage M28C_GNOMIX_FULL_B0_TRAIN_${replicate} \
      --input ${reference_vcf} --input ${sample_map} --input ${genetic_map} \
      --input ${gnomix_config} --input ${preregistration} --input ${validation_report} \
      --input ${runner_py} --input ${run_provenance} \
      --output training_${replicate}/models/m28c_b0_anchor_chm_22/m28c_b0_anchor_chm_22.pkl \
      --output training_${replicate}/m28c_gnomix_full_b0_train.public.json \
      --params-json '{"replicate":"${replicate}","seed":42,"target_input_present":false,"truth_accessed":false}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../../run_provenance.json \
      --out training_${replicate}/m28c_gnomix_full_b0_train.manifest.json
    """
}

process INFER_M28C_GNOMIX_FULL_B0 {
    tag "m28c_gnomix_full_b0_infer_${replicate}"
    publishDir "${params.m28c_full_results_dir}/full_b0", mode: 'copy', overwrite: false
    container params.m28c_full_container_image
    containerOptions params.m28c_full_container_options
    cpus params.m28c_full_infer_cpus
    memory params.m28c_full_infer_memory
    time params.m28c_full_infer_time

    input:
    tuple val(replicate), path(training_dir)
    path target_vcf
    path validation_report
    path gnomix_config
    path preregistration
    path runner_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    tuple val(replicate), path("inference_${replicate}"), emit: bundle

    script:
    """
    set -euo pipefail
    python3 ${runner_py} infer \
      --training-dir ${training_dir} \
      --train-report ${training_dir}/m28c_gnomix_full_b0_train.public.json \
      --target-vcf ${target_vcf} --prepare-report ${validation_report} \
      --gnomix-config ${gnomix_config} --gnomix-root /opt/gnomix \
      --replicate ${replicate} --preregistration ${preregistration} \
      --outdir inference_${replicate}

    python3 ${manifest_py} \
      --stage M28C_GNOMIX_FULL_B0_INFER_${replicate} \
      --input ${training_dir}/m28c_gnomix_full_b0_train.public.json \
      --input ${training_dir}/models/m28c_b0_anchor_chm_22/m28c_b0_anchor_chm_22.pkl \
      --input ${target_vcf} --input ${validation_report} --input ${gnomix_config} \
      --input ${preregistration} --input ${runner_py} --input ${run_provenance} \
      --output inference_${replicate}/results/query_results.msp \
      --output inference_${replicate}/results/query_results.fb \
      --output inference_${replicate}/m28c_gnomix_full_b0_inference.public.json \
      --params-json '{"replicate":"${replicate}","truth_accessed":false,"target_truth_accuracy_computed":false}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../../run_provenance.json \
      --out inference_${replicate}/m28c_gnomix_full_b0_inference.manifest.json
    """
}
