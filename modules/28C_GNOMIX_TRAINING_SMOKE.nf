nextflow.enable.dsl=2

process WRITE_M28C_GNOMIX_SMOKE_PROVENANCE {
    tag "m28c_gnomix_smoke_provenance"
    publishDir params.m28c_smoke_results_dir, mode: 'copy', overwrite: false
    container params.m28c_smoke_container_image
    containerOptions params.m28c_smoke_container_options
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

process PREPARE_M28C_GNOMIX_SMOKE {
    tag "m28c_gnomix_smoke_prepare"
    publishDir "${params.m28c_smoke_results_dir}/smoke", mode: 'copy', overwrite: false
    container params.m28c_smoke_container_image
    containerOptions params.m28c_smoke_container_options
    cpus params.m28c_smoke_prepare_cpus
    memory params.m28c_smoke_prepare_memory
    time params.m28c_smoke_prepare_time

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
    path smoke_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "prepared/m28c_b0_smoke_reference.vcf.gz", emit: reference
    path "prepared/m28c_b0_smoke_reference.vcf.gz.tbi", emit: reference_tbi
    path "prepared/m28c_b0_smoke_target.vcf.gz", emit: target
    path "prepared/m28c_b0_smoke_target.vcf.gz.tbi", emit: target_tbi
    path "prepared/m28c_gnomix_smoke_prepare.public.json", emit: report
    path "prepared/m28c_b0_smoke_markers.tsv", emit: markers
    path "prepared/m28c_b0_smoke_regions.tsv", emit: regions
    path "prepared/m28c_gnomix_smoke_prepare.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${smoke_py} prepare \
      --reference-vcf ${reference_vcf} \
      --reference-tbi ${reference_tbi} \
      --target-vcf ${target_vcf} \
      --target-tbi ${target_tbi} \
      --sample-map ${sample_map} \
      --b0-markers ${b0_markers} \
      --genetic-map ${genetic_map} \
      --gnomix-config ${gnomix_config} \
      --preregistration ${preregistration} \
      --outdir prepared

    python3 ${manifest_py} \
      --stage M28C_GNOMIX_TRAINING_SMOKE_PREPARE \
      --input ${reference_vcf} --input ${reference_tbi} \
      --input ${target_vcf} --input ${target_tbi} \
      --input ${sample_map} --input ${b0_markers} \
      --input ${genetic_map} --input ${gnomix_config} \
      --input ${preregistration} --input ${smoke_py} \
      --input ${run_provenance} \
      --output prepared/m28c_b0_smoke_reference.vcf.gz \
      --output prepared/m28c_b0_smoke_reference.vcf.gz.tbi \
      --output prepared/m28c_b0_smoke_target.vcf.gz \
      --output prepared/m28c_b0_smoke_target.vcf.gz.tbi \
      --output prepared/m28c_b0_smoke_markers.tsv \
      --output prepared/m28c_b0_smoke_regions.tsv \
      --output prepared/m28c_gnomix_smoke_prepare.public.json \
      --params-json '{"markers":10000,"bins":363,"truth_accessed":false}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../../run_provenance.json \
      --out prepared/m28c_gnomix_smoke_prepare.manifest.json
    """
}

process TRAIN_M28C_GNOMIX_SMOKE {
    tag "m28c_gnomix_smoke_train_${replicate}"
    publishDir "${params.m28c_smoke_results_dir}/smoke", mode: 'copy', overwrite: false
    container params.m28c_smoke_container_image
    containerOptions params.m28c_smoke_container_options
    cpus params.m28c_smoke_train_cpus
    memory params.m28c_smoke_train_memory
    time params.m28c_smoke_train_time

    input:
    val replicate
    path reference_vcf
    path sample_map
    path genetic_map
    path gnomix_config
    path preregistration
    path prepare_report
    path smoke_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    tuple val(replicate), path("training_${replicate}"), emit: bundle

    script:
    """
    set -euo pipefail
    python3 ${smoke_py} train \
      --reference-vcf ${reference_vcf} \
      --sample-map ${sample_map} \
      --genetic-map ${genetic_map} \
      --gnomix-config ${gnomix_config} \
      --prepare-report ${prepare_report} \
      --gnomix-root /opt/gnomix \
      --replicate ${replicate} \
      --preregistration ${preregistration} \
      --outdir training_${replicate}

    python3 ${manifest_py} \
      --stage M28C_GNOMIX_TRAINING_SMOKE_TRAIN_${replicate} \
      --input ${reference_vcf} --input ${sample_map} \
      --input ${genetic_map} --input ${gnomix_config} \
      --input ${preregistration} --input ${prepare_report} \
      --input ${smoke_py} --input ${run_provenance} \
      --output training_${replicate}/models/m28c_b0_smoke_chm_22/m28c_b0_smoke_chm_22.pkl \
      --output training_${replicate}/m28c_gnomix_smoke_train.public.json \
      --output training_${replicate}/gnomix_train.stdout.log \
      --output training_${replicate}/gnomix_train.stderr.log \
      --params-json '{"replicate":"${replicate}","seed":42,"truth_accessed":false,"target_truth_accuracy_computed":false,"internal_synthetic_validation_used_for_decision":false}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../../run_provenance.json \
      --out training_${replicate}/m28c_gnomix_smoke_train.manifest.json
    """
}

process INFER_M28C_GNOMIX_SMOKE {
    tag "m28c_gnomix_smoke_infer_${replicate}"
    publishDir "${params.m28c_smoke_results_dir}/smoke", mode: 'copy', overwrite: false
    container params.m28c_smoke_container_image
    containerOptions params.m28c_smoke_container_options
    cpus params.m28c_smoke_infer_cpus
    memory params.m28c_smoke_infer_memory
    time params.m28c_smoke_infer_time

    input:
    tuple val(replicate), path(training_dir)
    path target_vcf
    path prepare_report
    path gnomix_config
    path preregistration
    path smoke_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    tuple val(replicate), path("inference_${replicate}"), emit: bundle

    script:
    """
    set -euo pipefail
    python3 ${smoke_py} infer \
      --training-dir ${training_dir} \
      --train-report ${training_dir}/m28c_gnomix_smoke_train.public.json \
      --target-vcf ${target_vcf} \
      --prepare-report ${prepare_report} \
      --gnomix-config ${gnomix_config} \
      --gnomix-root /opt/gnomix \
      --replicate ${replicate} \
      --preregistration ${preregistration} \
      --outdir inference_${replicate}

    python3 ${manifest_py} \
      --stage M28C_GNOMIX_TRAINING_SMOKE_INFER_${replicate} \
      --input ${training_dir}/m28c_gnomix_smoke_train.public.json \
      --input ${training_dir}/models/m28c_b0_smoke_chm_22/m28c_b0_smoke_chm_22.pkl \
      --input ${target_vcf} --input ${prepare_report} \
      --input ${gnomix_config} --input ${preregistration} \
      --input ${smoke_py} --input ${run_provenance} \
      --output inference_${replicate}/results/query_results.msp \
      --output inference_${replicate}/results/query_results.fb \
      --output inference_${replicate}/m28c_gnomix_smoke_inference.public.json \
      --params-json '{"replicate":"${replicate}","truth_accessed":false,"target_truth_accuracy_computed":false}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../../run_provenance.json \
      --out inference_${replicate}/m28c_gnomix_smoke_inference.manifest.json
    """
}

process COMPARE_M28C_GNOMIX_SMOKE {
    tag "m28c_gnomix_smoke_compare"
    publishDir "${params.m28c_smoke_results_dir}/smoke", mode: 'copy', overwrite: false
    container params.m28c_smoke_container_image
    containerOptions params.m28c_smoke_container_options
    cpus params.m28c_smoke_compare_cpus
    memory params.m28c_smoke_compare_memory
    time params.m28c_smoke_compare_time

    input:
    path inference_a
    path inference_b
    path preregistration
    path smoke_py
    path manifest_py
    path run_provenance
    val provenance_b64

    output:
    path "comparison/m28c_gnomix_smoke_compare.public.json", emit: report
    path "comparison/m28c_gnomix_smoke_compare.manifest.json", emit: manifest

    script:
    """
    set -euo pipefail
    python3 ${smoke_py} compare \
      --inference-a ${inference_a} \
      --report-a ${inference_a}/m28c_gnomix_smoke_inference.public.json \
      --inference-b ${inference_b} \
      --report-b ${inference_b}/m28c_gnomix_smoke_inference.public.json \
      --preregistration ${preregistration} \
      --outdir comparison

    python3 ${manifest_py} \
      --stage M28C_GNOMIX_TRAINING_SMOKE_COMPARE \
      --input ${inference_a}/m28c_gnomix_smoke_inference.public.json \
      --input ${inference_b}/m28c_gnomix_smoke_inference.public.json \
      --input ${preregistration} --input ${smoke_py} --input ${run_provenance} \
      --output comparison/m28c_gnomix_smoke_compare.public.json \
      --params-json '{"replicates":["A","B"],"truth_accessed":false,"target_truth_accuracy_computed":false}' \
      --provenance-b64 '${provenance_b64}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref ../../run_provenance.json \
      --out comparison/m28c_gnomix_smoke_compare.manifest.json
    """
}
