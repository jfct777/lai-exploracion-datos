nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Module 25 — TRAIN-only descriptive rare-minor-allele window features
// ---------------------------------------------------------------------------

process WRITE_RARE_WINDOW_FEATURES_RUN_PROVENANCE {
    tag "rare_window_features_run_provenance"

    publishDir "${params.rare_window_features_results_dir}", mode: 'copy', overwrite: false

    cpus 1
    memory '1 GB'
    time '10m'

    input:
    val run_prov_b64

    output:
    path "run_provenance.json"

    script:
    """
    set -euo pipefail
    printf '%s' '${run_prov_b64}' | base64 -d > run_provenance.json
    """
}


process BUILD_RARE_WINDOW_FEATURES {
    tag "chr${chrom}_train_${params.rare_window_features_window_size_bp}bp"

    publishDir "${params.rare_window_features_results_dir}", mode: 'copy', overwrite: false

    cpus params.rare_window_features_cpus
    memory params.rare_window_features_memory
    time params.rare_window_features_time

    input:
    tuple val(chrom),
          path(vcf), path(vcf_tbi),
          path(reference_fai), path(split_manifest),
          path(upstream_qc), path(upstream_manifest),
          path(features_py), path(manifest_py)
    val prov_b64

    output:
    path "chr${chrom}.train_rare_sites.tsv.gz", emit: sites
    path "chr${chrom}.windows.tsv", emit: windows
    path "chr${chrom}.sample_window_features.tsv.gz", emit: features
    path "chr${chrom}.rare_window_qc.json", emit: qc
    path "chr${chrom}.rare_window_features.manifest.json", emit: manifest

    script:
    def outputs = [
        "chr${chrom}.train_rare_sites.tsv.gz",
        "chr${chrom}.windows.tsv",
        "chr${chrom}.sample_window_features.tsv.gz",
        "chr${chrom}.rare_window_qc.json",
    ]
    def outputArgs = outputs.collect { "--output ${it}" }.join(' ')

    """
    set -euo pipefail

    python3 ${features_py} \
      --vcf ${vcf} \
      --vcf-index ${vcf_tbi} \
      --reference-fai ${reference_fai} \
      --split-manifest ${split_manifest} \
      --upstream-qc ${upstream_qc} \
      --upstream-manifest ${upstream_manifest} \
      --chrom ${chrom} \
      --sample-id-col '${params.rare_window_features_sample_id_col}' \
      --split-col '${params.rare_window_features_split_col}' \
      --train-label '${params.rare_window_features_train_label}' \
      --test-label '${params.rare_window_features_test_label}' \
      --expected-train-samples ${params.rare_window_features_expected_train_samples} \
      --expected-input-sites ${params.rare_window_features_expected_input_sites} \
      --min-mac ${params.rare_window_features_min_mac} \
      --max-maf ${params.rare_window_features_max_maf} \
      --window-size-bp ${params.rare_window_features_window_size_bp} \
      --outdir .

    python3 ${manifest_py} \
      --stage BUILD_RARE_WINDOW_FEATURES \
      --input ${vcf} --input ${vcf_tbi} --input ${reference_fai} \
      --input ${split_manifest} --input ${upstream_qc} --input ${upstream_manifest} \
      --input ${features_py} \
      ${outputArgs} \
      --provenance-b64 ${prov_b64} \
      --params-json '{"chrom":"${chrom}","scope":"descriptive_train_transductive","expected_train_samples":${params.rare_window_features_expected_train_samples},"min_mac":${params.rare_window_features_min_mac},"max_maf":${params.rare_window_features_max_maf},"window_size_bp":${params.rare_window_features_window_size_bp},"coordinate_contract":"0-based_half-open"}' \
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      --run-provenance-ref run_provenance.json \
      --out chr${chrom}.rare_window_features.manifest.json
    """
}
