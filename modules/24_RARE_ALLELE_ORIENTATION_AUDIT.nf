nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Module 24 — audit of ALT-versus-minor-allele semantics
// ---------------------------------------------------------------------------
// A single-chromosome, no-training sensitivity analysis.  It reuses the M14
// segment/window functions and requires the historical arm to reproduce the
// published chr-level artefacts before comparing alternative orientations.

process WRITE_ALLELE_ORIENTATION_AUDIT_RUN_PROVENANCE {
    tag "run_provenance_chr${params.allele_orientation_audit_chromosome}"

    publishDir "${params.allele_orientation_audit_results_dir}", mode: 'copy', overwrite: false

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


process AUDIT_RARE_ALLELE_ORIENTATION {
    tag "chr${chrom}"

    publishDir "${params.allele_orientation_audit_results_dir}", mode: 'copy', overwrite: false

    cpus params.allele_orientation_audit_cpus
    memory params.allele_orientation_audit_memory
    time params.allele_orientation_audit_time

    input:
    tuple val(chrom),
          path(vcf), path(vcf_tbi),
          path(reference_fasta), path(reference_fai),
          path(canonical_windows), path(canonical_segments), path(canonical_summary),
          path(split_manifest),
          path(audit_py), path(orientation_py), path(painter_py), path(manifest_py)
    val prov_b64

    output:
    path "chr${chrom}.orientation_sites.tsv.gz", emit: orientation_sites
    path "chr${chrom}.orientation_summary.json", emit: orientation_summary
    path "chr${chrom}.historical_alt.sharing_windows.tsv.gz", emit: historical_windows
    path "chr${chrom}.historical_alt.pairwise_segments.tsv.gz", emit: historical_segments
    path "chr${chrom}.historical_alt.summary.json", emit: historical_summary
    path "chr${chrom}.minor_allele.sharing_windows.tsv.gz", emit: minor_windows
    path "chr${chrom}.minor_allele.pairwise_segments.tsv.gz", emit: minor_segments
    path "chr${chrom}.minor_allele.summary.json", emit: minor_summary
    path "chr${chrom}.exclude_alt_major.sharing_windows.tsv.gz", emit: excluded_windows
    path "chr${chrom}.exclude_alt_major.pairwise_segments.tsv.gz", emit: excluded_segments
    path "chr${chrom}.exclude_alt_major.summary.json", emit: excluded_summary
    path "chr${chrom}.mode_comparison.tsv", emit: comparison
    path "chr${chrom}.sample_burden_by_mode.tsv.gz", emit: burden
    path "chr${chrom}.audit_report.json", emit: report
    path "chr${chrom}.manifest.json", emit: manifest

    script:
    def outputs = [
        "chr${chrom}.orientation_sites.tsv.gz",
        "chr${chrom}.orientation_summary.json",
        "chr${chrom}.historical_alt.sharing_windows.tsv.gz",
        "chr${chrom}.historical_alt.pairwise_segments.tsv.gz",
        "chr${chrom}.historical_alt.summary.json",
        "chr${chrom}.minor_allele.sharing_windows.tsv.gz",
        "chr${chrom}.minor_allele.pairwise_segments.tsv.gz",
        "chr${chrom}.minor_allele.summary.json",
        "chr${chrom}.exclude_alt_major.sharing_windows.tsv.gz",
        "chr${chrom}.exclude_alt_major.pairwise_segments.tsv.gz",
        "chr${chrom}.exclude_alt_major.summary.json",
        "chr${chrom}.mode_comparison.tsv",
        "chr${chrom}.sample_burden_by_mode.tsv.gz",
        "chr${chrom}.audit_report.json",
    ]
    def outputArgs = outputs.collect { "--output ${it}" }.join(' \\\n      ')

    """
    set -euo pipefail

    PYTHONPATH=. python3 ${audit_py} \\
      --vcf ${vcf} \\
      --reference-fasta ${reference_fasta} \\
      --canonical-summary ${canonical_summary} \\
      --canonical-windows ${canonical_windows} \\
      --canonical-segments ${canonical_segments} \\
      --split-manifest ${split_manifest} \\
      --chrom ${chrom} \\
      --train-label '${params.allele_orientation_audit_train_label}' \\
      --sample-id-col '${params.allele_orientation_audit_sample_id_col}' \\
      --split-col '${params.allele_orientation_audit_split_col}' \\
      --n-jobs ${task.cpus} \\
      --outdir .

    python3 ${manifest_py} \\
      --stage AUDIT_RARE_ALLELE_ORIENTATION \\
      --input ${vcf} --input ${reference_fasta} \\
      --input ${canonical_windows} --input ${canonical_segments} \\
      --input ${canonical_summary} --input ${split_manifest} \\
      --input ${audit_py} --input ${orientation_py} --input ${painter_py} \\
      ${outputArgs} \\
      --provenance-b64 ${prov_b64} \\
      --params-json '{"chrom":"${chrom}","carrier_modes":["historical_alt","minor_allele","exclude_alt_major"],"n_jobs":${task.cpus}}' \\
      --stamp "\$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \\
      --out chr${chrom}.manifest.json
    """
}
