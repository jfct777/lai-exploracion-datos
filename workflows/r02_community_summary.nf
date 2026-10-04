nextflow.enable.dsl=2

// Summarize every saved partition; no clustering or scientific model fitting.
process R02_COMMUNITY_SUMMARY {
    tag 'saved_autosomal_community_summary'
    cache 'deep'
    cpus 1
    input:
    path source_bin
    path evidence_bundle
    output:
    path 'community_summary', emit: summary
    script:
    """
    export PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="\$PWD" SQLITE_TMPDIR="\$PWD"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 '${source_bin}/preprocess_storage_guard.py' --directory "\$PWD" --minimum-free-gib ${params.min_free_gib}
    python3 '${source_bin}/r02_community_summary.py' \
      --manifest '${evidence_bundle}/manifest.json' \
      --expected-manifest-sha256 '${params.evidence_manifest_sha256}' \
      --max-memory-mb ${params.max_memory_mb} \
      --output-dir community_summary
    """
}

workflow {
    ['source_bin', 'evidence_dir', 'evidence_manifest_sha256', 'min_free_gib', 'max_memory_mb'].each { key ->
        if (params[key] == null || params[key].toString().find(/[\n\r'"`$\\;|&<>]/))
            error "Missing or unsafe parameter: ${key}"
    }
    if (!(params.max_memory_mb.toString() ==~ /[1-9][0-9]*/)) error 'max_memory_mb must be a positive integer'
    if (!(params.evidence_manifest_sha256.toString() ==~ /[0-9a-f]{64}/)) error 'Invalid evidence manifest SHA256'
    if (!(params.min_free_gib.toString() ==~ /[0-9]+(\.[0-9]+)?/)) error 'min_free_gib must be nonnegative'
    def bundle = file(params.evidence_dir, checkIfExists:true)
    if (!bundle.isDirectory()) error 'evidence_dir must be a complete bundle directory'
    file("${params.evidence_dir}/manifest.json", checkIfExists:true)
    R02_COMMUNITY_SUMMARY(file(params.source_bin, checkIfExists:true), bundle)
}
