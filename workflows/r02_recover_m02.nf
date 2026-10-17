nextflow.enable.dsl=2

// Reuse the normalized M01 product. No normalization, rare selection, M14 or
// M16.5 is reachable from this entrypoint. Full staging authentication and
// create-only GCS publication belong to the external, versioned launcher.
include { PREPROCESS_FILTER_SNV_BIALLELIC_PASS } from '../modules/02_preprocess_filter_snv_biallelic_pass'

process RECOVER_M02_ADMISSION {
    tag "chr${chr}"
    cpus 1
    memory '1 GB'
    time '10m'
    input:
    tuple val(chr), path(norm), path(norm_index), path(raw_index)
    path inventory
    val inventory_sha256
    path admission_script
    path storage_guard
    path count_inspector
    path count_evidence
    val count_evidence_sha
    val expected_raw_count
    output:
    tuple val(chr), path(norm), path(norm_index), path(raw_index), path('admission.json')
    script:
    def countArgs = expected_raw_count == 0 ? '' : "--count-evidence '${count_evidence}' --count-evidence-sha256 '${count_evidence_sha}' --expected-raw-count '${expected_raw_count}'"
    """
    set -euo pipefail
    umask 077
    python3 '${admission_script}' --inventory '${inventory}' --inventory-sha256 '${inventory_sha256}' \
        --chromosome '${chr}' --vcf '${norm}' --index '${norm_index}' --raw-index '${raw_index}' \
        --scratch '${params.preprocess_large_temp_dir}' \
        --minimum-free-gib '${params.preprocess_minimum_free_gib}' \
        --required-free-bytes '${params.preprocess_required_free_bytes}' ${countArgs} > admission.json
    """
}

workflow {
    def chrom = params.r02_chrom.toString()
    if (!(chrom ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Expected one autosome, 1..22'
    if (params.preprocess_raw_index_only != true) error 'Recovery requires preprocess_raw_index_only=true'
    if (params.bcftools_min_alleles != 2 || params.plink_max_alleles != 2 ||
        params.plink_snps_only != true || params.keep_pass != true || params.max_maf != null) {
        error 'Recovery preserves exactly biallelic SNV/PASS without a MAF filter'
    }
    def cpus = params.cpus
    def threads = params.resources?.preprocess_filter_snv_biallelic_pass?.threads
    if (!(cpus instanceof Integer) || cpus < 2 || !(threads instanceof Integer) || threads < 1 || threads >= cpus) {
        error 'Declare cpus>=2 and 1<=M02 compression threads<cpus (zero selects the legacy default)'
    }
    for (key in ['r02_normalized_vcf', 'r02_raw_index', 'r02_source_receipt', 'r02_bin_dir', 'outdir', 'preprocess_large_temp_dir']) {
        def value = params[key]?.toString()
        if (!value || !value.startsWith('/') || value.contains("'") || value.contains('\n') || value.contains('\r')) {
            error "${key} must be an absolute safe local path"
        }
    }
    if (!(params.r02_source_receipt_sha256 ==~ /[0-9a-f]{64}/)) error 'Source inventory SHA256 is required'
    if (params.preprocess_minimum_free_gib == null || params.preprocess_minimum_free_gib < 0 ||
        params.preprocess_required_free_bytes == null || params.preprocess_required_free_bytes < 1) error 'Explicit disk admission budget required'
    def scripts = file(params.r02_bin_dir, checkIfExists:true)
    def normalized = file(params.r02_normalized_vcf, checkIfExists:true)
    def normalizedIndex = file(params.r02_normalized_vcf + '.tbi', checkIfExists:true)
    def rawIndex = file(params.r02_raw_index, checkIfExists:true)
    def source = Channel.of(tuple(chrom, normalized, normalizedIndex, rawIndex))
    def rawCount = params.preprocess_authenticated_raw_count
    if (rawCount != null && (!(rawCount instanceof Number) || rawCount <= 0 || rawCount != rawCount.toLong() ||
        !(params.r02_raw_count_evidence_sha256 ==~ /[0-9a-f]{64}/))) error 'Invalid authenticated RAW count contract'
    def countEvidence = rawCount == null ? [] : file(params.r02_raw_count_evidence, checkIfExists:true)
    def admitted = RECOVER_M02_ADMISSION(source,
        file(params.r02_source_receipt, checkIfExists:true), params.r02_source_receipt_sha256,
        scripts.resolve('r02_recover_m02_admission.py'), scripts.resolve('preprocess_storage_guard.py'),
        scripts.resolve('preprocess_count_validation.py'), countEvidence, params.r02_raw_count_evidence_sha256 ?: '', rawCount ?: 0)
    PREPROCESS_FILTER_SNV_BIALLELIC_PASS(admitted.map { c, vcf, tbi, raw, receipt -> tuple(c, [], raw, vcf, tbi) })
}
