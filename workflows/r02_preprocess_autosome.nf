nextflow.enable.dsl=2

// One chromosome per invocation. The outer runner bounds disk and memory,
// authenticates the selected inputs, and preserves each completed chromosome.
include { PREPROCESS_NORM_LEFTALIGN } from '../modules/01_preprocess_norm_leftalign'
include { PREPROCESS_NORM_LEFTALIGN_CHECKPOINTED } from '../modules/01_preprocess_checkpointed'
include { PREPROCESS_FILTER_SNV_BIALLELIC_PASS } from '../modules/02_preprocess_filter_snv_biallelic_pass'
include { LAI_RARE_BIALELIC_ONLY } from '../modules/lai_rare_bialelic_only'

workflow {
    def chrom = params.r02_chrom.toString()
    if (!(chrom ==~ /([1-9]|1[0-9]|2[0-2])/)) error 'Expected one autosome, 1..22'
    def scripts = file(params.r02_bin_dir, checkIfExists: true)
    def rawVcf = file(params.r02_raw_vcf, checkIfExists: true)
    def rawIndex = file(params.r02_raw_vcf + '.tbi', checkIfExists: true)
    def ref = file(params.ref_fasta, checkIfExists: true)
    raw = Channel.of(tuple(chrom, rawVcf, rawIndex))
    def normalized
    if (params.preprocess_checkpointed_m01 == true) {
        normalized = PREPROCESS_NORM_LEFTALIGN_CHECKPOINTED(
            raw.combine(Channel.value(ref)), scripts.resolve('mark_original_alleles.py'),
            scripts.resolve('preprocess_storage_guard.py'))
    } else {
        normalized = PREPROCESS_NORM_LEFTALIGN(raw.combine(Channel.value(ref)), scripts.resolve('mark_original_alleles.py'))
    }
    filtered = PREPROCESS_FILTER_SNV_BIALLELIC_PASS(raw.join(normalized[0]))
    LAI_RARE_BIALELIC_ONLY(filtered.map { c, counts, vcf, tbi -> tuple(c, vcf, tbi) }, [], scripts.resolve('select_rare_minor.py'))
}
