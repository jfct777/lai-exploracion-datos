nextflow.enable.dsl=2

def discoverLaiRareVcfs(String inputDir, String globPattern) {
    def reRare = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.rare(?:\.minor)?\.vcf\.gz$/
    return channel
        .fromPath("${inputDir}/${globPattern}", checkIfExists: true)
        .filter { p -> p.getName() ==~ reRare }
        .map { vcf_gz ->
            def m = (vcf_gz.getName() =~ reRare)
            def chr = m[0][1]
            def tbi = vcf_gz.resolveSibling("${vcf_gz.getName()}.tbi")
            if (!tbi.exists()) throw new IllegalStateException("Missing .tbi for rare VCF: ${vcf_gz}")
            tuple(chr, vcf_gz, tbi)
        }
}

def rareUnsupportedConsumers(config) {
    return [
        M12: config.enable_rare_snp_tracts || config.runRareSnpTractAnalysis,
        M13: config.enable_individual_snp_distance_modes,
        M17: config.enable_rare_in_lai, M19: config.enable_rare_on_lai_painting,
        M20: config.enable_feature_build, M21: config.enable_presence_channel,
        M24: config.enable_allele_orientation_audit || config.enable_allele_orientation_inventory,
    ].findAll { key, enabled -> enabled }.keySet().join(',')
}

def validateRareGenotypeOptions(config) {
    if (!config.enable_lai_rare || !config.lai_rare_keep_format ||
            config.lai_rare_keep_format.split(',').collect { it.trim() }.contains('GT')) return
    def consumers = [
        M13: config.enable_individual_snp_distance_modes,
        M14: config.enable_rare_allele_painting,
        M17: config.enable_rare_in_lai,
        M24: config.enable_allele_orientation_audit || config.enable_allele_orientation_inventory,
    ].findAll { name, enabled -> enabled }.keySet()
    if (consumers) throw new IllegalStateException("${consumers.join(',')} require FORMAT/GT; include GT in lai_rare_keep_format")
}

workflow VALIDATED_RARE_INPUT {
    take:
    source_vcfs
    unsupported_consumers
    painting_mode
    guard_script
    main:
    def checked = source_vcfs
    if (unsupported_consumers || painting_mode) {
        checked = VALIDATE_RARE_CONSUMERS(source_vcfs, unsupported_consumers, painting_mode, guard_script)
    }
    emit:
    vcfs = checked
}

process VALIDATE_RARE_CONSUMERS {
    tag "chr${chr}"
    cpus 1
    memory '512 MB'
    time '5m'
    input:
    tuple val(chr), path(vcf_gz), path(vcf_tbi)
    val unsupported_consumers
    val painting_mode
    path validate_rare_contract_py
    output:
    tuple val(chr), path(vcf_gz), path(vcf_tbi)
    script:
    """
    python3 '${validate_rare_contract_py}' --input '${vcf_gz}' \
        --unsupported-consumers '${unsupported_consumers}' --painting-mode '${painting_mode}'
    """
}
