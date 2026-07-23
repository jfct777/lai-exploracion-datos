nextflow.enable.dsl=2

include { PREPROCESS_NORM_LEFTALIGN } from './modules/01_preprocess_norm_leftalign'
include { PREPROCESS_FILTER_SNV_BIALLELIC_PASS } from './modules/02_preprocess_filter_snv_biallelic_pass'
include { QC_BCFTOOLS_STATS } from './modules/03_qc_bcftools_stats'
include { QC_PLINK_MAKE_PGEN } from './modules/04_qc_plink_make_pgen'
include { QC_PLINK_MISSING_HET } from './modules/05_qc_plink_missing_het'
include { QC_AGGREGATE_REPORT_PY } from './modules/06_qc_aggregate_report_py'
include { SFS_FROM_FILTERED_VCF } from './modules/07_SFS_FROM_FILTERED_VCF'
include { BUILD_ANCESTRAL_TSV_FROM_FASTA } from './modules/08_BUILD_ANCESTRAL_TSV_FROM_FASTA'
include { DAF_DSFS_FROM_ANCESTRAL_TSV } from './modules/09_DAF_DSFS_FROM_ANCESTRAL_TSV'
include { LD_DECAY_FROM_PGEN } from './modules/10_LD_DECAY_FROM_PGEN'
include { TAG_SNPS_FROM_PGEN } from './modules/11_TAG_SNPS_FROM_PGEN'

process MAKE_KEEP_SAMPLES {
    tag "keep_samples"

    cpus 1
    memory '1 GB'
    time '1h'

    input:
    path flags_tsv
    path make_keep_samples_py

    output:
    path("keep_samples.txt")

    script:
    """
    set -euo pipefail
    python3 ${make_keep_samples_py} \
      --flags ${flags_tsv} \
      --out keep_samples.txt \
      --flag_fail_col ${params.flag_fail_col} \
      --sample_id_col ${params.sample_id_col}
    """
}

// =========================================================================
// Helper: extract chr-value pattern from chr_regex capture group
//   e.g. 'dnabr\.hg38\.2723\.chr(22)\.vcf\.gz'        → /^(22)$/
//        'dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.vcf\.gz' → /^(\d+|X|Y|MT)$/
// =========================================================================
def extractChrValuePattern(chrRegex) {
    def m = (chrRegex =~ /\(([^)]+)\)/)
    if( m.find() ) {
        return java.util.regex.Pattern.compile('^(' + m[0][1] + ')$')
    }
    return null
}

// =========================================================================
// Helper: discover filtered VCFs from outdir/02_filter
// =========================================================================
def discoverFilteredVcfs(outdir, chrValuePattern) {
    def reVcf = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.snv\.bi\.pass\.vcf\.gz$/
    return channel
        .fromPath("${outdir}/02_filter/*.vcf.gz")
        .filter { p -> p.getName() ==~ reVcf }
        .map { vcf_gz ->
            def m = (vcf_gz.getName() =~ reVcf)
            if( !m.matches() ) {
                throw new IllegalArgumentException("Cannot extract chr from filtered VCF: ${vcf_gz.getName()}")
            }
            def chr = m[0][1]
            def tbi = file("${vcf_gz}.tbi")
            if( !tbi.exists() ) {
                throw new IllegalStateException("Missing index (.tbi) for filtered VCF: ${vcf_gz}")
            }
            tuple(chr, vcf_gz, tbi)
        }
        .filter { chr, _vcf_gz, _tbi -> chrValuePattern == null || (chr ==~ chrValuePattern) }
}

// =========================================================================
// Helper: discover PGEN triplets from outdir/04_plink_pgen
// =========================================================================
def discoverPgenTriplets(outdir, chrValuePattern) {
    def rePgen = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.pgen$/
    return channel
        .fromPath("${outdir}/04_plink_pgen/*.pgen")
        .filter { p -> p.getName() ==~ rePgen }
        .map { pgen ->
            def m = (pgen.getName() =~ rePgen)
            if( !m.matches() ) {
                throw new IllegalArgumentException("Cannot extract chr from pgen: ${pgen.getName()}")
            }
            def chr = m[0][1]
            def pvar = file(pgen.toString().replaceFirst(/\.pgen$/, '.pvar'))
            def psam = file(pgen.toString().replaceFirst(/\.pgen$/, '.psam'))
            if( !pvar.exists() || !psam.exists() ) {
                throw new IllegalStateException("Missing pvar/psam for ${pgen}")
            }
            tuple(chr, pgen, pvar, psam)
        }
        .filter { chr, _pgen, _pvar, _psam -> chrValuePattern == null || (chr ==~ chrValuePattern) }
}

// =========================================================================
// Helper: discover norm VCFs from outdir/01_norm
// =========================================================================
def discoverNormVcfs(outdir, chrValuePattern) {
    def reNorm = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.norm\.vcf\.gz$/
    return channel
        .fromPath("${outdir}/01_norm/*.norm.vcf.gz")
        .filter { p -> p.getName() ==~ reNorm }
        .map { norm_vcf ->
            def m = (norm_vcf.getName() =~ reNorm)
            if( !m.matches() ) {
                throw new IllegalArgumentException("Cannot extract chr from norm VCF: ${norm_vcf.getName()}")
            }
            def chr = m[0][1]
            def norm_tbi = file("${norm_vcf}.tbi")
            if( !norm_tbi.exists() ) {
                throw new IllegalStateException("Missing index (.tbi) for norm VCF: ${norm_vcf}")
            }
            tuple(chr, norm_vcf, norm_tbi)
        }
        .filter { chr, _norm_vcf, _norm_tbi -> chrValuePattern == null || (chr ==~ chrValuePattern) }
}

// =========================================================================
// Helper: discover ancestral TSVs from outdir/08_ancestral_polarization
// =========================================================================
def discoverAncestralTsvs(outdir, chrValuePattern) {
    def reAnc = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.ancestral\.tsv\.gz$/
    return channel
        .fromPath("${outdir}/08_ancestral_polarization/*.ancestral.tsv.gz")
        .filter { p -> p.getName() ==~ reAnc }
        .map { tsv_gz ->
            def m = (tsv_gz.getName() =~ reAnc)
            if( !m.matches() ) {
                throw new IllegalArgumentException("Cannot extract chr from ancestral TSV: ${tsv_gz.getName()}")
            }
            def chr = m[0][1]
            def summary = file(tsv_gz.toString().replaceFirst(/\.ancestral\.tsv\.gz$/, '.ancestral.summary.json'))
            if( !summary.exists() ) {
                throw new IllegalStateException("Missing ancestral summary JSON for ${tsv_gz}")
            }
            tuple(chr, tsv_gz, summary)
        }
        .filter { chr, _tsv_gz, _summary -> chrValuePattern == null || (chr ==~ chrValuePattern) }
}

// =========================================================================
// Main workflow
// =========================================================================
workflow {
    def chrPattern = java.util.regex.Pattern.compile(params.chr_regex)
    def chrValuePattern = extractChrValuePattern(params.chr_regex)
    def run_qc = (params.run_qc == null) ? true : params.run_qc

    // Resolve effective per-module flags:
    //   run_qc=true  → all QC 01-06 run (original behaviour)
    //   run_qc=false → honour individual enable_* flags; disabled modules read from outdir
    def do_norm        = run_qc ? params.enable_norm        : params.enable_norm
    def do_filter      = run_qc ? params.enable_filter      : params.enable_filter
    def do_bcfstats    = run_qc ? params.enable_bcfstats    : params.enable_bcfstats
    def do_pgen        = run_qc ? params.enable_pgen        : params.enable_pgen
    def do_missing_het = run_qc ? params.enable_missing_het : params.enable_missing_het
    def do_report      = run_qc ? params.enable_report      : params.enable_report

    // Determine what downstream needs even when run_qc=false
    def need_filtered = do_filter || do_bcfstats || do_pgen || do_missing_het || do_report ||
                        params.enable_sfs || params.enable_build_ancestral_tsv || params.enable_daf
    def need_pgen     = do_pgen || do_missing_het || do_report ||
                        params.enable_ld || params.enable_tag_snps
    def need_norm     = do_norm || do_filter

    // Scripts
    def parse_bcftools_stats_py = file("${projectDir}/bin/parse_bcftools_stats.py")
    def aggregate_qc_report_py  = file("${projectDir}/bin/aggregate_qc_report.py")
    def sfs_report_py            = file("${projectDir}/bin/sfs_report.py")
    def build_ancestral_tsv_py   = file("${projectDir}/bin/build_ancestral_tsv.py")
    def daf_dsfs_py              = file("${projectDir}/bin/daf_dsfs.py")
    def ld_decay_py              = file("${projectDir}/bin/ld_decay.py")
    def tag_summary_py           = file("${projectDir}/bin/tag_summary.py")

    // Placeholder empty file
    def empty_placeholder = file("${projectDir}/conf/empty.txt")
    if( !empty_placeholder.exists() ) {
        throw new IllegalStateException("Missing placeholder file: ${empty_placeholder}. Please keep conf/empty.txt in the repo (empty file).")
    }
    def ch_empty_file = channel.value(empty_placeholder)

    // =====================================================================
    // Discover raw VCFs from vcf_dir (needed when any upstream module runs)
    // =====================================================================
    def ch_vcfs = channel.empty()
    if( do_norm || do_filter ) {
        ch_vcfs = channel
            .fromPath("${params.vcf_dir}/*.vcf.gz")
            .filter { p -> p.getName() ==~ chrPattern }
            .map { vcf_gz ->
                def m = (vcf_gz.getName() =~ chrPattern)
                if( !m.matches() ) {
                    throw new IllegalArgumentException("VCF filename does not match chr_regex: ${vcf_gz.getName()}")
                }
                def chr = m[0][1]
                def vcf_tbi = file("${vcf_gz}.tbi")
                if( !vcf_tbi.exists() ) {
                    throw new IllegalStateException("Missing index (.tbi) for ${vcf_gz}")
                }
                tuple(chr, vcf_gz, vcf_tbi)
            }
    }

    // =====================================================================
    // 01: Normalization + left-align
    // =====================================================================
    def ch_norm_vcfs
    if( do_norm ) {
        def ch_ref = channel.value( file(params.ref_fasta) )
        ch_vcfs
            .combine(ch_ref)
            .set { ch_vcfs_with_ref }
        def (_ch_norm_vcfs, _ch_norm_logs) = PREPROCESS_NORM_LEFTALIGN(ch_vcfs_with_ref)
        ch_norm_vcfs = _ch_norm_vcfs
    } else if( need_norm || do_filter ) {
        // Read pre-existing norm VCFs from outdir
        ch_norm_vcfs = discoverNormVcfs(params.outdir, chrValuePattern)
    } else {
        ch_norm_vcfs = channel.empty()
    }

    // =====================================================================
    // 02: Filter SNV biallelic PASS
    // =====================================================================
    def ch_filtered
    if( do_filter ) {
        def ch_norm_keyed = ch_norm_vcfs.map { chr, norm_vcf, norm_tbi -> tuple(chr, norm_vcf, norm_tbi) }
        def ch_raw_keyed  = ch_vcfs.map { chr, vcf_gz, vcf_tbi -> tuple(chr, vcf_gz, vcf_tbi) }

        def ch_for_filter = ch_raw_keyed
            .join(ch_norm_keyed)
            .map { chr, vcf_gz, _vcf_tbi, norm_vcf, norm_tbi -> tuple(chr, vcf_gz, norm_vcf, norm_tbi) }

        ch_filtered = PREPROCESS_FILTER_SNV_BIALLELIC_PASS(ch_for_filter)
    } else {
        ch_filtered = channel.empty()
    }

    // Filtered VCFs channel: from process output when run, otherwise from outdir
    def ch_filtered_vcfs
    if( do_filter ) {
        ch_filtered_vcfs = ch_filtered
            .map { chr, _counts_tsv, vcf_pass, vcf_pass_tbi -> tuple(chr, vcf_pass, vcf_pass_tbi) }
    } else if( need_filtered ) {
        ch_filtered_vcfs = discoverFilteredVcfs(params.outdir, chrValuePattern)
    } else {
        ch_filtered_vcfs = channel.empty()
    }

    // =====================================================================
    // 03: bcftools stats
    // =====================================================================
    def ch_bcftools_stats = channel.empty()
    if( do_bcfstats ) {
        // Need filtered output (with counts_tsv)
        def ch_stats_input
        if( do_filter ) {
            ch_stats_input = ch_filtered
                .map { chr, counts_tsv, vcf_pass, vcf_pass_tbi ->
                    tuple(chr, counts_tsv, vcf_pass, vcf_pass_tbi, parse_bcftools_stats_py)
                }
        } else {
            // Discover from outdir — bcfstats needs counts_tsv + vcf from 02_filter
            ch_stats_input = discoverFilteredVcfs(params.outdir, chrValuePattern)
                .map { chr, vcf_gz, tbi ->
                    def counts_tsv = file("${params.outdir}/02_filter/dnabr.hg38.2723.chr${chr}.counts.tsv")
                    if( !counts_tsv.exists() ) {
                        throw new IllegalStateException("Missing counts TSV for chr${chr} in ${params.outdir}/02_filter")
                    }
                    tuple(chr, counts_tsv, vcf_gz, tbi, parse_bcftools_stats_py)
                }
        }
        def (_ch_bcftools_stats, _ch_bcftools_stats_txt) = QC_BCFTOOLS_STATS(ch_stats_input)
        ch_bcftools_stats = _ch_bcftools_stats
    }

    // =====================================================================
    // 04: PLINK make-pgen
    // =====================================================================
    def ch_pgen = channel.empty()
    if( do_pgen ) {
        def ch_pgen_input
        if( do_filter ) {
            ch_pgen_input = ch_filtered
        } else {
            // Discover filtered VCFs + counts from outdir for make-pgen input
            ch_pgen_input = discoverFilteredVcfs(params.outdir, chrValuePattern)
                .map { chr, vcf_gz, tbi ->
                    def counts_tsv = file("${params.outdir}/02_filter/dnabr.hg38.2723.chr${chr}.counts.tsv")
                    if( !counts_tsv.exists() ) {
                        throw new IllegalStateException("Missing counts TSV for chr${chr} in ${params.outdir}/02_filter")
                    }
                    tuple(chr, counts_tsv, vcf_gz, tbi)
                }
        }
        def (_ch_pgen, _ch_makepgen_logs) = QC_PLINK_MAKE_PGEN(ch_pgen_input)
        ch_pgen = _ch_pgen
    }

    // PGEN channel: from process output when run, otherwise from outdir
    def ch_pgen_triplets
    if( do_pgen ) {
        ch_pgen_triplets = ch_pgen
    } else if( need_pgen ) {
        ch_pgen_triplets = discoverPgenTriplets(params.outdir, chrValuePattern)
    } else {
        ch_pgen_triplets = channel.empty()
    }

    // =====================================================================
    // 05: PLINK missing + het QC
    // =====================================================================
    def ch_plink_qc = channel.empty()
    if( do_missing_het ) {
        def ch_missing_input = ch_pgen_triplets
            .map { chr, pgen, pvar, psam -> tuple(chr, pgen, pvar, psam, aggregate_qc_report_py) }
        def (_ch_plink_qc, _ch_plink_imiss, _ch_plink_het, _ch_plink_genoFilt_logs, _ch_plink_missing_logs, _ch_plink_het_logs) = QC_PLINK_MISSING_HET(ch_missing_input)
        ch_plink_qc = _ch_plink_qc
    }

    // =====================================================================
    // 06: Aggregate QC report
    // =====================================================================
    if( do_report ) {
        // Collect per-sample QC files
        def ch_qc_per_sample_all
        if( do_missing_het ) {
            ch_qc_per_sample_all = ch_plink_qc
                .map { _chr, qc_per_sample -> qc_per_sample }
                .collect()
        } else {
            // Discover from outdir/05_plink_qc
            ch_qc_per_sample_all = channel
                .fromPath("${params.outdir}/05_plink_qc/*.qc_per_sample.tsv")
                .collect()
        }

        // Collect counts files
        def ch_counts_all
        if( do_filter ) {
            ch_counts_all = ch_filtered
                .map { _chr, counts_tsv, _vcf_pass, _vcf_pass_tbi -> counts_tsv }
                .collect()
        } else {
            ch_counts_all = channel
                .fromPath("${params.outdir}/02_filter/*.counts.tsv")
                .collect()
        }

        // Collect bcftools stats files
        def ch_stats_all
        if( do_bcfstats ) {
            ch_stats_all = ch_bcftools_stats
                .map { _chr, stats_parsed -> stats_parsed }
                .collect()
        } else {
            ch_stats_all = channel
                .fromPath("${params.outdir}/03_bcftools_stats/*.bcftools.stats.parsed.tsv")
                .collect()
        }

        ch_qc_per_sample_all
            .combine(ch_stats_all)
            .combine(ch_counts_all)
            .map { qc_files, stats_files, counts_files -> tuple(qc_files, stats_files, counts_files) }
            .set { ch_agg_in }

        QC_AGGREGATE_REPORT_PY(
            ch_agg_in.map { qc_files, stats_files, counts_files -> tuple(qc_files, stats_files, counts_files, aggregate_qc_report_py) }
        )
    }

    // =====================================================================
    // Downstream population-genetics modules 07-11
    // =====================================================================
    def any_downstream = params.run_downstream || params.enable_sfs || params.enable_build_ancestral_tsv || params.enable_daf || params.enable_ld || params.enable_tag_snps

    if( any_downstream ) {
        // Resolve keep-samples and exclude-bed channels
        def make_keep_samples_py = file("${projectDir}/bin/make_keep_samples.py")

        def ch_keep_samples_opt
        if( params.use_keep_samples_from_qc ) {
            def flags = file(params.flags_tsv)
            if( !flags.exists() ) {
                throw new IllegalStateException("use_keep_samples_from_qc=true but flags_tsv not found: ${flags}")
            }
            ch_keep_samples_opt = MAKE_KEEP_SAMPLES(
                channel.value(flags),
                channel.value(make_keep_samples_py)
            )
        } else {
            ch_keep_samples_opt = ch_empty_file
        }

        def ch_exclude_bed_opt
        def exclude_bed_param = params.exclude_regions_bed
        if( exclude_bed_param instanceof Boolean ) {
            if( exclude_bed_param ) {
                throw new IllegalArgumentException("exclude_regions_bed was set to true (flag without a path). Provide a BED path, or disable with --exclude_regions_bed false")
            }
            ch_exclude_bed_opt = ch_empty_file
        }
        else if( exclude_bed_param ) {
            def bed = file(exclude_bed_param)
            if( bed.exists() ) {
                ch_exclude_bed_opt = channel.value(bed)
            } else {
                ch_exclude_bed_opt = ch_empty_file
            }
        }
        else {
            ch_exclude_bed_opt = ch_empty_file
        }

        // Filtered VCFs for downstream: prefer live channel from 02, fall back to outdir
        def ch_downstream_vcfs
        if( do_filter ) {
            ch_downstream_vcfs = ch_filtered
                .map { chr, _counts_tsv, vcf_pass, vcf_pass_tbi -> tuple(chr, vcf_pass, vcf_pass_tbi) }
        } else {
            ch_downstream_vcfs = discoverFilteredVcfs(params.outdir, chrValuePattern)
        }

        // PGEN triplets for downstream: prefer live channel from 04, fall back to outdir
        def ch_downstream_pgen
        if( do_pgen ) {
            ch_downstream_pgen = ch_pgen
        } else {
            ch_downstream_pgen = discoverPgenTriplets(params.outdir, chrValuePattern)
        }

        // 07: SFS
        if( params.enable_sfs ) {
            SFS_FROM_FILTERED_VCF(
                ch_downstream_vcfs
                    .map { chr, vcf_gz, tbi -> tuple(chr, vcf_gz, tbi, sfs_report_py) },
                ch_keep_samples_opt
            )
        }

        // 08: Build ancestral TSV from FASTA
        def ch_ancestral_tbl = channel.empty()
        if( params.enable_build_ancestral_tsv ) {
            def anc_fa = file(params.ancestral_fasta)
            def anc_fai = file("${params.ancestral_fasta}.fai")
            if( !anc_fa.exists() || !anc_fai.exists() ) {
                throw new IllegalStateException("Missing ancestral FASTA or .fai: ${anc_fa} / ${anc_fai}")
            }

            ch_ancestral_tbl = BUILD_ANCESTRAL_TSV_FROM_FASTA(
                ch_downstream_vcfs
                    .map { chr, vcf_gz, tbi -> tuple(chr, vcf_gz, tbi, anc_fa, anc_fai, build_ancestral_tsv_py) }
            )
        }

        // 09: DAF/DSFS — depends on 08 output (live or from outdir)
        if( params.enable_daf ) {
            def ch_ancestral_for_daf
            if( params.enable_build_ancestral_tsv ) {
                // Use live output from module 08
                ch_ancestral_for_daf = ch_ancestral_tbl
            } else {
                // Discover pre-existing ancestral TSVs from outdir
                ch_ancestral_for_daf = discoverAncestralTsvs(params.outdir, chrValuePattern)
            }

            DAF_DSFS_FROM_ANCESTRAL_TSV(
                ch_downstream_vcfs
                    .join(ch_ancestral_for_daf)
                    .map { chr, vcf_gz, tbi, ancestral_tsv, ancestral_summary -> tuple(chr, vcf_gz, tbi, ancestral_tsv, ancestral_summary, daf_dsfs_py) }
            )
        }

        // 10: LD decay
        if( params.enable_ld ) {
            LD_DECAY_FROM_PGEN(
                ch_downstream_pgen
                    .map { chr, pgen, pvar, psam -> tuple(chr, pgen, pvar, psam, ld_decay_py) },
                ch_keep_samples_opt,
                ch_exclude_bed_opt
            )
        }

        // 11: Tag SNPs
        if( params.enable_tag_snps ) {
            TAG_SNPS_FROM_PGEN(
                ch_downstream_pgen
                    .map { chr, pgen, pvar, psam -> tuple(chr, pgen, pvar, psam, tag_summary_py) },
                ch_keep_samples_opt,
                ch_exclude_bed_opt
            )
        }
    }
}
