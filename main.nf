nextflow.enable.dsl=2

import groovy.json.JsonOutput

include { SMOKE_TEST } from './modules/00_smoke_test'
include { PREPROCESS_NORM_LEFTALIGN } from './modules/01_preprocess_norm_leftalign'
include { PREPROCESS_FILTER_SNV_BIALLELIC_PASS } from './modules/02_preprocess_filter_snv_biallelic_pass'
include { LAI_RARE_BIALELIC_ONLY } from './modules/lai_rare_bialelic_only'
include { QC_BCFTOOLS_STATS } from './modules/03_qc_bcftools_stats'
include { QC_PLINK_MAKE_PGEN } from './modules/04_qc_plink_make_pgen'
include { QC_PLINK_MISSING_HET } from './modules/05_qc_plink_missing_het'
include { QC_AGGREGATE_REPORT_PY } from './modules/06_qc_aggregate_report_py'
include { SFS_FROM_FILTERED_VCF } from './modules/07_SFS_FROM_FILTERED_VCF'
include { BUILD_ANCESTRAL_TSV_FROM_FASTA } from './modules/08_BUILD_ANCESTRAL_TSV_FROM_FASTA'
include { DAF_DSFS_FROM_ANCESTRAL_TSV } from './modules/09_DAF_DSFS_FROM_ANCESTRAL_TSV'
include { LD_DECAY_FROM_PGEN } from './modules/10_LD_DECAY_FROM_PGEN'
include { TAG_SNPS_FROM_PGEN } from './modules/11_TAG_SNPS_FROM_PGEN'
include { ANALYZE_RARE_SNP_TRACTS } from './modules/12_RARE_SNP_TRACTS_FROM_RARE_VCF'
include { AGGREGATE_RARE_SNP_TRACTS } from './modules/12_RARE_SNP_TRACTS_FROM_RARE_VCF'
include { ANALYZE_INDIVIDUAL_SNP_DISTANCE_MODES } from './modules/13_INDIVIDUAL_SNP_DISTANCE_MODES'
include { AGGREGATE_INDIVIDUAL_SNP_DISTANCE_MODES } from './modules/13_INDIVIDUAL_SNP_DISTANCE_MODES'
include { ANALYZE_RARE_ALLELE_SHARING } from './modules/14_RARE_ALLELE_SHARING_PAINTER'
include { AGGREGATE_RARE_ALLELE_SHARING } from './modules/14_RARE_ALLELE_SHARING_PAINTER'
include { IBD_COMMUNITY_ENHANCED; IBD_ENHANCED_REPLOT } from './modules/16_5_IBD_COMMUNITY_ENHANCED'
include { ANALYZE_RARE_IN_LAI } from './modules/17_RARE_VARIANTS_IN_LAI_TRACTS'
include { AGGREGATE_RARE_IN_LAI } from './modules/17_RARE_VARIANTS_IN_LAI_TRACTS'
include { RARE_ON_LAI_PAINTING } from './modules/19_RARE_ON_LAI_PAINTING'
include { COMPARE_ASIBD_COMMON } from './modules/18_COMMON_ASIBD_COMPARATOR'
include { BUILD_PRESENCE_LCR_MASK; ANALYZE_PRESENCE_CHANNEL; AGGREGATE_PRESENCE_CHANNEL } from './modules/21_BUILD_PRESENCE_CHANNEL'
include { DEFINE_COHORT; COUNT_RARE_DENSITY; AGGREGATE_FEATURE_STORE } from './modules/20_BUILD_FEATURE_STORE'
include { BUILD_MODELING_MASTER; BUILD_SPLIT_MANIFEST; MODEL_PRIMARY_CV; EVALUATE_TEST; VERIFY_TEST_HASH } from './modules/22_MODEL_PIPELINE'
include { RARE_MATRIX_BENCHMARK } from './modules/23_RARE_MATRIX_BENCHMARK'

// ---------------------------------------------------------------------------
// Helper: discover normalized VCFs from outdir/01_norm/
// ---------------------------------------------------------------------------
def discoverNormVcfs(String outdir) {
    def reNorm = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.norm\.vcf\.gz$/
    return channel
        .fromPath("${outdir}/01_norm/*.norm.vcf.gz")
        .filter { p -> p.getName() ==~ reNorm }
        .map { norm_vcf ->
            def m = (norm_vcf.getName() =~ reNorm)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${norm_vcf.getName()}")
            def chr = m[0][1]
            def norm_tbi = file("${norm_vcf}.tbi")
            if( !norm_tbi.exists() ) throw new IllegalStateException("Missing .tbi for ${norm_vcf}")
            tuple(chr, norm_vcf, norm_tbi)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover filtered VCFs from outdir/02_filter/
// ---------------------------------------------------------------------------
def discoverFilteredVcfs(String outdir) {
    def reVcf = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.snv\.bi\.pass\.vcf\.gz$/
    return channel
        .fromPath("${outdir}/02_filter/*.snv.bi.pass.vcf.gz")
        .filter { p -> p.getName() ==~ reVcf }
        .map { vcf_gz ->
            def m = (vcf_gz.getName() =~ reVcf)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${vcf_gz.getName()}")
            def chr = m[0][1]
            def tbi = file("${vcf_gz}.tbi")
            if( !tbi.exists() ) throw new IllegalStateException("Missing .tbi for filtered VCF: ${vcf_gz}")
            tuple(chr, vcf_gz, tbi)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover counts TSVs from outdir/02_filter/
// ---------------------------------------------------------------------------
def discoverCountsTsvs(String outdir) {
    def reCounts = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.counts\.tsv$/
    return channel
        .fromPath("${outdir}/02_filter/*.counts.tsv")
        .filter { p -> p.getName() ==~ reCounts }
        .map { tsv ->
            def m = (tsv.getName() =~ reCounts)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${tsv.getName()}")
            def chr = m[0][1]
            tuple(chr, tsv)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover PGEN triplets from outdir/04_plink_pgen/
// ---------------------------------------------------------------------------
def discoverPgenTriplets(String outdir) {
    def rePgen = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.pgen$/
    return channel
        .fromPath("${outdir}/04_plink_pgen/*.pgen")
        .filter { p -> p.getName() ==~ rePgen }
        .map { pgen ->
            def m = (pgen.getName() =~ rePgen)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${pgen.getName()}")
            def chr = m[0][1]
            def pvar = file(pgen.toString().replaceFirst(/\.pgen$/, '.pvar'))
            def psam = file(pgen.toString().replaceFirst(/\.pgen$/, '.psam'))
            if( !pvar.exists() || !psam.exists() ) throw new IllegalStateException("Missing pvar/psam for ${pgen}")
            tuple(chr, pgen, pvar, psam)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover ancestral TSVs from outdir/08_ancestral_polarization/
// ---------------------------------------------------------------------------
def discoverAncestralTsvs(String outdir) {
    def reAnc = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.ancestral\.tsv\.gz$/
    return channel
        .fromPath("${outdir}/08_ancestral_polarization/*.ancestral.tsv.gz")
        .filter { p -> p.getName() ==~ reAnc }
        .map { tsv_gz ->
            def m = (tsv_gz.getName() =~ reAnc)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${tsv_gz.getName()}")
            def chr = m[0][1]
            def summary = file(tsv_gz.toString().replaceFirst(/\.ancestral\.tsv\.gz$/, '.ancestral.summary.json'))
            if( !summary.exists() ) throw new IllegalStateException("Missing summary JSON for ${tsv_gz}")
            tuple(chr, tsv_gz, summary)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover bcftools stats parsed TSVs from outdir/03_bcftools_stats/
// ---------------------------------------------------------------------------
def discoverBcftoolsStats(String outdir) {
    def reStats = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.bcftools\.stats\.parsed\.tsv$/
    return channel
        .fromPath("${outdir}/03_bcftools_stats/*.bcftools.stats.parsed.tsv")
        .filter { p -> p.getName() ==~ reStats }
        .map { tsv ->
            def m = (tsv.getName() =~ reStats)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${tsv.getName()}")
            def chr = m[0][1]
            tuple(chr, tsv)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover plink QC per-sample TSVs from outdir/05_plink_qc/
// ---------------------------------------------------------------------------
def discoverPlinkQcTsvs(String outdir) {
    def reQc = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.qc_per_sample\.tsv$/
    return channel
        .fromPath("${outdir}/05_plink_qc/*.qc_per_sample.tsv")
        .filter { p -> p.getName() ==~ reQc }
        .map { tsv ->
            def m = (tsv.getName() =~ reQc)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${tsv.getName()}")
            def chr = m[0][1]
            tuple(chr, tsv)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover rare-only VCFs from lai_rare outdir
// ---------------------------------------------------------------------------
def discoverLaiRareVcfs(String inputDir, String globPattern) {
    def reRare = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.rare\.vcf\.gz$/
    return channel
        .fromPath("${inputDir}/${globPattern}")
        .filter { p -> p.getName() ==~ reRare }
        .map { vcf_gz ->
            def m = (vcf_gz.getName() =~ reRare)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${vcf_gz.getName()}")
            def chr = m[0][1]
            def tbi = file("${vcf_gz}.tbi")
            if( !tbi.exists() ) throw new IllegalStateException("Missing .tbi for rare VCF: ${vcf_gz}")
            tuple(chr, vcf_gz, tbi)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover per-chromosome painting scan outputs from a previous run
// (publishDir layout of ANALYZE_RARE_ALLELE_SHARING: ``<per_chr_dir>/
//  dnabr.hg38.2723.chr<C>.pairwise_segments.tsv.gz`` + matching
//  ``*.sharing_scan.summary.json``).  Returned as ``tuple(chr, segments_tsv_gz,
// summary_json)``, mirroring the shape expected by the AGGREGATE step.
// ---------------------------------------------------------------------------
def discoverPaintingPerChr(String perChrDir) {
    def reSeg = ~/dnabr\.hg38\.2723\.chr(\d+|X|Y|MT)\.pairwise_segments\.tsv\.gz$/
    return channel
        .fromPath("${perChrDir}/*.pairwise_segments.tsv.gz")
        .filter { p -> p.getName() ==~ reSeg }
        .map { seg_gz ->
            def m = (seg_gz.getName() =~ reSeg)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${seg_gz.getName()}")
            def chr = m[0][1]
            def summary = file(seg_gz.toString().replaceFirst(/\.pairwise_segments\.tsv\.gz$/, '.sharing_scan.summary.json'))
            if( !summary.exists() ) throw new IllegalStateException("Missing scan summary JSON for ${seg_gz}")
            tuple(chr, seg_gz, summary)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover per-chromosome Gnomix painting .msp files for Module 17.
// Layout: ``<msp_dir>/chr<C>/dnabr.chr<C>.results.msp``.  Returns tuple(chr, msp)
// with ``chr`` as the suffix-only form ("1".."22","X","Y","MT") to match
// ``discoverLaiRareVcfs`` so the two channels join cleanly by key.
// ---------------------------------------------------------------------------
def discoverMspFiles(String mspDir, String globPattern) {
    def reMsp = ~/dnabr\.chr(\d+|X|Y|MT)\.results\.msp$/
    return channel
        .fromPath("${mspDir}/${globPattern}")
        .filter { p -> p.getName() ==~ reMsp }
        .map { msp ->
            def m = (msp.getName() =~ reMsp)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${msp.getName()}")
            tuple(m[0][1], msp)
        }
}

// ---------------------------------------------------------------------------
// Helper: discover pre-staged external panel VCFs for Module 21 (presence channel).
// Layout (tools/stage_presence_panels.sh): ``<panelDir>/<panelId>/chr<C>.vcf.gz`` + .tbi.
// Returns tuple(chr, vcf, tbi) with ``chr`` suffix-only ("1".."22","X","Y","MT") so it
// joins cleanly with ``discoverLaiRareVcfs``.  Fails loud on a missing .tbi (the panels
// must be staged first; the pipeline does NOT download from gs:// inside a process).
// ---------------------------------------------------------------------------
def discoverPresencePanelVcfs(String panelDir, String panelId) {
    def reVcf = ~/chr(\d+|X|Y|MT)\.vcf\.gz$/
    return channel
        .fromPath("${panelDir}/${panelId}/chr*.vcf.gz")
        .filter { p -> p.getName() ==~ reVcf }
        .map { vcf ->
            def m = (vcf.getName() =~ reVcf)
            if( !m.matches() ) throw new IllegalArgumentException("Cannot extract chr from ${vcf.getName()}")
            def tbi = file("${vcf}.tbi")
            if( !tbi.exists() ) throw new IllegalStateException(
                "Missing .tbi for panel VCF: ${vcf}. Run tools/stage_presence_panels.sh first.")
            tuple(m[0][1], vcf, tbi)
        }
}

// ---------------------------------------------------------------------------
// Helper: normalise ``params.painting_chromosomes`` to the suffix-only form
// used by ``discoverLaiRareVcfs`` / ``discoverPaintingPerChr`` (which extract
// the capture group ``(\d+|X|Y|MT)`` -> "1".."22","X","Y","MT", without the
// "chr" prefix).  Accepts a List or a comma-separated String; tolerates the
// optional "chr" prefix on each entry.  See the comment block on
// ``params.painting_chromosomes`` in ``nextflow.config`` for the biological
// rationale of the default (22 autosomes).
// ---------------------------------------------------------------------------
def parsePaintingChromosomes(value) {
    def items
    if( value instanceof List ) {
        items = value
    } else if( value instanceof CharSequence ) {
        items = value.toString().split(',') as List
    } else {
        throw new IllegalStateException(
            "params.painting_chromosomes must be a List or comma-separated String; got ${value?.class?.name}"
        )
    }
    return items
        .collect { it?.toString()?.trim() }
        .findAll { it }
        .collect { it.toLowerCase().startsWith('chr') ? it.substring(3) : it }
}

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

workflow {
    // -----------------------------------------------------------------------
    // Scripts & constants (declared once, before any conditional)
    // -----------------------------------------------------------------------
    def chrPattern              = java.util.regex.Pattern.compile(params.chr_regex)
    def parse_bcftools_stats_py = file("${projectDir}/bin/parse_bcftools_stats.py")
    def aggregate_qc_report_py  = file("${projectDir}/bin/aggregate_qc_report.py")
    def make_keep_samples_py    = file("${projectDir}/bin/make_keep_samples.py")
    def sfs_report_py           = file("${projectDir}/bin/sfs_report.py")
    def build_ancestral_tsv_py  = file("${projectDir}/bin/build_ancestral_tsv.py")
    def daf_dsfs_py             = file("${projectDir}/bin/daf_dsfs.py")
    def ld_decay_py             = file("${projectDir}/bin/ld_decay.py")
    def tag_summary_py          = file("${projectDir}/bin/tag_summary.py")
    def rare_snp_tract_py       = file("${projectDir}/bin/rare_snp_tract_distribution.py")
    def individual_distance_modes_py = file("${projectDir}/bin/individual_snp_distance_modes.py")
    def rare_allele_sharing_painter_py = file("${projectDir}/bin/rare_allele_sharing_painter.py")
    def ibd_community_enhanced_py = file("${projectDir}/bin/ibd_community_enhanced.py")
    def rare_in_lai_py          = file("${projectDir}/bin/rare_variants_in_lai_tracts.py")
    def aggregate_rare_in_lai_py = file("${projectDir}/bin/aggregate_rare_in_lai.py")
    def rare_on_lai_plot_py     = file("${projectDir}/scripts/plot_rare_karyogram_lai.py")
    def external_presence_audit_py = file("${projectDir}/bin/external_presence_audit.py")
    def genomic_context_py         = file("${projectDir}/bin/genomic_context.py")
    def aggregate_presence_py      = file("${projectDir}/bin/aggregate_presence_channel.py")
    def build_feature_store_py     = file("${projectDir}/bin/build_feature_store.py")

    def empty_placeholder = file("${projectDir}/conf/empty.txt")
    if( !empty_placeholder.exists() ) {
        throw new IllegalStateException("Missing placeholder: ${empty_placeholder}. Please keep conf/empty.txt in the repo.")
    }
    def ch_empty_file = channel.value(empty_placeholder)

    // -----------------------------------------------------------------------
    // Effective flags
    // -----------------------------------------------------------------------
    // QC modules 01–06 are gated by BOTH the per-module ``enable_*`` flag
    // AND the master ``run_qc`` switch.  This makes ``--run_qc false`` an
    // effective single-flag way to skip the entire QC stage when running
    // advanced modules (12+) on already-processed data.  Per-module flags
    // remain useful for running a subset of QC (e.g. only normalisation).
    def do_smoke_test  = params.enable_smoke_test
    def do_norm        = params.enable_norm        && params.run_qc
    def do_filter      = params.enable_filter      && params.run_qc
    def do_bcfstats    = params.enable_bcfstats    && params.run_qc
    def do_pgen        = params.enable_pgen        && params.run_qc
    def do_missing_het = params.enable_missing_het && params.run_qc
    def do_report      = params.enable_report      && params.run_qc
    def do_lai_rare    = params.enable_lai_rare
    def do_rare_tracts = params.enable_rare_snp_tracts || params.runRareSnpTractAnalysis
    def do_distance_modes = params.enable_individual_snp_distance_modes
    def do_painting = params.enable_rare_allele_painting
    def do_ibd_enhanced = params.enable_ibd_enhanced
    def do_rare_in_lai = params.enable_rare_in_lai
    def do_rare_on_lai = params.enable_rare_on_lai_painting
    def do_asibd_comparator = params.enable_asibd_comparator
    def do_presence_channel = params.enable_presence_channel
    def do_feature_build = params.enable_feature_build

    if( do_rare_tracts && params.rare_tract_input_format != 'vcf_rare' ) {
        throw new IllegalStateException("This project currently supports rare SNP tracts only from upstream rare VCFs (rare_tract_input_format='vcf_rare').")
    }
    if( do_rare_tracts && do_lai_rare && params.lai_rare_remove_info ) {
        throw new IllegalStateException("Rare SNP tract analysis requires INFO/AC, INFO/AN and INFO/AF in lai_rare outputs. Set lai_rare_remove_info=false.")
    }
    if( do_distance_modes && params.distance_mode_input_format != 'vcf_rare' ) {
        throw new IllegalStateException("This project currently supports per-individual distance modes only from upstream rare VCFs (distance_mode_input_format='vcf_rare').")
    }
    if( do_distance_modes && params.distance_mode_distance_units != 'bp' ) {
        throw new IllegalStateException("Per-individual distance modes currently support only bp distances (distance_mode_distance_units='bp').")
    }
    if( do_distance_modes && do_lai_rare && params.lai_rare_keep_format && !params.lai_rare_keep_format.split(',').collect { it.trim() }.contains('GT') ) {
        throw new IllegalStateException("Per-individual distance modes require FORMAT/GT in lai_rare outputs. Set lai_rare_keep_format to include GT.")
    }
    if( do_painting && params.painting_input_format != 'vcf_rare' ) {
        throw new IllegalStateException("Rare allele sharing painting currently supports only upstream rare VCFs (painting_input_format='vcf_rare').")
    }
    if( do_rare_in_lai && params.rare_in_lai_input_format != 'vcf_rare' ) {
        throw new IllegalStateException("Rare-in-LAI-tracts currently supports only upstream rare VCFs (rare_in_lai_input_format='vcf_rare').")
    }
    if( do_rare_in_lai && do_lai_rare && params.lai_rare_keep_format && !params.lai_rare_keep_format.split(',').collect { it.trim() }.contains('GT') ) {
        throw new IllegalStateException("Rare-in-LAI-tracts requires FORMAT/GT in lai_rare outputs. Set lai_rare_keep_format to include GT.")
    }
    if( do_painting && do_lai_rare && params.lai_rare_keep_format && !params.lai_rare_keep_format.split(',').collect { it.trim() }.contains('GT') ) {
        throw new IllegalStateException("Rare allele sharing painting requires FORMAT/GT in lai_rare outputs. Set lai_rare_keep_format to include GT.")
    }

    def any_downstream = params.run_downstream && (
        params.enable_sfs || params.enable_build_ancestral_tsv ||
        params.enable_daf || params.enable_ld || params.enable_tag_snps
    )

    // Who needs filtered VCFs (from 02 or outdir)?
    def need_filtered = do_bcfstats || do_pgen || do_report || do_lai_rare || (
        any_downstream && (params.enable_sfs || params.enable_build_ancestral_tsv || params.enable_daf)
    )
    // Who needs PGEN (from 04 or outdir)?
    def need_pgen = do_missing_het || (
        any_downstream && (params.enable_ld || params.enable_tag_snps)
    )

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // 00 — Smoke test (Slurm + Singularity sanity check)
    // -----------------------------------------------------------------------
    if( do_smoke_test ) {
        SMOKE_TEST()
    }

    // Raw VCFs (only if norm or filter will run)
    // -----------------------------------------------------------------------
    def ch_vcfs
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
    } else {
        ch_vcfs = channel.empty()
    }

    def ch_ref = channel.value( file(params.ref_fasta) )

    // -----------------------------------------------------------------------
    // 01  PREPROCESS_NORM_LEFTALIGN
    // -----------------------------------------------------------------------
    def ch_norm_vcfs
    if( do_norm ) {
        def (_ch_norm, _ch_norm_logs) = PREPROCESS_NORM_LEFTALIGN(
            ch_vcfs.combine(ch_ref)
        )
        ch_norm_vcfs = _ch_norm
    } else if( do_filter ) {
        // 02 needs norm VCFs but 01 is skipped → discover from outdir
        ch_norm_vcfs = discoverNormVcfs(params.outdir)
    } else {
        ch_norm_vcfs = channel.empty()
    }

    // -----------------------------------------------------------------------
    // 02  PREPROCESS_FILTER_SNV_BIALLELIC_PASS
    // -----------------------------------------------------------------------
    def ch_filtered  // tuple(chr, counts_tsv, vcf_pass, vcf_pass_tbi)
    if( do_filter ) {
        def ch_raw_keyed  = ch_vcfs.map      { chr, vcf, tbi  -> tuple(chr, vcf, tbi) }
        def ch_norm_keyed = ch_norm_vcfs.map  { chr, nvcf, ntbi -> tuple(chr, nvcf, ntbi) }
        def ch_for_filter = ch_raw_keyed
            .join(ch_norm_keyed)
            .map { chr, vcf, tbi, nvcf, ntbi -> tuple(chr, vcf, tbi, nvcf, ntbi) }
        ch_filtered = PREPROCESS_FILTER_SNV_BIALLELIC_PASS(ch_for_filter)
    } else {
        ch_filtered = channel.empty()
    }

    // Derive VCF-only and counts-only channels (from 02 live output or outdir)
    def ch_filtered_vcfs   // tuple(chr, vcf_pass, tbi)
    def ch_counts          // tuple(chr, counts_tsv)
    if( do_filter ) {
        ch_filtered_vcfs = ch_filtered.map { chr, cts, vcf, tbi -> tuple(chr, vcf, tbi) }
        ch_counts        = ch_filtered.map { chr, cts, vcf, tbi -> tuple(chr, cts) }
    } else if( need_filtered ) {
        def filtered_dir = params.filtered_input_dir ?: params.outdir
        ch_filtered_vcfs = discoverFilteredVcfs(filtered_dir)
        ch_counts        = discoverCountsTsvs(filtered_dir)
    } else {
        ch_filtered_vcfs = channel.empty()
        ch_counts        = channel.empty()
    }

    // -----------------------------------------------------------------------
    // 02.1  LAI_RARE_BIALELIC_ONLY
    // -----------------------------------------------------------------------
    def ch_lai_rare_vcfs
    if( do_lai_rare ) {
        def ch_lai_input = ch_filtered_vcfs.filter { chr, vcf, tbi ->
            boolean chr_matches  = "dnabr.hg38.2723.chr${chr}.vcf.gz" ==~ chrPattern
            // chrX/Y/MT: hemicigosis distorsiona Jaccard rare-variant en cohorts M/F mixtas
            boolean keep_sex_chr = params.lai_rare_exclude_sex_chr ? !(chr in ["X","Y","MT"]) : true
            return chr_matches && keep_sex_chr
        }
        def lai_rare_out = LAI_RARE_BIALELIC_ONLY(ch_lai_input)
        ch_lai_rare_vcfs = lai_rare_out.rare_vcfs
    } else if( do_rare_tracts || do_distance_modes || do_rare_in_lai || do_rare_on_lai || do_presence_channel || do_feature_build || (do_painting && !params.painting_aggregate_only) ) {
        // Discovery path: lai_rare no se genera live, se descubre del dir configurado.
        // Cada módulo consumidor declara (label, dir, glob). Regla general: todos los
        // Los consumidores habilitados deben apuntar al mismo directorio y glob. Si difieren, usa
        // un mensaje claro). Añadir un consumidor nuevo = una línea en este mapa.
        def lai_rare_consumers = [
            [enabled: do_rare_tracts,                                  label: 'rare_tract (M12)',   dir: params.rare_tract_input_dir,    glob: params.rare_tract_input_glob],
            [enabled: do_distance_modes,                               label: 'distance_mode (M13)', dir: params.distance_mode_input_dir, glob: params.distance_mode_input_glob],
            [enabled: do_painting && !params.painting_aggregate_only,  label: 'painting (M14)',      dir: params.painting_input_dir,      glob: params.painting_input_glob],
            [enabled: do_rare_in_lai,                                  label: 'rare_in_lai (M17)',   dir: params.rare_in_lai_input_dir,   glob: params.rare_in_lai_input_glob],
            [enabled: do_rare_on_lai,                                  label: 'rare_on_lai (M19)',   dir: params.rare_on_lai_input_dir,   glob: params.rare_on_lai_input_glob],
            [enabled: do_presence_channel,                             label: 'presence_channel (M21)', dir: params.presence_channel_input_dir, glob: params.presence_channel_input_glob],
            [enabled: do_feature_build,                                label: 'feature_build (M20)', dir: params.feature_build_input_dir,    glob: params.feature_build_input_glob],
        ].findAll { it.enabled }

        lai_rare_consumers.each { c ->
            if( !file(c.dir).exists() ) {
                throw new IllegalStateException("${c.label} input dir not found: ${c.dir}")
            }
        }
        def distinct_dirs  = lai_rare_consumers.collect { it.dir }.unique()
        def distinct_globs = lai_rare_consumers.collect { it.glob }.unique()
        if( distinct_dirs.size() > 1 || distinct_globs.size() > 1 ) {
            throw new IllegalStateException(
                "Multiple lai_rare-consuming modules are enabled without live lai_rare generation, but their " +
                "input dir/glob differ: ${lai_rare_consumers.collect { "${it.label}=${it.dir}/${it.glob}" }.join(' ; ')}. " +
                "Point them at the same upstream lai_rare dir/glob, or run enable_lai_rare=true, or run the modules separately."
            )
        }
        ch_lai_rare_vcfs = discoverLaiRareVcfs(lai_rare_consumers[0].dir, lai_rare_consumers[0].glob)
    } else {
        ch_lai_rare_vcfs = channel.empty()
    }

    // -----------------------------------------------------------------------
    // 03  QC_BCFTOOLS_STATS
    // -----------------------------------------------------------------------
    def ch_bcftools_stats  // tuple(chr, stats_parsed_tsv)
    if( do_bcfstats ) {
        def ch_for_bcfstats = ch_counts
            .join(ch_filtered_vcfs)
            .map { chr, cts, vcf, tbi -> tuple(chr, cts, vcf, tbi, parse_bcftools_stats_py) }
        def (_ch_stats, _ch_stats_txt) = QC_BCFTOOLS_STATS(ch_for_bcfstats)
        ch_bcftools_stats = _ch_stats
    } else if( do_report ) {
        // 06 needs stats but 03 is skipped → discover from outdir
        ch_bcftools_stats = discoverBcftoolsStats(params.outdir)
    } else {
        ch_bcftools_stats = channel.empty()
    }

    // -----------------------------------------------------------------------
    // 04  QC_PLINK_MAKE_PGEN
    // -----------------------------------------------------------------------
    def ch_pgen_out  // tuple(chr, pgen, pvar, psam)
    if( do_pgen ) {
        def ch_for_pgen = ch_counts
            .join(ch_filtered_vcfs)
            .map { chr, cts, vcf, tbi -> tuple(chr, cts, vcf, tbi) }
        def (_ch_pgen, _ch_pgen_logs) = QC_PLINK_MAKE_PGEN(ch_for_pgen)
        ch_pgen_out = _ch_pgen
    } else if( need_pgen ) {
        // 05 or downstream LD/tags need PGEN → discover from outdir
        ch_pgen_out = discoverPgenTriplets(params.outdir)
    } else {
        ch_pgen_out = channel.empty()
    }

    // -----------------------------------------------------------------------
    // 05  QC_PLINK_MISSING_HET
    // -----------------------------------------------------------------------
    def ch_plink_qc  // tuple(chr, qc_per_sample_tsv)
    if( do_missing_het ) {
        def (_ch_qc, _ch_imiss, _ch_het, _ch_gf_logs, _ch_miss_logs, _ch_het_logs) = QC_PLINK_MISSING_HET(
            ch_pgen_out.map { chr, pgen, pvar, psam -> tuple(chr, pgen, pvar, psam, aggregate_qc_report_py) }
        )
        ch_plink_qc = _ch_qc
    } else if( do_report ) {
        // 06 needs QC TSVs but 05 is skipped → discover from outdir
        ch_plink_qc = discoverPlinkQcTsvs(params.outdir)
    } else {
        ch_plink_qc = channel.empty()
    }

    // -----------------------------------------------------------------------
    // 06  QC_AGGREGATE_REPORT_PY
    // -----------------------------------------------------------------------
    if( do_report ) {
        ch_plink_qc
            .map { _chr, qc -> qc }
            .collect()
            .map { files -> tuple('agg', files) }
            .set { ch_qc_keyed }

        ch_counts
            .map { _chr, cts -> cts }
            .collect()
            .map { files -> tuple('agg', files) }
            .set { ch_counts_keyed }

        ch_bcftools_stats
            .map { _chr, stats -> stats }
            .collect()
            .map { files -> tuple('agg', files) }
            .set { ch_stats_keyed }

        ch_qc_keyed
            .join(ch_stats_keyed)
            .join(ch_counts_keyed)
            .map { _key, qc_files, stats_files, counts_files ->
                tuple(qc_files, stats_files, counts_files, aggregate_qc_report_py)
            }
            .set { ch_agg_in }

        QC_AGGREGATE_REPORT_PY(ch_agg_in)
    }

    // -----------------------------------------------------------------------
    // 12  Rare SNP tract analysis from upstream rare-only VCFs
    // -----------------------------------------------------------------------
    if( do_rare_tracts ) {
        def ch_metadata_opt
        if( params.rare_tract_metadata ) {
            def metadata = file(params.rare_tract_metadata)
            if( !metadata.exists() ) {
                throw new IllegalStateException("rare_tract_metadata not found: ${metadata}")
            }
            ch_metadata_opt = channel.value(metadata)
        } else {
            ch_metadata_opt = ch_empty_file
        }

        def ch_genetic_map_opt
        if( params.rare_tract_genetic_map ) {
            def genetic_map = file(params.rare_tract_genetic_map)
            if( !genetic_map.exists() ) {
                throw new IllegalStateException("rare_tract_genetic_map not found: ${genetic_map}")
            }
            ch_genetic_map_opt = channel.value(genetic_map)
        } else {
            ch_genetic_map_opt = ch_empty_file
        }

        def rare_scan_out = ANALYZE_RARE_SNP_TRACTS(
            ch_lai_rare_vcfs
                .map { chr, vcf_gz, tbi -> tuple(chr, vcf_gz, tbi, rare_snp_tract_py) }
        )

        rare_scan_out.window_scores
            .map { _chr, scores -> scores }
            .collect()
            .map { files -> tuple('agg', files) }
            .set { ch_rare_window_scores_keyed }

        rare_scan_out.scan_summaries
            .map { _chr, summary -> summary }
            .collect()
            .map { files -> tuple('agg', files) }
            .set { ch_rare_scan_summaries_keyed }

        ch_rare_window_scores_keyed
            .join(ch_rare_scan_summaries_keyed)
            .map { _key, window_score_files, scan_summary_files ->
                tuple(window_score_files, scan_summary_files, rare_snp_tract_py)
            }
            .set { ch_rare_agg_in }

        AGGREGATE_RARE_SNP_TRACTS(
            ch_rare_agg_in,
            ch_metadata_opt,
            ch_genetic_map_opt
        )
    }

    // -----------------------------------------------------------------------
    // 13  Per-individual distance modes from upstream rare-only VCFs
    // -----------------------------------------------------------------------
    if( do_distance_modes ) {
        def ch_distance_sample_ids_opt
        if( params.distance_mode_sample_ids_file ) {
            def sample_ids = file(params.distance_mode_sample_ids_file)
            if( !sample_ids.exists() ) {
                throw new IllegalStateException("distance_mode_sample_ids_file not found: ${sample_ids}")
            }
            def sample_ids_payload_b64 = sample_ids.getText('UTF-8').bytes.encodeBase64().toString()
            ch_distance_sample_ids_opt = channel.value(sample_ids_payload_b64)
        } else {
            ch_distance_sample_ids_opt = channel.value('')
        }

        def distance_scan_out = ANALYZE_INDIVIDUAL_SNP_DISTANCE_MODES(
            ch_lai_rare_vcfs
                .map { chr, vcf_gz, tbi -> tuple(chr, vcf_gz, tbi, individual_distance_modes_py) },
            ch_distance_sample_ids_opt
        )

        distance_scan_out.summary_files
            .flatMap { files -> files }
            .filter { f -> f.getName().endsWith(".individual_distance_summary.tsv") }
            .collect()
            .map { files -> tuple('agg', files) }
            .set { ch_distance_individual_summary_keyed }

        distance_scan_out.summary_files
            .flatMap { files -> files }
            .filter { f -> f.getName().endsWith(".cohort_distance_summary.json") }
            .collect()
            .map { files -> tuple('agg', files) }
            .set { ch_distance_cohort_summary_keyed }

        ch_distance_individual_summary_keyed
            .join(ch_distance_cohort_summary_keyed)
            .map { _key, individual_summary_files, cohort_summary_files ->
                tuple(individual_summary_files, cohort_summary_files, individual_distance_modes_py)
            }
            .set { ch_distance_agg_in }

        AGGREGATE_INDIVIDUAL_SNP_DISTANCE_MODES(ch_distance_agg_in)
    }

    // -----------------------------------------------------------------------
    // 14  Rare allele sharing painting from upstream rare-only VCFs
    // -----------------------------------------------------------------------
    if( do_painting ) {
        // ``ch_painting_segments_by_chr`` / ``ch_painting_summaries_by_chr``
        // carry the per-chromosome scan artefacts into the AGGREGATE step.
        // They are populated either by running ANALYZE live (default) or by
        // discovering the outputs of a previous scan on disk when
        // ``painting_aggregate_only`` is set.
        def ch_painting_segments_by_chr
        def ch_painting_summaries_by_chr

        // Resolve the effective chromosome whitelist once for both paths.
        // Filter is applied symmetrically before ANALYZE (live) and before the
        // AGGREGATE collect (aggregate-only) so the genome-wide aggregate that
        // M16.5 consumes never includes excluded chromosomes.  See the comment
        // block on ``params.painting_chromosomes`` for the biological rationale.
        def painting_chrs = parsePaintingChromosomes(params.painting_chromosomes)
        if( painting_chrs.isEmpty() ) {
            throw new IllegalStateException("params.painting_chromosomes resolved to empty list")
        }
        log.info "[M14] painting_chromosomes (${painting_chrs.size()}): ${painting_chrs.collect { 'chr' + it }.join(',')}"

        if( params.painting_aggregate_only ) {
            def per_chr_dir = params.painting_per_chr_dir ?: "${params.painting_results_dir}/per_chr"
            def per_chr_dir_f = file(per_chr_dir)
            if( !per_chr_dir_f.exists() ) {
                throw new IllegalStateException(
                    "painting_aggregate_only=true but per-chr scan directory not found: ${per_chr_dir_f}. " +
                    "Set params.painting_per_chr_dir or run the scan first."
                )
            }
            def ch_per_chr = discoverPaintingPerChr(per_chr_dir)
                .filter { chr, _seg, _sum -> painting_chrs.contains(chr) }
            ch_painting_segments_by_chr  = ch_per_chr.map { chr, seg, _sum -> tuple(chr, seg) }
            ch_painting_summaries_by_chr = ch_per_chr.map { chr, _seg, sum -> tuple(chr, sum) }
        } else {
            def ch_painting_sample_ids_opt
            if( params.painting_sample_ids_file ) {
                def sample_ids = file(params.painting_sample_ids_file)
                if( !sample_ids.exists() ) {
                    throw new IllegalStateException("painting_sample_ids_file not found: ${sample_ids}")
                }
                def sample_ids_payload_b64 = sample_ids.getText('UTF-8').bytes.encodeBase64().toString()
                ch_painting_sample_ids_opt = channel.value(sample_ids_payload_b64)
            } else {
                ch_painting_sample_ids_opt = channel.value('')
            }

            def painting_scan_out = ANALYZE_RARE_ALLELE_SHARING(
                ch_lai_rare_vcfs
                    .filter { chr, _vcf, _tbi -> painting_chrs.contains(chr) }
                    .map { chr, vcf_gz, tbi -> tuple(chr, vcf_gz, tbi, rare_allele_sharing_painter_py) },
                ch_painting_sample_ids_opt
            )
            ch_painting_segments_by_chr  = painting_scan_out.pairwise_segments
            ch_painting_summaries_by_chr = painting_scan_out.scan_summaries
        }

        ch_painting_segments_by_chr
            .map { _chr, segments -> segments }
            .collect()
            .map { files -> tuple('agg', files) }
            .set { ch_painting_segments_keyed }

        ch_painting_summaries_by_chr
            .map { _chr, summary -> summary }
            .collect()
            .map { files -> tuple('agg', files) }
            .set { ch_painting_summaries_keyed }

        ch_painting_segments_keyed
            .join(ch_painting_summaries_keyed)
            .map { _key, segment_files, summary_files ->
                tuple(segment_files, summary_files, rare_allele_sharing_painter_py)
            }
            .set { ch_painting_agg_in }

        AGGREGATE_RARE_ALLELE_SHARING(ch_painting_agg_in)
    }

    // -----------------------------------------------------------------------
    // 17  Rare variants in LAI tracts (Gnomix local-ancestry painting)
    // -----------------------------------------------------------------------
    // Join per-chr rare VCF (ch_lai_rare_vcfs) with the per-chr Gnomix .msp,
    // restricted to the configured chromosome whitelist (default 22 autosomes;
    // the Gnomix .msp covers only autosomes -- same biological rationale as
    // painting_chromosomes).  Per-chr ANALYZE -> genome-wide AGGREGATE.
    if( do_rare_in_lai ) {
        def rare_in_lai_chrs = parsePaintingChromosomes(params.rare_in_lai_chromosomes)
        if( rare_in_lai_chrs.isEmpty() ) {
            throw new IllegalStateException("params.rare_in_lai_chromosomes resolved to empty list")
        }
        log.info "[M17] rare_in_lai_chromosomes (${rare_in_lai_chrs.size()}): ${rare_in_lai_chrs.collect { 'chr' + it }.join(',')}"

        def metadata_file = file(params.rare_in_lai_metadata)
        if( !metadata_file.exists() ) {
            throw new IllegalStateException("rare_in_lai_metadata not found: ${metadata_file}")
        }

        def msp_dir = file(params.rare_in_lai_msp_dir)
        if( !msp_dir.exists() ) {
            throw new IllegalStateException("rare_in_lai_msp_dir not found: ${msp_dir}")
        }
        def ch_msp = discoverMspFiles(params.rare_in_lai_msp_dir, params.rare_in_lai_msp_glob)

        def ch_rare_in_lai_in = ch_lai_rare_vcfs
            .filter { chr, _vcf, _tbi -> rare_in_lai_chrs.contains(chr) }
            .join(ch_msp.filter { chr, _msp -> rare_in_lai_chrs.contains(chr) })
            .map { chr, vcf_gz, tbi, msp -> tuple(chr, vcf_gz, tbi, msp, metadata_file, rare_in_lai_py) }

        def rare_in_lai_out = ANALYZE_RARE_IN_LAI(ch_rare_in_lai_in)

        rare_in_lai_out.summaries
            .map { _chr, summary -> summary }
            .collect()
            .map { files -> tuple(files, aggregate_rare_in_lai_py) }
            .set { ch_rare_in_lai_agg_in }

        AGGREGATE_RARE_IN_LAI(ch_rare_in_lai_agg_in)
    }

    // -----------------------------------------------------------------------
    // 19  Rare variants on LAI painting (per-chromosome publication figure)
    // -----------------------------------------------------------------------
    // Une el VCF de raras por-crom con el .msp Gnomix por-crom (autosomas) y, por
    // cromosoma, computa la densidad estratificada (windowing, mismo bin que M17) y
    // dibuja la figura de 3 paneles. Self-contained: 1 proceso por cromosoma.
    if( do_rare_on_lai ) {
        def rare_on_lai_chrs = parsePaintingChromosomes(params.rare_on_lai_chromosomes)
        if( rare_on_lai_chrs.isEmpty() ) {
            throw new IllegalStateException("params.rare_on_lai_chromosomes resolved to empty list")
        }
        log.info "[M19] rare_on_lai_chromosomes (${rare_on_lai_chrs.size()}): ${rare_on_lai_chrs.collect { 'chr' + it }.join(',')}"

        def metadata_file_m19 = file(params.rare_on_lai_metadata)
        if( !metadata_file_m19.exists() ) {
            throw new IllegalStateException("rare_on_lai_metadata not found: ${metadata_file_m19}")
        }

        def msp_dir_m19 = file(params.rare_on_lai_msp_dir)
        if( !msp_dir_m19.exists() ) {
            throw new IllegalStateException("rare_on_lai_msp_dir not found: ${msp_dir_m19}")
        }
        def ch_msp_m19 = discoverMspFiles(params.rare_on_lai_msp_dir, params.rare_on_lai_msp_glob)

        def ch_rare_on_lai_in = ch_lai_rare_vcfs
            .filter { chr, _vcf, _tbi -> rare_on_lai_chrs.contains(chr) }
            .join(ch_msp_m19.filter { chr, _msp -> rare_on_lai_chrs.contains(chr) })
            .map { chr, vcf_gz, tbi, msp -> tuple(chr, vcf_gz, tbi, msp, metadata_file_m19, rare_in_lai_py, rare_on_lai_plot_py) }

        RARE_ON_LAI_PAINTING(ch_rare_on_lai_in)
    }

    // -----------------------------------------------------------------------
    // 21: canal de presencia externa
    // -----------------------------------------------------------------------
    // Para cada panel NAM externo preparado y cada cromosoma, clasifica si el alelo raro de
    // DNABR aparece afuera (PRESENT_ALLELE refuta privacidad; ABSENT no confirma founder).
    // Multi-panel parametrizable (params.presence_panels): NAMBR-128 VQSR vs 71-native .raw =
    // comparación apples-to-apples (--panel-pass-only por panel). Per (panel×chr) ANALYZE ->
    // per-panel AGGREGATE (suma cruda). El panel es input de referencia: tools/stage_presence_panels.sh.
    if( do_presence_channel ) {
        def presence_chrs = parsePaintingChromosomes(params.presence_channel_chromosomes)
        if( presence_chrs.isEmpty() ) {
            throw new IllegalStateException("params.presence_channel_chromosomes resolved to empty list")
        }
        log.info "[M21] presence_channel_chromosomes (${presence_chrs.size()}): ${presence_chrs.collect { 'chr' + it }.join(',')}"

        def panels = params.presence_panels
        if( !(panels instanceof List) || panels.isEmpty() ) {
            throw new IllegalStateException("params.presence_panels must be a non-empty List of panel maps")
        }

        def fasta_f = file(params.ref_fasta)
        def fai_f   = file("${params.ref_fasta}.fai")
        for( f in [fasta_f, fai_f] ) {
            if( !f.exists() ) throw new IllegalStateException("Missing FASTA/index for M21: ${f}")
        }

        if( !file(params.presence_panel_dir).exists() ) {
            throw new IllegalStateException(
                "presence_panel_dir not found: ${params.presence_panel_dir}. Run tools/stage_presence_panels.sh first.")
        }

        def segdup_f    = file("${params.presence_lcr_mask_dir}/genomicSuperDups.txt.gz")
        def simplerep_f = file("${params.presence_lcr_mask_dir}/simpleRepeat.txt.gz")
        def blacklist_f = file("${params.presence_lcr_mask_dir}/hg38-blacklist.v2.bed.gz")
        for( f in [segdup_f, simplerep_f, blacklist_f] ) {
            if( !f.exists() ) throw new IllegalStateException("Missing LCR mask input for M21: ${f}")
        }

        // (1) LCR mask genome-wide (una vez) -> el bin filtra por cromosoma en lectura
        def ch_lcr_bed = BUILD_PRESENCE_LCR_MASK(
            channel.value(tuple(segdup_f, simplerep_f, blacklist_f))
        ).bed

        // (2) tareas (panel × cromosoma): une el rare VCF por-chr con el panel VCF por-chr
        def ch_rare_for_presence = ch_lai_rare_vcfs.filter { chr, _v, _t -> presence_chrs.contains(chr) }
        def ch_tasks = channel.empty()
        for( panel in panels ) {
            if( !panel.id ) throw new IllegalStateException("presence_panels entry missing 'id': ${panel}")
            def pid   = panel.id
            def pver  = panel.version ?: panel.id
            def pdrop = panel.drop_sample ?: ''
            def ppass = panel.pass_only ? true : false
            def ch_panel = discoverPresencePanelVcfs(params.presence_panel_dir, pid)
                .filter { chr, _v, _t -> presence_chrs.contains(chr) }
            def ch_joined = ch_rare_for_presence
                .join(ch_panel)
                .map { chr, rvcf, rtbi, pvcf, ptbi ->
                    tuple(pid, pver, pdrop, ppass, chr, rvcf, rtbi, pvcf, ptbi) }
            ch_tasks = ch_tasks.mix(ch_joined)
        }

        // (3) adjunta mask + fasta + scripts; ANALYZE per (panel×chr)
        def ch_analyze_in = ch_tasks
            .combine(ch_lcr_bed)
            .map { pid, pver, pdrop, ppass, chr, rvcf, rtbi, pvcf, ptbi, lcr ->
                tuple(pid, pver, pdrop, ppass, chr, rvcf, rtbi, pvcf, ptbi,
                      lcr, fasta_f, fai_f, external_presence_audit_py, genomic_context_py) }

        def presence_out = ANALYZE_PRESENCE_CHANNEL(ch_analyze_in)

        // (4) AGGREGATE por panel (suma cruda de los summary.json por-chr de ese panel)
        presence_out.summary
            .map { pid, _chr, summary -> tuple(pid, summary) }
            .groupTuple()
            .map { pid, summaries -> tuple(pid, summaries, aggregate_presence_py) }
            .set { ch_presence_agg_in }

        AGGREGATE_PRESENCE_CHANNEL(ch_presence_agg_in)
    }

    // -----------------------------------------------------------------------
    // 20: feature store V.02-A con Q, Σℓ y densidad por individuo
    // -----------------------------------------------------------------------
    // Primera tabla oficial de features por individuo = piso reproducible de V.02.
    // Cohorte certificada (rare ∩ metadata, dedup) + Σℓ de M14 + densidad PSC por-chr.
    if( do_feature_build ) {
        def fb_chrs = parsePaintingChromosomes(params.feature_build_chromosomes)

        def fb_meta_f = file(params.feature_build_metadata_file)
        if( !fb_meta_f.exists() ) {
            throw new IllegalStateException("M20 enabled but metadata not found: ${fb_meta_f}")
        }
        def fb_sigma_f = file("${params.feature_build_m14_dir}/individual_sharing_summary.tsv")
        if( !fb_sigma_f.exists() ) {
            throw new IllegalStateException(
                "M20 enabled but M14 Σℓ summary not found: ${fb_sigma_f}. "
                + "Set params.feature_build_m14_dir or run M14."
            )
        }

        def ch_fb_rare = ch_lai_rare_vcfs.filter { chr, _v, _t -> fb_chrs.contains(chr) }

        // (1) cohorte certificada N (una muestra cualquiera del VCF de raras -> mismas samples)
        def ch_cohort = DEFINE_COHORT(
            ch_fb_rare.first().map { _chr, vcf, tbi -> tuple(vcf, tbi, fb_meta_f, build_feature_store_py) }
        ).cohort

        // (2) densidad PSC por cromosoma (paralelo)
        def ch_psc = COUNT_RARE_DENSITY(ch_fb_rare).psc

        // (3) AGGREGATE: cohorte + Σℓ + todos los PSC + metadata -> feature_store + manifest
        // ch_psc.collect() emite una lista con los N PSC y combine la aplana dentro de la tupla,
        // así que se reconstruye por slicing (all[0]=cohort, all[1..-1]=los N PSC como lista)
        // para que entren como UN solo path(psc_files) al proceso. sigma/meta/py = valores estáticos.
        def ch_fb_agg_in = ch_cohort
            .combine(ch_psc.collect())
            .map { all -> tuple(all[0], fb_sigma_f, fb_meta_f, all[1..-1], build_feature_store_py) }

        AGGREGATE_FEATURE_STORE(ch_fb_agg_in)
    }

    // -----------------------------------------------------------------------
    // 16.5  IBD community detection enhanced (biological interpretability)
    // -----------------------------------------------------------------------
    // Leiden multi-resolution communities + Laplacian-normalised Sym-NMF
    // (cophenetic K selection, ARI multi-seed stability), UMAP network layout,
    // hierarchical heatmap ordering and optional metadata (region/UF) overlay.
    // Runs on the M14 aggregate outputs (pair_sharing_summary, individual_
    // sharing_summary, all_pairwise_segments).
    if( do_ibd_enhanced ) {
        def ch_ibd_enh_segments
        def ch_ibd_enh_pair_summary
        def ch_ibd_enh_individual_summary

        if( do_painting ) {
            // Live path: reuse the AGGREGATE_RARE_ALLELE_SHARING outputs.
            ch_ibd_enh_segments           = AGGREGATE_RARE_ALLELE_SHARING.out.all_segments
            ch_ibd_enh_pair_summary       = AGGREGATE_RARE_ALLELE_SHARING.out.pair_summary
            ch_ibd_enh_individual_summary = AGGREGATE_RARE_ALLELE_SHARING.out.individual_summary
        } else {
            // Discovery path: locate the three TSVs on disk.  Allow
            // ibd_enhanced_input_dir to override, otherwise fall back to
            // the default M14 outdir.
            def ibd_enh_input_dir_str = params.ibd_enhanced_input_dir \
                ?: "${params.outdir}/14_rare_allele_sharing_painting"
            def ibd_enh_input_dir_f = file(ibd_enh_input_dir_str)
            if( !ibd_enh_input_dir_f.exists() ) {
                throw new IllegalStateException(
                    "M16.5 enabled but M14 aggregate directory not found: "
                    + "${ibd_enh_input_dir_f}. Set params.ibd_enhanced_input_dir "
                    + "or run M14."
                )
            }
            def seg_f  = file("${ibd_enh_input_dir_str}/all_pairwise_segments.tsv.gz")
            def pair_f = file("${ibd_enh_input_dir_str}/pair_sharing_summary.tsv")
            def ind_f  = file("${ibd_enh_input_dir_str}/individual_sharing_summary.tsv")
            for( f in [seg_f, pair_f, ind_f] ) {
                if( !f.exists() ) {
                    throw new IllegalStateException("Missing M14 aggregate output: ${f}")
                }
            }
            ch_ibd_enh_segments           = channel.value(seg_f)
            ch_ibd_enh_pair_summary       = channel.value(pair_f)
            ch_ibd_enh_individual_summary = channel.value(ind_f)
        }

        // Optional metadata: degrades to conf/empty.txt when absent.
        def ch_ibd_enh_meta
        if( params.ibd_enhanced_metadata_file ) {
            def meta_f = file(params.ibd_enhanced_metadata_file)
            if( !meta_f.exists() ) {
                throw new IllegalStateException(
                    "ibd_enhanced_metadata_file does not exist: ${meta_f}"
                )
            }
            ch_ibd_enh_meta = channel.value(meta_f)
        } else {
            ch_ibd_enh_meta = ch_empty_file
        }

        ch_ibd_enh_segments
            .combine(ch_ibd_enh_pair_summary)
            .combine(ch_ibd_enh_individual_summary)
            .map { segs, pair, ind -> tuple(segs, pair, ind, ibd_community_enhanced_py) }
            .set { ch_ibd_enh_in }

        IBD_COMMUNITY_ENHANCED(ch_ibd_enh_in, ch_ibd_enh_meta)

        // -------------------------------------------------------------------
        // M16.5 REPLOT — fan-out: re-render plots once per metadata column
        //
        // Re-uses the compute outputs from IBD_COMMUNITY_ENHANCED.  Each task
        // is one replot for one metadata colour, parallelised by Nextflow's
        // ``each`` operator.  No re-computation of the graph / Leiden / NMF.
        // Activated when ``--ibd_enhanced_extra_color_columns`` is non-empty
        // (comma-separated, e.g. 'Region,State,finestructure_bigclusters').
        // -------------------------------------------------------------------
        def extra_cols_str = params.containsKey('ibd_enhanced_extra_color_columns') \
                             ? (params.ibd_enhanced_extra_color_columns ?: '') : ''
        def extra_cols = extra_cols_str.toString().split(',').collect { it.trim() }
                                       .findAll { it }
        if( extra_cols ) {
            // Bundle compute outputs into a single channel item so the
            // replot process consumes them via one `path` tuple.  The
            // M14 aggregate files (segments + pair_summary + individual_
            // summary) are appended because the python script's plot
            // mode reconstructs the pair-segment table at startup.
            def ch_replot_inputs = IBD_COMMUNITY_ENHANCED.out.graph_edges
                .combine(IBD_COMMUNITY_ENHANCED.out.graph_nodes)
                .combine(IBD_COMMUNITY_ENHANCED.out.graph_matrix)
                .combine(IBD_COMMUNITY_ENHANCED.out.leiden_assignments)
                .combine(IBD_COMMUNITY_ENHANCED.out.leiden_modularity)
                .combine(IBD_COMMUNITY_ENHANCED.out.nmf_err)
                .combine(IBD_COMMUNITY_ENHANCED.out.global_summary)
                .combine(IBD_COMMUNITY_ENHANCED.out.validation)
                .combine(ch_ibd_enh_segments)
                .combine(ch_ibd_enh_pair_summary)
                .combine(ch_ibd_enh_individual_summary)
                .map { ge, gn, gm, la, lm, ne, gs, vl, seg, pair, ind ->
                    tuple(ge, gn, gm, la, lm, ne, gs, vl,
                           seg, pair, ind, ibd_community_enhanced_py)
                }
            // Metadata file: required for replot (replotting without
            // metadata defeats the purpose — fall back to empty placeholder
            // so the process still runs and emits the community-only panel).
            IBD_ENHANCED_REPLOT(ch_replot_inputs, ch_ibd_enh_meta, extra_cols)
        }
    }

    // -----------------------------------------------------------------------
    // M18 — Common-variant asIBD comparator (standalone: consumes Nunes' asIBD
    // from disk + M16.5 leiden_assignments; does NOT consume lai_rare, so it is
    // wired here, not in the lai_rare consumer map).
    // -----------------------------------------------------------------------
    if( do_asibd_comparator ) {
        def asibd_dir_f = file(params.asibd_input_dir)
        if( !asibd_dir_f.exists() ) {
            throw new IllegalStateException(
                "M18 enabled but asibd_input_dir not found: ${asibd_dir_f}. "
                + "Download Nunes' asIBD (anc{1,2,3}.gapfilled_ibd) from "
                + "gs://projects-usp/dna-do-brasil/dnabr-ibd/combined/ first."
            )
        }
        def asibd_files = files("${params.asibd_input_dir}/${params.asibd_glob}")
        if( !asibd_files ) {
            throw new IllegalStateException(
                "No asIBD files matching ${params.asibd_glob} in ${asibd_dir_f}"
            )
        }
        def leiden_str = params.asibd_leiden_assignments \
            ?: "${params.ibd_enhanced_results_dir}/leiden_assignments.tsv"
        def leiden_f = file(leiden_str)
        if( !leiden_f.exists() ) {
            throw new IllegalStateException(
                "M18 needs leiden_assignments.tsv: ${leiden_f} not found. "
                + "Run M16.5 or set --asibd_leiden_assignments."
            )
        }
        def asibd_meta_f = file(params.asibd_metadata)
        if( !asibd_meta_f.exists() ) {
            throw new IllegalStateException("asibd_metadata not found: ${asibd_meta_f}")
        }
        // Se preparan ambos scripts porque asibd_comparator.py importa ibd_community_enhanced.py
        // para reutilizar la configuración de Leiden de M16.5.
        def asibd_py     = file("${projectDir}/bin/asibd_comparator.py")
        def asibd_lib_py = file("${projectDir}/bin/ibd_community_enhanced.py")
        channel.value(asibd_files)
            .map { fs -> tuple(fs, leiden_f, asibd_meta_f, asibd_py, asibd_lib_py) }
            .set { ch_asibd_in }
        COMPARE_ASIBD_COMMON(ch_asibd_in)
    }

    // -----------------------------------------------------------------------
    // Downstream population-genetics modules 07-11
    // -----------------------------------------------------------------------
    if( any_downstream ) {
        // --- keep-samples / exclude-bed ---
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
        if( params.exclude_regions_bed ) {
            def bed = file(params.exclude_regions_bed)
            ch_exclude_bed_opt = bed.exists() ? channel.value(bed) : ch_empty_file
        } else {
            ch_exclude_bed_opt = ch_empty_file
        }

        // Reuse the already-resolved channels (live from 02/04 or discovered from outdir).
        // ch_filtered_vcfs and ch_pgen_out were populated in sections 02/04 above
        // based on need_filtered / need_pgen, so they are guaranteed to be non-empty
        // when any downstream module requires them.
        def ch_downstream_vcfs = ch_filtered_vcfs
        def ch_downstream_pgen = ch_pgen_out

        // 07: SFS
        if( params.enable_sfs ) {
            SFS_FROM_FILTERED_VCF(
                ch_downstream_vcfs
                    .map { chr, vcf_gz, tbi -> tuple(chr, vcf_gz, tbi, sfs_report_py) },
                ch_keep_samples_opt
            )
        }

        // 08: build ancestral TSV from FASTA
        def ch_ancestral_tbl
        if( params.enable_build_ancestral_tsv ) {
            def anc_fa  = file(params.ancestral_fasta)
            def anc_fai = file("${params.ancestral_fasta}.fai")
            if( !anc_fa.exists() || !anc_fai.exists() ) {
                throw new IllegalStateException("Missing ancestral FASTA or .fai: ${anc_fa} / ${anc_fai}")
            }
            ch_ancestral_tbl = BUILD_ANCESTRAL_TSV_FROM_FASTA(
                ch_downstream_vcfs
                    .map { chr, vcf_gz, tbi -> tuple(chr, vcf_gz, tbi, anc_fa, anc_fai, build_ancestral_tsv_py) }
            )
        }

        // 09: DAF/DSFS
        if( params.enable_daf ) {
            // If 08 ran live, use its output; otherwise discover from outdir
            def ch_anc = params.enable_build_ancestral_tsv
                ? ch_ancestral_tbl
                : discoverAncestralTsvs(params.outdir)

            DAF_DSFS_FROM_ANCESTRAL_TSV(
                ch_downstream_vcfs
                    .join(ch_anc)
                    .map { chr, vcf_gz, tbi, anc_tsv, anc_summary ->
                        tuple(chr, vcf_gz, tbi, anc_tsv, anc_summary, daf_dsfs_py)
                    }
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

    // -----------------------------------------------------------------------
    // 22  Pipeline de modelado (modeling_master → split → CV interna → [TEST])
    // -----------------------------------------------------------------------
    // Cadena lineal de procesos únicos (value channels). EVALUATE_TEST desactivado por
    // defecto + doble llave (force+reason); publica en un subdirectorio del resultado,
    // y nunca al artefacto canónico. VERIFY_TEST_HASH sólo comprueba el sha256 del fold de test
    // congelado (independiente, no reabre el fold 3).
    def do_build_master     = params.enable_build_modeling_master && params.run_model_pipeline
    def do_build_split      = params.enable_build_split_manifest  && params.run_model_pipeline
    def do_model_cv         = params.enable_model_primary_cv      && params.run_model_pipeline
    def do_evaluate_test    = params.enable_evaluate_test         && params.run_model_pipeline
    def do_verify_test_hash = params.enable_verify_test_hash      && params.run_model_pipeline

    if( do_build_split && !do_build_master && !file("${params.model_pipeline_results_dir}/modeling_master/modeling_master.tsv").exists() )
        throw new IllegalStateException("M22: build_split_manifest requiere build_modeling_master (o su output ya publicado).")
    if( do_model_cv && !do_build_split && !file("${params.model_pipeline_results_dir}/split/split_manifest.tsv").exists() )
        throw new IllegalStateException("M22: model_primary_cv requiere build_split_manifest (o su output ya publicado).")
    if( do_evaluate_test && !do_model_cv && !file("${params.model_pipeline_results_dir}/model_primary/model_primary_cv_results.json").exists() )
        throw new IllegalStateException("M22: evaluate_test requiere model_primary_cv (ancla el candidato).")
    if( do_evaluate_test && params.model_pipeline_force_evaluate_test && !params.model_pipeline_force_evaluate_test_reason )
        throw new IllegalStateException("M22: --model_pipeline_force_evaluate_test=true exige --model_pipeline_force_evaluate_test_reason no vacío (auditable).")

    if( do_build_master || do_build_split || do_model_cv || do_evaluate_test || do_verify_test_hash ) {
        def build_modeling_master_py = file("${projectDir}/bin/build_modeling_master.py")
        def build_split_manifest_py  = file("${projectDir}/bin/build_split_manifest.py")
        def model_primary_cv_py      = file("${projectDir}/bin/model_primary_cv.py")
        def evaluate_test_py         = file("${projectDir}/bin/evaluate_test.py")

        // requiere un valor no-nulo para un param de dato; falla claro si falta (sin inferencias no sustentadas)
        def req = { val, name -> if( val == null ) throw new IllegalStateException("M22: falta --${name}"); return val }

        // --- Información reproducible para los manifiestos de cada etapa ---
        // Se calcula una vez en el nodo principal cuando corre M22, porque el sha256 del .sif no es
        // visible desde dentro del contenedor. Llega a cada proceso como JSON en base64 (shell-safe:
        // el comando Nextflow literal puede llevar comillas/espacios). Las versiones de librerías y
        // de Python las lee write_stage_manifest.py dentro del contenedor. Si
        // git o sha256sum no responden se registra 'unknown'/'unavailable', nunca un valor inventado.
        // sha256 del contenedor: coreutils siempre está en el nodo. El commit, en cambio, se resuelve
        // leyendo .git/ directamente con Groovy, porque el binario `git` puede no estar en el
        // PATH del nodo de cómputo. La lectura directa funciona también en el sistema compartido.
        def shOut = { cmd -> try { def p = ['bash','-c',cmd].execute(); p.waitFor(); return p.exitValue()==0 ? p.text.trim() : '' } catch( ignored ) { return '' } }
        def resolveGitCommit = { dir ->
            try {
                def head = new File("${dir}/.git/HEAD").text.trim()
                if( !head.startsWith('ref:') ) return head            // HEAD desprendido = sha directo
                def ref = head.substring(4).trim()
                def refFile = new File("${dir}/.git/${ref}")
                if( refFile.exists() ) return refFile.text.trim()
                def packed = new File("${dir}/.git/packed-refs")       // fallback: refs empaquetadas
                if( packed.exists() )
                    for( line in packed.readLines() )
                        if( line.endsWith(" ${ref}") ) return line.split(' ')[0]
                return 'unknown'
            } catch( ignored ) { return 'unknown' }
        }
        def git_commit    = resolveGitCommit(projectDir.toString()) ?: 'unknown'
        def container_sha = shOut("sha256sum '${params.container_image}' 2>/dev/null | cut -d' ' -f1") ?: 'unavailable'
        if( !container_sha ) container_sha = 'unavailable'
        // Procedencia estable del resultado: entra al cache-key de cada proceso vía
        // `val prov_b64`. El comando de Nextflow queda fuera porque `workflow.commandLine` cambia con `-resume`
        // y, al ser input `val` interpolado en el script, rompería el cache. El comando se guarda
        // en run_provenance.json, fuera del cache-key.
        def prov_map = [
            git_commit        : git_commit,
            nextflow_version  : workflow.nextflow.version.toString(),
            container_path    : params.container_image,
            container_sha256  : container_sha,
        ]
        def prov_b64 = JsonOutput.toJson(prov_map).bytes.encodeBase64().toString()
        def ch_prov  = channel.value(prov_b64)
        // Comando Nextflow literal y contexto de ejecución en run_provenance.json, escrito una vez
        // junto a los subdirectorios de etapa. Al no ser entrada de ningún proceso, -resume
        // no lo considera y los procesos de cómputo se reutilizan.
        def run_prov = [
            git_commit       : git_commit,
            nextflow_command : workflow.commandLine,
            nextflow_version : workflow.nextflow.version.toString(),
            container_path   : params.container_image,
            container_sha256 : container_sha,
            launch_dir       : workflow.launchDir.toString(),
            project_dir      : projectDir.toString(),
        ]
        def rp_dir = new File("${params.model_pipeline_results_dir}")
        rp_dir.mkdirs()
        new File(rp_dir, 'run_provenance.json').text = JsonOutput.prettyPrint(JsonOutput.toJson(run_prov))

        // --- 22a  BUILD_MODELING_MASTER (o descubre su output previo) ---
        def ch_modeling_master
        if( do_build_master ) {
            def ch_master_in = channel.value(tuple(
                file(req(params.model_pipeline_feature_store_tsv, 'model_pipeline_feature_store_tsv')),
                file(req(params.model_pipeline_leiden_tsv,        'model_pipeline_leiden_tsv')),
                file(req(params.model_pipeline_graph_nodes_tsv,   'model_pipeline_graph_nodes_tsv')),
                file(req(params.model_pipeline_pcrelate_kin_tsv,  'model_pipeline_pcrelate_kin_tsv')),
                file(params.model_pipeline_metadata_file),
                build_modeling_master_py))
            req(params.model_pipeline_red_samples, 'model_pipeline_red_samples')
            ch_modeling_master = BUILD_MODELING_MASTER(ch_master_in, ch_prov).modeling_master
        } else if( do_build_split || do_model_cv ) {
            def mm = file("${params.model_pipeline_results_dir}/modeling_master/modeling_master.tsv")
            if( !mm.exists() ) throw new IllegalStateException("M22: modeling_master.tsv no encontrado en ${mm}.")
            ch_modeling_master = channel.value(mm)
        }

        // --- 22b  BUILD_SPLIT_MANIFEST (o descubre) ---
        def ch_split_manifest
        if( do_build_split ) {
            req(params.model_pipeline_red_samples, 'model_pipeline_red_samples')
            ch_split_manifest = BUILD_SPLIT_MANIFEST(
                ch_modeling_master.map { mm -> tuple(mm, build_split_manifest_py) },
                ch_prov
            ).split_manifest
        } else if( do_model_cv ) {
            def sm = file("${params.model_pipeline_results_dir}/split/split_manifest.tsv")
            if( !sm.exists() ) throw new IllegalStateException("M22: split_manifest.tsv no encontrado en ${sm}.")
            ch_split_manifest = channel.value(sm)
        }

        // --- 22c  MODEL_PRIMARY_CV (solo TRAIN; TEST no se abre) ---
        if( do_model_cv ) {
            MODEL_PRIMARY_CV(
                ch_modeling_master.combine(ch_split_manifest)
                    .map { mm, sp -> tuple(mm, sp, model_primary_cv_py) },
                ch_prov
            )
        }

        // --- 22d  EVALUATE_TEST (fold de test cerrado; desactivado por defecto) ---
        if( do_evaluate_test ) {
            // La orquestación limita la evaluación a una ejecución; la validación de Python vive en
            // `--outdir .` (work dir efímero, siempre vacío) → no protege dentro de Nextflow. Aquí sí:
            // si el resultado YA está publicado, exige --model_pipeline_force_evaluate_test (+ reason).
            def published = file("${params.model_pipeline_results_dir}/test_eval/evaluate_test_results.json")
            if( published.exists() && !params.model_pipeline_force_evaluate_test )
                throw new IllegalStateException(
                    "M22: evaluate_test_results.json ya existe en ${published}; la evaluación del fold de test se "
                    + "realiza una sola vez. Para repetirla usa --model_pipeline_force_evaluate_test true + "
                    + "--model_pipeline_force_evaluate_test_reason '<razón>'.")
            def mm = file("${params.model_pipeline_results_dir}/modeling_master/modeling_master.tsv")
            def sm = file("${params.model_pipeline_results_dir}/split/split_manifest.tsv")
            def sa = file("${params.model_pipeline_results_dir}/split/split_manifest_audit.json")
            def cv = file("${params.model_pipeline_results_dir}/model_primary/model_primary_cv_results.json")
            for( f in [mm, sm, sa, cv] ) if( !f.exists() ) throw new IllegalStateException("M22 evaluate_test: falta ${f}.")
            EVALUATE_TEST(
                channel.value(tuple(mm, sm, sa, cv, model_primary_cv_py, evaluate_test_py)),
                channel.value(params.model_pipeline_force_evaluate_test),
                ch_prov
            )
        }

        // --- 22e  VERIFY_TEST_HASH (integridad sha256 del TEST congelado; nunca lo reabre) ---
        if( do_verify_test_hash ) {
            def frozen_json = file(req(params.model_pipeline_frozen_test_json, 'model_pipeline_frozen_test_json'))
            def frozen_sha  = file(params.model_pipeline_frozen_test_sha256)
            for( f in [frozen_json, frozen_sha] ) if( !f.exists() ) throw new IllegalStateException("M22 verify: falta ${f}.")
            VERIFY_TEST_HASH(channel.value(tuple(frozen_json, frozen_sha)))
        }
    }

    RARE_MATRIX_BENCHMARK()
}
