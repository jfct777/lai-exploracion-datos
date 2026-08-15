#!/usr/bin/env Rscript

# M27D pass0: provisional PCA and the first PC-Relate pass over every eligible sample.
#
# This pass exists only to find out who is related to whom.  GENESIS accepts the whole
# cohort as its fitting set when no independent subset is known yet, which is exactly
# the situation here, and that is the reason the result is provisional: relatives inside
# the fitting set bias the very allele frequencies used to remove ancestry.  Its single
# durable output is the list of related pairs that the next stage turns into a
# training set.  No candidate is certified here.
#
# KING is not used at any point.  PC-AiR is not used either: it needs a kinship matrix
# to initialise, and in this project the only available one came from KING.

suppressPackageStartupMessages({
  library(BiocParallel)
  library(GENESIS)
  library(GWASTools)
  library(SNPRelate)
  library(gdsfmt)
  library(jsonlite)
})

source_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(source_dir, "m27d_common.R"))

args <- commandArgs(trailingOnly = TRUE)
gds_path <- value_after(args, "gds")
snp_rds <- value_after(args, "snp-rds")
strata_path <- value_after(args, "strata")
preregistration <- value_after(args, "preregistration")
outdir <- value_after(args, "outdir", ".")
threads <- as.integer(value_after(args, "threads", "4"))

require_files(c(gds_path, snp_rds, strata_path, preregistration))
if (is.na(threads) || threads < 1L) stop("Invalid thread count")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

contract <- read_contract(preregistration)
anchor <- contract_configuration(contract, "anchor_pc8_r2_020")
anchor_pcs <- as.integer(anchor$n_pcs)
max_pcs <- max(vapply(contract$configurations, function(config) as.integer(config$n_pcs), integer(1)))
report_threshold <- min(as.numeric(unlist(contract$pcrelate$descriptive_phi_thresholds)))

gds <- snpgdsOpen(gds_path)
on.exit(try(snpgdsClose(gds), silent = TRUE), add = TRUE)
sample_ids <- read.gdsn(index.gdsn(gds, "sample.id"))
expected_samples <- as.integer(contract$scope$official_panel_samples_expected)
if (length(sample_ids) != expected_samples) {
  stop("Panel sample count mismatch: expected ", expected_samples, ", observed ", length(sample_ids))
}

snp_ids <- readRDS(snp_rds)
snp_ids <- validate_snp_set(gds, snp_ids)

# Genotypes exist for every panel member, including the samples whose metadata row is
# missing.  Dropping them here would silently remove people from a kinship audit
# because of a bookkeeping gap, so only an explicit metadata exclusion removes anyone.
strata <- read_strata(strata_path, sample_ids)
included <- eligible_samples(strata, sample_ids)
if (length(included) < 2L) stop("Fewer than two eligible samples remain for pass0")
assert_small_sample_correction_applies(contract, length(included))

call_rate <- 1 - snpgdsSampMissRate(gds, sample.id = included, snp.id = snp_ids, with.id = FALSE)
call_rate_table <- data.frame(
  sample_id = included,
  call_rate = as.numeric(call_rate),
  stringsAsFactors = FALSE
)
write_private_tsv(call_rate_table, file.path(outdir, "m27d_pass0_sample_call_rate.private.tsv"))
writeLines(included, file.path(outdir, "m27d_pass0_sample_universe.private.txt"))

started <- Sys.time()
seed <- apply_seed(contract)
pca_time <- system.time({
  pca <- snpgdsPCA(
    gds,
    sample.id = included,
    snp.id = snp_ids,
    autosome.only = TRUE,
    remove.monosnp = TRUE,
    maf = NaN,
    missing.rate = NaN,
    num.thread = threads,
    eigen.cnt = max_pcs,
    algorithm = "randomized",
    verbose = FALSE
  )
})
pcs <- pca$eigenvect
rownames(pcs) <- pca$sample.id
variance_explained <- pca$varprop[seq_len(max_pcs)]
snpgdsClose(gds)

geno_reader <- GdsGenotypeReader(filename = gds_path)
geno_data <- GenotypeData(geno_reader)
geno_iter <- GenotypeBlockIterator(
  geno_data,
  snpInclude = snp_ids,
  snpBlock = as.integer(contract$pcrelate$snp_block_size)
)
pcrelate_time <- system.time({
  related <- pcrelate(
    geno_iter,
    pcs = pcs[, seq_len(anchor_pcs), drop = FALSE],
    scale = contract$pcrelate$scale,
    ibd.probs = TRUE,
    sample.include = included,
    training.set = included,
    maf.thresh = as.numeric(contract$pcrelate$maf_thresh),
    maf.bound.method = contract$pcrelate$maf_bound_method,
    small.samp.correct = isTRUE(contract$pcrelate$small_sample_correction),
    BPPARAM = MulticoreParam(workers = threads, progressbar = FALSE),
    verbose = FALSE
  )
})
close(geno_data)

pairs <- related$kinBtwn
n_pairs <- nrow(pairs)
expected_pairs <- length(included) * (length(included) - 1) / 2
if (n_pairs != expected_pairs) {
  stop("Unexpected PC-Relate pair count: expected ", expected_pairs, ", observed ", n_pairs)
}

# The closed form above is self-consistent: it recomputes the expectation from the same
# n it just measured, so it cannot notice that the eligible universe changed.  The
# checkpoint compares against the absolute number fixed before the run, which is what
# the stop-rule actually means.
checkpoint <- contract$pass0_checkpoint
if (!is.null(checkpoint$expected_eligible_samples)) {
  if (length(included) != as.integer(checkpoint$expected_eligible_samples)) {
    stop(
      "Eligible universe changed: the checkpoint fixes ",
      checkpoint$expected_eligible_samples, " samples, observed ", length(included)
    )
  }
}
if (!is.null(checkpoint$expected_pairs)) {
  if (n_pairs != as.numeric(checkpoint$expected_pairs)) {
    stop("Pair count differs from the preregistered checkpoint ", checkpoint$expected_pairs)
  }
}

# Only pairs at or above the lowest preregistered reporting threshold are written out.
# The full 6.8 million rows carry no information the audit uses and the total count is
# reported separately, so the stop-rule on pair count still has its number.
reported <- pairs[pairs$kin >= report_threshold, , drop = FALSE]
write_private_tsv_gz(
  reported[order(-reported$kin), , drop = FALSE],
  file.path(outdir, "m27d_pass0_related_pairs.private.tsv.gz")
)

# GENESIS computes self-kinship in the same pass and the code used to discard it.  Its
# diagonal is the inbreeding coefficient, which is the only observable in this module
# that can separate a recent pedigree from drift inside an endogamous population: two
# members of a small isolate look related to PC-Relate whether or not they share a
# recent ancestor, and f tells those cases apart.  Recovering it later costs a full pass.
write_private_tsv(
  related$kinSelf,
  file.path(outdir, "m27d_pass0_inbreeding.private.tsv")
)

scores <- data.frame(sample_id = rownames(pcs), pcs, stringsAsFactors = FALSE)
colnames(scores) <- c("sample_id", paste0("PC", seq_len(ncol(pcs))))
write_private_tsv_gz(scores, file.path(outdir, "m27d_pass0_pca_scores.private.tsv.gz"))

summary <- list(
  stage = "M27D_PASS0_PCRELATE",
  scientific_result = FALSE,
  provisional = TRUE,
  king_executed = FALSE,
  pcair_used = FALSE,
  n_eligible_samples = length(included),
  n_excluded_by_metadata = length(sample_ids) - length(included),
  n_markers = length(snp_ids),
  random_seed = seed,
  n_pcs_used = anchor_pcs,
  n_pcs_computed = max_pcs,
  n_pairs_total = n_pairs,
  n_pairs_reported = nrow(reported),
  report_threshold = report_threshold,
  pair_counts_by_threshold = threshold_counts(pairs$kin, contract),
  variance_explained = as.numeric(variance_explained),
  timing_seconds = list(
    pca = unname(pca_time[["elapsed"]]),
    pcrelate = unname(pcrelate_time[["elapsed"]]),
    total = elapsed_seconds(started)
  ),
  threads = threads,
  sample_ids_emitted = FALSE,
  software = software_versions()
)
write_json(summary, file.path(outdir, "m27d_pass0_pcrelate.json"), pretty = TRUE, auto_unbox = TRUE)
