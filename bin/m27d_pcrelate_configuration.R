#!/usr/bin/env Rscript

# M27D final PC-Relate pass for one preregistered configuration.
#
# Every configuration reuses the same training set, so the only thing that changes
# between them is the factor written in the preregistration: the number of principal
# components, or the LD-pruned marker set.  Refitting the training set per
# configuration would change two things at once and make the comparison unreadable.

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
training_set_path <- value_after(args, "training-set")
pca_scores_path <- value_after(args, "pca-scores")
preregistration <- value_after(args, "preregistration")
configuration_id <- value_after(args, "configuration-id")
marker_set_id <- value_after(args, "marker-set-id")
outdir <- value_after(args, "outdir", ".")
threads <- as.integer(value_after(args, "threads", "4"))

require_files(c(gds_path, snp_rds, strata_path, training_set_path, pca_scores_path, preregistration))
if (is.null(configuration_id) || !nzchar(configuration_id)) stop("Missing --configuration-id")
if (is.na(threads) || threads < 1L) stop("Invalid thread count")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

contract <- read_contract(preregistration)
configuration <- contract_configuration(contract, configuration_id)
n_pcs <- as.integer(configuration$n_pcs)

# The scores and the marker set arrive as two separate files, so nothing in the file
# names stops a run from scoring the strict-LD configuration on the anchor PCA.  The
# expected marker set is derived from the configuration's own r2 and checked against
# both the identifier passed in and the file actually staged.
expected_marker_set <- if (isTRUE(all.equal(as.numeric(configuration$ld_r2_max), 0.2))) "anchor" else "strict"
if (is.null(marker_set_id) || !nzchar(marker_set_id)) stop("Missing --marker-set-id")
if (!identical(marker_set_id, expected_marker_set)) {
  stop(
    "Configuration ", configuration_id, " has r2=", configuration$ld_r2_max,
    ", which needs the '", expected_marker_set, "' marker set, but '", marker_set_id, "' was passed"
  )
}
if (!grepl(expected_marker_set, basename(snp_rds), fixed = TRUE)) {
  stop("Marker file ", basename(snp_rds), " does not belong to the '", expected_marker_set, "' set")
}
if (!grepl(expected_marker_set, basename(pca_scores_path), fixed = TRUE)) {
  stop("PCA scores ", basename(pca_scores_path), " were not fitted on the '", expected_marker_set, "' set")
}
report_threshold <- min(as.numeric(unlist(contract$pcrelate$descriptive_phi_thresholds)))

gds <- snpgdsOpen(gds_path)
sample_ids <- read.gdsn(index.gdsn(gds, "sample.id"))
snp_ids <- validate_snp_set(gds, readRDS(snp_rds))
strata <- read_strata(strata_path, sample_ids)
included <- eligible_samples(strata, sample_ids)
snpgdsClose(gds)

training_set <- readLines(training_set_path)
training_set <- training_set[nzchar(training_set)]
if (!all(training_set %in% included)) stop("Training set contains samples outside the eligible universe")
assert_small_sample_correction_applies(contract, length(included))

scores <- read.delim(pca_scores_path, stringsAsFactors = FALSE, check.names = FALSE)
if (!"sample_id" %in% colnames(scores)) stop("PCA score table has no sample_id column")
pc_columns <- paste0("PC", seq_len(n_pcs))
missing_pcs <- setdiff(pc_columns, colnames(scores))
if (length(missing_pcs)) {
  stop("PCA score table lacks components: ", paste(missing_pcs, collapse = ", "))
}
if (!all(included %in% scores$sample_id)) stop("PCA score table does not cover every eligible sample")

pcs <- as.matrix(scores[match(included, scores$sample_id), pc_columns, drop = FALSE])
rownames(pcs) <- included
storage.mode(pcs) <- "double"
if (anyNA(pcs)) stop("PCA score table contains missing values")

started <- Sys.time()
seed <- apply_seed(contract)
geno_reader <- GdsGenotypeReader(filename = gds_path)
geno_data <- GenotypeData(geno_reader)
geno_iter <- GenotypeBlockIterator(geno_data, snpInclude = snp_ids)
related <- pcrelate(
  geno_iter,
  pcs = pcs,
  scale = contract$pcrelate$scale,
  ibd.probs = TRUE,
  sample.include = included,
  training.set = training_set,
  maf.thresh = as.numeric(contract$pcrelate$maf_thresh),
  maf.bound.method = contract$pcrelate$maf_bound_method,
  small.samp.correct = isTRUE(contract$pcrelate$small_sample_correction),
  BPPARAM = MulticoreParam(workers = threads, progressbar = FALSE),
  verbose = FALSE
)
close(geno_data)

pairs <- related$kinBtwn
n_pairs <- nrow(pairs)
expected_pairs <- length(included) * (length(included) - 1) / 2
if (n_pairs != expected_pairs) {
  stop("Unexpected PC-Relate pair count: expected ", expected_pairs, ", observed ", n_pairs)
}

reported <- pairs[pairs$kin >= report_threshold, , drop = FALSE]
write_private_tsv_gz(
  reported[order(-reported$kin), , drop = FALSE],
  file.path(outdir, sprintf("m27d_pcrelate_%s_pairs.private.tsv.gz", configuration_id))
)

summary <- list(
  stage = "M27D_PCRELATE_CONFIGURATION",
  configuration_id = configuration_id,
  role = configuration$role,
  n_pcs = n_pcs,
  ld_r2_max = as.numeric(configuration$ld_r2_max),
  scientific_result = FALSE,
  king_executed = FALSE,
  pcair_used = FALSE,
  training_set_reused_from_pass0 = TRUE,
  n_training_samples = length(training_set),
  n_eligible_samples = length(included),
  n_markers = length(snp_ids),
  random_seed = seed,
  n_pairs_total = n_pairs,
  n_pairs_reported = nrow(reported),
  report_threshold = report_threshold,
  pair_counts_by_threshold = threshold_counts(pairs$kin, contract),
  elapsed_seconds = elapsed_seconds(started),
  threads = threads,
  sample_ids_emitted = FALSE,
  software = software_versions()
)
write_json(
  summary,
  file.path(outdir, sprintf("m27d_pcrelate_%s.json", configuration_id)),
  pretty = TRUE,
  auto_unbox = TRUE
)
