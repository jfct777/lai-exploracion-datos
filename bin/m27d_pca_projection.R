#!/usr/bin/env Rscript

# M27D PCA fitted on the pass0 training set and projected onto everybody else.
#
# The point of refitting is that pass0 estimated its axes with relatives inside the
# fitting set.  A family large enough to matter pulls an axis towards itself, and an
# axis that encodes a family instead of an ancestry cannot remove ancestry from the
# kinship estimate.  Fitting only on the independent set and projecting the rest with
# the same SNP loadings keeps the axes a property of the population.
#
# One fit is produced per LD-pruned marker set, not per configuration.  The number of
# principal components is a slice of a single fit, so changing it changes exactly one
# factor; changing the marker set genuinely requires its own fit, which is the one
# factor that the strict-LD configuration varies.

suppressPackageStartupMessages({
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
preregistration <- value_after(args, "preregistration")
marker_set_id <- value_after(args, "marker-set-id")
outdir <- value_after(args, "outdir", ".")
threads <- as.integer(value_after(args, "threads", "4"))

require_files(c(gds_path, snp_rds, strata_path, training_set_path, preregistration))
if (is.null(marker_set_id) || !nzchar(marker_set_id)) stop("Missing --marker-set-id")
if (is.na(threads) || threads < 1L) stop("Invalid thread count")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

contract <- read_contract(preregistration)
max_pcs <- max(vapply(contract$configurations, function(config) as.integer(config$n_pcs), integer(1)))

gds <- snpgdsOpen(gds_path)
on.exit(try(snpgdsClose(gds), silent = TRUE), add = TRUE)
sample_ids <- read.gdsn(index.gdsn(gds, "sample.id"))
snp_ids <- validate_snp_set(gds, readRDS(snp_rds))
strata <- read_strata(strata_path, sample_ids)
included <- eligible_samples(strata, sample_ids)

training_set <- readLines(training_set_path)
training_set <- training_set[nzchar(training_set)]
if (length(training_set) < max_pcs + 1L) {
  stop("Training set is too small to support ", max_pcs, " principal components")
}
if (!all(training_set %in% included)) {
  stop("Training set contains samples outside the eligible universe")
}
if (anyDuplicated(training_set)) stop("Training set contains duplicate identifiers")

started <- Sys.time()
seed <- apply_seed(contract)
fit <- snpgdsPCA(
  gds,
  sample.id = training_set,
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

# The projection reuses the SNP loadings of the training-set fit, so a sample outside
# the training set never contributes to the axes it is scored on.
snp_loadings <- snpgdsPCASNPLoading(fit, gds, num.thread = threads, verbose = FALSE)
projected <- snpgdsPCASampLoading(
  snp_loadings,
  gds,
  sample.id = setdiff(included, training_set),
  num.thread = threads,
  verbose = FALSE
)

training_scores <- fit$eigenvect[, seq_len(max_pcs), drop = FALSE]
rownames(training_scores) <- fit$sample.id
projected_scores <- projected$eigenvect[, seq_len(max_pcs), drop = FALSE]
rownames(projected_scores) <- projected$sample.id

scores <- rbind(training_scores, projected_scores)
if (nrow(scores) != length(included)) {
  stop("PCA scores cover ", nrow(scores), " samples but ", length(included), " are eligible")
}
scores <- scores[match(included, rownames(scores)), , drop = FALSE]

score_table <- data.frame(
  sample_id = rownames(scores),
  in_training_set = rownames(scores) %in% training_set,
  scores,
  stringsAsFactors = FALSE
)
colnames(score_table) <- c("sample_id", "in_training_set", paste0("PC", seq_len(max_pcs)))
write_private_tsv_gz(
  score_table,
  file.path(outdir, sprintf("m27d_pca_%s_scores.private.tsv.gz", marker_set_id))
)

# An axis that separates the technical source or one population far more than the
# others is the failure mode gate G2 exists to catch, so the association of every axis
# with source, ancestry and country is reported next to the variance it explains.
axis_association <- function(scores, grouping) {
  grouping <- as.factor(grouping)
  if (nlevels(grouping) < 2L) return(rep(NA_real_, ncol(scores)))
  vapply(seq_len(ncol(scores)), function(index) {
    values <- scores[, index]
    total <- sum((values - mean(values))^2)
    if (total <= 0) return(NA_real_)
    between <- sum(vapply(split(values, grouping), function(group) {
      length(group) * (mean(group) - mean(values))^2
    }, numeric(1)))
    between / total
  }, numeric(1))
}

# Only samples with a resolved metadata row can be cross-tabulated against a group; the
# ones without one would otherwise form a phantom stratum made of a bookkeeping gap.
strata_included <- strata[match(included, strata$sample_id), , drop = FALSE]
annotated <- toupper(trimws(strata_included$population_interpretable)) == "TRUE"
association <- list()
for (column in c("Source", "Ancestry", "Country", "Population")) {
  if (!column %in% colnames(strata_included)) next
  association[[column]] <- as.numeric(
    axis_association(scores[annotated, , drop = FALSE], strata_included[[column]][annotated])
  )
}

# G2 used to be one verdict over three different questions, and it aborted runs on the
# third of them under the name of the first.  They are separated here.
#
# G2A asks whether the computation is sound: finite scores, the eligible samples once
# each and in the eligible order, one fit projected with its own SNP loadings, the
# declared number of components, the declared seed.  Those are properties of the
# arithmetic, so a failure aborts and no calibration can rescue it.
#
# G2B measures how many individuals carry each axis.  The participation ratio is a real,
# label-free measurement, but it names no cause: a localised axis can be a family, a small
# differentiated population, an isolate, a technical group or the samples with no metadata
# row.  PC-Relate needs axes that describe small differentiated populations in order to
# estimate individual-specific allele frequencies there, so treating localisation as
# breakage would remove exactly the structure the estimator has to condition on.  Below
# the fraction an axis is marked REVIEW, which is a request to look, not a verdict.
#
# G2C asks whether the fitting set still represents the small populations.  It is not
# adjudicated here: that answer needs the per-population survival of the training set, and
# declaring a threshold before measuring it would be choosing a number to fit a result.
axis_contract <- contract$pca_axis_contract
localization_contract <- axis_contract$g2b_axis_localization

# Honest accounting of this receipt: most of these restate a guard the script already
# enforced with stop() before reaching here, so they document what was checked rather than
# add a new way to fail.  The one that can genuinely fire on correct-looking input is
# scores_are_finite: snpgdsPCASampLoading can return NaN for a sample the loadings cannot
# score, and nothing upstream looks.  A check that only repeats an earlier stop() is worth
# publishing, but calling the whole list a gate would overstate it.
g2a_checks <- list(
  scores_are_finite = all(is.finite(scores)),
  scores_cover_eligible_samples_in_order = identical(rownames(scores), included),
  scores_have_no_duplicate_samples = !anyDuplicated(rownames(scores)),
  component_count_matches_contract = identical(ncol(scores), as.integer(max_pcs)),
  # snpgdsPCASNPLoading carries the sample set of the fit it was derived from, so this is
  # what distinguishes reusing one fit from silently running a second one.
  projection_reused_the_training_fit = identical(snp_loadings$sample.id, fit$sample.id),
  training_set_inside_eligible_universe = all(training_set %in% included)
)
# The seed is reported, not checked. `apply_seed` returns the contract value it just
# applied, so comparing the two was identical(x, x); and whether set.seed actually reached
# the RNG is not observable from here. Publishing the number is the honest version.
g2a_failed <- names(Filter(isFALSE, g2a_checks))
g2a_status <- if (length(g2a_failed)) "FAIL" else "PASS"

# The bound scales with the cohort: an absolute count calibrated for 3685 samples would
# fire on any smaller panel for a reason that has nothing to do with the data.
review_fraction <- as.numeric(localization_contract$review_fraction_per_axis)
review_bound <- review_fraction * length(included)
participation_ratio <- function(vector) {
  weights <- vector^2
  total <- sum(weights)
  if (!is.finite(total) || total <= 0) return(NA_real_)
  weights <- weights / total
  1 / sum(weights^2)
}
effective_individuals <- vapply(seq_len(ncol(scores)), function(i) participation_ratio(scores[, i]), numeric(1))
axis_status <- vapply(effective_individuals, function(value) {
  if (is.na(value)) "NOT_EVALUATED" else if (value >= review_bound) "PASS" else "REVIEW"
}, character(1))

# Every configuration is judged on the prefix it actually reads.  A run with four
# components is not affected by an eleventh axis it never passes to PC-Relate, and the
# old aggregate over max_pcs condemned all four configurations for one of them.
prefixes <- sort(unique(vapply(contract$configurations, function(c) as.integer(c$n_pcs), integer(1))))
g2b_by_prefix <- lapply(prefixes, function(k) {
  used <- seq_len(min(k, length(effective_individuals)))
  flagged <- used[axis_status[used] == "REVIEW"]
  list(
    n_pcs = k,
    min_effective_individuals = min(effective_individuals[used]),
    axes_under_review = paste0("PC", flagged),
    status = if (any(axis_status[used] == "NOT_EVALUATED")) {
      "NOT_EVALUATED"
    } else if (length(flagged)) "REVIEW" else "PASS"
  )
})
names(g2b_by_prefix) <- paste0("n_pcs_", prefixes)

# A ratio is only interpretable next to who those individuals are, and the groups have to
# include the samples with no metadata row: an axis they dominate is a bookkeeping gap
# before it is anything biological.
carriers_for_axis <- function(index, column) {
  weights <- scores[, index]^2
  total <- sum(weights)
  if (!is.finite(total) || total <= 0) return(list())
  weights <- weights / total
  labels <- if (column %in% colnames(strata_included)) strata_included[[column]] else NULL
  if (is.null(labels)) return(list())
  labels <- as.character(labels)
  labels[is.na(labels) | !nzchar(trimws(labels))] <- "(unlabelled)"
  by_label <- sort(tapply(weights, factor(labels), sum), decreasing = TRUE)
  lapply(seq_len(min(3L, length(by_label))), function(i) {
    list(label = names(by_label)[i], weight_fraction = as.numeric(by_label[i]))
  })
}
grouping_columns <- as.character(localization_contract$grouping_columns_reported)
unlabelled_fraction_for_axis <- function(index) {
  weights <- scores[, index]^2
  total <- sum(weights)
  if (!is.finite(total) || total <= 0) return(NA_real_)
  sum(weights[!annotated]) / total
}
axis_localization <- lapply(seq_len(ncol(scores)), function(index) {
  carriers <- lapply(grouping_columns, function(column) carriers_for_axis(index, column))
  names(carriers) <- grouping_columns
  list(
    participation_ratio = effective_individuals[index],
    status = axis_status[index],
    variance_explained = as.numeric(fit$varprop[index]),
    fraction_carried_by_samples_without_metadata = unlabelled_fraction_for_axis(index),
    carried_by = carriers
  )
})
names(axis_localization) <- paste0("PC", seq_len(ncol(scores)))
# Kept under its historical name so existing readers of the score summary do not break.
axis_carriers <- lapply(seq_len(ncol(scores)), function(i) carriers_for_axis(i, "Population"))
names(axis_carriers) <- paste0("PC", seq_len(ncol(scores)))

summary <- list(
  stage = "M27D_PCA_PROJECTION",
  marker_set_id = marker_set_id,
  g2a_technical_integrity_status = g2a_status,
  g2a_checks = g2a_checks,
  g2a_failed_checks = g2a_failed,
  g2a_enforcement = "abort_on_fail",
  g2b_axis_localization = axis_localization,
  g2b_status_by_preregistered_n_pcs = g2b_by_prefix,
  g2b_effective_individuals_by_axis = as.numeric(effective_individuals),
  g2b_min_effective_individuals = min(effective_individuals),
  g2b_axes_under_review = paste0("PC", which(axis_status == "REVIEW")),
  g2b_axis_carriers = axis_carriers,
  g2b_enforcement = "report_only_review_does_not_abort",
  g2b_review_bound = review_bound,
  g2b_review_fraction = review_fraction,
  g2b_review_blocks = as.character(localization_contract$review_blocks),
  g2c_ancestry_representativeness_status = axis_contract$g2c_ancestry_representativeness$status,
  scientific_result = FALSE,
  king_executed = FALSE,
  pcair_used = FALSE,
  n_training_samples = length(training_set),
  n_projected_samples = length(included) - length(training_set),
  n_eligible_samples = length(included),
  n_markers = length(snp_ids),
  n_pcs = max_pcs,
  random_seed = seed,
  pca_fitted_only_on_training_set = TRUE,
  variance_explained = as.numeric(fit$varprop[seq_len(max_pcs)]),
  axis_variance_fraction_by_group = association,
  elapsed_seconds = elapsed_seconds(started),
  threads = threads,
  sample_ids_emitted = FALSE,
  software = software_versions()
)
write_json(
  summary,
  file.path(outdir, sprintf("m27d_pca_%s.json", marker_set_id)),
  pretty = TRUE,
  auto_unbox = TRUE
)

# Only the technical check stops the stage, and it is adjudicated here rather than at
# selection time so a broken fit does not get paid for four times.  The summary is written
# first so the failure keeps its evidence.  G2B never stops anything: it is a measurement
# whose flagged axes have to be read next to the populations carrying them, and the run
# that certifies donors is the one that has to answer for them.
if (identical(g2a_status, "FAIL")) {
  stop(
    "G2A failed for the '", marker_set_id, "' marker set. Failed checks: ",
    paste(g2a_failed, collapse = ", "),
    ". These are properties of the computation, not of the cohort, so the stage cannot ",
    "continue. The full receipt is in the summary written above."
  )
}

if (any(axis_status == "REVIEW")) {
  under_review <- which(axis_status == "REVIEW")
  message(
    "G2B marks ", length(under_review), " axis/axes for review in the '", marker_set_id,
    "' marker set: ",
    paste0(
      "PC", under_review, " (", round(effective_individuals[under_review], 1),
      " effective individuals)",
      collapse = ", "
    ),
    "; the contract flags below ", round(review_bound, 1),
    ". This is not a failure. A localised axis may be a family, a small differentiated ",
    "population, an isolate, a technical group or the samples without a metadata row, and ",
    "the ratio alone does not separate them. The carrying populations are in the summary ",
    "written above, and the operating value still has to be justified against that ",
    "distribution before any donor is certified."
  )
}
