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

# G2 asks whether an axis encodes ancestry rather than one family.  Associating axes
# with the Source column cannot answer that here: in this panel Source and ancestry are
# confounded by construction, so a real ancestry axis separates sources strongly and a
# bound on that association would fail on correct behaviour.
#
# The participation ratio needs no group labels.  For a unit-norm eigenvector it equals
# the number of individuals effectively carrying the axis: about n when the axis is
# spread across the cohort, about k when k people dominate it.  An axis carried by a
# handful of people is a family axis, and removing it would remove that family rather
# than ancestry.  The group associations stay in the report as description only.
participation_ratio <- function(vector) {
  weights <- vector^2
  total <- sum(weights)
  if (!is.finite(total) || total <= 0) return(NA_real_)
  weights <- weights / total
  1 / sum(weights^2)
}

# The bound scales with the cohort: an absolute count calibrated for 3685 samples would
# fail on any smaller panel for a reason that has nothing to do with the data.
axis_contract <- contract$pca_axis_contract
min_effective <- as.numeric(axis_contract$min_effective_individual_fraction_per_axis) * length(included)
effective_individuals <- vapply(seq_len(ncol(scores)), function(i) participation_ratio(scores[, i]), numeric(1))
g2_status <- if (anyNA(effective_individuals)) {
  "NOT_EVALUATED"
} else if (min(effective_individuals) >= min_effective) {
  "PASS"
} else {
  "FAIL"
}

# The verdict is also reported per preregistered component count.  A configuration that
# uses eight components is not made wrong by a degenerate twelfth axis it never reads, and
# a single aggregate verdict over max_pcs cannot say which configurations are affected.
# Enforcement stays as the contract declares it; this only makes the reason legible.
prefixes <- sort(unique(vapply(contract$configurations, function(c) as.integer(c$n_pcs), integer(1))))
g2_by_prefix <- lapply(prefixes, function(k) {
  used <- effective_individuals[seq_len(min(k, length(effective_individuals)))]
  list(
    n_pcs = k,
    min_effective_individuals = min(used),
    status = if (anyNA(used)) "NOT_EVALUATED" else if (min(used) >= min_effective) "PASS" else "FAIL"
  )
})
names(g2_by_prefix) <- paste0("n_pcs_", prefixes)

# The populations carrying each axis are the calibration the contract asks for before any
# donor is certified: a bound on effective individuals is only interpretable next to who
# those individuals are.
carriers_for_axis <- function(index) {
  weights <- scores[, index]^2
  total <- sum(weights)
  if (!is.finite(total) || total <= 0) return(list())
  weights <- weights / total
  labels <- strata_included$Population
  labels[is.na(labels) | !nzchar(trimws(labels))] <- "(unlabelled)"
  by_label <- tapply(weights, factor(labels), sum)
  by_label <- sort(by_label, decreasing = TRUE)
  lapply(seq_len(min(3L, length(by_label))), function(i) {
    list(label = names(by_label)[i], weight_fraction = as.numeric(by_label[i]))
  })
}
axis_carriers <- lapply(seq_len(ncol(scores)), carriers_for_axis)
names(axis_carriers) <- paste0("PC", seq_len(ncol(scores)))

summary <- list(
  stage = "M27D_PCA_PROJECTION",
  marker_set_id = marker_set_id,
  g2_status = g2_status,
  g2_min_effective_individuals = min(effective_individuals),
  g2_effective_individuals_by_axis = as.numeric(effective_individuals),
  g2_status_by_preregistered_n_pcs = g2_by_prefix,
  g2_axis_carriers = axis_carriers,
  g2_enforcement = "abort_if_any_preregistered_prefix_fails",
  g2_bound = min_effective,
  g2_bound_fraction = as.numeric(axis_contract$min_effective_individual_fraction_per_axis),
  g2_bound_status = axis_contract$status,
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

# G2 is adjudicated here rather than only at selection time.  An axis carried by a
# handful of people means PC-Relate would remove that family instead of ancestry, so
# every configuration downstream would be estimating the wrong thing; letting the run
# continue would pay for four PC-Relate passes before saying so.  The summary is written
# first so the failure keeps its evidence.
if (identical(g2_status, "FAIL")) {
  failing <- which(effective_individuals < min_effective)
  blocked <- names(Filter(function(entry) identical(entry$status, "FAIL"), g2_by_prefix))
  stop(
    "G2 failed for the '", marker_set_id, "' marker set. Degenerate axes: ",
    paste0(
      "PC", failing, " (", round(effective_individuals[failing], 1), " effective individuals)",
      collapse = ", "
    ),
    "; the preregistration requires at least ", round(min_effective, 1),
    ". Blocked configurations: ", paste(blocked, collapse = ", "),
    ". The per-axis ratios and their carrying populations are in the summary written above; ",
    "the contract asks for the operating value to be justified against exactly that ",
    "distribution before any donor is certified."
  )
}
