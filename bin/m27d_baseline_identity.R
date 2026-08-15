#!/usr/bin/env Rscript

# M27D reconciliation of the frozen LAI baseline donors against the official panel.
#
# Disjunction has two levels and this stage establishes the first one.  Identity keeps
# the audit from scoring the same person twice, once as a baseline donor and once as a
# fresh candidate.  Kinship, handled later by PC-Relate, keeps a sibling of a baseline
# donor from passing as independent evidence.  Identity alone is not enough, but
# without it the kinship comparison has nothing to compare against.
#
# Matching is by genotype, not by name.  Two cohorts can spell the same person
# differently and can spell different people the same way, so every name-level match is
# confirmed with dosage concordance over jointly called autosomal markers.

suppressPackageStartupMessages({
  library(SNPRelate)
  library(gdsfmt)
  library(jsonlite)
})

source_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(source_dir, "m27d_common.R"))

args <- commandArgs(trailingOnly = TRUE)
panel_gds_path <- value_after(args, "panel-gds")
baseline_vcfs <- split_csv(value_after(args, "baseline-vcfs"))
snp_rds <- value_after(args, "snp-rds")
strata_path <- value_after(args, "strata")
preregistration <- value_after(args, "preregistration")
outdir <- value_after(args, "outdir", ".")
threads <- as.integer(value_after(args, "threads", "4"))

require_files(c(panel_gds_path, snp_rds, strata_path, preregistration))
require_files(baseline_vcfs)
if (length(baseline_vcfs) != 22L) stop("Expected 22 autosomal baseline VCFs; found ", length(baseline_vcfs))
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

contract <- read_contract(preregistration)
identity <- contract$identity_contract
expected_shared <- as.integer(identity$expected_shared_baseline_identities)
min_joint <- as.integer(identity$joint_autosomal_genotypes_min_per_shared_identity)
min_concordance <- as.numeric(identity$dosage_concordance_min_per_shared_identity)
expected_baseline <- as.integer(contract$scope$baseline_samples_expected)

seed <- apply_seed(contract)
baseline_gds_path <- file.path(outdir, "m27d_baseline_donors.gds")
snpgdsVCF2GDS(
  baseline_vcfs,
  baseline_gds_path,
  method = "biallelic.only",
  snpfirstdim = FALSE,
  ignore.chr.prefix = "chr",
  verbose = FALSE
)

panel <- snpgdsOpen(panel_gds_path)
on.exit(try(snpgdsClose(panel), silent = TRUE), add = TRUE)
baseline <- snpgdsOpen(baseline_gds_path)
on.exit(try(snpgdsClose(baseline), silent = TRUE), add = TRUE)

panel_samples <- read.gdsn(index.gdsn(panel, "sample.id"))
baseline_samples <- read.gdsn(index.gdsn(baseline, "sample.id"))
if (length(baseline_samples) != expected_baseline) {
  stop("Baseline donor count mismatch: expected ", expected_baseline, ", observed ", length(baseline_samples))
}

marker_key <- function(gds, snp_ids = NULL) {
  ids <- read.gdsn(index.gdsn(gds, "snp.id"))
  keep <- if (is.null(snp_ids)) seq_along(ids) else match(snp_ids, ids)
  data.frame(
    snp_id = ids[keep],
    key = paste(
      read.gdsn(index.gdsn(gds, "snp.chromosome"))[keep],
      read.gdsn(index.gdsn(gds, "snp.position"))[keep],
      read.gdsn(index.gdsn(gds, "snp.allele"))[keep],
      sep = ":"
    ),
    stringsAsFactors = FALSE
  )
}

panel_markers <- marker_key(panel, validate_snp_set(panel, readRDS(snp_rds)))
baseline_markers <- marker_key(baseline)
shared <- merge(panel_markers, baseline_markers, by = "key", suffixes = c("_panel", "_baseline"))
if (nrow(shared) < min_joint) {
  stop("Only ", nrow(shared), " markers share exact position and alleles with the baseline")
}

panel_dosage <- snpgdsGetGeno(panel, snp.id = shared$snp_id_panel, with.id = TRUE)
baseline_dosage <- snpgdsGetGeno(baseline, snp.id = shared$snp_id_baseline, with.id = TRUE)
panel_matrix <- panel_dosage$genotype
baseline_matrix <- baseline_dosage$genotype
panel_matrix <- panel_matrix[, match(shared$snp_id_panel, panel_dosage$snp.id), drop = FALSE]
baseline_matrix <- baseline_matrix[, match(shared$snp_id_baseline, baseline_dosage$snp.id), drop = FALSE]
rownames(panel_matrix) <- panel_dosage$sample.id
rownames(baseline_matrix) <- baseline_dosage$sample.id

strata <- read_strata(strata_path, panel_samples)

# The name-level proposal only narrows the search; the genotype decides.  Every baseline
# donor is compared against every panel member, which removes any dependence on spelling.
#
# The comparison is written as matrix algebra over blocks of markers rather than as a
# loop over pairs.  A loop would rebuild a 3685 x ~100000 subset once per donor, which is
# gigabytes of copying and hours of interpreted work for an arithmetic that BLAS does in
# seconds.  Agreement between two genotype calls is counted by summing, over the three
# possible dosages, the product of the indicator matrices; joint callability is the same
# product over the non-missing indicators.
concordance_matrices <- function(panel, base, block_size = 20000L) {
  n_panel <- nrow(panel)
  n_base <- nrow(base)
  agree <- matrix(0, n_panel, n_base)
  joint <- matrix(0, n_panel, n_base)
  starts <- seq(1L, ncol(panel), by = block_size)
  for (start in starts) {
    stop_at <- min(start + block_size - 1L, ncol(panel))
    columns <- start:stop_at
    panel_block <- panel[, columns, drop = FALSE]
    base_block <- base[, columns, drop = FALSE]
    panel_called <- !is.na(panel_block) & panel_block != 3L
    base_called <- !is.na(base_block) & base_block != 3L
    joint <- joint + tcrossprod(panel_called * 1, base_called * 1)
    for (dosage in 0:2) {
      agree <- agree + tcrossprod(
        (panel_called & panel_block == dosage) * 1,
        (base_called & base_block == dosage) * 1
      )
    }
  }
  list(agree = agree, joint = joint)
}

counts <- concordance_matrices(panel_matrix, baseline_matrix)
concordance <- ifelse(counts$joint >= min_joint, counts$agree / counts$joint, NA_real_)

best_for_donor <- function(column) {
  values <- concordance[, column]
  if (all(is.na(values))) {
    return(list(best = NA_character_, concordance = NA_real_, joint = 0L, runner_up = NA_real_))
  }
  best_index <- which.max(values)
  ordered <- sort(values, decreasing = TRUE, na.last = NA)
  list(
    best = rownames(panel_matrix)[best_index],
    concordance = unname(values[best_index]),
    joint = as.integer(counts$joint[best_index, column]),
    runner_up = if (length(ordered) > 1L) unname(ordered[2]) else NA_real_
  )
}

matches <- lapply(seq_len(nrow(baseline_matrix)), best_for_donor)
resolved <- data.frame(
  baseline_sample = rownames(baseline_matrix),
  panel_sample = vapply(matches, function(m) if (is.null(m$best)) NA_character_ else m$best, character(1)),
  dosage_concordance = vapply(matches, function(m) m$concordance, numeric(1)),
  jointly_called_markers = vapply(matches, function(m) m$joint, integer(1)),
  runner_up_concordance = vapply(matches, function(m) m$runner_up, numeric(1)),
  stringsAsFactors = FALSE
)
resolved$identity_confirmed <- !is.na(resolved$dosage_concordance) &
  resolved$dosage_concordance >= min_concordance &
  resolved$jointly_called_markers >= min_joint
resolved$panel_sample[!resolved$identity_confirmed] <- NA_character_

if (anyDuplicated(stats::na.omit(resolved$panel_sample))) {
  stop("Two baseline donors resolved to the same panel sample")
}

n_confirmed <- sum(resolved$identity_confirmed)
write_private_tsv(resolved, file.path(outdir, "m27d_baseline_identity.private.tsv"))
writeLines(
  stats::na.omit(resolved$panel_sample),
  file.path(outdir, "m27d_baseline_panel_identities.private.txt")
)

summary <- list(
  stage = "M27D_BASELINE_IDENTITY",
  scientific_result = FALSE,
  king_executed = FALSE,
  n_baseline_donors = nrow(resolved),
  random_seed = seed,
  n_shared_markers = nrow(shared),
  n_identities_confirmed = n_confirmed,
  n_identities_expected = expected_shared,
  n_baseline_donors_absent_from_panel = nrow(resolved) - n_confirmed,
  identity_matches_expectation = n_confirmed == expected_shared,
  min_confirmed_dosage_concordance = if (n_confirmed > 0) {
    min(resolved$dosage_concordance[resolved$identity_confirmed])
  } else NA_real_,
  max_runner_up_concordance = suppressWarnings(max(resolved$runner_up_concordance, na.rm = TRUE)),
  min_jointly_called_markers = if (n_confirmed > 0) {
    min(resolved$jointly_called_markers[resolved$identity_confirmed])
  } else NA_integer_,
  dosage_concordance_min_required = min_concordance,
  jointly_called_markers_min_required = min_joint,
  # A donor without a panel twin cannot be compared to any candidate by kinship.  That
  # is a real blind spot in the disjunction argument, not a rounding error, so it is
  # reported rather than absorbed into the confirmed count.
  unmatched_baseline_donor_blocks_full_kinship_disjointness = n_confirmed < nrow(resolved),
  sample_ids_emitted = FALSE,
  software = software_versions()
)
write_json(summary, file.path(outdir, "m27d_baseline_identity.json"), pretty = TRUE, auto_unbox = TRUE)

# Stop-rule 1 of the contract: identities must reconcile.  The summary above is written
# first so that a failure leaves its evidence behind, and only then does the stage refuse
# to continue.  Everything downstream reads the identity file, so proceeding with fewer
# confirmations than the preregistration expects would let G4 pass on an incomplete
# comparison instead of failing on an unreconciled one.
if (n_confirmed != expected_shared) {
  stop(
    "Baseline identity reconciliation failed: expected ", expected_shared,
    " confirmed identities, observed ", n_confirmed,
    ". The disjointness audit cannot run against an incomplete identity set."
  )
}
