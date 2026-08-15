#!/usr/bin/env Rscript

# Helpers shared by the M27D productive stages.
#
# The two stages that already ran in the cloud (marker preparation and the resource
# benchmark) keep their own inline copies on purpose: their outputs are published and
# their manifests bind to the exact bytes of those files, so rewriting them now would
# break the provenance chain of a finished run for a cosmetic gain.  Every stage added
# afterwards sources this file instead.

value_after <- function(args, flag, default = NULL) {
  index <- which(args == paste0("--", flag))
  if (length(index) == 0L) return(default)
  if (index[1] == length(args)) stop("Missing value for --", flag)
  args[index[1] + 1L]
}

split_csv <- function(value) {
  if (is.null(value)) return(character(0))
  parts <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
  parts[nzchar(parts)]
}

elapsed_seconds <- function(started) {
  as.numeric(difftime(Sys.time(), started, units = "secs"))
}

require_files <- function(paths) {
  if (any(is.na(paths)) || any(!nzchar(paths))) stop("A required input path was not provided")
  missing <- paths[!file.exists(paths)]
  if (length(missing)) stop("Missing required input: ", paste(basename(missing), collapse = ", "))
  invisible(TRUE)
}

read_contract <- function(path) {
  contract <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(contract$stage, "M27D_DONOR_KINSHIP_AUDIT")) stop("Unexpected preregistration stage")
  if (isTRUE(contract$pcrelate$king_allowed)) stop("M27D forbids KING")
  if ("KING" %in% unlist(contract$scope$allowed_actions)) stop("M27D forbids KING")
  contract
}

# snpgdsPCA(algorithm = "randomized") draws a random test matrix, so an unseeded run is
# not reproducible: on the synthetic fixture two runs over identical input differed by
# up to 0.97 in the eigenvectors, and seeding drove that to exactly zero.  Those scores
# reach PC-Relate, the training set and the candidate list, so the seed comes from the
# preregistration and is written into every stage summary.
contract_seed <- function(contract) {
  seed <- contract$determinism$random_seed
  if (is.null(seed)) stop("Preregistration does not fix determinism$random_seed")
  seed <- as.integer(seed)
  if (is.na(seed)) stop("Preregistered random seed is not an integer")
  seed
}

apply_seed <- function(contract) {
  seed <- contract_seed(contract)
  set.seed(seed)
  seed
}

contract_configuration <- function(contract, id) {
  match <- Filter(function(config) identical(config$id, id), contract$configurations)
  if (length(match) != 1L) stop("Missing unique preregistered configuration: ", id)
  match[[1]]
}

validate_snp_set <- function(gds, snp_ids) {
  snp_ids <- unname(unlist(snp_ids, use.names = FALSE))
  if (length(snp_ids) == 0L) stop("Prepared marker set is empty")
  if (anyDuplicated(snp_ids)) stop("Prepared marker set contains duplicates")
  available <- read.gdsn(index.gdsn(gds, "snp.id"))
  if (!all(snp_ids %in% available)) stop("Prepared marker set contains SNP IDs absent from the GDS")
  snp_ids
}

read_strata <- function(path, sample_ids) {
  strata <- read.delim(path, stringsAsFactors = FALSE, check.names = FALSE, colClasses = "character")
  required <- c("sample_id", "match_status", "resolution_method", "population_interpretable", "Exclude")
  missing <- setdiff(required, colnames(strata))
  if (length(missing)) stop("Strata table is missing columns: ", paste(missing, collapse = ", "))
  if (anyDuplicated(strata$sample_id)) stop("Strata table contains duplicate sample identifiers")
  if (!all(sample_ids %in% strata$sample_id)) stop("Strata table does not cover every panel sample")
  if (any(strata$resolution_method == "AMBIGUOUS_FAIL_CLOSED")) {
    stop("Strata table still contains unresolved alias collisions")
  }
  strata[match(sample_ids, strata$sample_id), , drop = FALSE]
}

# Only an explicit metadata exclusion removes a sample.  A missing metadata row is a
# bookkeeping gap, not a reason to drop somebody from a kinship audit, so those samples
# stay in the genotypic analysis and lose only their population interpretation.
eligible_samples <- function(strata, sample_ids) {
  excluded <- toupper(trimws(strata$Exclude)) %in% c("TRUE", "T", "1", "YES", "Y")
  sample_ids[!excluded]
}

interpretable_samples <- function(strata, sample_ids) {
  flag <- toupper(trimws(strata$population_interpretable)) == "TRUE"
  sample_ids[flag]
}

# GENESIS silently drops the small-sample correction, with only a warning, when the
# cohort does not fit in a single sample block: inside .pcrelate the flag is rewritten to
# `(nsampblock == 1) & (scale != "none")`.  The preregistration asks for the correction,
# so the condition that makes it real is checked here instead of trusted.  With 3685
# samples and the default block size of 5000 there is one block, but a larger panel would
# turn a contract term into a warning nobody reads.
PCRELATE_SAMPLE_BLOCK_SIZE <- 5000L

assert_small_sample_correction_applies <- function(contract, n_samples) {
  if (!isTRUE(contract$pcrelate$small_sample_correction)) return(invisible(FALSE))
  if (identical(contract$pcrelate$scale, "none")) {
    stop("Preregistration asks for the small-sample correction but scale='none' disables it")
  }
  if (n_samples > PCRELATE_SAMPLE_BLOCK_SIZE) {
    stop(
      "Preregistration asks for the small-sample correction, but ", n_samples,
      " samples exceed the PC-Relate sample block size of ", PCRELATE_SAMPLE_BLOCK_SIZE,
      ", so GENESIS would silently disable it"
    )
  }
  invisible(TRUE)
}

threshold_counts <- function(kinship, contract) {
  thresholds <- sort(unique(c(
    as.numeric(contract$pcrelate$primary_phi_threshold),
    as.numeric(unlist(contract$pcrelate$descriptive_phi_thresholds))
  )))
  stats::setNames(
    lapply(thresholds, function(threshold) sum(kinship >= threshold, na.rm = TRUE)),
    paste0("phi_ge_", format(thresholds, scientific = FALSE, trim = TRUE))
  )
}

write_private_tsv <- function(table, path) {
  write.table(table, path, sep = "\t", quote = FALSE, row.names = FALSE)
}

write_private_tsv_gz <- function(table, path) {
  connection <- gzfile(path, "wt")
  on.exit(close(connection), add = TRUE)
  write.table(table, connection, sep = "\t", quote = FALSE, row.names = FALSE)
}

software_versions <- function() {
  installed <- rownames(utils::installed.packages())
  versions <- list(R = as.character(getRversion()))
  for (package in c("GENESIS", "SNPRelate", "GWASTools", "gdsfmt", "BiocParallel")) {
    if (package %in% installed) versions[[package]] <- as.character(utils::packageVersion(package))
  }
  versions
}
