#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SNPRelate)
  library(gdsfmt)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag, default = NULL) {
  index <- which(args == paste0("--", flag))
  if (length(index) == 0L) return(default)
  if (index[1] == length(args)) stop("Missing value for --", flag)
  args[index[1] + 1L]
}

split_csv <- function(value) {
  parts <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
  parts[nzchar(parts)]
}

elapsed_seconds <- function(started) {
  as.numeric(difftime(Sys.time(), started, units = "secs"))
}

panel_vcfs <- split_csv(value_after("panel-vcfs"))
exclude_bed <- value_after("exclude-bed")
preregistration <- value_after("preregistration")
outdir <- value_after("outdir", ".")
threads <- as.integer(value_after("threads", "16"))

if (length(panel_vcfs) != 22L) stop("Expected 22 autosomal panel VCFs; found ", length(panel_vcfs))
if (any(!file.exists(panel_vcfs))) stop("At least one panel VCF is missing")
if (is.na(threads) || threads < 1L) stop("Invalid thread count")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

contract <- fromJSON(preregistration, simplifyVector = FALSE)
if (!identical(contract$stage, "M27D_DONOR_KINSHIP_AUDIT")) stop("Unexpected preregistration stage")
if (isTRUE(contract$pcrelate$king_allowed)) stop("M27D forbids KING")

config_by_id <- setNames(contract$configurations, vapply(contract$configurations, `[[`, "", "id"))
anchor_r2 <- as.numeric(config_by_id[["anchor_pc8_r2_020"]]$ld_r2_max)
strict_r2 <- as.numeric(config_by_id[["pc8_r2_010"]]$ld_r2_max)
if (!isTRUE(all.equal(anchor_r2, 0.20)) || !isTRUE(all.equal(strict_r2, 0.10))) {
  stop("Unexpected preregistered LD thresholds")
}

gds_path <- file.path(outdir, "m27d_official_panel_autosomes.gds")
total_started <- Sys.time()
conversion_started <- Sys.time()
snpgdsVCF2GDS(
  panel_vcfs,
  gds_path,
  method = "biallelic.only",
  snpfirstdim = FALSE,
  ignore.chr.prefix = "chr",
  verbose = TRUE
)
conversion_seconds <- elapsed_seconds(conversion_started)

gds <- snpgdsOpen(gds_path)
on.exit(try(snpgdsClose(gds), silent = TRUE), add = TRUE)
sample_ids <- read.gdsn(index.gdsn(gds, "sample.id"))
snp_ids_all <- read.gdsn(index.gdsn(gds, "snp.id"))
snp_chr_all <- read.gdsn(index.gdsn(gds, "snp.chromosome"))
snp_pos_all <- read.gdsn(index.gdsn(gds, "snp.position"))

expected_samples <- as.integer(contract$scope$official_panel_samples_expected)
if (length(sample_ids) != expected_samples) {
  stop("Panel sample count mismatch: expected ", expected_samples, ", observed ", length(sample_ids))
}
if (!identical(sort(unique(as.integer(snp_chr_all))), 1:22)) {
  stop("Imported GDS does not contain exactly autosomes 1-22")
}

qc_started <- Sys.time()
qc_pre_ld_ids <- snpgdsSelectSNP(
  gds,
  autosome.only = TRUE,
  remove.monosnp = TRUE,
  maf = as.numeric(contract$marker_contract$global_maf_min),
  missing.rate = 1 - as.numeric(contract$marker_contract$variant_call_rate_min),
  verbose = TRUE
)

bed <- read.delim(exclude_bed, header = FALSE, comment.char = "#", stringsAsFactors = FALSE)
if (ncol(bed) < 3L) stop("Long-range LD BED must have at least three columns")
bed_chr <- suppressWarnings(as.integer(sub("^chr", "", bed[[1]], ignore.case = TRUE)))
in_long_ld <- rep(FALSE, length(snp_ids_all))
for (row in seq_len(nrow(bed))) {
  if (is.na(bed_chr[row])) next
  in_long_ld <- in_long_ld |
    (snp_chr_all == bed_chr[row] & snp_pos_all > bed[[2]][row] & snp_pos_all <= bed[[3]][row])
}
qc_ids <- qc_pre_ld_ids[!qc_pre_ld_ids %in% snp_ids_all[in_long_ld]]
if (length(qc_ids) == 0L) stop("No SNPs remain after common-marker QC")
qc_seconds <- elapsed_seconds(qc_started)

prune_ids <- function(r2_max) {
  pruned <- snpgdsLDpruning(
    gds,
    sample.id = sample_ids,
    snp.id = qc_ids,
    autosome.only = TRUE,
    remove.monosnp = TRUE,
    maf = NaN,
    missing.rate = NaN,
    method = "corr",
    slide.max.bp = as.integer(contract$marker_contract$ld_window_bp),
    ld.threshold = sqrt(r2_max),
    start.pos = "first",
    num.thread = threads,
    verbose = TRUE
  )
  unname(unlist(pruned, use.names = FALSE))
}

anchor_started <- Sys.time()
anchor_ids <- prune_ids(anchor_r2)
anchor_seconds <- elapsed_seconds(anchor_started)
strict_started <- Sys.time()
strict_ids <- prune_ids(strict_r2)
strict_seconds <- elapsed_seconds(strict_started)
if (length(anchor_ids) == 0L || length(strict_ids) == 0L) stop("LD pruning returned no SNPs")
if (!all(strict_ids %in% qc_ids) || !all(anchor_ids %in% qc_ids)) {
  stop("LD-pruned SNP IDs are not a subset of the QC marker set")
}

saveRDS(anchor_ids, file.path(outdir, "m27d_ld_pruned_anchor_snp_ids.rds"))
saveRDS(strict_ids, file.path(outdir, "m27d_ld_pruned_strict_snp_ids.rds"))

marker_qc <- data.frame(
  step = c(
    "imported_biallelic",
    "common_callable_before_long_ld",
    "common_callable_outside_long_ld",
    "anchor_r2_0.20",
    "strict_r2_0.10"
  ),
  n_snps = c(
    length(snp_ids_all),
    length(qc_pre_ld_ids),
    length(qc_ids),
    length(anchor_ids),
    length(strict_ids)
  )
)
write.table(
  marker_qc,
  file.path(outdir, "m27d_marker_qc.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

summary <- list(
  stage = "M27D_MARKER_PREPARATION",
  scientific_result = FALSE,
  king_executed = FALSE,
  official_panel_samples = length(sample_ids),
  imported_biallelic_snps = length(snp_ids_all),
  common_callable_snps_before_long_ld = length(qc_pre_ld_ids),
  common_callable_snps_outside_long_ld = length(qc_ids),
  anchor_ld_pruned_snps = length(anchor_ids),
  strict_ld_pruned_snps = length(strict_ids),
  timing_seconds = list(
    conversion = conversion_seconds,
    marker_qc = qc_seconds,
    anchor_ld_pruning = anchor_seconds,
    strict_ld_pruning = strict_seconds,
    total = elapsed_seconds(total_started)
  ),
  threads = threads,
  sample_ids_emitted = FALSE,
  software = list(
    R = as.character(getRversion()),
    SNPRelate = as.character(packageVersion("SNPRelate")),
    gdsfmt = as.character(packageVersion("gdsfmt"))
  )
)
write_json(summary, file.path(outdir, "m27d_marker_preparation.json"), pretty = TRUE, auto_unbox = TRUE)

snpgdsClose(gds)
