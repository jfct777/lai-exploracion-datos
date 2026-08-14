#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(BiocParallel)
  library(GENESIS)
  library(GWASTools)
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

panel_vcfs <- split_csv(value_after("panel-vcfs"))
metadata_strata <- value_after("metadata-strata")
exclude_bed <- value_after("exclude-bed")
preregistration <- value_after("preregistration")
outdir <- value_after("outdir", ".")
thread_grid <- as.integer(split_csv(value_after("thread-grid", "4,8,16")))

if (length(panel_vcfs) != 22L) stop("Expected 22 autosomal panel VCFs; found ", length(panel_vcfs))
if (any(!file.exists(panel_vcfs))) stop("At least one panel VCF is missing")
if (any(is.na(thread_grid)) || any(thread_grid < 1L)) stop("Invalid thread grid")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

contract <- fromJSON(preregistration, simplifyVector = FALSE)
if (!identical(contract$stage, "M27D_DONOR_KINSHIP_AUDIT")) stop("Unexpected preregistration stage")
if (isTRUE(contract$pcrelate$king_allowed)) stop("M27D forbids KING")

gds_path <- file.path(outdir, "m27d_official_panel_autosomes.gds")
started <- Sys.time()
snpgdsVCF2GDS(
  panel_vcfs,
  gds_path,
  method = "biallelic.only",
  snpfirstdim = FALSE,
  ignore.chr.prefix = "chr",
  verbose = TRUE
)
conversion_seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))

gds <- snpgdsOpen(gds_path)
sample_ids <- read.gdsn(index.gdsn(gds, "sample.id"))
snp_ids_all <- read.gdsn(index.gdsn(gds, "snp.id"))
snp_chr_all <- read.gdsn(index.gdsn(gds, "snp.chromosome"))
snp_pos_all <- read.gdsn(index.gdsn(gds, "snp.position"))

expected_samples <- as.integer(contract$scope$official_panel_samples_expected)
if (length(sample_ids) != expected_samples) {
  stop("Panel sample count mismatch: expected ", expected_samples, ", observed ", length(sample_ids))
}

qc_ids <- snpgdsSelectSNP(
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
qc_ids <- qc_ids[!qc_ids %in% snp_ids_all[in_long_ld]]
if (length(qc_ids) == 0L) stop("No SNPs remain after common-marker QC")

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
    num.thread = max(thread_grid),
    verbose = TRUE
  )
  unname(unlist(pruned, use.names = FALSE))
}

anchor_ids <- prune_ids(0.20)
strict_ids <- prune_ids(0.10)
saveRDS(anchor_ids, file.path(outdir, "m27d_ld_pruned_anchor_snp_ids.rds"))
saveRDS(strict_ids, file.path(outdir, "m27d_ld_pruned_strict_snp_ids.rds"))

snp_index <- match(anchor_ids, snp_ids_all)
anchor_chr <- snp_chr_all[snp_index]

even_snps <- function(ids, chromosomes, target_n) {
  target_n <- min(as.integer(target_n), length(ids))
  if (target_n == length(ids)) return(ids)
  selected <- integer(0)
  chromosomes_present <- sort(unique(chromosomes))
  base_n <- target_n %/% length(chromosomes_present)
  remainder <- target_n %% length(chromosomes_present)
  for (offset in seq_along(chromosomes_present)) {
    chromosome <- chromosomes_present[offset]
    candidates <- ids[chromosomes == chromosome]
    take_n <- min(length(candidates), base_n + as.integer(offset <= remainder))
    if (take_n > 0L) {
      positions <- unique(pmax(1L, pmin(length(candidates), round(seq(1, length(candidates), length.out = take_n)))))
      selected <- c(selected, candidates[positions])
    }
  }
  if (length(selected) < target_n) {
    selected <- c(selected, head(setdiff(ids, selected), target_n - length(selected)))
  }
  selected[seq_len(min(target_n, length(selected)))]
}

arm_full_snps <- even_snps(
  anchor_ids,
  anchor_chr,
  as.integer(contract$resource_smoke$arm_full_n$snps)
)
arm_marker_snps <- even_snps(
  anchor_ids,
  anchor_chr,
  as.integer(contract$resource_smoke$arm_marker_scaling$max_snps)
)

strata <- read.delim(metadata_strata, stringsAsFactors = FALSE, check.names = FALSE)
strata <- strata[strata$sample_id %in% sample_ids, , drop = FALSE]
strata$stratum <- paste(strata$Source, strata$Ancestry, sep = "|")

stratified_samples <- function(table, target_n, all_ids) {
  target_n <- min(as.integer(target_n), length(all_ids))
  matched <- table[table$match_status == "MATCHED" & nzchar(table$stratum), , drop = FALSE]
  matched <- matched[order(matched$stratum, matched$sample_id), , drop = FALSE]
  groups <- split(matched$sample_id, matched$stratum)
  selected <- character(0)
  while (length(selected) < target_n && any(lengths(groups) > 0L)) {
    for (name in sort(names(groups))) {
      if (length(selected) >= target_n) break
      if (length(groups[[name]]) == 0L) next
      selected <- c(selected, groups[[name]][1])
      groups[[name]] <- groups[[name]][-1]
    }
  }
  if (length(selected) < target_n) {
    selected <- c(selected, head(setdiff(sort(all_ids), selected), target_n - length(selected)))
  }
  selected
}

arm_marker_samples <- stratified_samples(
  strata,
  as.integer(contract$resource_smoke$arm_marker_scaling$samples),
  sample_ids
)
snpgdsClose(gds)

benchmark_run <- function(label, samples, snps, threads) {
  pca_gds <- snpgdsOpen(gds_path)
  pca_time <- system.time({
    pca <- snpgdsPCA(
      pca_gds,
      sample.id = samples,
      snp.id = snps,
      autosome.only = TRUE,
      remove.monosnp = TRUE,
      maf = NaN,
      missing.rate = NaN,
      num.thread = threads,
      eigen.cnt = 12L,
      algorithm = "randomized",
      verbose = FALSE
    )
  })
  snpgdsClose(pca_gds)
  pcs <- pca$eigenvect
  rownames(pcs) <- pca$sample.id

  geno_reader <- GdsGenotypeReader(filename = gds_path)
  geno_data <- GenotypeData(geno_reader)
  geno_iter <- GenotypeBlockIterator(geno_data, snpInclude = snps)
  pcrelate_time <- system.time({
    related <- pcrelate(
      geno_iter,
      pcs = pcs[, seq_len(8L), drop = FALSE],
      scale = "overall",
      ibd.probs = TRUE,
      sample.include = samples,
      training.set = samples,
      maf.thresh = 0.01,
      maf.bound.method = "filter",
      small.samp.correct = TRUE,
      BPPARAM = MulticoreParam(workers = threads, progressbar = FALSE),
      verbose = FALSE
    )
  })
  n_pairs <- nrow(related$kinBtwn)
  rm(related)
  close(geno_data)
  gc(verbose = FALSE)
  data.frame(
    arm = label,
    n_samples = length(samples),
    n_snps = length(snps),
    n_pairs = n_pairs,
    threads = threads,
    pca_elapsed_seconds = unname(pca_time[["elapsed"]]),
    pcrelate_elapsed_seconds = unname(pcrelate_time[["elapsed"]]),
    total_elapsed_seconds = unname(pca_time[["elapsed"]] + pcrelate_time[["elapsed"]]),
    stringsAsFactors = FALSE
  )
}

benchmarks <- do.call(
  rbind,
  lapply(thread_grid, function(threads) benchmark_run("full_n", sample_ids, arm_full_snps, threads))
)
fastest <- min(benchmarks$total_elapsed_seconds)
eligible <- benchmarks[benchmarks$total_elapsed_seconds <= 1.20 * fastest, , drop = FALSE]
selected_threads <- min(eligible$threads)
benchmarks <- rbind(
  benchmarks,
  benchmark_run("marker_scaling", arm_marker_samples, arm_marker_snps, selected_threads)
)

write.table(
  benchmarks,
  file.path(outdir, "m27d_resource_smoke.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

summary <- list(
  stage = "M27D_RESOURCE_SMOKE",
  scientific_result = FALSE,
  king_executed = FALSE,
  official_panel_samples = length(sample_ids),
  imported_biallelic_snps = length(snp_ids_all),
  common_callable_snps_outside_long_ld = length(qc_ids),
  anchor_ld_pruned_snps = length(anchor_ids),
  strict_ld_pruned_snps = length(strict_ids),
  conversion_seconds = conversion_seconds,
  selected_threads = selected_threads,
  thread_selection_rule = contract$resource_smoke$selection_rule,
  peak_ram_gate_requires_nextflow_trace = TRUE,
  benchmark_rows = lapply(seq_len(nrow(benchmarks)), function(i) as.list(benchmarks[i, ])),
  software = list(
    R = as.character(getRversion()),
    GENESIS = as.character(packageVersion("GENESIS")),
    SNPRelate = as.character(packageVersion("SNPRelate")),
    GWASTools = as.character(packageVersion("GWASTools")),
    BiocParallel = as.character(packageVersion("BiocParallel"))
  ),
  sample_ids_emitted = FALSE
)
write_json(summary, file.path(outdir, "m27d_resource_smoke.json"), pretty = TRUE, auto_unbox = TRUE)

marker_qc <- data.frame(
  step = c("imported_biallelic", "common_callable_outside_long_ld", "anchor_r2_0.20", "strict_r2_0.10"),
  n_snps = c(length(snp_ids_all), length(qc_ids), length(anchor_ids), length(strict_ids))
)
write.table(
  marker_qc,
  file.path(outdir, "m27d_marker_qc.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
