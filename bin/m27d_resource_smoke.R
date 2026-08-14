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

gds_path <- value_after("gds")
anchor_rds <- value_after("anchor-rds")
strict_rds <- value_after("strict-rds")
metadata_strata <- value_after("metadata-strata")
preregistration <- value_after("preregistration")
outdir <- value_after("outdir", ".")
thread_grid <- as.integer(split_csv(value_after("thread-grid", "4,8,16")))

required_files <- c(gds_path, anchor_rds, strict_rds, metadata_strata, preregistration)
if (any(is.na(required_files)) || any(!file.exists(required_files))) stop("At least one prepared input is missing")
if (any(is.na(thread_grid)) || any(thread_grid < 1L)) stop("Invalid thread grid")
if (anyDuplicated(thread_grid)) stop("Thread grid contains duplicates")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

contract <- fromJSON(preregistration, simplifyVector = FALSE)
if (!identical(contract$stage, "M27D_DONOR_KINSHIP_AUDIT")) stop("Unexpected preregistration stage")
if (isTRUE(contract$pcrelate$king_allowed)) stop("M27D forbids KING")
expected_grid <- sort(as.integer(unlist(contract$resource_smoke$thread_screen)))
if (!identical(sort(thread_grid), expected_grid)) stop("Thread grid differs from preregistration")

gds <- snpgdsOpen(gds_path)
sample_ids <- read.gdsn(index.gdsn(gds, "sample.id"))
snp_ids_all <- read.gdsn(index.gdsn(gds, "snp.id"))
snp_chr_all <- read.gdsn(index.gdsn(gds, "snp.chromosome"))
expected_samples <- as.integer(contract$scope$official_panel_samples_expected)
if (length(sample_ids) != expected_samples) {
  stop("Panel sample count mismatch: expected ", expected_samples, ", observed ", length(sample_ids))
}

anchor_ids <- readRDS(anchor_rds)
strict_ids <- readRDS(strict_rds)
if (length(anchor_ids) == 0L || length(strict_ids) == 0L) stop("Prepared LD sets are empty")
if (!all(anchor_ids %in% snp_ids_all) || !all(strict_ids %in% snp_ids_all)) {
  stop("Prepared LD set contains SNP IDs absent from GDS")
}
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
required_strata_columns <- c("sample_id", "match_status", "Source", "Ancestry")
if (!all(required_strata_columns %in% colnames(strata))) stop("Prepared strata table is incomplete")
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

anchor_config <- Filter(function(config) identical(config$id, "anchor_pc8_r2_020"), contract$configurations)
if (length(anchor_config) != 1L) stop("Missing unique anchor configuration")
anchor_pcs <- as.integer(anchor_config[[1]]$n_pcs)
max_pcs <- max(vapply(contract$configurations, function(config) as.integer(config$n_pcs), integer(1)))

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
      eigen.cnt = max_pcs,
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
      pcs = pcs[, seq_len(anchor_pcs), drop = FALSE],
      scale = contract$pcrelate$scale,
      ibd.probs = TRUE,
      sample.include = samples,
      training.set = samples,
      maf.thresh = as.numeric(contract$pcrelate$maf_thresh),
      maf.bound.method = contract$pcrelate$maf_bound_method,
      small.samp.correct = isTRUE(contract$pcrelate$small_sample_correction),
      BPPARAM = MulticoreParam(workers = threads, progressbar = FALSE),
      verbose = FALSE
    )
  })
  n_pairs <- nrow(related$kinBtwn)
  expected_pairs <- length(samples) * (length(samples) - 1) / 2
  if (n_pairs != expected_pairs) stop("Unexpected PC-Relate pair count")
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

started <- Sys.time()
benchmarks <- do.call(
  rbind,
  lapply(thread_grid, function(threads) benchmark_run("full_n", sample_ids, arm_full_snps, threads))
)
fastest <- min(benchmarks$total_elapsed_seconds)
eligible <- benchmarks[benchmarks$total_elapsed_seconds <= 1.20 * fastest, , drop = FALSE]
candidate_threads <- min(eligible$threads)
benchmarks <- rbind(
  benchmarks,
  benchmark_run("marker_scaling", arm_marker_samples, arm_marker_snps, candidate_threads)
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
  anchor_ld_pruned_snps = length(anchor_ids),
  strict_ld_pruned_snps = length(strict_ids),
  candidate_threads_by_time = candidate_threads,
  final_thread_selection_requires_peak_ram = TRUE,
  thread_selection_rule = contract$resource_smoke$selection_rule,
  elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
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
