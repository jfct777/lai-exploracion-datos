#!/usr/bin/env Rscript
# Private descriptive donor/reference audit. No donor removal or independence claim.
# Two Nextflow stages reuse the same authenticated KING -> PC-AiR checkpoint.
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1",
           VECLIB_MAXIMUM_THREADS = "1", NUMEXPR_NUM_THREADS = "1")
suppressPackageStartupMessages({
  library(GENESIS); library(SNPRelate); library(GWASTools); library(gdsfmt)
  library(BiocParallel); library(jsonlite); library(digest)
})

kin_hash <- function(path) digest::digest(file = path, algo = "sha256")

kin_auth <- function(path, sha, label) {
  if (!is.character(path) || length(path) != 1L || !file.exists(path) || dir.exists(path))
    stop("Missing file: ", label)
  if (!is.character(sha) || length(sha) != 1L || !grepl("^[0-9a-f]{64}$", sha) ||
      !identical(kin_hash(path), sha)) stop("SHA256 mismatch: ", label)
  invisible(TRUE)
}

kin_integer <- function(x, name, minimum = 1L) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x != floor(x) ||
      x < minimum || x > .Machine$integer.max) stop("Invalid integer: ", name)
  as.integer(x)
}

kin_parameters <- function(p) {
  required <- c("pc_counts", "kin_threshold", "div_threshold", "maf_threshold",
                "sample_block_size", "snp_block_size", "threads", "small_sample_correction", "seed")
  if (!is.list(p) || anyDuplicated(names(p)) || !setequal(names(p), required)) stop("Unexpected parameter fields")
  counts <- unlist(p$pc_counts, use.names = FALSE)
  if (!length(counts) || anyDuplicated(counts)) stop("Invalid pc_counts")
  p$pc_counts <- sort(vapply(counts, kin_integer, integer(1), name = "pc_counts"))
  for (name in c("sample_block_size", "snp_block_size", "threads", "seed"))
    p[[name]] <- kin_integer(p[[name]], name, if (name == "seed") 0L else 1L)
  for (name in c("kin_threshold", "div_threshold", "maf_threshold")) {
    value <- p[[name]]
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value))
      stop("Invalid numerical parameter: ", name)
  }
  if (p$kin_threshold <= 0 || p$kin_threshold >= 0.5 || p$div_threshold >= 0 ||
      p$div_threshold <= -1 || p$maf_threshold <= 0 || p$maf_threshold >= 0.5)
    stop("Parameter outside declared domain")
  if (!is.logical(p$small_sample_correction) || length(p$small_sample_correction) != 1L ||
      is.na(p$small_sample_correction)) stop("small_sample_correction must be Boolean")
  p
}

kin_contract <- function(path, sha) {
  kin_auth(path, sha, "contract")
  c <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(c$schema, "r02_lai_panel_kinship_v1")) stop("Unexpected contract schema")
  if (anyDuplicated(names(c)) || anyDuplicated(names(c$inputs)) ||
      !setequal(names(c$inputs), c("gds", "roles", "snp_ids"))) stop("Unexpected input descriptors")
  for (name in names(c$inputs))
    kin_auth(c$inputs[[name]]$path, c$inputs[[name]]$sha256, name)
  c$parameters <- kin_parameters(c$parameters)
  c$contract_sha256 <- sha
  c
}

kin_inputs <- function(c) {
  roles <- read.delim(c$inputs$roles$path, colClasses = "character", check.names = FALSE,
                      stringsAsFactors = FALSE, na.strings = c("", "NA"))
  required <- c("sample_id", "role", "ancestry", "population")
  if (!all(required %in% names(roles)) || anyNA(roles[, required]) ||
      anyDuplicated(roles$sample_id) || any(!nzchar(roles$sample_id)))
    stop("Invalid or incomplete role table")
  if (any(!roles$role %in% c("REF_TRAIN", "SOURCE_VALID")))
    stop("Forbidden role: only REF_TRAIN and SOURCE_VALID may be opened")
  g <- snpgdsOpen(c$inputs$gds$path, readonly = TRUE)
  on.exit(snpgdsClose(g), add = TRUE)
  ids <- as.character(read.gdsn(index.gdsn(g, "sample.id")))
  if (anyDuplicated(ids) || !setequal(ids, roles$sample_id) || length(ids) != nrow(roles))
    stop("GDS IDs do not match the exact permitted role universe")
  if (length(ids) <= max(c$parameters$pc_counts) + 2L) stop("Insufficient sample count for PCs")
  markers <- readRDS(c$inputs$snp_ids$path)
  available <- read.gdsn(index.gdsn(g, "snp.id"))
  chromosomes <- read.gdsn(index.gdsn(g, "snp.chromosome"))
  if (!is.atomic(markers) || !length(markers) || anyNA(markers) || anyDuplicated(markers) ||
      !all(markers %in% available)) stop("Invalid LD-pruned SNP IDs")
  if (!all(chromosomes[match(markers, available)] %in% 1:22)) stop("Non-autosomal marker")
  if (length(markers) <= max(c$parameters$pc_counts)) stop("Insufficient SNPs for PCs")
  if (c$parameters$small_sample_correction && c$parameters$sample_block_size < length(ids))
    stop("Small-sample correction requires a single sample block in GENESIS")
  list(ids = ids, roles = roles[match(ids, roles$sample_id), , drop = FALSE], markers = markers)
}

kin_write <- function(table, path, gzip = FALSE) {
  con <- if (gzip) gzfile(path, "wt") else file(path, "wt")
  on.exit(close(con), add = TRUE)
  write.table(table, con, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
}

kin_versions <- function() {
  packages <- c("GENESIS", "SNPRelate", "GWASTools", "gdsfmt", "BiocParallel", "jsonlite", "digest")
  c(list(R = as.character(getRversion())), setNames(lapply(packages, function(x)
    as.character(packageVersion(x))), packages))
}

kin_pcs <- function(cp, inputs, n) {
  if (!identical(as.character(cp$sample_ids), inputs$ids) ||
      !identical(cp$markers, inputs$markers)) stop("Checkpoint identity mismatch")
  pcs <- cp$pcs
  if (!is.matrix(pcs) || !identical(rownames(pcs), inputs$ids) || ncol(pcs) < n ||
      any(!is.finite(pcs))) stop("Invalid checkpoint PC matrix")
  u <- as.character(cp$unrelated)
  r <- as.character(cp$related)
  if (anyDuplicated(c(u, r)) || !setequal(c(u, r), inputs$ids) || length(u) <= n + 1L)
    stop("Invalid PC-AiR fitting/projection partition")
  result <- pcs[, seq_len(n), drop = FALSE]
  if (qr(cbind(1, result[match(u, inputs$ids), , drop = FALSE]))$rank != n + 1L)
    stop("Rank-deficient training PCs")
  result
}

kin_representation <- function(roles, unrelated) {
  roles$pcair_status <- ifelse(roles$sample_id %in% unrelated, "FIT_UNRELATED", "PROJECTED_RELATED")
  keys <- unique(roles[, c("role", "ancestry", "population"), drop = FALSE])
  keys$n_total <- keys$n_fit <- keys$n_projected <- integer(nrow(keys))
  for (i in seq_len(nrow(keys))) {
    m <- roles$role == keys$role[i] & roles$ancestry == keys$ancestry[i] & roles$population == keys$population[i]
    keys$n_total[i] <- sum(m)
    keys$n_fit[i] <- sum(m & roles$pcair_status == "FIT_UNRELATED")
    keys$n_projected[i] <- keys$n_total[i] - keys$n_fit[i]
  }
  list(individuals = roles, summary = keys)
}

kin_pcair <- function(c, inputs, outdir) {
  p <- c$parameters
  set.seed(p$seed)
  g <- snpgdsOpen(c$inputs$gds$path, readonly = TRUE)
  on.exit(snpgdsClose(g), add = TRUE)
  # KING only initializes the PC-AiR kinship/divergence matrices. It is not the final estimate.
  king <- snpgdsIBDKING(g, sample.id = inputs$ids, snp.id = inputs$markers,
                       type = "KING-robust", family.id = NULL, autosome.only = TRUE,
                       remove.monosnp = TRUE, maf = NaN, missing.rate = NaN,
                       num.thread = p$threads, verbose = FALSE)
  k <- king$kinship
  dimnames(k) <- list(as.character(king$sample.id), as.character(king$sample.id))
  if (!setequal(rownames(k), inputs$ids) || !identical(dim(k), c(length(inputs$ids), length(inputs$ids))))
    stop("KING matrix IDs or dimensions are invalid")
  k <- k[inputs$ids, inputs$ids, drop = FALSE]
  if (any(!is.finite(k)) || !isTRUE(all.equal(k, t(k), tolerance = 1e-12)))
    stop("KING matrix must be finite and symmetric")
  pca <- pcair(g, kinobj = k, divobj = k, kin.thresh = p$kin_threshold,
               div.thresh = p$div_threshold, sample.include = inputs$ids,
               snp.include = inputs$markers, num.cores = p$threads,
               eigen.cnt = max(p$pc_counts), remove.monosnp = TRUE,
               maf = NaN, missing.rate = NaN, verbose = FALSE)
  if (!setequal(as.character(pca$sample.id), inputs$ids)) stop("PC-AiR dropped sample IDs")
  pcs <- pca$vectors[match(inputs$ids, as.character(pca$sample.id)), , drop = FALSE]
  rownames(pcs) <- inputs$ids
  colnames(pcs) <- paste0("PC", seq_len(ncol(pcs)))
  cp <- list(schema = "r02_lai_pcair_checkpoint_v1", contract_sha256 = c$contract_sha256,
             input_sha256 = lapply(c$inputs, function(x) x$sha256), parameters = p,
             sample_ids = inputs$ids, markers = inputs$markers, pcs = pcs,
             eigenvalues = pca$values,
             variance_proportion = pca$varprop[seq_along(pca$values)],
             unrelated = as.character(pca$unrels), related = as.character(pca$rels),
             king_initializer = k, software = kin_versions())
  kin_pcs(cp, inputs, max(p$pc_counts))
  if (length(cp$eigenvalues) < max(p$pc_counts) || any(!is.finite(cp$eigenvalues)))
    stop("Invalid PC-AiR eigenvalues")
  saveRDS(cp, file.path(outdir, "pcair.checkpoint.rds"), version = 3)
  kin_write(data.frame(sample_id = inputs$ids, pcs, check.names = FALSE), file.path(outdir, "pcs.private.tsv"))
  kin_write(data.frame(component = seq_along(cp$eigenvalues), eigenvalue = cp$eigenvalues,
                       variance_proportion = cp$variance_proportion), file.path(outdir, "eigenvalues.tsv"))
  representation <- kin_representation(inputs$roles, cp$unrelated)
  kin_write(representation$individuals, file.path(outdir, "pcair_roles.private.tsv"))
  kin_write(representation$summary, file.path(outdir, "representation.private.tsv"))
  list(status = "PCAIR_CHECKPOINT_NOT_DONOR_INDEPENDENCE", n_fit = length(cp$unrelated),
       n_projected = length(cp$related), checkpoint_sha256 = kin_hash(file.path(outdir, "pcair.checkpoint.rds")))
}

kin_pcrelate <- function(c, inputs, checkpoint, checkpoint_sha, n_pcs, outdir) {
  kin_auth(checkpoint, checkpoint_sha, "PC-AiR checkpoint")
  cp <- readRDS(checkpoint)
  if (!identical(cp$schema, "r02_lai_pcair_checkpoint_v1") ||
      !identical(cp$contract_sha256, c$contract_sha256) ||
      !identical(cp$input_sha256, lapply(c$inputs, function(x) x$sha256)) ||
      !identical(cp$parameters, c$parameters)) stop("Checkpoint does not belong to this contract")
  if (!identical(cp$software, kin_versions())) stop("Checkpoint runtime software differs")
  n_pcs <- kin_integer(n_pcs, "n_pcs")
  if (!n_pcs %in% c$parameters$pc_counts) stop("PC count is not declared in contract")
  pcs <- kin_pcs(cp, inputs, n_pcs)
  p <- c$parameters
  set.seed(p$seed)
  data <- GenotypeData(GdsGenotypeReader(c$inputs$gds$path))
  on.exit(close(data), add = TRUE)
  iterator <- GenotypeBlockIterator(data, snpBlock = p$snp_block_size, snpInclude = inputs$markers)
  parallel <- if (p$threads == 1L) SerialParam() else MulticoreParam(workers = p$threads,
    progressbar = FALSE, RNGseed = p$seed)
  rel <- pcrelate(iterator, pcs = pcs, training.set = cp$unrelated, sample.include = inputs$ids,
    scale = "overall", ibd.probs = TRUE, maf.thresh = p$maf_threshold, maf.bound.method = "filter",
    sample.block.size = p$sample_block_size, small.samp.correct = p$small_sample_correction,
    BPPARAM = parallel, verbose = FALSE)
  pairs <- rel$kinBtwn
  required <- c("ID1", "ID2", "kin", "k0", "k2")
  if (!all(required %in% names(pairs)) || nrow(pairs) != length(inputs$ids) * (length(inputs$ids) - 1) / 2)
    stop("Incomplete PC-Relate pair matrix")
  a <- match(pairs$ID1, inputs$ids); b <- match(pairs$ID2, inputs$ids)
  if (anyNA(c(a, b)) || any(a == b) || anyDuplicated(paste(pmin(a, b), pmax(a, b), sep = ":")))
    stop("Invalid or duplicate PC-Relate pairs")
  if (!all(c("ID", "f") %in% names(rel$kinSelf)) || anyDuplicated(rel$kinSelf$ID) ||
      !setequal(as.character(rel$kinSelf$ID), inputs$ids)) stop("Incomplete inbreeding table")
  pairs$role1 <- inputs$roles$role[a]; pairs$role2 <- inputs$roles$role[b]
  # Preserve non-finite estimates and their denominator; never turn them into unrelated pairs.
  pairs$estimate_finite <- is.finite(pairs$kin) & is.finite(pairs$k0) & is.finite(pairs$k2)
  kin_write(pairs, file.path(outdir, "pcrelate_pairs.private.tsv.gz"), gzip = TRUE)
  kin_write(rel$kinSelf, file.path(outdir, "inbreeding.private.tsv"))
  list(status = "PCRELATE_DESCRIPTIVE_NOT_DONOR_INDEPENDENCE", n_pcs = n_pcs,
       n_pairs = nrow(pairs), n_pairs_finite = sum(pairs$estimate_finite),
       n_inbreeding_finite = sum(is.finite(rel$kinSelf$f)), checkpoint_sha256 = checkpoint_sha,
       small_sample_correction_effective = p$small_sample_correction,
       sample_blocks = ceiling(length(inputs$ids) / p$sample_block_size))
}

kin_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  allowed <- c("stage", "contract", "contract-sha256", "outdir", "checkpoint", "checkpoint-sha256", "n-pcs")
  if (length(args) %% 2L != 0L || !length(args)) stop("Arguments require --name value pairs")
  keys <- sub("^--", "", args[seq(1L, length(args), by = 2L)])
  if (anyDuplicated(keys) || any(!keys %in% allowed) || any(!startsWith(args[seq(1L, length(args), by = 2L)], "--")))
    stop("Unknown or duplicate arguments")
  opts <- setNames(as.list(args[seq(2L, length(args), by = 2L)]), keys)
  if (!all(c("stage", "contract", "contract-sha256", "outdir") %in% keys) ||
      !opts$stage %in% c("pcair", "pcrelate")) stop("Missing required stage arguments")
  if (opts$stage == "pcrelate" && !all(c("checkpoint", "checkpoint-sha256", "n-pcs") %in% keys))
    stop("Missing checkpoint arguments")
  c <- kin_contract(opts$contract, opts[["contract-sha256"]])
  inputs <- kin_inputs(c)
  if (dir.exists(opts$outdir) && length(list.files(opts$outdir, all.files = TRUE, no.. = TRUE)))
    stop("Output directory is not empty; use Nextflow cache rather than overwrite")
  dir.create(opts$outdir, recursive = TRUE, showWarnings = FALSE)
  started <- Sys.time()
  result <- if (opts$stage == "pcair") kin_pcair(c, inputs, opts$outdir) else
    kin_pcrelate(c, inputs, opts$checkpoint, opts[["checkpoint-sha256"]],
                 suppressWarnings(as.numeric(opts[["n-pcs"]])), opts$outdir)
  result$schema <- "r02_lai_panel_kinship_result_v1"
  result$stage <- opts$stage
  result$contract_sha256 <- c$contract_sha256
  result$input_sha256 <- lapply(c$inputs, function(x) x$sha256)
  result$parameters <- c$parameters
  result$n_samples <- length(inputs$ids); result$n_markers <- length(inputs$markers)
  result$software <- kin_versions()
  result$elapsed_seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  result$source_test_opened <- FALSE
  result$donors_removed <- FALSE
  result$independence_certified <- FALSE
  result$population_structure_confirmed <- FALSE
  result$created_utc <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  # A source altered while calculating invalidates publication, not merely the next run.
  kin_auth(opts$contract, opts[["contract-sha256"]], "contract after computation")
  for (name in names(c$inputs)) kin_auth(c$inputs[[name]]$path, c$inputs[[name]]$sha256, name)
  if (opts$stage == "pcrelate") kin_auth(opts$checkpoint, opts[["checkpoint-sha256"]], "checkpoint after computation")
  write_json(result, file.path(opts$outdir, "summary.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
  outputs <- sort(list.files(opts$outdir, all.files = TRUE, no.. = TRUE))
  manifest <- list(schema = if (opts$stage == "pcair") "r02_lai_panel_pcair_v1" else "r02_lai_panel_pcrelate_v1",
    status = paste0("COMPLETE_", result$status), stage = opts$stage,
    contract_sha256 = c$contract_sha256, input_sha256 = result$input_sha256,
    outputs_sha256 = setNames(lapply(outputs, function(name) kin_hash(file.path(opts$outdir, name))), outputs),
    n_samples = length(inputs$ids), n_markers = length(inputs$markers),
    source_test_opened = FALSE, donors_removed = FALSE, independence_certified = FALSE,
    created_utc = result$created_utc)
  write_json(manifest, file.path(opts$outdir, "manifest.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat("Completed", opts$stage, "samples", length(inputs$ids), "markers", length(inputs$markers), "\n")
  invisible(result)
}

if (sys.nframe() == 0L) kin_main()
