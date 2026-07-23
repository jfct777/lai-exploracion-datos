#!/usr/bin/env Rscript
# Parentesco corregido por ancestria: PC-AiR + PC-Relate (Conomos et al. 2016, AJHG 98:127).
#
# Por que no basta KING-robust en DNABR: KING asume estructura poblacional entre poblaciones
# DISCRETAS. En una cohorte admixta CONTINUA, dos individuos con la misma composicion ancestral
# comparten alelos por ancestria compartida, y KING los contabiliza como co-descendencia -> phi
# inflado. Medido en este dataset: a phi>=0.0442 el 98.7% de los 2601 individuos cae en UNA sola
# componente conexa (grado medio 10.5 "primos segundos" por persona), lo que hace inejecutable
# cualquier GroupKFold por bloque familiar.
#
# PC-Relate residualiza las frecuencias alelicas individuales por los PCs de ancestria (obtenidos
# con PC-AiR, que a su vez es robusto al parentesco) -> estima parentesco NETO de la ancestria.
#
# Salida: <out>.pcrelate.kin.tsv  (ID1 ID2 kin k0 k2)  +  <out>.pcair.eigenvect.tsv  +  <out>.summary.json

suppressPackageStartupMessages({
  library(SNPRelate); library(GENESIS); library(GWASTools); library(gdsfmt)
})

args <- commandArgs(trailingOnly = TRUE)
kv <- function(k, default = NULL) {
  i <- which(args == paste0("--", k))
  if (length(i) == 0) return(default)
  args[i + 1]
}
bed_prefix <- kv("bed_prefix")                      # prefijo de {bed,bim,fam}
prune_in   <- kv("prune_in")                        # lista de SNPs LD-pruned (una por linea)
out_prefix <- kv("out_prefix")
n_pcs      <- as.integer(kv("n_pcs", "8"))          # PCs de ancestria a condicionar
kin_thresh <- as.numeric(kv("kin_thresh", "0.0442"))# 3er grado: umbral del divisor unrelated/related de PC-AiR
n_threads  <- as.integer(kv("threads", "8"))

stopifnot(!is.null(bed_prefix), !is.null(out_prefix))
cat("[pcrelate] bed_prefix =", bed_prefix, "\n")
cat("[pcrelate] n_pcs =", n_pcs, " kin_thresh(PC-AiR) =", kin_thresh, "\n")

gds_file <- paste0(out_prefix, ".gds")
if (!file.exists(gds_file)) {
  snpgdsBED2GDS(paste0(bed_prefix, ".bed"), paste0(bed_prefix, ".fam"),
                paste0(bed_prefix, ".bim"), gds_file, cvt.chr = "int")
}
gds <- snpgdsOpen(gds_file)

snp_ids <- if (!is.null(prune_in)) {
  keep <- readLines(prune_in)
  all_snp <- read.gdsn(index.gdsn(gds, "snp.rs.id"))
  read.gdsn(index.gdsn(gds, "snp.id"))[all_snp %in% keep]
} else NULL
cat("[pcrelate] SNPs usados =", if (is.null(snp_ids)) "todos" else length(snp_ids), "\n")

# --- Paso 1: KING-robust como medida de DIVERGENCIA ancestral (no como parentesco final).
# PC-AiR lo necesita para dos cosas distintas: (a) kinobj = quien esta emparentado (para excluirlos
# del set de entrenamiento de los PCs), (b) divobj = quien es ancestralmente divergente.
cat("[pcrelate] corriendo KING-robust (semilla de PC-AiR)...\n")
king <- snpgdsIBDKING(gds, snp.id = snp_ids, num.thread = n_threads, verbose = TRUE)
kingMat <- king$kinship
dimnames(kingMat) <- list(king$sample.id, king$sample.id)
# SNPRelate y GWASTools no pueden tener el MISMO .gds abierto a la vez: hay que soltar el handle
# de SNPRelate antes de que GdsGenotypeReader lo reabra (si no: "has been created or opened").
snpgdsClose(gds)

# --- Paso 2: PC-AiR = PCs de ancestria robustos al parentesco.
# Particiona la muestra en un set "unrelated" (que define los ejes) y proyecta el resto.
# Sin esto, los PCs de un PCA normal capturan familias, no ancestria.
cat("[pcrelate] PC-AiR...\n")
geno <- GdsGenotypeReader(filename = gds_file)
genoData <- GenotypeData(geno)
pca <- pcair(genoData, kinobj = kingMat, divobj = kingMat,
             kin.thresh = kin_thresh, div.thresh = -kin_thresh,
             snp.include = snp_ids)
cat("[pcrelate] PC-AiR: unrelated =", length(pca$unrels), " related =", length(pca$rels), "\n")

ev <- data.frame(sample.id = pca$sample.id, pca$vectors[, seq_len(n_pcs), drop = FALSE])
colnames(ev)[-1] <- paste0("PC", seq_len(n_pcs))
write.table(ev, paste0(out_prefix, ".pcair.eigenvect.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- Paso 3: PC-Relate = parentesco condicionado a los PCs de ancestria.
# training.set = los "unrelated" de PC-AiR: sobre ellos se estiman las frecuencias alelicas
# individuales especificas de ancestria; el parentesco es el residuo sobre esas frecuencias.
cat("[pcrelate] PC-Relate (condicionando en", n_pcs, "PCs)...\n")
genoIter <- GenotypeBlockIterator(genoData, snpInclude = snp_ids)
pcrel <- pcrelate(genoIter, pcs = pca$vectors[, seq_len(n_pcs), drop = FALSE],
                  training.set = pca$unrels, ibd.probs = TRUE)
close(genoData)

kin <- pcrel$kinBtwn[, c("ID1", "ID2", "kin", "k0", "k2")]
write.table(kin, paste0(out_prefix, ".pcrelate.kin.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- Resumen comparativo: cuantos pares cruzan cada umbral de grado de parentesco.
grades <- c("4to" = 0.0221, "3ro" = 0.0442, "2do" = 0.0884, "1ro" = 0.1770, "MZ" = 0.2500)
counts <- sapply(grades, function(t) sum(kin$kin >= t, na.rm = TRUE))
cat("\n[pcrelate] pares por umbral (PC-Relate):\n")
for (g in names(grades)) cat(sprintf("    phi>=%.4f (%s grado): %d pares\n", grades[[g]], g, counts[[g]]))

json <- sprintf('{"n_samples": %d, "n_snps": %d, "n_pcs": %d, "n_unrelated_pcair": %d, "n_related_pcair": %d, "pairs_by_threshold": {%s}}',
                length(pca$sample.id), if (is.null(snp_ids)) NA else length(snp_ids), n_pcs,
                length(pca$unrels), length(pca$rels),
                paste(sprintf('"%s": %d', unname(grades), counts), collapse = ", "))
writeLines(json, paste0(out_prefix, ".summary.json"))
cat("[pcrelate] OK ->", paste0(out_prefix, ".pcrelate.kin.tsv"), "\n")
