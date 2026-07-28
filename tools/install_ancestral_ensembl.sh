#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Descarga y preparación del FASTA ancestral de Ensembl para el HPC.
# ============================================================================
# Uso: bash install_ancestral_ensembl.sh [RELEASE]
# Ejemplo: bash install_ancestral_ensembl.sh 113
# ============================================================================

DEST_DIR="/scratch/trusted/genbr/genbr_bioinfo/projects/DNABR_QC/reference/ensembl_ancestral"
OUT_FA="${DEST_DIR}/homo_sapiens_ancestor_GRCh38.fa"
OUT_FAI="${OUT_FA}.fai"

RELEASE="${1:-${RELEASE:-}}"

mkdir -p "${DEST_DIR}"

if [[ -s "${OUT_FA}" && -s "${OUT_FAI}" ]]; then
  echo "[OK] Found existing ancestral FASTA and index:"
  ls -lh "${OUT_FA}" "${OUT_FAI}"
  echo ""
  echo "[INFO] To re-download, delete these files first:"
  echo "  rm ${OUT_FA} ${OUT_FAI}"
  exit 0
fi

if [[ -n "${RELEASE}" ]]; then
  URL="ftp://ftp.ensembl.org/pub/release-${RELEASE}/fasta/ancestral_alleles/homo_sapiens_ancestor_GRCh38.tar.gz"
else
  URL="ftp://ftp.ensembl.org/pub/current_fasta/ancestral_alleles/homo_sapiens_ancestor_GRCh38.tar.gz"
fi

TARBALL="${DEST_DIR}/homo_sapiens_ancestor_GRCh38.tar.gz"
WORK_DIR="${DEST_DIR}/_tmp_extract"

if [[ ! -s "${TARBALL}" ]]; then
  echo "[INFO] Downloading: ${URL}"
  echo "[INFO] Destination: ${TARBALL}"
  curl -L --fail --retry 3 --retry-delay 5 -o "${TARBALL}" "${URL}"
else
  echo "[INFO] Using existing tarball: ${TARBALL}"
fi

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

echo "[INFO] Extracting tarball..."
tar -xzf "${TARBALL}" -C "${WORK_DIR}"

# Ensembl distribuye el FASTA ancestral en un archivo por cromosoma.
# Se concatenan 1-22, X, Y y MT en ese orden.
EXTRACTED_DIR=$(find "${WORK_DIR}" -maxdepth 2 -type d -name "homo_sapiens_ancestor_GRCh38" | head -n 1)

if [[ -z "${EXTRACTED_DIR}" || ! -d "${EXTRACTED_DIR}" ]]; then
  echo "[ERROR] No se encontró el directorio extraído 'homo_sapiens_ancestor_GRCh38'" >&2
  echo "[ERROR] Revisa el directorio: ${WORK_DIR}" >&2
  ls -lah "${WORK_DIR}" >&2 || true
  exit 1
fi

echo "[INFO] Found extracted directory: ${EXTRACTED_DIR}"
echo "[INFO] Concatenating individual chromosome FASTA files..."

# Construye la lista de FASTA en el orden 1-22, X, Y y MT.
FASTA_FILES=()
for CHR in {1..22} X Y MT; do
  FA="${EXTRACTED_DIR}/homo_sapiens_ancestor_${CHR}.fa"
  if [[ -f "${FA}" ]]; then
    FASTA_FILES+=("${FA}")
    echo "  [OK] Found: homo_sapiens_ancestor_${CHR}.fa"
  else
    echo "  [SKIP] Missing: homo_sapiens_ancestor_${CHR}.fa"
  fi
done

if [[ ${#FASTA_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No se encontraron archivos FASTA por cromosoma en ${EXTRACTED_DIR}" >&2
  ls -lh "${EXTRACTED_DIR}"/*.fa 2>&1 || true
  exit 1
fi

echo "[INFO] Concatenating ${#FASTA_FILES[@]} chromosome FASTA files into: ${OUT_FA}"
cat "${FASTA_FILES[@]}" > "${OUT_FA}"

echo "[INFO] Concatenation complete. Verifying output..."
if [[ ! -s "${OUT_FA}" ]]; then
  echo "[ERROR] El FASTA de salida está vacío o no existe: ${OUT_FA}" >&2
  exit 1
fi

# Busca samtools en el host y, si no está, intenta usar el contenedor.
SAMTOOLS_CMD="samtools"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/images/dnabr-qc_27-01-2026.sif}"

if ! command -v samtools >/dev/null 2>&1; then
  if [[ -f "${CONTAINER_IMAGE}" ]]; then
    echo "[INFO] samtools not found on host, using Singularity container..."
    SAMTOOLS_CMD="singularity exec ${CONTAINER_IMAGE} samtools"
  else
    echo "[ERROR] No se encontró samtools en el host ni un contenedor disponible." >&2
    echo "[ERROR] Instala samtools o carga el módulo bcftools/samtools:" >&2
    echo "  module load samtools" >&2
    echo "  # or" >&2
    echo "  micromamba activate nf" >&2
    exit 2
  fi
fi

echo "[INFO] Indexing FASTA with samtools faidx..."
${SAMTOOLS_CMD} faidx "${OUT_FA}"

echo "[INFO] Final checks:"
ls -lh "${OUT_FA}" "${OUT_FAI}"

test -s "${OUT_FA}"
test -s "${OUT_FAI}"

if command -v md5sum >/dev/null 2>&1; then
  echo "[INFO] md5sum (FASTA):"
  md5sum "${OUT_FA}" || true
fi

echo ""
echo "[OK] Ensembl ancestral FASTA installed successfully!"
echo ""
echo "======================================================================"
echo "IMPORTANT: Ensembl ancestral FASTA format"
echo "======================================================================"
echo "The Ensembl ancestral FASTA uses non-standard contig names like:"
echo "  'ANCESTOR_for_chromosome:GRCh38:1:1:248956422:1'"
echo ""
echo "Your workflow expects simple chromosome names like: '1', '2', ..., 'X', 'Y'"
echo ""
echo "To check the contig names in the downloaded file:"
echo "  grep '^>' ${OUT_FA} | head -n 5"
echo ""
echo "If contig names are incompatible, you may need to:"
echo "  1. Disable ancestral modules: --enable_build_ancestral_tsv false --enable_daf false"
echo "  2. Use alternative ancestral allele data with standard chromosome names"
echo "======================================================================"
echo ""
echo "Output file: ${OUT_FA}"
echo "Index file:  ${OUT_FAI}"
