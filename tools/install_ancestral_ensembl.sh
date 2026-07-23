#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/mnt/d/ega/DNABR/meta/ancestral_alleles"
OUT_FA="${DEST_DIR}/homo_sapiens_ancestor_GRCh38.fa"
OUT_FAI="${OUT_FA}.fai"

RELEASE="${RELEASE:-}"

mkdir -p "${DEST_DIR}"

if [[ -s "${OUT_FA}" && -s "${OUT_FAI}" ]]; then
  echo "[OK] Found existing ancestral FASTA and index:"
  ls -lh "${OUT_FA}" "${OUT_FAI}"
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

FA_CANDIDATE=""
FA_GZ_CANDIDATE=""

FA_CANDIDATE=$(find "${WORK_DIR}" -maxdepth 4 -type f \( -name "*.fa" -o -name "*.fasta" \) | head -n 1 || true)
FA_GZ_CANDIDATE=$(find "${WORK_DIR}" -maxdepth 4 -type f \( -name "*.fa.gz" -o -name "*.fasta.gz" \) | head -n 1 || true)

if [[ -n "${FA_GZ_CANDIDATE}" ]]; then
  echo "[INFO] Found gz FASTA: ${FA_GZ_CANDIDATE}"
  echo "[INFO] Writing normalized FASTA: ${OUT_FA}"
  gunzip -c "${FA_GZ_CANDIDATE}" > "${OUT_FA}"
elif [[ -n "${FA_CANDIDATE}" ]]; then
  echo "[INFO] Found FASTA: ${FA_CANDIDATE}"
  echo "[INFO] Writing normalized FASTA: ${OUT_FA}"
  cp -f "${FA_CANDIDATE}" "${OUT_FA}"
else
  echo "[ERROR] Could not locate extracted FASTA (.fa/.fa.gz) inside tarball." >&2
  echo "[ERROR] Inspect directory: ${WORK_DIR}" >&2
  ls -lah "${WORK_DIR}" >&2 || true
  exit 1
fi

if ! command -v samtools >/dev/null 2>&1; then
  echo "[ERROR] samtools not found on host. Install samtools to build the FASTA index (.fai)." >&2
  echo "[ERROR] Example (Ubuntu): sudo apt-get install samtools" >&2
  exit 2
fi

echo "[INFO] Indexing FASTA with samtools faidx..."
samtools faidx "${OUT_FA}"

echo "[INFO] Final checks:"
ls -lh "${OUT_FA}" "${OUT_FAI}"

test -s "${OUT_FA}"
test -s "${OUT_FAI}"

if command -v md5sum >/dev/null 2>&1; then
  echo "[INFO] md5sum (FASTA):"
  md5sum "${OUT_FA}" || true
fi

echo "[OK] Ensembl ancestral FASTA installed."
