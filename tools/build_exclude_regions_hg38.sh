#!/usr/bin/env bash
# ============================================================================
# build_exclude_regions_hg38.sh
#
# Genera exclude_regions_hg38.bed combinando tres fuentes:
#   1. ENCODE Blacklist v2 (artefactos sistemáticos)
#   2. UCSC Gap Track (centrómeros, telómeros, heterocromatina)
#   3. Regiones de LD extendido conocidas (HLA, inversiones, pericentroméricas)
#
# Uso:
#   micromamba activate nf
#   bash tools/build_exclude_regions_hg38.sh [OUTDIR]
#
#   OUTDIR  directorio destino (default: ruta del proyecto en /scratch)
#
# Requiere: wget, gunzip, awk, sort, bedtools (env nf de micromamba)
#
# Referencia:
#   - ENCODE Blacklist: Amemiya et al. 2019, Sci Rep 9:9354
#   - UCSC gap track: hgdownload.cse.ucsc.edu/goldenpath/hg38/database/
#   - Regiones LD extendido: Price et al. 2008; Anderson et al. 2010
# ============================================================================
set -euo pipefail

# ── Parámetros ──────────────────────────────────────────────────────────────
DEFAULT_OUTDIR="/scratch/trusted/genbr/genbr_bioinfo/projects/DNABR_QC/reference/exclude_regions_hg38"
OUTDIR="${1:-${DEFAULT_OUTDIR}}"
OUTFILE="${OUTDIR}/exclude_regions_hg38.bed"

echo "==> Directorio destino: ${OUTDIR}"
echo "==> Archivo de salida:  ${OUTFILE}"

# Verificar bedtools
command -v bedtools >/dev/null 2>&1 || { echo "ERROR: bedtools no encontrado. Activa el env: micromamba activate nf"; exit 1; }
echo "==> bedtools $(bedtools --version 2>&1 | head -1)"

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

# ── Backup del archivo anterior (si existe) ─────────────────────────────────
if [ -f "${OUTFILE}" ]; then
    cp "${OUTFILE}" "${OUTFILE}.bak"
    echo "==> Backup guardado: ${OUTFILE}.bak"
fi

# ── 1. ENCODE Blacklist v2 ──────────────────────────────────────────────────
echo "[1/6] Descargando ENCODE Blacklist v2 ..."
wget -qO- https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz | \
    gunzip > _blacklist.bed
echo "      $(wc -l < _blacklist.bed) regiones"

# ── 2. UCSC Gap Track ───────────────────────────────────────────────────────
echo "[2/6] Descargando UCSC Gap Track (centrómeros, telómeros, heterocromatina) ..."
wget -qO- http://hgdownload.cse.ucsc.edu/goldenpath/hg38/database/gap.txt.gz | \
    gunzip | \
    awk 'BEGIN{OFS="\t"} {print $2,$3,$4}' > _gaps.bed
echo "      $(wc -l < _gaps.bed) regiones"

# ── 3. Regiones de LD extendido conocidas (todos los cromosomas) ────────────
echo "[3/6] Creando regiones de LD extendido conocidas ..."
cat > _ld_extended.bed << 'EOF'
chr2	86088342	101041482
chr6	25726063	33400644
chr8	7962590	11963571
chr11	46043424	57243424
chr17	40900000	44900000
chr20	25600000	33400000
EOF
echo "      $(wc -l < _ld_extended.bed) regiones"

# ── 4. Combinar, ordenar y mergear ──────────────────────────────────────────
echo "[4/6] Combinando y mergeando con bedtools ..."
cat _blacklist.bed _gaps.bed _ld_extended.bed | \
    cut -f1-3 | \
    sort -k1,1 -k2,2n | \
    bedtools merge -i - > "${OUTFILE}"

# ── 5. Estadísticas del resultado ───────────────────────────────────────────
echo "[5/6] Verificando resultado ..."
echo ""
echo "--- Regiones por cromosoma ---"
cut -f1 "${OUTFILE}" | sort -V | uniq -c
echo ""
TOTAL_BP=$(awk '{sum += $3-$2} END {print sum}' "${OUTFILE}")
TOTAL_MB=$(awk '{sum += $3-$2} END {printf "%.1f", sum/1e6}' "${OUTFILE}")
TOTAL_LINES=$(wc -l < "${OUTFILE}")
echo "Total regiones: ${TOTAL_LINES}"
echo "Total bases excluidas: ${TOTAL_BP} bp (${TOTAL_MB} Mb)"
echo ""

# ── 6. Limpiar temporales ───────────────────────────────────────────────────
echo "[6/6] Limpiando archivos temporales ..."
rm -f _blacklist.bed _gaps.bed _ld_extended.bed

echo ""
echo "==> Listo: ${OUTFILE}"
echo "==> Para usar en el workflow, verifica que nextflow.config apunte a:"
echo "    exclude_regions_bed = \"${OUTFILE}\""
