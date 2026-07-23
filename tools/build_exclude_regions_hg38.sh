#!/usr/bin/env bash
# ============================================================================
# build_exclude_regions_hg38.sh
# ----------------------------------------------------------------------------
# Genera un BED unificado de regiones a excluir para análisis de LD y QC
# genómico en GRCh38/hg38, combinando tres fuentes:
#
#   1. ENCODE Blacklist v2  (Amemiya HM, Kundaje A, Boyle AP. The ENCODE
#      Blacklist: Identification of Problematic Regions of the Genome.
#      Sci Rep. 2019;9(1):9354. doi:10.1038/s41598-019-45839-z)
#
#   2. UCSC Gap Track  (centrómeros, telómeros, short-arms, heterochromatina)
#      http://hgdownload.cse.ucsc.edu/goldenpath/hg38/database/gap.txt.gz
#
#   3. Regiones de LD extendido conocidas en GRCh38, basadas en:
#      - Price AL et al. Long-Range LD Can Confound Genome Scans in Admixed
#        Populations. Am J Hum Genet. 2008;83(1):132-135.
#      - Anderson CA et al. Data quality control in genetic case-control
#        association studies. Nat Protoc. 2010;5(9):1564-1573.
#      Coordenadas liftOver hg19→hg38 (fuentes: literature + UCSC liftOver).
#
# Entorno local (WSL / Docker):
#   - bedtools instalado en el env de conda "ega"
#   - Rutas bajo /mnt/d/ega/DNABR/meta
#
# Uso:
#   conda activate ega && bash tools/build_exclude_regions_hg38.sh
#   # o bien:
#   conda run --no-banner -n ega bash tools/build_exclude_regions_hg38.sh
# ============================================================================
set -euo pipefail

# ── Configuración ──────────────────────────────────────────────────────────
DEFAULT_OUTDIR="/mnt/d/ega/DNABR/meta"
OUTDIR="${1:-$DEFAULT_OUTDIR}"
OUTFILE="${OUTDIR}/exclude_regions_hg38.bed"
TMPDIR_BASE="${OUTDIR}/.tmp_exclude_build"

BLACKLIST_URL="https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz"
GAP_URL="http://hgdownload.cse.ucsc.edu/goldenpath/hg38/database/gap.txt.gz"

# ── Verificaciones ─────────────────────────────────────────────────────────
if ! command -v bedtools &>/dev/null; then
    echo "ERROR: bedtools no está disponible en el PATH."
    echo "       Activa el env conda 'ega' antes de ejecutar este script:"
    echo "         conda activate ega && bash $0"
    exit 1
fi
echo "✓ bedtools encontrado: $(command -v bedtools) ($(bedtools --version 2>&1 | head -1))"

if ! command -v wget &>/dev/null; then
    echo "ERROR: wget no está disponible. Instálalo o usa el env conda 'ega'."
    exit 1
fi

mkdir -p "${OUTDIR}"
mkdir -p "${TMPDIR_BASE}"

# ── Backup del BED anterior ───────────────────────────────────────────────
if [[ -f "${OUTFILE}" ]]; then
    BACKUP="${OUTFILE}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "→ Backup del BED anterior: ${BACKUP}"
    cp "${OUTFILE}" "${BACKUP}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " Construyendo ${OUTFILE}"
echo "═══════════════════════════════════════════════════════════════════"

# ── 1. ENCODE Blacklist v2 ─────────────────────────────────────────────────
echo ""
echo "── [1/4] Descargando ENCODE Blacklist v2 ──"
BL_GZ="${TMPDIR_BASE}/_blacklist.bed.gz"
BL_BED="${TMPDIR_BASE}/_blacklist.bed"
wget -q --show-progress -O "${BL_GZ}" "${BLACKLIST_URL}"
gunzip -f "${BL_GZ}"
# Mantener solo las 3 primeras columnas (chr, start, end)
awk 'BEGIN{OFS="\t"} {print $1,$2,$3}' "${BL_BED}" > "${TMPDIR_BASE}/_01_blacklist.bed"
echo "  → $(wc -l < "${TMPDIR_BASE}/_01_blacklist.bed") regiones blacklist"

# ── 2. UCSC Gap Track ──────────────────────────────────────────────────────
echo ""
echo "── [2/4] Descargando UCSC Gap Track (centrómeros, telómeros, etc.) ──"
GAP_GZ="${TMPDIR_BASE}/_gap.txt.gz"
GAP_BED="${TMPDIR_BASE}/_02_gaps.bed"
wget -q --show-progress -O "${GAP_GZ}" "${GAP_URL}"
# gap.txt: col2=chrom, col3=chromStart, col4=chromEnd, col8=type
# Filtrar solo cromosomas canónicos (chr1-22, chrX, chrY)
gunzip -c "${GAP_GZ}" | \
    awk 'BEGIN{OFS="\t"} $2 ~ /^chr([0-9]+|[XY])$/ {print $2,$3,$4}' | \
    sort -k1,1 -k2,2n > "${GAP_BED}"
echo "  → $(wc -l < "${GAP_BED}") regiones gap (centrómeros, telómeros, heterochromatina)"

# ── 3. Regiones de LD extendido (GRCh38) ──────────────────────────────────
echo ""
echo "── [3/4] Escribiendo regiones de LD extendido (GRCh38 / hg38) ──"
LD_BED="${TMPDIR_BASE}/_03_ld_extended.bed"

cat > "${LD_BED}" <<'EOF'
chr1	48000000	52000000
chr2	86000000	100500000
chr2	134500000	138000000
chr2	183000000	190000000
chr3	47500000	50000000
chr3	83500000	87000000
chr3	89000000	97500000
chr5	98000000	100500000
chr5	129000000	132000000
chr5	135500000	138500000
chr6	25500000	33500000
chr6	57000000	64000000
chr6	140000000	142500000
chr7	55000000	66000000
chr8	7000000	13000000
chr8	43000000	50000000
chr8	112000000	115000000
chr10	37000000	43000000
chr11	46000000	57000000
chr11	87500000	90500000
chr12	33000000	40000000
chr12	109500000	112000000
chr15	28000000	30500000
chr17	43500000	46000000
chr20	32000000	34500000
chr22	22500000	25500000
EOF

echo "  → $(wc -l < "${LD_BED}") regiones de LD extendido"
echo "    Cromosomas: $(awk '{print $1}' "${LD_BED}" | sort -u | tr '\n' ' ')"

# ── 4. Merge ──────────────────────────────────────────────────────────────
echo ""
echo "── [4/4] Combinando, ordenando y fusionando con bedtools merge ──"
COMBINED="${TMPDIR_BASE}/_combined_sorted.bed"

cat "${TMPDIR_BASE}/_01_blacklist.bed" \
    "${GAP_BED}" \
    "${LD_BED}" | \
    awk 'BEGIN{OFS="\t"} $1 ~ /^chr([0-9]+|[XY])$/ {print $1,$2,$3}' | \
    sort -k1,1V -k2,2n > "${COMBINED}"

bedtools merge -i "${COMBINED}" > "${OUTFILE}"

echo ""
echo "✓ BED final escrito en: ${OUTFILE}"
echo "  → $(wc -l < "${OUTFILE}") regiones fusionadas"

# ── 5. Estadísticas ───────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " Estadísticas"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Regiones por cromosoma:"
awk '{print $1}' "${OUTFILE}" | sort -V | uniq -c | awk '{printf "  %-8s %d\n", $2, $1}'

TOTAL_BP=$(awk 'BEGIN{s=0} {s += ($3-$2)} END{print s}' "${OUTFILE}")
TOTAL_MB=$(awk "BEGIN{printf \"%.2f\", ${TOTAL_BP}/1000000}")
TOTAL_REGIONS=$(wc -l < "${OUTFILE}")

echo ""
echo "  Total regiones:  ${TOTAL_REGIONS}"
echo "  Total bases:     ${TOTAL_BP} bp (${TOTAL_MB} Mb)"
echo ""

# ── 6. Limpieza ───────────────────────────────────────────────────────────
echo "── Limpiando archivos temporales ──"
rm -rf "${TMPDIR_BASE}"
echo "✓ Listo."
