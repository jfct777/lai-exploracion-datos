#!/bin/bash
#SBATCH --job-name=stage_panels
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=stage_panels_%j.log
set -euo pipefail

# ---------------------------------------------------------------------------
# Preparación de los paneles externos de referencia para el módulo 21.
# (canal de presencia externa). Baja del bucket cada panel × cromosoma y lo
# normaliza a la convención que consume el pipeline:
#     <PANEL_DIR>/<panel_id>/chr<C>.vcf.gz (+ .tbi)
# Esta preparación de datos de referencia se mantiene separada del cómputo de Nextflow.
# Por eso se mantiene en tools/ y se ejecuta fuera del pipeline, igual que install_ancestral.
# Idempotente: salta lo ya staged. Corre en Slurm (partición cpu), nunca en login.
#
# Los valores panel_id, drop_sample y pass_only de cada panel deben coincidir
# con params.presence_panels en nextflow.config (única fuente de verdad del consumo).
# Aquí solo vive la URI del bucket (provisioning), que el pipeline no necesita conocer.
# ---------------------------------------------------------------------------

GSUTIL=${GSUTIL:-"$HOME/micromamba/envs/gcloud/bin/gsutil"}
SIF=${SIF:-"$HOME/images/dnabr-qc-hpc.sif"}
PANEL_DIR=${PANEL_DIR:-/scratch/datalake/refined/genbr/genbr_bioinfo/projects/DNABR_QC/reference_external_panels}
CHROMS=${CHROMS:-$(seq 1 22)}

mkdir -p "$PANEL_DIR"

# Paneles: id | plantilla-de-URI (CHR = placeholder del número) | tbi-en-bucket(yes/no)
PANELS=(
  "NAMBR_128_hg38_vqsr|gs://projects-usp/nambr/chr/joint_germline_recalibrated.normalized.chrCHR.vcf.gz|no"
  "NAM_native_71_raw|gs://projects-usp/nam-diversity/nat_jointvcf_hg38/chr_hg38/natwgs.hg38.71.chrCHR.raw.vcf.gz|yes"
)

stage_one () {  # $1=panel_id  $2=src_uri  $3=tbi_in_bucket  $4=dest_vcf
  local panel_id="$1" src="$2" tbi_bucket="$3" dest="$4"
  if [ -s "$dest" ] && [ -s "${dest}.tbi" ]; then
    echo "[skip] $dest ya staged"; return 0
  fi
  echo "[stage] $panel_id <- $src"
  "$GSUTIL" cp "$src" "$dest"
  if [ "$tbi_bucket" = "yes" ]; then
    "$GSUTIL" cp "${src}.tbi" "${dest}.tbi"
  else
    singularity exec -B /scratch -B /home "$SIF" tabix -f -p vcf "$dest"
  fi
}

for entry in "${PANELS[@]}"; do
  IFS='|' read -r panel_id template tbi_bucket <<< "$entry"
  mkdir -p "$PANEL_DIR/$panel_id"
  for c in $CHROMS; do
    src="${template/CHR/$c}"
    dest="$PANEL_DIR/$panel_id/chr${c}.vcf.gz"
    stage_one "$panel_id" "$src" "$tbi_bucket" "$dest"
  done
done

echo "Listo: $PANEL_DIR"
