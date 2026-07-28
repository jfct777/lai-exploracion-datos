#!/bin/bash
#SBATCH --job-name=m23_refit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=120:00:00
#SBATCH --requeue
#SBATCH --nodelist=c002
#SBATCH --output=m23_refit_%j.log

# Control acotado de convergencia para los modelos finales de E_full_elasticnet.
# Reajusta los cuatro modelos finales con los hiperparámetros elegidos en la corrida base. Estos
# valores se leen de los JSON publicados y no se transcriben a mano. Se usa max_iter=2000 sin repetir
# la grilla, seleccionar nuevos parámetros ni recalcular A/B/C. Las métricas de C se leen de
# abc_results.json.
#
# En la corrida base, 28 de los 36 ajustes internos de cada fold de E llegaron a max_iter=1000 sin
# converger. Como el JSON no guardaba n_iter_, no era posible comprobar la convergencia del modelo
# final.
#
# El trabajo se fija en c002 porque es el nodo que tiene montado el datalake. Al reutilizar el mismo
# workDir y usar -resume, la concatenación queda en caché y cada combinación de set y fold puede
# reanudarse de forma independiente.
# En el peor caso, los cuatro folds usan 48 horas en paralelo y el reintento tarda otras 60 horas.
set -euo pipefail

export PATH="$HOME/micromamba/envs/nf/bin:$PATH"
export JAVA_HOME="$HOME/micromamba/envs/nf"

# Bajo sbatch, $0 apunta a la copia del script guardada por Slurm y no al archivo original.
# SLURM_SUBMIT_DIR conserva el directorio desde el que se lanzó sbatch; M23_REPO permite indicar otra
# raíz de forma explícita. Si main.nf no está allí, el script termina antes de ejecutar Nextflow.
REPO="${M23_REPO:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}}"
REPO="$(readlink -f "$REPO")"
if [ ! -f "$REPO/main.nf" ]; then
  echo "No encuentro main.nf en $REPO. Lanza el sbatch desde la raiz del repo, o exporta M23_REPO=<raiz>." >&2
  exit 1
fi
cd "$REPO"
mkdir -p logs

BASE=/scratch/datalake/refined/genbr/genbr_bioinfo/projects/DNABR_QC
OUTDIR="$BASE/results_m23_extract_22chr"
WORKDIR=/scratch/datalake/transient/genbr/genbr_bioinfo/projects/DNABR_QC/tmp/nxf_work_m23_cv
SPLIT="$BASE/model_pipeline_canonical/22_model_pipeline/split/split_manifest.tsv"
MASTER="$BASE/model_pipeline_canonical/22_model_pipeline/modeling_master/modeling_master.tsv"
# De la corrida base se leen los hiperparámetros elegidos, las métricas de C y el input_sha256 que
# permite comprobar que se usaron las mismas entradas.
BASELINE="$OUTDIR/23_rare_matrix_benchmark/cv"
REPORTDIR="$REPO/logs/m23_refit_reports"
mkdir -p "$REPORTDIR"

nextflow run main.nf \
  -profile slurm_singularity \
  -c hpc.config \
  -c conf/auto_resources.config \
  -c scripts/nodelist_c002.config \
  -w "$WORKDIR" \
  -with-trace "$REPORTDIR/trace.txt" \
  -with-report "$REPORTDIR/report.html" \
  -with-timeline "$REPORTDIR/timeline.html" \
  --run_qc false --run_downstream false --run_model_pipeline false \
  --run_rare_bench true \
  --enable_rare_bench_extract false --enable_rare_bench_smoke false \
  --enable_rare_bench_concat true \
  --enable_rare_bench_cv false \
  --enable_rare_bench_refit true \
  --rare_bench_stage_subdir refit_check \
  --rare_bench_refit_baseline_dir "$BASELINE" \
  --rare_bench_refit_sets E_full_elasticnet \
  --rare_bench_refit_folds "0,1,2,4" \
  --rare_bench_refit_max_iter 2000 \
  --rare_bench_split_manifest "$SPLIT" \
  --rare_bench_modeling_master "$MASTER" \
  --rare_bench_chromosomes "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22" \
  --rare_bench_run_svd false \
  --outdir "$OUTDIR" \
  -resume

nextflow log -q 2>/dev/null | tail -n 1 | tee -a "$REPORTDIR/session_history.txt" || true
