#!/bin/bash
#SBATCH --job-name=m23_refit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --nodelist=c002
#SBATCH --output=/home/jose.tantalean/projects/lai-exploracion-datos/logs/m23_refit_%j.log

# Control ACOTADO de convergencia del solver sobre los modelos finales de E_full_elasticnet.
# Reajusta los 4 modelos finales con los hiperparametros que la corrida base YA eligio (se leen de sus
# JSON, no se transcriben) y max_iter=2000. Ni grilla, ni seleccion nueva, ni TEST, ni recomputo de
# A/B/C: las metricas de C se reutilizan del abc_results.json de la base.
#
# Motivo: en la corrida base 28/36 ajustes internos de cada fold de E tocaron max_iter=1000 sin
# converger y el JSON no guardaba n_iter_, asi que no se podia saber si el modelo ganador convergio.
#
# Fijado a c002 (unico nodo con el datalake). Mismo WORKDIR que la base + -resume: CONCAT queda
# cacheado (no depende de rare_bench_cv.py) y cada (set,fold) es una tarea independiente reanudable.
# Peor caso: 4 folds x 24h en paralelo + reintento de 48h = ~72h < 96h del padre.
set -euo pipefail

export PATH="$HOME/micromamba/envs/nf/bin:$PATH"
export JAVA_HOME="$HOME/micromamba/envs/nf"

cd /home/jose.tantalean/projects/lai-exploracion-datos
mkdir -p logs

BASE=/scratch/datalake/refined/genbr/genbr_bioinfo/projects/DNABR_QC
OUTDIR="$BASE/results_m23_extract_22chr"
WORKDIR=/scratch/datalake/transient/genbr/genbr_bioinfo/projects/DNABR_QC/tmp/nxf_work_m23_cv
SPLIT="$BASE/model_pipeline_canonical/22_model_pipeline/split/split_manifest.tsv"
MASTER="$BASE/model_pipeline_canonical/22_model_pipeline/modeling_master/modeling_master.tsv"
# Corrida BASE: de aqui salen los hiperparametros ganadores, las metricas de C y el input_sha256 que
# prueba que se usaron EXACTAMENTE estas mismas entradas.
BASELINE="$OUTDIR/23_rare_matrix_benchmark/cv"
REPORTDIR=/home/jose.tantalean/projects/lai-exploracion-datos/logs/m23_refit_reports
mkdir -p "$REPORTDIR"

nextflow run main.nf \
  -profile slurm_singularity \
  -c hpc.config \
  -c conf/auto_resources.config \
  -c /home/jose.tantalean/m23_launch/nodelist_c002.config \
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
