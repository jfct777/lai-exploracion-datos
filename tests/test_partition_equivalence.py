#!/usr/bin/env python3
"""Equivalencia entre los modos monolítico y particionado del benchmark M23 con datos sintéticos.

Exige que el modo particionado (abc + 8 folds por set y fold + aggregate) produzca los mismos
campos científicos que el monolítico de referencia. Ignora `operational` (tiempos/modo/traza).
Valida únicamente la equivalencia lógica del mecanismo de reanudación. Debe ejecutarse dentro del
contenedor del pipeline:

  srun -p cpu -c 4 --mem=8G -t 00:20:00 singularity exec -B "$HOME" <sif> \
    bash -c 'export TMPDIR=<dir-compartido>; python3 tests/test_partition_equivalence.py'

Las comprobaciones con datos reales y consumo de memoria corresponden a la ejecución de Nextflow.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BIN = Path(__file__).resolve().parents[1] / "bin" / "rare_bench_cv.py"
PY = sys.executable
WORK = Path(tempfile.mkdtemp(prefix="m23eq_"))
print(f"[test] bin={BIN}\n[test] workdir={WORK}")

rng = np.random.default_rng(0)
n_train, n_var = 160, 1000
folds_train = np.array([[0, 1, 2, 4][i % 4] for i in range(n_train)])
sample_ids = [f"S{i:04d}" for i in range(n_train)]

rows, cols, vals = [], [], []
for j in range(n_var):
    k = int(rng.choice([1, 2, 3, 5], p=[0.45, 0.30, 0.15, 0.10]))
    for w in rng.choice(n_train, size=k, replace=False):
        rows.append(int(w)); cols.append(j); vals.append(int(rng.choice([1, 2])))
X = sp.csr_matrix((vals, (rows, cols)), shape=(n_train, n_var), dtype=np.int16)
dens = np.asarray(X[:, :50].sum(axis=1)).ravel()
y = ((dens + rng.normal(0, 2, n_train)) > np.median(dens)).astype(int)

sp.save_npz(WORK / "matrix.npz", X.tocsc())
pd.DataFrame({"sample_id": sample_ids}).to_csv(WORK / "samples.tsv", sep="\t", index=False)

test_ids = [f"T{i:04d}" for i in range(20)]
pd.DataFrame({
    "sample_id": sample_ids + test_ids,
    "split": ["TRAIN"] * n_train + ["TEST"] * len(test_ids),
    "fold": list(folds_train) + [3] * len(test_ids),
    "y": list(y) + list((rng.random(len(test_ids)) < 0.3).astype(int)),
    "split_group_key": [f"g{i}" for i in range(n_train)] + [f"gt{i}" for i in range(len(test_ids))],
}).to_csv(WORK / "split.tsv", sep="\t", index=False)

pd.DataFrame({
    "sample_id": sample_ids,
    "Q_NAM": rng.random(n_train), "Q_EUR": rng.random(n_train),
    "Q_EAS": rng.random(n_train), "Q_AFR": rng.random(n_train),
    "sex": rng.integers(0, 2, n_train),
    "rare_density": rng.random(n_train),
    "rare_carrier_site_count": rng.integers(1, 100, n_train),
    "rare_gt_nonmissing_sites": rng.integers(120, 200, n_train),
}).to_csv(WORK / "mm.tsv", sep="\t", index=False)

sci = ["--c-grid", "0.01,1", "--l1-ratios", "0.5", "--inner-splits", "2",
       "--n-jobs", "1", "--seed", "42", "--container-sha256", "deadbeefcafe"]
data = ["--matrix-npz", str(WORK / "matrix.npz"), "--samples-tsv", str(WORK / "samples.tsv"),
        "--split-manifest", str(WORK / "split.tsv"), "--modeling-master", str(WORK / "mm.tsv")]


def run(tag, extra):
    """Ejecuta un modo del benchmark y termina si el proceso falla."""
    r = subprocess.run([PY, str(BIN)] + data + sci + extra, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[test] falló el modo {tag}:\n{r.stdout}\n{r.stderr}")
        sys.exit(1)
    print(f"[test] ok {tag}")


run("monolithic", ["--mode", "monolithic", "--outdir", str(WORK / "mono")])
run("preflight", ["--mode", "preflight", "--outdir", str(WORK / "pf")])
pf = str(WORK / "pf" / "preflight.json")
run("abc", ["--mode", "abc", "--outdir", str(WORK / "part"), "--preflight-json", pf])
fold_jsons = []
for s in ["D_matrix_elasticnet", "E_full_elasticnet"]:
    for k in [0, 1, 2, 4]:
        run(f"fold {s} {k}", ["--mode", "fold", "--set", s, "--fold", str(k),
                              "--outdir", str(WORK / "part"), "--preflight-json", pf])
        fold_jsons += ["--fold-json", str(WORK / "part" / f"{s}.fold{k}.json")]
run("aggregate", ["--mode", "aggregate", "--abc-json", str(WORK / "part" / "abc_results.json"),
                  "--outdir", str(WORK / "part_out")] + fold_jsons)

mono = json.loads((WORK / "mono" / "rare_bench_cv_results.json").read_text())
part = json.loads((WORK / "part_out" / "rare_bench_cv_results.json").read_text())

SCIENTIFIC = ["per_set_summary", "contrasts", "anti_leakage_retained_cols_differ_across_folds",
              "retention_rule_fold_fitted", "cv_design", "compute_plan", "n_train",
              "n_variants_input", "outer_folds", "prevalence_train"]
mismatches = [k for k in SCIENTIFIC if mono.get(k) != part.get(k)]

print("\n[test] resultado")
print(f"[test] E-C monolitico:   {mono['contrasts']['PRIMARY_delta_balacc_E_minus_C']}")
print(f"[test] E-C particionado: {part['contrasts']['PRIMARY_delta_balacc_E_minus_C']}")
if mismatches:
    print(f"[test] hay diferencias en los campos científicos: {mismatches}")
    for k in mismatches:
        print(f"   {k}: mono={json.dumps(mono.get(k))[:300]} part={json.dumps(part.get(k))[:300]}")
    sys.exit(1)
print("[test] ok: los campos científicos del modo monolítico y el particionado son idénticos")
