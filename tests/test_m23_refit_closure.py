#!/usr/bin/env python3
"""Prueba el control acotado de convergencia de M23 (`refit` y `refit_aggregate`).

El control reajusta el modelo final de cada combinación de set y fold con los hiperparámetros elegidos
en la corrida base y un techo de iteraciones más alto. No repite la grilla ni selecciona parámetros
nuevos. Esta prueba cubre el ensamblado y el criterio de cierre, no el ajuste del modelo. Los JSON de
entrada se crean a partir de una corrida base y usan números sintéticos para recorrer cada caso.

Se prueba en dos bloques:

  Veredictos
    1. Converge y el cambio es menor al umbral material: CIERRE_CON_CAVEAT.
    2. Sigue alcanzando el nuevo techo: SIGUE_SIN_CONVERGER.
    3. El cambio supera el umbral material: DETENERSE_Y_REVISAR.
    4. El solver no expone n_iter_: INDETERMINADO_SIN_n_iter.
    La convergencia se revisa antes del cierre. Si el modelo alcanza max_iter, el caso no se cierra
    aunque el delta continúe negativo y estable.

  Validaciones que deben rechazar la entrada
    5. abc pertenece a otra corrida base.
    6. Falta uno de los folds requeridos.
    7. max_iter no es mayor que el valor de la base.
    8. max_iter no coincide con config.max_iter.
    9. input_sha256 no coincide con el de la corrida base.
   10. input_sha256 está ausente.

También comprueba que, si las métricas no cambian, el delta frente a C reconstruido por el agregador
coincida con el publicado por la corrida base.

Se ejecuta dentro del contenedor del pipeline y desde un nodo de cómputo:
  singularity exec -B "$HOME" -B /scratch/datalake <sif> \
    python3 tests/test_m23_refit_closure.py --baseline-dir <dir de la corrida base>
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CV_PY = ROOT / "bin" / "rare_bench_cv.py"
SET_E = "E_full_elasticnet"
FOLDS = (0, 1, 2, 4)

# Parámetros científicos del control. Solo max_iter cambia frente a la corrida base.
SCI = ("--sample-id-col sample_id --split-col split --label-col y --fold-col fold "
       "--group-col split_group_key --train-label TRAIN --test-fold 3 --min-mac-train 2 "
       "--min-alt-carriers-train 2 --min-variance-train 0.0 --c-grid 1e-3,1e-2,1e-1,1 "
       "--l1-ratios 0.1,0.5,0.9 --inner-splits 3 --inner-scorer balanced_accuracy "
       "--decision-threshold 0.5 --class-weight balanced --svd-components 50 --tol 1e-4 --seed 42")


def build_refits(base_dir, outdir, scenario, max_iter=2000):
    """Crea cuatro archivos <set>.fold<K>.refit.json a partir de la corrida base."""
    inputs = json.loads((base_dir / "preflight.json").read_text())["fingerprint"]["input_sha256"]
    for fold in FOLDS:
        base = json.loads((base_dir / f"{SET_E}.fold{fold}.json").read_text())
        entry = copy.deepcopy(base["per_fold_entry"])
        if scenario == "converge_igual":
            entry["n_iter"], entry["converged"] = 1400, True
            entry["balanced_accuracy"] = round(entry["balanced_accuracy"] + 0.002, 6)
        elif scenario == "no_converge":
            entry["n_iter"], entry["converged"] = max_iter, False
            entry["balanced_accuracy"] = round(entry["balanced_accuracy"] + 0.002, 6)
        elif scenario == "material":
            entry["n_iter"], entry["converged"] = 1500, True
            entry["balanced_accuracy"] = round(entry["balanced_accuracy"] + 0.09, 6)
        elif scenario == "sin_n_iter":
            entry["n_iter"], entry["converged"] = None, None
        elif scenario == "identico":  # Las métricas no cambian y el delta frente a C debe coincidir.
            entry["n_iter"], entry["converged"] = 1400, True
        else:
            raise SystemExit(f"escenario desconocido: {scenario}")
        entry.update(fit_seconds=12345.6, peak_rss_self_gb=61.2, peak_rss_children_gb=0.0)
        cfg = dict(base["config"])
        cfg["max_iter"] = max_iter
        obj = {"stage": "RARE_BENCH_REFIT_FOLD", "set": base["set"], "held_out_fold": fold,
               "family": "elasticnet", "per_fold_entry": entry,
               "baseline": {"per_fold_entry": base["per_fold_entry"],
                            "max_iter": int(base["config"]["max_iter"]),
                            "fingerprint_id": base["fingerprint_id"],
                            "config": base["config"], "input_sha256": inputs},
               "input_sha256": inputs, "max_iter": max_iter, "meta": base["meta"],
               "config": cfg, "fingerprint_id": "fp_refit_sintetico"}
        (outdir / f"{SET_E}.fold{fold}.refit.json").write_text(json.dumps(obj, indent=2))


def aggregate(outdir, abc_json, refits, max_iter=2000):
    """Corre --mode refit_aggregate. Devuelve (returncode, stdout+stderr)."""
    cmd = [sys.executable, str(CV_PY), "--mode", "refit_aggregate", *SCI.split(),
           "--max-iter", str(max_iter), "--abc-json", str(abc_json),
           "--refit-material-delta", "0.01", "--outdir", str(outdir)]
    for r in refits:
        cmd += ["--refit-json", str(r)]
    p = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def edit_refits(outdir, fn):
    """Aplica una modificación controlada a los JSON sintéticos de reajuste."""
    for f in sorted(outdir.glob("*.refit.json")):
        obj = json.loads(f.read_text())
        fn(obj)
        f.write_text(json.dumps(obj, indent=2))


def main():
    """Ejecuta los casos de cierre, ensamblado y validación estricta."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True, type=Path,
                    help="directorio publicado de la corrida base (abc_results.json, preflight.json, "
                         "<set>.fold<K>.json)")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="directorio de trabajo; por defecto uno temporal")
    args = ap.parse_args()

    base_dir = args.baseline_dir
    abc = base_dir / "abc_results.json"
    for f in [abc, base_dir / "preflight.json"] + [base_dir / f"{SET_E}.fold{k}.json" for k in FOLDS]:
        if not f.exists():
            raise SystemExit(f"falta el artefacto de la corrida base: {f}")

    import tempfile
    tmp = args.workdir or Path(tempfile.mkdtemp(prefix="m23_refit_test_"))
    tmp.mkdir(parents=True, exist_ok=True)
    refits = [tmp / f"{SET_E}.fold{k}.refit.json" for k in FOLDS]
    failures = []

    def check(name, ok, detail=""):
        """Registra una comprobación y acumula los casos fallidos."""
        print(f"  {'ok   ' if ok else 'falla'} {name}{('  -> ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print("Veredictos")
    for scenario, expected in [("converge_igual", "CIERRE_CON_CAVEAT"),
                               ("no_converge", "SIGUE_SIN_CONVERGER"),
                               ("material", "DETENERSE_Y_REVISAR"),
                               ("sin_n_iter", "INDETERMINADO_SIN_n_iter")]:
        build_refits(base_dir, tmp, scenario)
        rc, out = aggregate(tmp, abc, refits)
        got = None
        if rc == 0:
            got = json.loads((tmp / "rare_bench_refit_results.json").read_text())["closure"]["verdict"]
        check(f"{scenario} -> {expected}", got == expected, f"rc={rc} veredicto={got}")

    print("Invariante de ensamblado")
    build_refits(base_dir, tmp, "identico")
    rc, out = aggregate(tmp, abc, refits)
    ok = rc == 0
    if ok:
        rep = json.loads((tmp / "rare_bench_refit_results.json").read_text())
        got = rep["per_set"][SET_E]["delta_balacc_vs_C"]["per_fold"]
        # Referencia: E - C por fold, calculado de los artefactos publicados de la base.
        c_by = {int(e["held_out_fold"]): e["balanced_accuracy"]
                for e in json.loads(abc.read_text())["sets"]["C_Q_sex_burden"]["per_fold"]}
        want = [round(json.loads((base_dir / f"{SET_E}.fold{k}.json").read_text())
                      ["per_fold_entry"]["balanced_accuracy"] - c_by[k], 6) for k in FOLDS]
        ok = got == want
        check("delta vs C reproduce el publicado", ok, f"{got} vs {want}")
    else:
        check("delta vs C reproduce el publicado", False, f"rc={rc}")

    print("Comprobaciones que deben abortar")
    build_refits(base_dir, tmp, "converge_igual")
    abc_otro = tmp / "abc_otro.json"
    d = json.loads(abc.read_text())
    d["fingerprint_id"] = "fp_de_otra_corrida"
    abc_otro.write_text(json.dumps(d))
    rc, out = aggregate(tmp, abc_otro, refits)
    check("abc de otra corrida base", rc != 0, out.strip().splitlines()[-1] if out.strip() else "")

    rc, out = aggregate(tmp, abc, refits[:1])
    check("falta un fold", rc != 0, out.strip().splitlines()[-1] if out.strip() else "")

    build_refits(base_dir, tmp, "converge_igual", max_iter=1000)  # == max_iter de la base
    rc, out = aggregate(tmp, abc, refits, max_iter=1000)
    check("max_iter no subido", rc != 0, out.strip().splitlines()[-1] if out.strip() else "")

    build_refits(base_dir, tmp, "converge_igual")
    edit_refits(tmp, lambda o: o.update(max_iter=5000))  # config.max_iter sigue en 2000
    rc, out = aggregate(tmp, abc, refits)
    check("max_iter declarado != config", rc != 0, out.strip().splitlines()[-1] if out.strip() else "")

    build_refits(base_dir, tmp, "converge_igual")
    def _tamper(o):
        o["input_sha256"] = dict(o["input_sha256"])
        o["input_sha256"]["split_manifest"] = "0" * 64
    edit_refits(tmp, _tamper)
    rc, out = aggregate(tmp, abc, refits)
    check("input_sha256 distinto del baseline", rc != 0, out.strip().splitlines()[-1] if out.strip() else "")

    build_refits(base_dir, tmp, "converge_igual")
    edit_refits(tmp, lambda o: o.pop("input_sha256", None))
    rc, out = aggregate(tmp, abc, refits)
    check("input_sha256 ausente", rc != 0, out.strip().splitlines()[-1] if out.strip() else "")

    print()
    print(f"directorio de trabajo: {tmp}")
    if failures:
        print(f"Fallaron {len(failures)} pruebas: {failures}")
        return 1
    print("Todas las pruebas pasaron")
    return 0


if __name__ == "__main__":
    sys.exit(main())
