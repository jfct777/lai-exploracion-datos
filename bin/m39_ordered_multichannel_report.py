#!/usr/bin/env python3
"""Read audited multichannel primaries and write a bounded Spanish SELECT report.

No fitting, genotype/store reads, new truth extraction or cloud operations occur.
Calibration predictions are joined by exact sample/locus keys and rescored on
the neural axis. Final reporting requires identical calibration-selection and
neural anchor universes; a larger-anchor fitted calibrator is not substituted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_exclusive_json
from m39_anchor_screen import metrics, sha256
import m39_calibration as calibration_transform
from m39_ordered_models import STATE_NAMES
from m39_ordered_multichannel_training import ARMS, SOURCE_FILES as TRAINING_SOURCE_FILES, load_config
from m39_ordered_multichannel_calibration import SOURCE_FILES as CALIBRATION_SOURCE_FILES
import m39_ordered_multichannel_sweep as sweep
from m39_ordered_training_data import require
from m39_ordered_training_sweep import verify_recomputed_metrics


SCHEMA = "m39-ordered-multichannel-report-v1"
COMPARATORS = ("FMINUS", "FFULL", "CALIBRATION_ONLY")
ANCESTRIES = ("AFR", "EUR", "NAM")
GEOMETRY = ("train_manifest_sha256", "select_manifest_sha256", "development_sha256",
            "anchor_count", "anchor_seed", "batch_size", "multichannel")
SOURCE_FILES = tuple(sorted(set(TRAINING_SOURCE_FILES) | set(CALIBRATION_SOURCE_FILES) | {
    "m39_ordered_multichannel_report.py", "m39_ordered_multichannel_sweep.py",
    "m39_ordered_training_sweep.py", "m39_ordered_gpu_manifest.py"}))


def axis_hash(sample_keys: np.ndarray, locus_ids: np.ndarray) -> str:
    """Hash the typed ordered axes, not just their unordered set or row count."""
    digest = hashlib.sha256()
    for name, values in (("sample_key_sha256", sample_keys), ("locus_id", locus_ids)):
        values = np.ascontiguousarray(values)
        require(values.ndim == 1 and values.size > 0 and not values.dtype.hasobject
                and len(np.unique(values)) == len(values), f"invalid or duplicate {name}")
        digest.update(json.dumps([name, values.dtype.str, values.shape]).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _join(source, target, name):
    require(source.dtype == target.dtype and source.ndim == target.ndim == 1
            and len(np.unique(source)) == len(source)
            and len(np.unique(target)) == len(target), f"invalid {name} join axis")
    lookup = {value: index for index, value in enumerate(source.tolist())}
    require(all(value in lookup for value in target.tolist()), f"missing calibration {name}")
    return np.asarray([lookup[value] for value in target.tolist()], dtype=np.int64)


def align_calibration(arrays: dict, neural: dict) -> tuple[dict, dict]:
    rows = _join(arrays["select_sample_key_sha256"], neural["sample_key_sha256"], "sample keys")
    columns = _join(arrays["locus_id"], neural["locus_id"], "locus IDs")
    # SELECT is a fixed role, not another tunable subset of persons.
    require(len(rows) == len(arrays["select_sample_key_sha256"]), "SELECT person universe differs")
    truth = arrays["select_truth_state"][rows][:, columns]
    require(truth.dtype == neural["truth_state"].dtype and np.array_equal(truth, neural["truth_state"]),
            "calibration/neural truth differs after exact join")
    aligned = {name: metrics(arrays["select_" + name][rows][:, columns], truth) for name in COMPARATORS}
    return aligned, {"source_axis_sha256": axis_hash(arrays["select_sample_key_sha256"], arrays["locus_id"]),
                     "evaluated_axis_sha256": axis_hash(neural["sample_key_sha256"], neural["locus_id"]),
                     "source_anchors": len(arrays["locus_id"]), "evaluated_anchors": len(columns),
                     "source_people": len(rows), "exact_locus_and_sample_join": True,
                     "subset_rescored": len(columns) != len(arrays["locus_id"]),
                     "calibration_refitted_or_reselected": False,
                     "selection_used_larger_anchor_universe": len(columns) < len(arrays["locus_id"])}


def _flatten(metric):
    return {"brier": metric["brier"], "log_loss": metric["log_loss"],
            **{f"dosage_mae_{a}": metric["dosage_mae"][i] for i, a in enumerate(ANCESTRIES)}}


def _write_csv(path, rows):
    require(bool(rows), "empty report table")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(screen_plan: Path, screen_outputs: Path, screen_comparison: Path,
                 followup_plan: Path, followup_outputs: Path, followup_comparison: Path,
                 calibration_dir: Path, calibration_report_sha256: str, outdir: Path) -> dict:
    require(not outdir.exists() and not outdir.is_symlink(), "report output already exists")
    sources = {}

    def register(path):
        require(path.is_file() and not path.is_symlink(), "report source must be a regular file")
        digest = sha256(path)
        sources[str(path.resolve())] = digest
        return digest

    def read_json(path):
        register(path)
        return json.loads(path.read_text())

    for name in SOURCE_FILES:
        register(Path(__file__).with_name(name))
    calibration_path = calibration_dir / "report.json"
    require(register(calibration_path) == calibration_report_sha256, "calibration report hash differs")
    cal = read_json(calibration_path)
    require(cal["schema_version"] == "m39-multichannel-calibration-v1"
            and cal["status"] == "EXPLORATORY_CALIBRATION_ONLY_NOT_VALIDATION"
            and cal["score_read"] is False and cal["source_test_or_valid_opened"] is False
            and cal["genotypes_used_as_predictors"] is False and cal["reopened_predictions_exact"] is True,
            "calibration scope or validation differs")
    require(set(cal["outputs_sha256"]) == {"calibration.json", "candidates.csv", "predictions.npz"},
            "calibration artifact inventory differs")
    for name, digest in cal["outputs_sha256"].items():
        require(register(calibration_dir / name) == digest, "calibration artifact hash differs")
    require(read_json(calibration_dir / "calibration.json") == cal["calibration"],
            "calibration parameters differ from report")
    with np.load(calibration_dir / "predictions.npz", allow_pickle=False) as saved:
        require(np.array_equal(saved["state_names"], np.asarray(STATE_NAMES, dtype="S2")),
                "calibration six-state order differs")
        names = ["locus_id", "anchor_indices", "select_sample_key_sha256", "select_truth_state",
                 *["select_" + name for name in COMPARATORS]]
        calibration_arrays = {name: saved[name].copy() for name in names}
    for name in COMPARATORS:
        verify_recomputed_metrics(metrics(calibration_arrays["select_" + name],
                                  calibration_arrays["select_truth_state"]), cal["role_metrics"]["SELECT"][name])
    parameter = cal["calibration"]
    transformed = calibration_transform.transform(calibration_arrays["select_FMINUS"], parameter["family"],
        parameter["temperature"], parameter["alpha"], np.asarray(parameter["prior"]))
    require(np.array_equal(transformed, calibration_arrays["select_CALIBRATION_ONLY"]),
            "calibrated predictions do not reproduce frozen transform")
    plans, comparisons, cases, curves = {}, {}, [], []
    reference_geometry = reference_axis = reference_truth = None
    axis_bindings, controls = {}, {}
    for label, plan_path, outputs, comparison_path, stage in (
        ("A", screen_plan, screen_outputs, screen_comparison, "multichannel_screen"),
        ("B", followup_plan, followup_outputs, followup_comparison, "multichannel_followup")):
        plan = sweep.load_plan(plan_path)
        require(plan["stage"] == stage, "report requires scientific A and B, never a technical profile")
        plans[label] = plan
        register(plan_path)
        comparison = read_json(comparison_path)
        reopened = sweep.audit_results(plan_path, outputs)
        # Cross-runtime roundoff diagnostics may differ; all scientific values,
        # inventories, axes, selection and hashes remain exactly the audited ones.
        def without_roundoff_diagnostics(value):
            return {**value, "metric_recomputation": {k: v for k, v in value["metric_recomputation"].items()
                                                     if k != "nonzero_differences"}}
        require(without_roundoff_diagnostics(comparison) == without_roundoff_diagnostics(reopened),
                "verified comparison does not match reopened primaries")
        comparisons[label] = comparison
        for group in plan["groups"]:
            directory = outputs / ("training-" + group["id"])
            register(directory / "group.completion.json")
            for spec in group["configs"]:
                config_path = plan_path.parent / spec["file"]
                register(config_path)
                cfg = load_config(config_path)
                geometry = {key: cfg[key] for key in GEOMETRY}
                geometry["model_without_family"] = {k: v for k, v in cfg["model"].items() if k != "family"}
                require(reference_geometry is None or reference_geometry == geometry,
                        "A/B input, geometry or capacity differs")
                reference_geometry = geometry
                require(cfg["development_sha256"] == cal["input_sha256"]["development"]
                        and cfg["train_manifest_sha256"] == cal["input_sha256"]["train_manifest"]
                        and cfg["select_manifest_sha256"] == cal["input_sha256"]["select_manifest"],
                        "neural/calibration input binding differs")
                case_dir = directory / cfg["arm"]
                receipt = read_json(case_dir / "training.receipt.json")
                prediction_path = case_dir / "select.predictions.npz"
                register(prediction_path)
                register(case_dir / "checkpoint.pt")
                with np.load(prediction_path, allow_pickle=False) as saved:
                    neural = {name: saved[name].copy() for name in
                              ("sample_key_sha256", "locus_id", "truth_state", "anchor_indices")}
                current_axis = axis_hash(neural["sample_key_sha256"], neural["locus_id"])
                require(reference_axis is None or reference_axis == current_axis,
                        "A/B or group prediction axes differ")
                require(reference_truth is None or np.array_equal(reference_truth, neural["truth_state"]),
                        "A/B or group truth differs")
                reference_axis, reference_truth = current_axis, neural["truth_state"]
                aligned, binding = align_calibration(calibration_arrays, neural)
                require(not binding["selection_used_larger_anchor_universe"],
                        "calibration selection anchor universe differs; refit on fixed neural anchors")
                joined = _join(calibration_arrays["locus_id"], neural["locus_id"], "locus IDs")
                require(np.array_equal(calibration_arrays["anchor_indices"][joined], neural["anchor_indices"])
                        and parameter["anchor_indices"] == receipt["anchor_indices"],
                        "calibration/neural fixed anchor indices differ")
                for name, field in (("FMINUS", "Fminus_SELECT"), ("FFULL", "Ffull_SELECT")):
                    verify_recomputed_metrics(aligned[name], receipt[field])
                axis_bindings[label + "/" + group["id"]] = binding
                controls[label + "/" + group["id"]] = aligned
                row = {"stage": label, "group": group["id"], "family": cfg["model"]["family"],
                       "seed": cfg["seed"], "arm": cfg["arm"].upper(), "learning_rate": cfg["learning_rate"],
                       **{key: cfg[key] for key in ("steps", "evaluate_every_steps", "max_runtime_seconds",
                                                    "max_input_bytes", "max_device_bytes", "max_rss_bytes")},
                       **_flatten(receipt["selected_SELECT_metrics"]), "selected_step": receipt["selected_step"],
                       "complete_passes": receipt["TRAIN_exposure"]["complete_passes_over_declared_pairs"],
                       "selected_at_budget_end": receipt["budget_diagnostics"]["best_checkpoint_at_budget_end"],
                       "people": len(neural["sample_key_sha256"]), "anchors": len(neural["locus_id"])}
                cases.append(row)
                for point in receipt["curve"]:
                    register(case_dir / f"curve-step-{point['step']:07d}.json")
                    curves.append({"stage": label, "group": group["id"], "family": row["family"],
                        "seed": cfg["seed"], "arm": row["arm"], "step": point["step"],
                        "checkpoint_eligible": point["checkpoint_eligible"],
                        "selected": point["step"] == receipt["selected_step"], **_flatten(point["SELECT"]),
                        "train_online_cross_entropy": point["train_cross_entropy_window"],
                        "train_probe_brier": point["TRAIN_probe"]["brier"] if point["TRAIN_probe"] else None,
                        "gradient_norm": point["gradient_norm_last_step"],
                        "train_observations": point["train_observations_cumulative"],
                        "complete_passes": point["TRAIN_exposure"]["complete_passes_over_declared_pairs"],
                        "training_seconds": point["training_seconds_cumulative"],
                        "evaluation_seconds": point["evaluation_seconds_cumulative"]})
    chosen = sweep.selected_learning_rates(comparisons["A"])
    seeds = {family: {row["seed"] for row in cases if row["stage"] == "B" and row["family"] == family}
             for family in ("cnn", "attention")}
    require(seeds["cnn"] == seeds["attention"] and len(seeds["cnn"]) >= 2,
            "followup must contain both families and every declared optimization seed")
    require(not seeds["cnn"] & {row["seed"] for row in cases if row["stage"] == "A"},
            "followup reused screening optimization seeds")
    contrasts, tables = [], []
    seen = set()
    for group in plans["B"]["groups"]:
        group_cases = [row for row in cases if row["stage"] == "B" and row["group"] == group["id"]]
        require([row["arm"] for row in group_cases] == [arm.upper() for arm in ARMS], "incomplete followup arms")
        template = group_cases[0]
        identity = (template["family"], template["seed"])
        require(identity not in seen, "duplicate followup family/seed group")
        seen.add(identity)
        require(template["learning_rate"] == chosen[template["family"]]["learning_rate"],
                "followup learning rate differs from joint NONE/BOTH selection")
        baselines = [{**template, "arm": name, **_flatten(metric), "learning_rate": None,
                      "selected_step": None, "complete_passes": None, "selected_at_budget_end": None}
                     for name, metric in controls["B/" + group["id"]].items()]
        table = group_cases + baselines
        tables.extend(table)
        focal = next(row for row in group_cases if row["arm"] == "BOTH")
        for control in table:
            if control["arm"] != "BOTH":
                contrasts.append({"group": group["id"], "family": template["family"], "seed": template["seed"],
                    "focal": "BOTH", "comparator": control["arm"],
                    **{f"delta_{name}": focal[name] - control[name] for name in _flatten({
                        "brier": 0, "log_loss": 0, "dosage_mae": [0, 0, 0]})}})
    require(len(seen) == 2 * len(seeds["cnn"]), "followup family/seed inventory differs")
    report = {"schema_version": SCHEMA, "status": "AUDITED_EXPLORATORY_SELECT_ONLY",
        "counts": {"screen_groups": len(plans["A"]["groups"]), "screen_fits": sum(r["stage"] == "A" for r in cases),
                   "followup_groups": len(seen), "followup_fits": len(seen) * len(ARMS),
                   "SELECT_people": len(reference_truth), "anchors": reference_truth.shape[1]},
        "selected_learning_rates": chosen, "same_A_B_geometry": True, "geometry": reference_geometry,
        "prediction_axis_sha256": reference_axis, "calibration_alignment": axis_bindings,
        "calibration": parameter, "calibration_primary_metric": "brier", "all_neural_cases": cases,
        "followup_table": tables, "BOTH_minus_comparator": contrasts, "learning_curves": curves,
        "units": {"brier": "sum_squared_error_six_states_dimensionless", "log_loss": "natural_log_nats",
                  "dosage_mae": "ancestral_copies_0_to_2", "weighting": "equal_people_equal_selected_anchors",
                  "delta": "BOTH_minus_comparator_negative_is_lower_error", "log_loss_floor": 1e-12},
        "limits": {"SELECT_reused_exploratory": True, "optimization_seeds_are_biological_replicates": False,
                   "initial_checkpoint_selectable": False, "budget_end_does_not_establish_convergence": True,
                   "dense_LAI_or_border_F1_evaluated": False, "SCORE_VALID_TEST_opened": False,
                   "NAM_population_generalization_established": False, "significance_threshold_applied": False,
                   "population_uncertainty_estimated": False, "SHAM_exact_exchangeability_established": False},
        "source_sha256": sources, "generator_sha256": sha256(Path(__file__))}
    require(all(sha256(Path(path)) == digest for path, digest in sources.items()), "report input changed during reading")
    outdir.mkdir(parents=True, mode=0o700, exist_ok=False)
    write_exclusive_json(outdir / "report.json", report)
    _write_csv(outdir / "cases.csv", cases)
    _write_csv(outdir / "followup-table.csv", tables)
    _write_csv(outdir / "contrasts.csv", contrasts)
    _write_csv(outdir / "learning-curves.csv", curves)
    with (outdir / "report.md").open("x") as stream:
        stream.write(render_markdown(report))
    return report


def render_markdown(report: dict) -> str:
    count = report["counts"]
    lines = ["# Comparación multicanal: SELECT exploratorio", "",
        f"Se verificaron {count['screen_fits']} ajustes de pantalla y {count['followup_fits']} ajustes posteriores "
        f"({count['followup_groups']} grupos completos), sobre {count['SELECT_people']} personas simuladas "
        f"y {count['anchors']} anclas por persona. Se muestran todos los brazos y semillas, sin seleccionar una semilla favorable.", "",
        "Brier: error cuadrático sumado sobre seis estados; log-loss: logaritmo natural (nats, piso 10⁻¹²). "
        "MAE ancestral: copias 0–2, no proporciones 0–1. Ponderación igual por ancla dentro de persona y después por persona. "
        "En todas las diferencias, Δ = BOTH − comparador: negativo indica menor error, no significación estadística.", "",
        "## Selección y exposición", "",
        "La tasa se eligió por media equiponderada del Brier NONE/BOTH, después log-loss y menor tasa. "
        "Los checkpoints se eligieron por Brier, log-loss y paso más temprano al completar pasadas; "
        "la evaluación inicial sólo diagnostica. A y B conservan entradas, anclas, radio y capacidad.", ""]
    for family, choice in report["selected_learning_rates"].items():
        lines.append(f"- {family}: tasa {choice['learning_rate']:g}; Brier medio NONE/BOTH {choice['mean_NONE_BOTH_brier']:.6f}.")
    lines.extend(["", "La calibración ajustó parámetros sólo en TRAIN y eligió familia en SELECT sobre "
        "el mismo conjunto fijo de anclas de A/B. El informe une exactamente las claves de personas y loci "
        "y recalcula las métricas, sin reajustar ni reseleccionar."])
    lines.extend(["", "## Resultados por familia y semilla", ""])
    for group in dict.fromkeys(row["group"] for row in report["followup_table"]):
        rows = [row for row in report["followup_table"] if row["group"] == group]
        first = rows[0]
        lines.extend([f"### {first['family']} · semilla {first['seed']}", "",
            "FMINUS equivale a OFF; FFULL es el FLARE completo. CALIBRATION_ONLY no usa genotipos como predictores. "
            "Los cinco brazos neuronales se entrenaron por separado con los mismos lotes.", "",
            "| Brazo | Brier | Log-loss | MAE AFR | MAE EUR | MAE NAM | Paso elegido |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in rows:
            lines.append(f"| {row['arm']} | {row['brier']:.6f} | {row['log_loss']:.6f} | "
                f"{row['dosage_mae_AFR']:.6f} | {row['dosage_mae_EUR']:.6f} | {row['dosage_mae_NAM']:.6f} | "
                f"{row['selected_step'] if row['selected_step'] is not None else '—'} |")
        lines.extend(["", "| BOTH − comparador | ΔBrier | ΔLog-loss | ΔMAE AFR | ΔMAE EUR | ΔMAE NAM |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in report["BOTH_minus_comparator"]:
            if row["group"] == group:
                lines.append(f"| {row['comparator']} | {row['delta_brier']:+.6f} | {row['delta_log_loss']:+.6f} | "
                    f"{row['delta_dosage_mae_AFR']:+.6f} | {row['delta_dosage_mae_EUR']:+.6f} | "
                    f"{row['delta_dosage_mae_NAM']:+.6f} |")
        lines.append("")
    endpoints = sum(row["selected_at_budget_end"] is True for row in report["all_neural_cases"])
    lines.extend(["## Curvas y límites de lectura", "",
        f"{endpoints}/{len(report['all_neural_cases'])} checkpoints neuronales quedaron en el final de su presupuesto. "
        "Las curvas completas TRAIN/SELECT y la exposición están en learning-curves.csv; el entrenamiento online "
        "no es una pérdida de checkpoint congelado. Llegar al límite no demuestra convergencia ni ausencia de señal.", "",
        "Sólo chr22/R0 simulado y SELECT reutilizado: las semillas repiten optimización, no poblaciones independientes. "
        "No se evaluaron LAI denso ni F1 de bordes; no se demuestra generalización NAM ni DNABR observado. "
        "Una mejora de calibración o de log-loss no sustituye al contraste raro ni garantiza mejora de dosis ancestral. "
        "SUMMARY cubre el panel REF global; DETAIL usa candidatos recuperados. SHAM conserva el resumen y "
        "no constituye un test exacto de ausencia de toda información rara.", "",
        "Las diferencias son descriptivas; no se aplicó un umbral arbitrario de significación ni se estimó "
        "incertidumbre poblacional. Cualquier ampliación requiere revisar conjuntamente controles, daños ancestrales, "
        "curvas y presupuesto; no abrir otros roles para rescatar un signo favorable.", "",
        "Fuentes y hashes exactos: report.json. No se modificaron primarios.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen-plan", "screen-outputs", "screen-comparison", "followup-plan", "followup-outputs",
                 "followup-comparison", "calibration-dir", "outdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--calibration-report-sha256", required=True)
    args = parser.parse_args()
    report = build_report(**vars(args))
    print(json.dumps({"status": report["status"], **report["counts"]}))


if __name__ == "__main__":
    main()
