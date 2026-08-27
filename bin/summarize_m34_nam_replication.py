#!/usr/bin/env python3
"""Validate and summarize the frozen M34 NAM replication.

The summary treats each mosaic realization root as the replication unit.  It
applies the decision rule recorded before execution and never pools synthetic
people as if they were independent genealogies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


RADII = ("0.1", "0.2", "0.5")
GUARDRAILS = (
    "macro_ancestry_dose_MAE",
    "NAM_truth_present_MAE",
    "false_transitions_per_cM",
    "haplotype_Brier",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON object required: {path}")
    return payload


def finite(value: Any, label: str) -> float:
    number = float(value)
    require(math.isfinite(number), f"non-finite value for {label}")
    return number


def parse_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        require("=" in value, f"root must use ROTATION=PATH: {value}")
        rotation, raw_path = value.split("=", 1)
        rotation = rotation.upper()
        require(rotation not in roots, f"duplicate root: {rotation}")
        path = Path(raw_path)
        require(path.is_dir(), f"root directory does not exist: {path}")
        roots[rotation] = path
    return roots


def metric_key(path: Path, payload: dict[str, Any]) -> tuple[str, str, str]:
    task = payload.get("task")
    if isinstance(task, dict):
        return str(task["family"]), str(task["config_id"]), str(task["arm"])
    require("/FLARE/" in str(path), f"metric lacks task identity: {path}")
    return "FLARE", "F0", "F0"


def load_metrics(root: Path) -> dict[tuple[str, str, str], tuple[Path, dict[str, Any]]]:
    metrics: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]] = {}
    for path in sorted((root / "metrics").rglob("*.metrics.json")):
        payload = load_json(path)
        require(payload.get("status") == "PASS_SCORED", f"metric is not PASS_SCORED: {path}")
        require(payload.get("truth_opened_only_by_scorer") is True,
                f"truth barrier is not recorded: {path}")
        key = metric_key(path, payload)
        require(key not in metrics, f"duplicate metric {key}: {path}")
        metrics[key] = path, payload
    return metrics


def load_receipts(root: Path) -> dict[tuple[str, str, str], tuple[Path, dict[str, Any]]]:
    receipts: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]] = {}
    for path in sorted((root / "models").rglob("train.receipt.json")):
        payload = load_json(path)
        task = payload.get("task", {})
        key = str(task.get("family")), str(task.get("config_id")), str(task.get("arm"))
        require(key not in receipts, f"duplicate training receipt {key}: {path}")
        require(payload.get("status") == "PASS_TRAINED_VALID_ONLY_FACTORIZED_LAZY",
                f"training receipt is not complete: {path}")
        require(payload.get("test_opened") is False, f"TEST was opened: {path}")
        receipts[key] = path, payload
    return receipts


def guardrail_value(payload: dict[str, Any], name: str) -> float:
    if name == "false_transitions_per_cM":
        return finite(payload["boundary"]["0.2"][name], name)
    return finite(payload[name], name)


def validate_root(
    rotation: str,
    root: Path,
    plan: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = load_metrics(root)
    receipts = load_receipts(root)
    expected_models = {
        (str(task["family"]), str(task["config_id"]))
        for task in plan["tasks"] if task["rotation"] == rotation
    }
    expected_metric_keys = {("FLARE", "F0", "F0")}
    expected_receipt_keys: set[tuple[str, str, str]] = set()
    for family, config_id in expected_models:
        for arm in ("RD", "RE"):
            expected_metric_keys.add((family, config_id, arm))
            expected_receipt_keys.add((family, config_id, arm))
    require(set(metrics) == expected_metric_keys,
            f"metric grid differs for {rotation}: {sorted(set(metrics) ^ expected_metric_keys)}")
    require(set(receipts) == expected_receipt_keys,
            f"receipt grid differs for {rotation}: {sorted(set(receipts) ^ expected_receipt_keys)}")

    truth_hashes = {payload["input_sha256"]["truth"] for _path, payload in metrics.values()}
    marker_counts = {int(payload["marker_count"]) for _path, payload in metrics.values()}
    sample_counts = {int(payload["sample_count"]) for _path, payload in metrics.values()}
    require(len(truth_hashes) == 1, f"truth differs within {rotation}")
    require(len(marker_counts) == 1, f"marker axes differ within {rotation}")
    require(sample_counts == {int(plan["target_size"]["valid"])},
            f"VALID size differs within {rotation}: {sample_counts}")

    f0 = metrics[("FLARE", "F0", "F0")][1]
    thresholds = contract["selection"]["maximum_guardrail_worsening"]
    rows: list[dict[str, Any]] = []
    for family, config_id in sorted(expected_models):
        rd_path, rd = metrics[(family, config_id, "RD")]
        re_path, re_payload = metrics[(family, config_id, "RE")]
        rd_receipt_path, rd_receipt = receipts[(family, config_id, "RD")]
        re_receipt_path, re_receipt = receipts[(family, config_id, "RE")]
        require(rd_receipt["input_sha256"] == re_receipt["input_sha256"],
                f"RD/RE inputs differ for {rotation}/{family}/{config_id}")
        require(rd_receipt["paired_task_sha256_without_arm"] ==
                re_receipt["paired_task_sha256_without_arm"],
                f"RD/RE pairing digest differs for {rotation}/{family}/{config_id}")
        require(int(rd_receipt["fit_sample_count"]) == int(plan["target_size"]["fit"]),
                f"FIT size differs in {rd_receipt_path}")
        require(int(re_receipt["fit_sample_count"]) == int(plan["target_size"]["fit"]),
                f"FIT size differs in {re_receipt_path}")
        require(int(rd_receipt["updates_executed"]) == int(plan["maximum_updates"]),
                f"RD update budget differs in {rd_receipt_path}")
        require(int(re_receipt["updates_executed"]) == int(plan["maximum_updates"]),
                f"RE update budget differs in {re_receipt_path}")

        boundary = {}
        for radius in RADII:
            rd_f1 = finite(rd["boundary"][radius]["f1"], f"{rotation} RD F1 {radius}")
            re_f1 = finite(re_payload["boundary"][radius]["f1"], f"{rotation} RE F1 {radius}")
            f0_f1 = finite(f0["boundary"][radius]["f1"], f"{rotation} F0 F1 {radius}")
            boundary[radius] = {
                "F0": f0_f1,
                "RD": rd_f1,
                "RE": re_f1,
                "RE_minus_RD": re_f1 - rd_f1,
                "RE_minus_F0": re_f1 - f0_f1,
            }
        guardrails = {}
        for name in GUARDRAILS:
            rd_value = guardrail_value(rd, name)
            re_value = guardrail_value(re_payload, name)
            delta = re_value - rd_value
            threshold = finite(thresholds[name], f"threshold {name}")
            guardrails[name] = {
                "RD": rd_value,
                "RE": re_value,
                "RE_minus_RD": delta,
                "maximum_worsening": threshold,
                "pass": delta <= threshold,
            }
        rows.append({
            "rotation": rotation,
            "family": family,
            "config_id": config_id,
            "boundary": boundary,
            "guardrails": guardrails,
            "all_guardrails_pass": all(item["pass"] for item in guardrails.values()),
            "selected_update": {
                "RD": int(rd_receipt["selected_update"]),
                "RE": int(re_receipt["selected_update"]),
            },
            "parameter_count": int(re_receipt["parameter_count"]),
            "input_sha256": {
                "RD_metrics": sha256_file(rd_path),
                "RE_metrics": sha256_file(re_path),
                "RD_receipt": sha256_file(rd_receipt_path),
                "RE_receipt": sha256_file(re_receipt_path),
            },
        })
    audit = {
        "rotation": rotation,
        "metric_count": len(metrics),
        "training_receipt_count": len(receipts),
        "truth_sha256": next(iter(truth_hashes)),
        "marker_count": next(iter(marker_counts)),
        "valid_sample_count": next(iter(sample_counts)),
    }
    return rows, audit


def plot_summary(result: dict[str, Any], output: Path) -> None:
    """Write a dependency-free SVG of paired effects by simulation root."""
    roots = [item["rotation"] for item in result["root_audit"]]
    colors = {"bilstm": "#2b6cb0", "unet_1d": "#dd6b20"}
    labels = {"bilstm": "BiLSTM", "unet_1d": "U-Net 1D"}
    panels = (
        ("RE_minus_RD_F1_0.2cM", "Aporte incremental de variantes raras"),
        ("RE_minus_F0_F1_0.2cM", "Resultado frente a FLARE"),
    )
    all_values = [0.0, 0.005]
    for decision in result["model_decisions"]:
        for key, _title in panels:
            all_values.extend(decision[key]["per_root"].values())
    low, high = min(all_values), max(all_values)
    padding = max((high - low) * 0.12, 0.002)
    low -= padding
    high += padding

    width, height = 1100, 470
    panel_width, panel_height = 430, 300
    panel_lefts = (105, 625)
    panel_top = 80

    def y_position(value: float) -> float:
        return panel_top + (high - value) * panel_height / (high - low)

    def line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
        attributes = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
        return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {attributes}/>'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:bold}'
        '.panel{font-size:15px;font-weight:bold}.axis{font-size:12px}.legend{font-size:13px}</style>',
        '<text x="550" y="34" text-anchor="middle" class="title">'
        'M34 NAM: replicación exploratoria en cromosoma 22</text>',
    ]
    ticks = 5
    for panel_index, (key, title) in enumerate(panels):
        left = panel_lefts[panel_index]
        right = left + panel_width
        svg.append(f'<text x="{(left + right) / 2}" y="62" text-anchor="middle" class="panel">{title}</text>')
        for tick in range(ticks + 1):
            value = low + tick * (high - low) / ticks
            y = y_position(value)
            svg.append(line(left, y, right, y, stroke="#dddddd", stroke_width="1"))
            svg.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis">{value:+.3f}</text>')
        svg.append(line(left, panel_top, left, panel_top + panel_height,
                        stroke="#333333", stroke_width="1"))
        svg.append(line(left, y_position(0.0), right, y_position(0.0),
                        stroke="#333333", stroke_width="1.2"))
        if panel_index == 0:
            svg.append(line(left, y_position(0.005), right, y_position(0.005),
                            stroke="#666666", stroke_width="1", stroke_dasharray="6 4"))
        x_positions = [left + index * panel_width / (len(roots) - 1) for index in range(len(roots))]
        for x, root in zip(x_positions, roots, strict=True):
            svg.append(f'<text x="{x:.2f}" y="{panel_top + panel_height + 22}" text-anchor="middle" class="axis">{root}</text>')
        for decision in result["model_decisions"]:
            family = decision["family"]
            points = [(x, y_position(decision[key]["per_root"][root]))
                      for x, root in zip(x_positions, roots, strict=True)]
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{colors[family]}" '
                       'stroke-width="2.5"/>')
            for x, y in points:
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{colors[family]}"/>')
        svg.append(f'<text x="{(left + right) / 2}" y="{panel_top + panel_height + 46}" '
                   'text-anchor="middle" class="axis">Raíz de simulación</text>')
    svg.append('<text x="22" y="230" text-anchor="middle" class="axis" '
               'transform="rotate(-90 22 230)">Diferencia en F1 de bordes (0,2 cM)</text>')
    legend_y = 450
    for index, family in enumerate(("bilstm", "unet_1d")):
        x = 350 + index * 170
        svg.append(line(x, legend_y - 4, x + 28, legend_y - 4,
                        stroke=colors[family], stroke_width="3"))
        svg.append(f'<text x="{x + 36}" y="{legend_y}" class="legend">{labels[family]}</text>')
    svg.append(line(690, legend_y - 4, 718, legend_y - 4, stroke="#666666",
                    stroke_width="1", stroke_dasharray="6 4"))
    svg.append('<text x="726" y="450" class="legend">mínimo preregistrado</text>')
    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_json(args.plan)
    contract = load_json(args.contract)
    require(sha256_file(args.contract) == plan["inputs"]["adaptive_contract_sha256"],
            "adaptive contract hash differs from the frozen plan")
    require(plan.get("test_opened") is False, "frozen plan opened TEST")
    roots = parse_roots(args.root)
    expected_roots = set(map(str, plan["roots"]))
    require(set(roots) == expected_roots,
            f"root set differs: expected={sorted(expected_roots)}, observed={sorted(roots)}")

    rows: list[dict[str, Any]] = []
    audits = []
    for rotation in sorted(roots):
        root_rows, audit = validate_root(rotation, roots[rotation], plan, contract)
        rows.extend(root_rows)
        audits.append(audit)

    models = sorted({(row["family"], row["config_id"]) for row in rows})
    decisions = []
    minimum = finite(contract["selection"]["provisional_minimum_delta_F1"], "minimum delta")
    for family, config_id in models:
        selected = [row for row in rows if (row["family"], row["config_id"]) ==
                    (family, config_id)]
        primary = [row["boundary"]["0.2"]["RE_minus_RD"] for row in selected]
        versus_f0 = [row["boundary"]["0.2"]["RE_minus_F0"] for row in selected]
        sensitivity = {
            radius: [row["boundary"][radius]["RE_minus_RD"] for row in selected]
            for radius in ("0.1", "0.5")
        }
        checks = {
            "primary_positive_all_roots": all(value > 0.0 for value in primary),
            "primary_at_least_0.005_in_two_roots": sum(value >= minimum for value in primary) >= 2,
            "primary_root_median_at_least_0.005": statistics.median(primary) >= minimum,
            "sensitivity_0.1_nonnegative_in_two_roots":
                sum(value >= 0.0 for value in sensitivity["0.1"]) >= 2,
            "sensitivity_0.5_nonnegative_in_two_roots":
                sum(value >= 0.0 for value in sensitivity["0.5"]) >= 2,
            "guardrails_pass_all_roots": all(row["all_guardrails_pass"] for row in selected),
            "RE_minus_F0_root_median_positive": statistics.median(versus_f0) > 0.0,
            "RE_minus_F0_nonnegative_in_two_roots": sum(value >= 0.0 for value in versus_f0) >= 2,
        }
        decisions.append({
            "family": family,
            "config_id": config_id,
            "root_count": len(selected),
            "RE_minus_RD_F1_0.2cM": {
                "per_root": dict(zip(sorted(roots), primary, strict=True)),
                "mean": statistics.mean(primary),
                "median": statistics.median(primary),
            },
            "RE_minus_F0_F1_0.2cM": {
                "per_root": dict(zip(sorted(roots), versus_f0, strict=True)),
                "mean": statistics.mean(versus_f0),
                "median": statistics.median(versus_f0),
            },
            "checks": checks,
            "promotion_pass": all(checks.values()),
        })

    status = "PASS_PROMOTE_EXPLORATORY" if any(item["promotion_pass"] for item in decisions) \
        else "STOP_FROZEN_DIPLOID_FINALISTS"
    result = {
        "schema_version": "1.0.0",
        "stage": "M34_NAM_128_REPLICATION_SUMMARY",
        "status": status,
        "claim_level": "exploratory",
        "test_opened": False,
        "replication_unit": "independent_mosaic_realization_root",
        "plan_sha256": sha256_file(args.plan),
        "contract_sha256": sha256_file(args.contract),
        "root_audit": audits,
        "rows": rows,
        "model_decisions": decisions,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = [
        "rotation", "family", "config_id", "F0_F1_0.2cM", "RD_F1_0.2cM",
        "RE_F1_0.2cM", "RE_minus_RD_F1_0.2cM", "RE_minus_F0_F1_0.2cM",
        "RE_minus_RD_false_transitions_per_cM", "RE_minus_RD_macro_MAE",
        "RE_minus_RD_NAM_MAE", "RE_minus_RD_Brier", "all_guardrails_pass",
        "selected_update_RD", "selected_update_RE",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        primary = row["boundary"]["0.2"]
        values = [
            row["rotation"], row["family"], row["config_id"], primary["F0"],
            primary["RD"], primary["RE"], primary["RE_minus_RD"],
            primary["RE_minus_F0"],
            row["guardrails"]["false_transitions_per_cM"]["RE_minus_RD"],
            row["guardrails"]["macro_ancestry_dose_MAE"]["RE_minus_RD"],
            row["guardrails"]["NAM_truth_present_MAE"]["RE_minus_RD"],
            row["guardrails"]["haplotype_Brier"]["RE_minus_RD"],
            row["all_guardrails_pass"], row["selected_update"]["RD"],
            row["selected_update"]["RE"],
        ]
        lines.append("\t".join(map(str, values)))
    args.output_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_figure = getattr(args, "output_figure", None)
    if output_figure is not None:
        plot_summary(result, output_figure)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, metavar="ROTATION=PATH")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    summary = summarize(parse_args())
    print(json.dumps({
        "status": summary["status"],
        "models": len(summary["model_decisions"]),
        "roots": len(summary["root_audit"]),
    }, sort_keys=True))
