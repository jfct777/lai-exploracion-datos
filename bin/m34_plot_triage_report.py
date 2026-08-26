#!/usr/bin/env python3
"""Summarize and plot the audited M34 triage comparison table."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PRIMARY_METRIC = "boundary_F1_0.2cM"
REQUIRED_COLUMNS = {
    "family", "config_id", "seed", "root", "radius_cM", "sweep_stage",
    "maximum_updates", "metric", "F0", "RD", "RE", "RE_minus_RD",
    "RE_minus_F0",
}


class ReportError(ValueError):
    """Raised when the comparison table cannot support an M34 report."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ReportError(f"non-finite JSON constant in {path}: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read JSON contract: {path}") from error
    require(isinstance(value, dict), "contract root must be an object")
    return value


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{label} must be numeric") from error
    require(math.isfinite(result), f"{label} must be finite")
    return result


def read_contract(path: Path) -> dict[str, Any]:
    contract = strict_json(path)
    require(contract.get("experiment_id") == "M34_NAM_ADAPTIVE_MODEL_SWEEP",
            "unexpected adaptive sweep contract")
    stages = contract.get("stages")
    metrics = contract.get("metrics")
    selection = contract.get("selection")
    families = contract.get("families")
    require(isinstance(stages, dict) and isinstance(metrics, dict) and
            isinstance(selection, dict) and isinstance(families, dict),
            "contract is missing reporting members")
    require(metrics.get("primary") == PRIMARY_METRIC,
            "contract primary metric differs from the report")
    guardrails = metrics.get("guardrails")
    thresholds = selection.get("maximum_guardrail_worsening")
    require(isinstance(guardrails, list) and len(guardrails) > 0 and
            isinstance(thresholds, dict) and set(guardrails) == set(thresholds),
            "contract guardrails and thresholds differ")
    require(all(finite_float(thresholds[name], f"threshold/{name}") > 0.0
                for name in guardrails),
            "guardrail thresholds must be positive")
    return contract


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(reader.fieldnames is not None and
                    REQUIRED_COLUMNS.issubset(reader.fieldnames),
                    "comparison table columns differ")
            rows = list(reader)
    except OSError as error:
        raise ReportError(f"cannot read comparison table: {path}") from error
    require(rows, "comparison table is empty")
    return rows


def validate_aggregate_receipt(path: Path, comparison_path: Path) -> dict[str, Any]:
    receipt = strict_json(path)
    require(receipt.get("stage") == "M34_AGGREGATE_ADAPTIVE_STAGE_METRICS" and
            receipt.get("source_plan_stage") == "M34_TRIAGE_PLAN" and
            receipt.get("status") == "PASS_EXACT_TRIAGE_GRID_F0_RD_RE" and
            receipt.get("claim_level") == "exploratory",
            "aggregate receipt identity differs")
    require(receipt.get("evaluation_split") == "VALID" and
            receipt.get("test_opened") is False,
            "aggregate receipt does not close TEST")
    output_hashes = receipt.get("output_sha256")
    require(isinstance(output_hashes, dict) and
            output_hashes.get("table") == sha256_file(comparison_path),
            "aggregate receipt is not bound to the comparison table")
    require(receipt.get("pair_count") == 21 and
            receipt.get("long_table_row_count") == 147,
            "aggregate receipt triage dimensions differ")
    return receipt


def triage_config_order(contract: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    ordered: list[tuple[str, str, int]] = []
    for family, specification in contract["families"].items():
        triage_ids = specification.get("triage_ids")
        configs = specification.get("configs")
        require(isinstance(triage_ids, list) and isinstance(configs, list),
                f"{family} configuration space is malformed")
        ranks = {
            str(config.get("id")): int(config.get("complexity_rank"))
            for config in configs
        }
        require(set(triage_ids).issubset(ranks),
                f"{family} triage IDs are absent from the declared space")
        ordered.extend((family, config_id, ranks[config_id])
                       for config_id in triage_ids)
    require(len(ordered) == len({(family, config) for family, config, _ in ordered}),
            "contract contains duplicate triage configurations")
    return ordered


def summarize(rows: Sequence[Mapping[str, str]],
              contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build one exact, ordered report row per family and configuration."""
    guardrails = list(contract["metrics"]["guardrails"])
    thresholds = contract["selection"]["maximum_guardrail_worsening"]
    required_metrics = {PRIMARY_METRIC, *guardrails}
    expected = triage_config_order(contract)
    expected_pairs = {(family, config) for family, config, _ in expected}
    by_config: dict[tuple[str, str], dict[str, Mapping[str, str]]] = {}
    metadata: dict[tuple[str, str], tuple[str, ...]] = {}

    for index, row in enumerate(rows, start=2):
        pair = (str(row["family"]), str(row["config_id"]))
        require(pair in expected_pairs,
                f"undeclared triage configuration on line {index}: {pair}")
        metric = str(row["metric"])
        if metric not in required_metrics:
            continue
        require(metric not in by_config.setdefault(pair, {}),
                f"duplicate {metric} row for {pair}")
        by_config[pair][metric] = row
        identity = tuple(str(row[name]) for name in
                         ("seed", "root", "radius_cM", "sweep_stage",
                          "maximum_updates"))
        if pair in metadata:
            require(metadata[pair] == identity,
                    f"inconsistent task identity for {pair}")
        else:
            metadata[pair] = identity

    require(set(by_config) == expected_pairs,
            "comparison table contains an incomplete triage configuration grid")
    summaries: list[dict[str, Any]] = []
    for family, config_id, complexity_rank in expected:
        pair = (family, config_id)
        observed = by_config[pair]
        require(set(observed) == required_metrics,
                f"required metrics are incomplete for {pair}")
        seed, root, radius, stage, updates = metadata[pair]
        require(stage == "triage", f"unexpected sweep stage for {pair}: {stage}")
        primary = metric_values(observed[PRIMARY_METRIC], pair, PRIMARY_METRIC)
        summary: dict[str, Any] = {
            "family": family,
            "config_id": config_id,
            "complexity_rank": complexity_rank,
            "seed": int(seed),
            "root": root,
            "radius_cM": finite_float(radius, f"{pair}/radius_cM"),
            "sweep_stage": stage,
            "maximum_updates": int(updates),
            **prefixed_values("boundary_F1_0.2cM", primary),
        }
        failed = []
        for metric in guardrails:
            values = metric_values(observed[metric], pair, metric)
            threshold = finite_float(thresholds[metric], f"threshold/{metric}")
            passed = values["RE_minus_RD"] <= threshold + 1e-15
            summary.update(prefixed_values(metric, values))
            summary[f"threshold_{metric}"] = threshold
            summary[f"pass_{metric}"] = passed
            if not passed:
                failed.append(metric)
        minimum_delta = finite_float(
            contract["selection"]["provisional_minimum_delta_F1"],
            "provisional_minimum_delta_F1",
        )
        summary["primary_delta_pass"] = primary["RE_minus_RD"] >= minimum_delta
        summary["beats_F0"] = primary["RE_minus_F0"] > 0.0
        summary["all_guardrails_pass"] = not failed
        summary["failed_guardrails"] = ";".join(failed)
        summaries.append(summary)
    return summaries


def metric_values(row: Mapping[str, str], pair: tuple[str, str],
                  metric: str) -> dict[str, float]:
    values = {
        name: finite_float(row[name], f"{pair}/{metric}/{name}")
        for name in ("F0", "RD", "RE", "RE_minus_RD", "RE_minus_F0")
    }
    tolerance = 64.0 * math.ulp(max(1.0, abs(values["RE"]), abs(values["RD"]),
                                    abs(values["F0"])))
    require(abs((values["RE"] - values["RD"]) - values["RE_minus_RD"]) <= tolerance,
            f"RE-RD arithmetic differs for {pair}/{metric}")
    require(abs((values["RE"] - values["F0"]) - values["RE_minus_F0"]) <= tolerance,
            f"RE-F0 arithmetic differs for {pair}/{metric}")
    return values


def prefixed_values(metric: str, values: Mapping[str, float]) -> dict[str, float]:
    return {f"{name}_{metric}": value for name, value in values.items()}


def summary_columns(contract: Mapping[str, Any]) -> list[str]:
    columns = [
        "family", "config_id", "complexity_rank", "seed", "root",
        "radius_cM", "sweep_stage", "maximum_updates",
    ]
    value_names = ("F0", "RD", "RE", "RE_minus_RD", "RE_minus_F0")
    columns.extend(f"{name}_{PRIMARY_METRIC}" for name in value_names)
    for metric in contract["metrics"]["guardrails"]:
        columns.extend(f"{name}_{metric}" for name in value_names)
        columns.extend((f"threshold_{metric}", f"pass_{metric}"))
    columns.extend(("primary_delta_pass", "beats_F0", "all_guardrails_pass",
                    "failed_guardrails"))
    return columns


def format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def summary_tsv(summaries: Sequence[Mapping[str, Any]],
                contract: Mapping[str, Any]) -> str:
    columns = summary_columns(contract)
    lines = ["\t".join(columns)]
    lines.extend("\t".join(format_cell(row[name]) for name in columns)
                 for row in summaries)
    return "\n".join(lines) + "\n"


def render_figure(summaries: Sequence[Mapping[str, Any]],
                  contract: Mapping[str, Any], png_path: Path,
                  pdf_path: Path) -> None:
    """Render absolute performance, paired effects and guardrails together."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    guardrails = list(contract["metrics"]["guardrails"])
    labels = [f"{row['family']} | {row['config_id']}" for row in summaries]
    y = np.arange(len(summaries))
    f0 = np.asarray([row[f"F0_{PRIMARY_METRIC}"] for row in summaries])
    rd = np.asarray([row[f"RD_{PRIMARY_METRIC}"] for row in summaries])
    re = np.asarray([row[f"RE_{PRIMARY_METRIC}"] for row in summaries])
    delta_rd = np.asarray([row[f"RE_minus_RD_{PRIMARY_METRIC}"]
                           for row in summaries])
    delta_f0 = np.asarray([row[f"RE_minus_F0_{PRIMARY_METRIC}"]
                           for row in summaries])
    guardrail_ratio = np.asarray([
        [row[f"RE_minus_RD_{metric}"] / row[f"threshold_{metric}"]
         for metric in guardrails]
        for row in summaries
    ])

    ink = "#26333D"
    muted = "#667681"
    grid = "#D9E0E5"
    f0_color = "#424B54"
    rd_color = "#D0832F"
    re_color = "#2C6E9B"
    fig = plt.figure(figsize=(18.0, 11.8), facecolor="white")
    layout = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 1.1), wspace=0.23)
    ax_absolute = fig.add_subplot(layout[0, 0])
    ax_delta = fig.add_subplot(layout[0, 1], sharey=ax_absolute)
    ax_guard = fig.add_subplot(layout[0, 2], sharey=ax_absolute)

    ax_absolute.scatter(rd, y, s=36, marker="o", facecolors="white",
                        edgecolors=rd_color, linewidths=1.5, label="RD")
    ax_absolute.scatter(re, y, s=36, marker="s", color=re_color, label="RE")
    for index in range(len(y)):
        ax_absolute.plot([rd[index], re[index]], [y[index], y[index]],
                         color=grid, linewidth=1.0, zorder=0)
    f0_reference = float(f0[0])
    require(np.allclose(f0, f0_reference, rtol=0.0, atol=1e-12),
            "F0 differs across triage configurations")
    ax_absolute.axvline(f0_reference, color=f0_color, linestyle="--",
                        linewidth=1.4, label=f"F0 = {f0_reference:.4f}")
    ax_absolute.set_xlim(0.0, 1.0)
    ax_absolute.set_xlabel("F1 de bordes a 0,2 cM", color=ink)
    ax_absolute.set_yticks(y, labels, fontsize=7.4)
    ax_absolute.invert_yaxis()
    ax_absolute.set_title("A. Rendimiento absoluto", loc="left", color=ink,
                          fontweight="bold")
    ax_absolute.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")

    ax_delta.axvline(0.0, color=f0_color, linestyle="--", linewidth=1.0)
    ax_delta.scatter(delta_rd, y, s=34, marker="o", color=re_color,
                     label="RE − RD")
    ax_delta.scatter(delta_f0, y, s=34, marker="D", facecolors="white",
                     edgecolors=ink, linewidths=1.2, label="RE − F0")
    delta_extent = max(0.005, float(np.max(np.abs(
        np.concatenate((delta_rd, delta_f0))))) * 1.18)
    ax_delta.set_xlim(-delta_extent, delta_extent)
    ax_delta.set_xlabel("Diferencia en F1 (positivo = mejor)", color=ink)
    ax_delta.set_title("B. Diferencia al habilitar valores raros", loc="left", color=ink,
                       fontweight="bold")
    ax_delta.legend(frameon=False, fontsize=8, loc="lower left")
    ax_delta.tick_params(axis="y", labelleft=False)

    limit = max(1.25, float(np.max(np.abs(guardrail_ratio))))
    image = ax_guard.imshow(guardrail_ratio, aspect="auto", cmap="PuOr_r",
                            vmin=-limit, vmax=limit)
    short_names = {
        "macro_ancestry_dose_MAE": "MAE\nmacro",
        "NAM_truth_present_MAE": "MAE NAM\ncon verdad",
        "false_transitions_per_cM": "Transiciones\nfalsas/cM",
        "haplotype_Brier": "Brier\nhaplotipo",
    }
    ax_guard.set_xticks(range(len(guardrails)),
                        [short_names.get(name, name) for name in guardrails],
                        fontsize=7.6)
    ax_guard.tick_params(axis="y", labelleft=False)
    ax_guard.set_title("C. Controles de error", loc="left", color=ink,
                       fontweight="bold")
    for row_index in range(guardrail_ratio.shape[0]):
        for column_index in range(guardrail_ratio.shape[1]):
            ratio = guardrail_ratio[row_index, column_index]
            delta = summaries[row_index][f"RE_minus_RD_{guardrails[column_index]}"]
            marker = "!" if ratio > 1.0 else ""
            ax_guard.text(column_index, row_index, f"{delta:+.3g}{marker}",
                          ha="center", va="center", fontsize=6.4,
                          fontweight="bold" if marker else "normal",
                          color="white" if abs(ratio) > 0.58 * limit else ink)
    colorbar = fig.colorbar(image, ax=ax_guard, fraction=0.045, pad=0.03)
    colorbar.set_label("(RE − RD) / límite permitido", fontsize=8, color=ink)
    colorbar.ax.tick_params(labelsize=7)

    families = [row["family"] for row in summaries]
    for index in range(1, len(families)):
        if families[index] != families[index - 1]:
            for axis in (ax_absolute, ax_delta, ax_guard):
                axis.axhline(index - 0.5, color=grid, linewidth=0.9)
    for axis in (ax_absolute, ax_delta):
        axis.grid(axis="x", color=grid, linewidth=0.7)
        axis.set_axisbelow(True)
    for axis in (ax_absolute, ax_delta, ax_guard):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=muted)

    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.20, right=0.96)
    fig.suptitle("M34 chr22 — comparación exploratoria AFR/EUR/NAM",
                 x=0.055, y=0.965, ha="left", fontsize=17,
                 fontweight="bold", color=ink)
    fig.text(
        0.055, 0.925,
        "Triaje exploratorio con una raíz (R0) y una semilla (1103); VALID "
        "separado de FIT. F0: FLARE; RD: valores raros desactivados, conservando "
        "loci y máscaras; RE: valores raros habilitados.",
        ha="left", fontsize=10.2, color=muted,
    )
    minimum_delta = float(contract["selection"]["provisional_minimum_delta_F1"])
    fig.text(
        0.055, 0.025,
        f"El contrato de triaje exige RE−RD ≥ {minimum_delta:.3f} y controles "
        "dentro de sus umbrales. Para superar F0 también se necesita RE−F0 > 0; "
        "aun así, la señal sigue siendo exploratoria. En los controles de error, "
        "valores menores son mejores; "
        "! indica que RE−RD supera el umbral preespecificado. TEST no se abrió.",
        ha="left", fontsize=9.0, color=ink,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, format="png", dpi=220, bbox_inches="tight",
                facecolor="white", metadata={"Software": "M34 report"})
    fixed_date = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white",
                metadata={"Creator": "M34 report", "CreationDate": fixed_date,
                          "ModDate": fixed_date})
    plt.close(fig)


def write_artifacts(comparison_path: Path, contract_path: Path,
                    aggregate_receipt_path: Path,
                    summary_path: Path, png_path: Path, pdf_path: Path,
                    receipt_path: Path) -> dict[str, Any]:
    outputs = (summary_path, png_path, pdf_path, receipt_path)
    require(len({path.resolve() for path in outputs}) == len(outputs),
            "report output paths must be distinct")
    require(not any(path.exists() for path in outputs),
            "refusing to overwrite report outputs")
    contract = read_contract(contract_path)
    validate_aggregate_receipt(aggregate_receipt_path, comparison_path)
    summaries = summarize(read_rows(comparison_path), contract)
    temporary = tuple(path.with_name(f".{path.stem}.tmp{path.suffix}")
                      for path in outputs)
    require(not any(path.exists() for path in temporary),
            "temporary report output already exists")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary[0].write_text(summary_tsv(summaries, contract), encoding="utf-8")
        render_figure(summaries, contract, temporary[1], temporary[2])
        receipt = {
            "schema_version": "1.0.0",
            "stage": "M34_TRIAGE_REPORT",
            "status": "PASS_REPORT_RENDERED",
            "claim_level": contract["scope"]["claim_level"],
            "evaluation_split": "VALID",
            "test_opened": False,
            "configuration_count": len(summaries),
            "primary_metric": PRIMARY_METRIC,
            "input_sha256": {
                "comparison_table": sha256_file(comparison_path),
                "adaptive_contract": sha256_file(contract_path),
                "aggregate_receipt": sha256_file(aggregate_receipt_path),
            },
            "output_sha256": {
                "summary_table": sha256_file(temporary[0]),
                "figure_png": sha256_file(temporary[1]),
                "figure_pdf": sha256_file(temporary[2]),
            },
        }
        temporary[3].write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for source, target in zip(temporary, outputs):
            os.replace(source, target)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--aggregate-receipt", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = write_artifacts(
        args.comparison, args.contract, args.aggregate_receipt, args.summary,
        args.png, args.pdf, args.receipt,
    )
    print(json.dumps({
        "status": receipt["status"],
        "configuration_count": receipt["configuration_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
