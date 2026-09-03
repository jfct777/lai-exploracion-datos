#!/usr/bin/env python3
"""Deterministic promotion of complete, paired M37 candidate families."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from m37_trace_core import require


ARMS = ("RE", "RD", "POOLED", "SHAM", "GEOMETRY")
COMPARATORS = ("F0", "RD", "POOLED", "SHAM", "GEOMETRY")


def bootstrap_interval(values: list[float], seed: int, draws: int,
                       minimum_roots: int) -> list[float]:
    values_array = np.asarray(values, dtype=float)
    require(len(values_array) >= minimum_roots,
            f"promotion needs at least {minimum_roots} independent mosaic roots")
    require(draws > 0, "bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    medians = np.median(
        values_array[rng.integers(0, len(values_array), size=(draws, len(values_array)))], axis=1
    )
    return [float(np.quantile(medians, .025)), float(np.quantile(medians, .975))]


def _value(metric: dict, name: str, boundary_tolerance_cm: float) -> float:
    if name == "f1":
        key = str(boundary_tolerance_cm)
        require("f1_boundary" in metric and key in metric["f1_boundary"],
                f"TRACE metrics lack F1 at {key} cM")
        return float(metric["f1_boundary"][key])
    require(name == "log_loss" and "log_loss" in metric, "TRACE metrics lack log_loss")
    return float(metric["log_loss"])


def _baseline(metric: dict) -> dict:
    baseline = metric.get("baseline", metric.get("F0"))
    require(isinstance(baseline, dict), "TRACE metrics lack generic F0 baseline")
    return baseline


def promote(metrics: list[dict], keep_fraction: float, boundary_tolerance_cm: float = .2,
            minimum_f1_gain: float = 0.0, maximum_log_loss_increase: float = 0.0,
            bootstrap_seed: int = 1103, bootstrap_draws: int = 2000,
            minimum_replication_roots: int = 3) -> tuple[list[dict], dict[str, object]]:
    """Apply the preregistered positive-F1/nonworse-log-loss gate.

    ``minimum_f1_gain=0`` implements the contract's strict positive-increment
    rule; a zero permitted log-loss increase implements its non-worsening rule.
    """
    require(0 < keep_fraction <= 1, "keep fraction differs")
    require(boundary_tolerance_cm > 0 and minimum_f1_gain >= 0 and
            maximum_log_loss_increase >= 0 and minimum_replication_roots >= 3,
            "TRACE promotion criteria differ from the preregistered domain")
    paired: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in metrics:
        require(set(("candidate_id", "arm", "root", "metrics")).issubset(row),
                "TRACE metric row differs")
        candidate, root, arm = str(row["candidate_id"]), str(row["root"]), str(row["arm"])
        require(arm in ARMS, "TRACE metric arm differs")
        require(arm not in paired[candidate][root], "duplicate TRACE candidate/root/arm row")
        paired[candidate][root][arm] = row

    ranked: list[dict] = []
    for candidate, roots in paired.items():
        gains: dict[str, list[float]] = {name: [] for name in COMPARATORS}
        log_loss_deltas: dict[str, list[float]] = {name: [] for name in COMPARATORS}
        root_gates: dict[str, dict[str, object]] = {}
        incomplete: dict[str, list[str]] = {}
        for root, arms in sorted(roots.items()):
            missing = sorted(set(ARMS) - set(arms))
            if missing:
                incomplete[root] = missing
                continue
            re_metric = arms["RE"]["metrics"]
            baseline = _baseline(re_metric)
            for arm in ARMS:
                arm_baseline = _baseline(arms[arm]["metrics"])
                require(_value(arm_baseline, "f1", boundary_tolerance_cm) ==
                        _value(baseline, "f1", boundary_tolerance_cm) and
                        _value(arm_baseline, "log_loss", boundary_tolerance_cm) ==
                        _value(baseline, "log_loss", boundary_tolerance_cm),
                        "F0 baseline differs across paired arms")
            comparator_metrics = {"F0": baseline, **{
                arm: arms[arm]["metrics"] for arm in ARMS if arm != "RE"
            }}
            root_gain: dict[str, float] = {}
            root_safety: dict[str, float] = {}
            for name in COMPARATORS:
                root_gain[name] = (_value(re_metric, "f1", boundary_tolerance_cm) -
                                   _value(comparator_metrics[name], "f1", boundary_tolerance_cm))
                root_safety[name] = (_value(re_metric, "log_loss", boundary_tolerance_cm) -
                                     _value(comparator_metrics[name], "log_loss", boundary_tolerance_cm))
                gains[name].append(root_gain[name])
                log_loss_deltas[name].append(root_safety[name])
            root_gates[root] = {
                "RE_minus_comparator_F1": root_gain,
                "RE_minus_comparator_log_loss": root_safety,
                "pass": (all(value > minimum_f1_gain for value in root_gain.values()) and
                         all(value <= maximum_log_loss_increase for value in root_safety.values())),
            }

        complete_roots = sorted(root_gates)
        if not complete_roots:
            ranked.append({"candidate_id": candidate, "roots": sorted(roots),
                           "complete_roots": [], "incomplete_roots": incomplete,
                           "promote": False, "reason": "NO_COMPLETE_PAIRED_ROOT"})
            continue
        replication = len(complete_roots) >= minimum_replication_roots
        median_gains = {name: float(np.median(values)) for name, values in gains.items()}
        median_safety = {name: float(np.median(values)) for name, values in log_loss_deltas.items()}
        intervals = ({name: bootstrap_interval(values, bootstrap_seed, bootstrap_draws,
                                                minimum_replication_roots)
                      for name, values in gains.items()} if replication else None)
        if replication:
            gate = (not incomplete and
                    all(intervals[name][0] > minimum_f1_gain for name in COMPARATORS) and
                    all(value <= maximum_log_loss_increase for value in median_safety.values()))
        else:
            gate = not incomplete and all(bool(row["pass"]) for row in root_gates.values())
        ranked.append({
            "candidate_id": candidate,
            "roots": sorted(roots),
            "complete_roots": complete_roots,
            "incomplete_roots": incomplete,
            "root_gates": root_gates,
            "median_RE_minus_comparator_F1": median_gains,
            "median_RE_minus_comparator_log_loss": median_safety,
            "RE_minus_comparator_F1_bootstrap95": intervals,
            "rung_mode": "replication" if replication else "triage_or_expansion",
            "worst_median_F1_gain": min(median_gains.values()),
            "promote": bool(gate),
            "reason": "PASS_ALL_COMPARATORS" if gate else "FAIL_PAIRED_GATE",
        })

    ranked.sort(key=lambda row: (
        not row["promote"], -float(row.get("worst_median_F1_gain", -np.inf)), row["candidate_id"]
    ))
    allowed = max(1, int(len(ranked) * keep_fraction)) if ranked else 0
    for index, row in enumerate(ranked):
        if row["promote"] and index >= allowed:
            row["promote"] = False
            row["reason"] = "PASS_GATE_BELOW_KEEP_FRACTION"
    criteria = {
        "boundary_tolerance_cM": boundary_tolerance_cm,
        "minimum_strict_F1_gain": minimum_f1_gain,
        "maximum_log_loss_increase": maximum_log_loss_increase,
        "comparators": list(COMPARATORS),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws": bootstrap_draws,
        "minimum_replication_roots": minimum_replication_roots,
    }
    return ranked, criteria


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--metrics-receipt", type=Path, required=True)
    parser.add_argument("--keep-fraction", type=float, required=True)
    parser.add_argument("--boundary-tolerance-cm", type=float, required=True)
    parser.add_argument("--minimum-f1-gain", type=float, required=True)
    parser.add_argument("--maximum-log-loss-increase", type=float, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--bootstrap-draws", type=int, required=True)
    parser.add_argument("--minimum-replication-roots", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite TRACE promotion plan")
    collection = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    collection_receipt = json.loads(args.metrics_receipt.read_text(encoding="utf-8"))
    collection_sha256 = hashlib.sha256(args.metrics_json.read_bytes()).hexdigest()
    require(isinstance(collection, dict) and collection.get("stage") == "M37_TRACE_COLLECT_METRICS" and
            collection_receipt.get("stage") == "M37_TRACE_COLLECT_METRICS" and
            collection_receipt.get("output_sha256") == collection_sha256 and
            collection_receipt.get("root") == collection.get("root") and
            collection_receipt.get("row_count") == len(collection.get("rows", [])),
            "TRACE metric collection receipt differs")
    rows = collection.get("rows")
    require(isinstance(rows, list) and rows, "TRACE metrics collection is empty")
    ranked, criteria = promote(
        rows, args.keep_fraction, args.boundary_tolerance_cm, args.minimum_f1_gain,
        args.maximum_log_loss_increase, args.bootstrap_seed, args.bootstrap_draws,
        args.minimum_replication_roots,
    )
    payload = {"schema_version": "1.0.0", "stage": "M37_SUCCESSIVE_HALVING",
               "root": collection["root"], "criteria": criteria, "ranked": ranked}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".receipt.json").write_text(json.dumps({
        "schema_version": "1.0.0", "stage": "M37_SUCCESSIVE_HALVING",
        "root": collection["root"],
        "metrics_collection_sha256": collection_sha256,
        "metrics_collection_receipt_sha256": hashlib.sha256(args.metrics_receipt.read_bytes()).hexdigest(),
        "criteria": criteria,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
