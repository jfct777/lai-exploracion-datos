#!/usr/bin/env python3
"""Compare canonical ALT and minor-allele M16.5 partitions without label bias."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


RESOLUTIONS = ("0.5", "0.8", "1", "1.2", "1.5", "2", "3")
EXPECTED_GRAPH = {
    "n_nodes": 2601,
    "n_edges": 1604,
    "n_isolated": 1796,
    "min_edge_bp": 5_000_000,
    "min_max_segment_bp": 500_000,
    "weight_transform": "log1p",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-dir", required=True, type=Path)
    p.add_argument("--minor-dir", required=True, type=Path)
    p.add_argument("--cohort-summary", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    return p.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_edges(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip")


def edge_map(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    required = {"sample_a", "sample_b", "weight"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"Edge table lacks {sorted(required - set(frame.columns))}")
    result: dict[tuple[str, str], float] = {}
    for row in frame.itertuples(index=False):
        pair = tuple(sorted((str(row.sample_a), str(row.sample_b))))
        if pair[0] == pair[1] or pair in result:
            raise SystemExit(f"Invalid or duplicate undirected edge: {pair}")
        weight = float(row.weight)
        if not math.isfinite(weight) or weight < 0:
            raise SystemExit(f"Invalid edge weight for {pair}: {weight}")
        result[pair] = weight
    return result


def variation_of_information(left: np.ndarray, right: np.ndarray) -> float:
    contingency = pd.crosstab(pd.Series(left), pd.Series(right)).to_numpy(dtype=float)
    pxy = contingency / contingency.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    denom = px[:, None] * py[None, :]
    mask = pxy > 0
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))
    return hx + hy - 2 * mi


def status(label: float | int | None) -> str:
    if label is None or pd.isna(label):
        return "ABSENT_FROM_GRAPH"
    return "UNASSIGNED" if int(label) < 0 else "ASSIGNED"


def match_communities(hist: pd.Series, minor: pd.Series, resolution: str) -> list[dict]:
    active = (hist >= 0) & (minor >= 0)
    h = sorted(map(int, hist[active].unique()))
    m = sorted(map(int, minor[active].unique()))
    if not h or not m:
        return []
    overlap = np.zeros((len(m), len(h)), dtype=int)
    for i, mc in enumerate(m):
        for j, hc in enumerate(h):
            overlap[i, j] = int(((minor == mc) & (hist == hc)).sum())
    rows, cols = linear_sum_assignment(-overlap)
    matches = []
    for i, j in zip(rows, cols):
        mc, hc = m[i], h[j]
        inter = int(overlap[i, j])
        minor_size = int((minor == mc).sum())
        hist_size = int((hist == hc).sum())
        union = minor_size + hist_size - inter
        matches.append({
            "resolution": resolution,
            "minor_community": mc,
            "historical_community": hc,
            "intersection": inter,
            "minor_size": minor_size,
            "historical_size": hist_size,
            "jaccard": inter / union if union else 1.0,
        })
    return matches


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cohort = list(map(str, load_json(args.cohort_summary).get("selected_samples", [])))
    if len(cohort) != 2619 or len(set(cohort)) != 2619:
        raise SystemExit(f"Expected 2619 unique canonical samples; observed {len(cohort)}")

    hist_graph = load_json(args.historical_dir / "graph_summary.json")
    minor_graph = load_json(args.minor_dir / "graph_summary.json")
    anchor_ok = all(hist_graph.get(k) == v for k, v in EXPECTED_GRAPH.items())
    if not anchor_ok:
        observed = {k: hist_graph.get(k) for k in EXPECTED_GRAPH}
        raise SystemExit(f"Historical 5 Mb graph anchor failed: {observed}")
    for key in ("min_edge_bp", "min_max_segment_bp", "weight_transform"):
        if minor_graph.get(key) != EXPECTED_GRAPH[key]:
            raise SystemExit(f"Minor graph parameter drift: {key}={minor_graph.get(key)}")

    hist_edges = edge_map(load_edges(args.historical_dir / "graph_edges.tsv.gz"))
    minor_edges = edge_map(load_edges(args.minor_dir / "graph_edges.tsv.gz"))
    new_edges = set(minor_edges) - set(hist_edges)
    increased = [p for p in set(minor_edges) & set(hist_edges)
                 if minor_edges[p] > hist_edges[p] + 1e-12]
    if new_edges or increased:
        raise SystemExit(
            f"Minor graph is not a weighted subgraph: new={len(new_edges)}, increased={len(increased)}"
        )

    hist = pd.read_csv(args.historical_dir / "leiden_assignments.tsv", sep="\t")
    minor = pd.read_csv(args.minor_dir / "leiden_assignments.tsv", sep="\t")
    if hist["sample_id"].duplicated().any() or minor["sample_id"].duplicated().any():
        raise SystemExit("Duplicate sample_id in Leiden assignments")
    universe = pd.DataFrame({"sample_id": cohort})
    hist = universe.merge(hist, on="sample_id", how="left", validate="one_to_one")
    minor = universe.merge(minor, on="sample_id", how="left", validate="one_to_one")

    transitions: list[dict] = []
    matches: list[dict] = []
    metrics: dict[str, dict] = {}
    for res in RESOLUTIONS:
        column = f"community_res_{res}"
        if column not in hist or column not in minor:
            raise SystemExit(f"Missing resolution column: {column}")
        h_status = hist[column].map(status)
        m_status = minor[column].map(status)
        counts = Counter(zip(h_status, m_status))
        transitions.extend({
            "resolution": res,
            "historical_status": h_status_value,
            "minor_status": minor_status_value,
            "n_samples": count,
        } for (h_status_value, minor_status_value), count in sorted(counts.items()))
        common = hist[column].notna() & minor[column].notna() & (hist[column] >= 0) & (minor[column] >= 0)
        h_active = hist.loc[common, column].astype(int).to_numpy()
        m_active = minor.loc[common, column].astype(int).to_numpy()
        metrics[res] = {
            "n_full_cohort": 2619,
            "historical": dict(Counter(h_status)),
            "minor": dict(Counter(m_status)),
            "n_common_assigned": int(common.sum()),
            "ari_common_assigned": (
                float(adjusted_rand_score(h_active, m_active)) if common.sum() > 1 else None
            ),
            "nmi_common_assigned": (
                float(normalized_mutual_info_score(h_active, m_active)) if common.sum() > 1 else None
            ),
            "vi_nats_common_assigned": (
                float(variation_of_information(h_active, m_active)) if common.sum() > 1 else None
            ),
        }
        matches.extend(match_communities(hist[column], minor[column], res))

    pd.DataFrame(transitions).to_csv(
        args.outdir / "m16_5_status_transitions.tsv", sep="\t", index=False
    )
    pd.DataFrame(matches).to_csv(
        args.outdir / "m16_5_community_matches.tsv", sep="\t", index=False
    )
    report = {
        "status": "PASS",
        "scope": "internal_stability_alt_vs_minor_no_biological_validation_no_test",
        "historical_graph_anchor": {"expected": EXPECTED_GRAPH, "observed": hist_graph, "pass": True},
        "minor_graph": minor_graph,
        "weighted_subgraph_gate": {
            "historical_edges": len(hist_edges),
            "minor_edges": len(minor_edges),
            "new_edges": 0,
            "increased_common_edge_weights": 0,
            "pass": True,
        },
        "partition_metrics_by_resolution": metrics,
        "primary_resolution": "1",
        "interpretation_limits": [
            "ARI/NMI/VI measure internal stability, not independent biological validation.",
            "Absent and unassigned samples are retained in the full-cohort transition table.",
            "Community identifiers were matched by overlap and have no intrinsic identity.",
        ],
    }
    (args.outdir / "m16_5_orientation_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
