#!/usr/bin/env python3
"""How much of the M27D donor count is a property of the data, and how much of a choice.

Everything here is computed from kinship pairs that already exist, so it costs nothing to
run and adds no PC-Relate pass.  It answers one question: if the retained set drops from
3685 individuals to roughly 3400, and Native American drops from 199 to about 100, how
much of that is the relatedness threshold, how much is the tie-break order, and how much
is the greedy construction settling for a maximal set instead of a maximum one.

What this can and cannot establish, stated up front because the numbers invite more:

* It **can** show how strongly the final composition depends on phi and on the selection
  algorithm.  A composition that swings with phi is not a fact about the panel.
* It **cannot** decide the threshold.  Choosing phi after seeing which value retains more
  Native American donors would be selecting the parameter on the outcome, which the
  preregistration forbids.  The thresholds reported here are the ones fixed in advance.
* It **cannot** attribute the loss.  An edge above phi is a kinship estimate, and in an
  isolated population recent pedigree relatedness, endogamy and population structure that
  eight components did not absorb all push it up.  Nothing in a single PC-Relate pass
  separates them.

The inbreeding coefficient is reported next to the retained and removed groups for the
same reason: it is a diagnostic that is *consistent with* endogamy raising kinship, not a
test that isolates it.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m27d_kinship_graph import (  # noqa: E402
    adjacency,
    maximal_independent_set,
    maximum_independent_set_size,
    open_text,
    read_call_rates,
    read_sample_universe,
    read_strata_table,
    stratum_coverage,
)


COVERAGE_COLUMNS = ("Ancestry", "Population", "Source", "Country")
TRUE_WORDS = {"true", "t", "1", "yes", "y"}


def contract_thresholds(contract: dict) -> list[float]:
    """The preregistered thresholds, primary included, never a value chosen here."""
    block = contract["pcrelate"]
    values = {float(block["primary_phi_threshold"])}
    values.update(float(value) for value in block.get("descriptive_phi_thresholds", []))
    return sorted(values)


def read_inbreeding(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with open_text(path) as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            try:
                values[record["ID"]] = float(record["f"])
            except (KeyError, TypeError, ValueError):
                continue
    return values


def describe(values: list[float]) -> dict[str, object]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    quantiles = statistics.quantiles(ordered, n=4) if len(ordered) >= 4 else [None, None, None]
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "q1": None if quantiles[0] is None else round(quantiles[0], 6),
        "q3": None if quantiles[2] is None else round(quantiles[2], 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
    }


def edge_homogeneity(
    edges: set[tuple[str, str]], strata: dict[str, dict[str, str]], column: str
) -> dict[str, object]:
    """Fraction of edges that stay inside one label, over an explicit denominator.

    Only edges whose two endpoints both have a resolved metadata row can be classified,
    so that count is reported rather than folded into the denominator.  A high fraction
    is what both a family and a small endogamous population produce; it does not
    distinguish them.
    """
    same = 0
    classifiable = 0
    for left, right in edges:
        left_row, right_row = strata.get(left), strata.get(right)
        if left_row is None or right_row is None:
            continue
        left_label = str(left_row.get(column, "")).strip()
        right_label = str(right_row.get(column, "")).strip()
        if not left_label or not right_label:
            continue
        classifiable += 1
        if left_label == right_label:
            same += 1
    return {
        "n_edges_classifiable": classifiable,
        "n_edges_within_same_label": same,
        "fraction_within_same_label": round(same / classifiable, 6) if classifiable else None,
    }


def evaluate_threshold(
    threshold: float,
    nodes: list[str],
    all_edges: list[tuple[tuple[str, str], float]],
    call_rate: dict[str, float],
    strata: dict[str, dict[str, str]],
    inbreeding: dict[str, float],
    max_component_nodes: int,
) -> dict[str, object]:
    edges = {pair for pair, kinship in all_edges if kinship >= threshold}
    graph = adjacency(edges)
    primary = maximal_independent_set(nodes, edges, call_rate)
    alternate = maximal_independent_set(nodes, edges, call_rate, descending_hash=True)
    exact = maximum_independent_set_size(nodes, edges, max_component_nodes=max_component_nodes)

    chosen = set(primary)
    removed = [node for node in nodes if node not in chosen]
    coverage = {
        column: stratum_coverage(primary, nodes, strata, column) for column in COVERAGE_COLUMNS
    }
    lost = {
        column: sorted(
            label
            for label, counts in coverage[column].items()
            if counts["available"] > 0 and counts["retained"] == 0
        )
        for column in COVERAGE_COLUMNS
    }
    return {
        "phi_threshold": threshold,
        "n_edges": len(edges),
        "n_samples_with_at_least_one_edge": len(graph),
        "n_retained_primary_order": len(primary),
        "n_retained_alternate_order": len(alternate),
        "n_shared_by_both_orders": len(chosen & set(alternate)),
        "n_symmetric_difference_between_orders": len(chosen ^ set(alternate)),
        "order_sensitivity_count_delta": len(primary) - len(alternate),
        "maximum_independent_set": exact,
        "greedy_shortfall_vs_maximum": (
            exact["size"] - len(primary) if exact["size"] is not None else None
        ),
        "stratum_coverage": coverage,
        "strata_with_no_survivor": lost,
        "inbreeding_retained": describe([inbreeding[n] for n in primary if n in inbreeding]),
        "inbreeding_removed": describe([inbreeding[n] for n in removed if n in inbreeding]),
        "edge_homogeneity": {
            column: edge_homogeneity(edges, strata, column) for column in ("Population", "Ancestry")
        },
    }


def write_threshold_table(rows: list[dict], path: Path) -> None:
    columns = [
        "phi_threshold",
        "n_edges",
        "n_samples_with_at_least_one_edge",
        "n_retained_primary_order",
        "n_retained_alternate_order",
        "n_shared_by_both_orders",
        "n_symmetric_difference_between_orders",
        "maximum_independent_set_size",
        "maximum_independent_set_is_exact",
        "greedy_shortfall_vs_maximum",
        "largest_component_nodes",
        "inbreeding_median_retained",
        "inbreeding_median_removed",
        "fraction_edges_within_same_population",
        "fraction_edges_within_same_ancestry",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "phi_threshold": row["phi_threshold"],
                    "n_edges": row["n_edges"],
                    "n_samples_with_at_least_one_edge": row["n_samples_with_at_least_one_edge"],
                    "n_retained_primary_order": row["n_retained_primary_order"],
                    "n_retained_alternate_order": row["n_retained_alternate_order"],
                    "n_shared_by_both_orders": row["n_shared_by_both_orders"],
                    "n_symmetric_difference_between_orders": row[
                        "n_symmetric_difference_between_orders"
                    ],
                    "maximum_independent_set_size": row["maximum_independent_set"]["size"],
                    "maximum_independent_set_is_exact": row["maximum_independent_set"]["exact"],
                    "greedy_shortfall_vs_maximum": row["greedy_shortfall_vs_maximum"],
                    "largest_component_nodes": row["maximum_independent_set"][
                        "largest_component_nodes"
                    ],
                    "inbreeding_median_retained": row["inbreeding_retained"].get("median"),
                    "inbreeding_median_removed": row["inbreeding_removed"].get("median"),
                    "fraction_edges_within_same_population": row["edge_homogeneity"]["Population"][
                        "fraction_within_same_label"
                    ],
                    "fraction_edges_within_same_ancestry": row["edge_homogeneity"]["Ancestry"][
                        "fraction_within_same_label"
                    ],
                }
            )


def write_coverage_table(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["phi_threshold", "column", "label", "available", "retained", "lost"])
        for row in rows:
            for column in COVERAGE_COLUMNS:
                for label, counts in row["stratum_coverage"][column].items():
                    writer.writerow(
                        [
                            row["phi_threshold"],
                            column,
                            label,
                            counts["available"],
                            counts["retained"],
                            counts["available"] - counts["retained"],
                        ]
                    )


def draw_figure(rows: list[dict], path: Path, focus: list[str]) -> str:
    """Optional. The tables are the result; the figure only makes the slope visible."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return f"not emitted: {error}"

    thresholds = [row["phi_threshold"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(thresholds, [row["n_retained_primary_order"] for row in rows], marker="o",
                 label="orden principal")
    axes[0].plot(thresholds, [row["n_retained_alternate_order"] for row in rows], marker="s",
                 linestyle="--", label="orden alterno")
    axes[0].plot(thresholds, [row["maximum_independent_set"]["size"] for row in rows], marker="^",
                 linestyle=":", label="máximo exacto")
    axes[0].set_xscale("log")
    axes[0].set_xticks(thresholds)
    axes[0].set_xticklabels([f"{value:g}" for value in thresholds])
    axes[0].set_xlabel("umbral φ")
    axes[0].set_ylabel("individuos retenidos")
    axes[0].set_title("Retención total")
    axes[0].legend(fontsize=8)

    for label in focus:
        retained, available = [], None
        for row in rows:
            counts = row["stratum_coverage"]["Ancestry"].get(label)
            if counts is None:
                retained.append(None)
                continue
            available = counts["available"]
            retained.append(100.0 * counts["retained"] / counts["available"])
        if available:
            axes[1].plot(thresholds, retained, marker="o", label=f"{label} (n={available})")
    axes[1].set_xscale("log")
    axes[1].set_xticks(thresholds)
    axes[1].set_xticklabels([f"{value:g}" for value in thresholds])
    axes[1].set_xlabel("umbral φ")
    axes[1].set_ylabel("% del estrato retenido")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Retención por ancestría")
    axes[1].legend(fontsize=8)

    figure.suptitle(
        "M27D · sensibilidad de la composición al umbral y al orden de selección "
        "(diagnóstico, no elección de umbral)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return "emitted"


def run(args: argparse.Namespace) -> dict[str, object]:
    contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if contract.get("pcrelate", {}).get("king_allowed"):
        raise SystemExit("M27D forbids KING")
    thresholds = contract_thresholds(contract)

    nodes = read_sample_universe(args.samples)
    if len(set(nodes)) != len(nodes):
        raise SystemExit("Sample universe contains duplicate identifiers")
    call_rate = read_call_rates(args.call_rates)
    strata = read_strata_table(args.strata)
    inbreeding = read_inbreeding(args.inbreeding)

    lowest = min(thresholds)
    # The pair table was written at a reporting threshold. Evaluating below it would run
    # on a truncated graph and silently overstate every retained count.
    reported_at = float(args.reported_threshold) if args.reported_threshold is not None else None
    if reported_at is not None and lowest < reported_at:
        raise SystemExit(
            f"Pairs were reported at phi>={reported_at}; {lowest} cannot be evaluated from them"
        )

    ranked = list(_pairs_with_kinship(args.pairs, lowest))
    node_set = set(nodes)
    external = {node for (edge, _kin) in ranked for node in edge} - node_set
    if external:
        raise SystemExit(f"{len(external)} kinship identifiers are outside the sample universe")
    rows = [
        evaluate_threshold(
            threshold, nodes, ranked, call_rate, strata, inbreeding, args.max_component_nodes
        )
        for threshold in thresholds
    ]

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    write_threshold_table(rows, args.out_thresholds)
    write_coverage_table(rows, args.out_coverage)
    figure_status = "not requested"
    if args.out_figure is not None:
        figure_status = draw_figure(rows, args.out_figure, args.figure_ancestries)

    summary = {
        "stage": "M27D_KINSHIP_THRESHOLD_SENSITIVITY",
        "scientific_result": False,
        "derived_from_existing_pairs_only": True,
        "king_executed": False,
        "pcair_used": False,
        "new_pcrelate_pass_executed": False,
        "n_samples": len(nodes),
        "phi_thresholds": thresholds,
        "thresholds_read_from_preregistration": True,
        "threshold_selected_by_this_analysis": False,
        "figure": figure_status,
        "sample_ids_emitted": False,
        "interpretation_limits": [
            "Shows dependence on phi and on the selection order; does not choose phi.",
            "An edge above phi does not separate recent pedigree relatedness from endogamy "
            "or from population structure the components did not absorb.",
            "The inbreeding coefficient is reported as a concomitant diagnostic, not as a "
            "test that isolates endogamy.",
            "The maximum independent set is measured to price the greedy choice; it never "
            "feeds a selection, because a retained count must not be a function of a search.",
        ],
        "by_threshold": rows,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _pairs_with_kinship(path: Path, threshold: float):
    """Edges kept once with their kinship, so every threshold reuses one read."""
    with open_text(path) as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            try:
                kinship = float(record["kin"])
            except (TypeError, ValueError, KeyError):
                continue
            if kinship < threshold:
                continue
            left, right = record["ID1"], record["ID2"]
            if left == right:
                continue
            yield ((left, right) if left <= right else (right, left)), kinship


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--call-rates", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--inbreeding", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--reported-threshold", type=float, default=None)
    parser.add_argument("--max-component-nodes", type=int, default=60)
    parser.add_argument("--figure-ancestries", nargs="*", default=["Native_American", "European"])
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-thresholds", type=Path, required=True)
    parser.add_argument("--out-coverage", type=Path, required=True)
    parser.add_argument("--out-figure", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
