#!/usr/bin/env python3
"""Compare two PC-Relate configurations that differ only in the number of components.

The question is narrow on purpose: if the ancestry adjustment uses twelve principal
components instead of eight, does the relatedness graph, and therefore the surviving
donor panel, change materially?  Everything else is held fixed — the same individuals,
the same markers, one shared PCA fit sliced at eight and at twelve, the same seed, the
same thread count, the same thresholds.  The first thing this script does is *verify*
that, and refuse to compare if anything besides the component count differs.  A
one-factor claim that nobody checked is just a claim.

What this can establish: how sensitive the edges and the retained composition are to the
number of components.  What it cannot: whether an edge is recent pedigree relatedness or
population structure the components never absorbed.  Both arms use the same estimator on
the same data, so agreement between them is evidence of insensitivity to this knob, not
evidence that either answer is right.

On coverage of the pair space.  Each arm publishes pairs at or above the lowest
preregistered threshold, 0.0221.  That truncation costs nothing for this comparison: a
pair can only cross a preregistered threshold, all of which are at or above 0.0221, if it
reaches 0.0221 in at least one arm, so the union of the two tails contains every crossing
by construction.  The marginal counts over all 6,787,770 pairs come from each arm's own
summary, which counted them before truncating.
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
    maximal_independent_set,
    maximum_independent_set_size,
    open_text,
    read_call_rates,
    read_sample_universe,
    read_strata_table,
    stratum_coverage,
)


# Held fixed by design.  Anything here that differs between the arms means more than one
# factor moved and the comparison is not interpretable, so it aborts instead of reporting.
INVARIANT_FIELDS = (
    "n_eligible_samples",
    "n_markers",
    "n_training_samples",
    "n_pairs_total",
    "random_seed",
    "threads",
    "ld_r2_max",
    "report_threshold",
)
VARYING_FIELD = "n_pcs"
COVERAGE_COLUMNS = ("Ancestry", "Population")


def parse_labelled(values: list[str], flag: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} expects configuration_id=path, got {value!r}")
        key, path = value.split("=", 1)
        mapping[key] = Path(path)
    return mapping


def read_kinship(path: Path) -> dict[tuple[str, str], float]:
    pairs: dict[tuple[str, str], float] = {}
    with open_text(path) as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            left, right = record["ID1"], record["ID2"]
            key = (left, right) if left <= right else (right, left)
            pairs[key] = float(record["kin"])
    return pairs


def read_f(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with open_text(path) as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            try:
                values[record["ID"]] = float(record["f"])
            except (KeyError, TypeError, ValueError):
                continue
    return values


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2 + 1
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    return pearson(ranks(left), ranks(right))


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    left_ss = sum((a - mean_left) ** 2 for a in left)
    right_ss = sum((b - mean_right) ** 2 for b in right)
    if left_ss <= 0 or right_ss <= 0:
        return None
    return numerator / (left_ss * right_ss) ** 0.5


def _round(value: float | None, digits: int = 8) -> float | None:
    return None if value is None else round(value, digits)


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def relative_change(reference: float, other: float) -> float | None:
    """Signed change of the second arm against the first, which is the primary one."""
    if reference == 0:
        return None
    return (other - reference) / reference


def selection_for(
    kinship: dict[tuple[str, str], float],
    threshold: float,
    nodes: list[str],
    call_rate: dict[str, float],
    max_component_nodes: int,
) -> dict[str, object]:
    edges = {pair for pair, value in kinship.items() if value >= threshold}
    primary = maximal_independent_set(nodes, edges, call_rate)
    alternate = maximal_independent_set(nodes, edges, call_rate, descending_hash=True)
    exact = maximum_independent_set_size(nodes, edges, max_component_nodes=max_component_nodes)
    return {
        "edges": edges,
        "primary": primary,
        "n_primary": len(primary),
        "n_alternate": len(alternate),
        "maximum_independent_set": exact,
    }


def compare_threshold(
    threshold: float,
    arms: list[str],
    kinship: dict[str, dict[tuple[str, str], float]],
    summaries: dict[str, dict],
    nodes: list[str],
    call_rate: dict[str, float],
    strata: dict[str, dict[str, str]],
    max_component_nodes: int,
) -> dict[str, object]:
    reference, other = arms
    selections = {
        arm: selection_for(kinship[arm], threshold, nodes, call_rate, max_component_nodes)
        for arm in arms
    }
    edges_ref = selections[reference]["edges"]
    edges_other = selections[other]["edges"]
    union = edges_ref | edges_other
    intersection = edges_ref & edges_other

    # The whole-set counts come from each arm's own summary, so they cover all pairs and
    # not only the published tail.
    key = f"phi_ge_{threshold:.4f}".rstrip("0").rstrip(".")
    full_counts = {
        arm: summaries[arm].get("pair_counts_by_threshold", {}).get(key) for arm in arms
    }

    retained = {arm: set(selections[arm]["primary"]) for arm in arms}
    flipped = retained[reference] ^ retained[other]

    coverage = {
        column: {
            arm: stratum_coverage(selections[arm]["primary"], nodes, strata, column)
            for arm in arms
        }
        for column in COVERAGE_COLUMNS
    }
    coverage_delta = {
        column: {
            label: coverage[column][other].get(label, {}).get("retained", 0)
            - counts.get("retained", 0)
            for label, counts in coverage[column][reference].items()
        }
        for column in COVERAGE_COLUMNS
    }

    return {
        "phi_threshold": threshold,
        "n_edges_in_published_tail": {arm: len(selections[arm]["edges"]) for arm in arms},
        "n_edges_over_all_pairs": full_counts,
        "n_edges_union": len(union),
        "n_edges_shared": len(intersection),
        "jaccard_of_edge_sets": round(len(intersection) / len(union), 6) if union else None,
        "n_edges_only_in_reference": len(edges_ref - edges_other),
        "n_edges_only_in_other": len(edges_other - edges_ref),
        "n_retained_primary_order": {arm: selections[arm]["n_primary"] for arm in arms},
        "n_retained_alternate_order": {arm: selections[arm]["n_alternate"] for arm in arms},
        "maximum_independent_set": {
            arm: selections[arm]["maximum_independent_set"] for arm in arms
        },
        "n_individuals_changing_retention_status": len(flipped),
        "relative_change_in_edges": relative_change(
            len(edges_ref), len(edges_other)
        ),
        "relative_change_in_retained": relative_change(
            selections[reference]["n_primary"], selections[other]["n_primary"]
        ),
        "stratum_coverage": coverage,
        "stratum_retained_delta": coverage_delta,
    }


def compare_kinship_values(
    arms: list[str], kinship: dict[str, dict[tuple[str, str], float]]
) -> dict[str, object]:
    reference, other = arms
    shared = sorted(set(kinship[reference]) & set(kinship[other]))
    left = [kinship[reference][pair] for pair in shared]
    right = [kinship[other][pair] for pair in shared]
    deltas = [b - a for a, b in zip(left, right)]
    absolute = [abs(value) for value in deltas]
    return {
        "n_pairs_in_both_tails": len(shared),
        "n_pairs_only_in_reference_tail": len(set(kinship[reference]) - set(kinship[other])),
        "n_pairs_only_in_other_tail": len(set(kinship[other]) - set(kinship[reference])),
        # Both estimators are undefined on too few pairs or on a constant arm, and a
        # comparison that reports a correlation it could not compute is worse than one
        # that says so.
        "pearson_r": _round(pearson(left, right)),
        "spearman_rho": _round(spearman(left, right)),
        "delta_phi_median": None if not deltas else round(statistics.median(deltas), 8),
        "delta_phi_mean": None if not deltas else round(statistics.fmean(deltas), 8),
        "absolute_delta_phi_median": None if not absolute else round(statistics.median(absolute), 8),
        "absolute_delta_phi_q90": round(quantile(absolute, 0.90), 8) if absolute else None,
        "absolute_delta_phi_q99": round(quantile(absolute, 0.99), 8) if absolute else None,
        "absolute_delta_phi_max": round(max(absolute), 8) if absolute else None,
        "note": (
            "Computed on the union of the published tails at phi>=0.0221, which contains "
            "every crossing of every preregistered threshold. It is not the correlation "
            "over all pairs, where most of the mass sits near zero and would inflate it."
        ),
    }


def compare_inbreeding(arms: list[str], inbreeding: dict[str, dict[str, float]]) -> dict[str, object]:
    reference, other = arms
    shared = sorted(set(inbreeding[reference]) & set(inbreeding[other]))
    left = [inbreeding[reference][sample] for sample in shared]
    right = [inbreeding[other][sample] for sample in shared]
    absolute = [abs(b - a) for a, b in zip(left, right)]
    return {
        "n_samples": len(shared),
        "pearson_r": _round(pearson(left, right)),
        "absolute_delta_f_median": round(statistics.median(absolute), 8) if absolute else None,
        "absolute_delta_f_q90": round(quantile(absolute, 0.90), 8) if absolute else None,
        "absolute_delta_f_max": round(max(absolute), 8) if absolute else None,
        "median_f": {arm: round(statistics.median(inbreeding[arm].values()), 8) for arm in arms},
        "note": (
            "f is estimated by the same PC-Relate call whose component count is being "
            "varied, so it is a concomitant readout of the same fit, never an independent "
            "check on it."
        ),
    }


def stratum_stability(
    threshold_rows: list[dict], arms: list[str], label: str, column: str
) -> dict[str, object]:
    rows = {}
    for row in threshold_rows:
        coverage = row["stratum_coverage"][column]
        entry = {arm: coverage[arm].get(label) for arm in arms}
        if any(value is None for value in entry.values()):
            continue
        rows[str(row["phi_threshold"])] = {
            "available": entry[arms[0]]["available"],
            "retained": {arm: entry[arm]["retained"] for arm in arms},
            "delta": entry[arms[1]]["retained"] - entry[arms[0]]["retained"],
        }
    return rows


def verify_one_factor(arms: list[str], summaries: dict[str, dict]) -> dict[str, object]:
    reference, other = arms
    differing = {}
    for field in INVARIANT_FIELDS:
        left, right = summaries[reference].get(field), summaries[other].get(field)
        if left != right:
            differing[field] = {reference: left, other: right}
    if differing:
        raise SystemExit(
            "The two configurations differ in more than the component count: "
            + json.dumps(differing, sort_keys=True)
        )
    if summaries[reference].get(VARYING_FIELD) == summaries[other].get(VARYING_FIELD):
        raise SystemExit("Both configurations use the same number of components")
    for arm in arms:
        if summaries[arm].get("king_executed") or summaries[arm].get("pcair_used"):
            raise SystemExit(f"{arm} reports a forbidden KING or PC-AiR execution")
        if not summaries[arm].get("training_set_reused_from_pass0"):
            raise SystemExit(f"{arm} did not reuse the pass0 training set")
    return {
        "one_factor_verified": True,
        "held_fixed": {field: summaries[reference][field] for field in INVARIANT_FIELDS},
        "n_pcs": {arm: summaries[arm][VARYING_FIELD] for arm in arms},
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if contract.get("pcrelate", {}).get("king_allowed"):
        raise SystemExit("M27D forbids KING")
    thresholds = sorted(
        {float(contract["pcrelate"]["primary_phi_threshold"])}
        | {float(v) for v in contract["pcrelate"].get("descriptive_phi_thresholds", [])}
    )

    pair_files = parse_labelled(args.pairs, "--pairs")
    f_files = parse_labelled(args.inbreeding, "--inbreeding")
    summaries = {}
    for path in args.summaries:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        summaries[payload["configuration_id"]] = payload

    arms = [args.reference_configuration]
    arms += sorted(set(pair_files) - {args.reference_configuration})
    if len(arms) != 2:
        raise SystemExit(f"Expected exactly two configurations, got {sorted(pair_files)}")
    missing = [arm for arm in arms if arm not in summaries or arm not in f_files]
    if missing:
        raise SystemExit(f"Missing summary or inbreeding table for: {', '.join(missing)}")

    guard = verify_one_factor(arms, summaries)

    nodes = read_sample_universe(args.samples)
    call_rate = read_call_rates(args.call_rates)
    strata = read_strata_table(args.strata)
    kinship = {arm: read_kinship(pair_files[arm]) for arm in arms}
    inbreeding = {arm: read_f(f_files[arm]) for arm in arms}

    rows = [
        compare_threshold(
            threshold, arms, kinship, summaries, nodes, call_rate, strata, args.max_component_nodes
        )
        for threshold in thresholds
    ]

    primary_threshold = float(contract["pcrelate"]["primary_phi_threshold"])
    primary_row = next(row for row in rows if row["phi_threshold"] == primary_threshold)

    summary = {
        "stage": "M27D_PC_COUNT_SENSITIVITY",
        "scientific_result": False,
        "king_executed": False,
        "pcair_used": False,
        "reference_configuration": arms[0],
        "compared_configuration": arms[1],
        "one_factor_guard": guard,
        "kinship_agreement": compare_kinship_values(arms, kinship),
        "inbreeding_agreement": compare_inbreeding(arms, inbreeding),
        "by_threshold": rows,
        "primary_threshold": primary_threshold,
        "primary_relative_change_in_edges": primary_row["relative_change_in_edges"],
        "primary_relative_change_in_retained": primary_row["relative_change_in_retained"],
        "native_american_stability": stratum_stability(rows, arms, "Native_American", "Ancestry"),
        "administrative_nam_candidates": {
            "status": "NOT_EVALUATED_MISSING_BASELINE_IDENTITY",
            "reason": (
                "The 173 administrative NAM candidates are the panel-and-metadata NAM samples "
                "that are disjoint from the frozen baseline. Disjointness is decided by the "
                "baseline identity audit, which has not run, and M27 published only aggregate "
                "counts, never the identifiers. Reconstructing the set from metadata alone "
                "would silently substitute a different denominator for it."
            ),
        },
        "sample_ids_emitted": False,
        "interpretation_limits": [
            "Agreement between the arms shows insensitivity to the component count only.",
            "Neither arm is a ground truth; both use the same estimator on the same data.",
            "Kinship agreement is measured on the union of the published tails, not on all pairs.",
            "f comes from the same fit whose component count varies, so it is concomitant.",
        ],
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_table(rows, arms, args.out_table)
    return summary


def write_table(rows: list[dict], arms: list[str], path: Path) -> None:
    reference, other = arms
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["phi_threshold", f"edges_{reference}", f"edges_{other}", "jaccard_of_edge_sets",
             "edges_only_reference", "edges_only_other", f"retained_{reference}",
             f"retained_{other}", "individuals_changing_status", "relative_change_in_edges",
             "relative_change_in_retained"]
        )
        for row in rows:
            writer.writerow([
                row["phi_threshold"],
                row["n_edges_in_published_tail"][reference],
                row["n_edges_in_published_tail"][other],
                row["jaccard_of_edge_sets"],
                row["n_edges_only_in_reference"],
                row["n_edges_only_in_other"],
                row["n_retained_primary_order"][reference],
                row["n_retained_primary_order"][other],
                row["n_individuals_changing_retention_status"],
                row["relative_change_in_edges"],
                row["relative_change_in_retained"],
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", required=True, metavar="ID=PATH")
    parser.add_argument("--inbreeding", nargs="+", required=True, metavar="ID=PATH")
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--reference-configuration", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--call-rates", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--max-component-nodes", type=int, default=60)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-table", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
