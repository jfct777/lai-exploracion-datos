#!/usr/bin/env python3
"""Narrow down *why* whole populations disappear from the M27D donor pool.

Three explanations competed for the loss of, for instance, 12 of 13 Pima at phi>=0.0442:
recent pedigree relatedness, endogamy and population structure the components never
absorbed, and the behaviour of the greedy selection.  Everything here runs on artifacts
pass0 already produced, so it separates what can be separated for free and says plainly
what it cannot separate at all.

Three readings, each answering a different question:

``structural_ceiling``
    The exact maximum independent set *inside each population*.  When a population's
    kinship graph is a complete clique, no selection algorithm retains more than one
    member, and blaming the greedy order is arithmetic nonsense.  This either exonerates
    the algorithm per population or names the individuals it actually costs.

``pedigree_locus``
    Whether an edge carries the IBS0 deficit that real identity-by-descent sharing
    implies.  With no IBD2, pedigree relatedness satisfies k0 = 1 - 4*phi.  An estimate
    inflated by a mis-specified allele frequency raises phi while leaving k0 near one, so
    the residual k0 - (1 - 4*phi) separates genuine sharing from an estimation artifact.
    It does **not** separate recent pedigree from old drift: both are real sharing, and
    only segment *length*, which a moment estimator never sees, distinguishes their time
    depth.  That question stays open here by construction.

``axis_degeneracy``
    The participation ratio of every principal component, next to the populations that
    carry it.  This is the calibration the preregistration demands before any donor is
    certified, and it is also a preflight: an axis carried by a handful of individuals
    trips gate G2, and a configuration that would fail it should be known before a
    machine is paid for rather than after.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m27d_kinship_graph import (  # noqa: E402
    maximal_independent_set,
    maximum_independent_set_size,
    open_text,
    read_call_rates,
    read_sample_universe,
    read_strata_table,
)


def read_pairs_with_ibd(path: Path, threshold: float) -> list[dict[str, object]]:
    required = ("ID1", "ID2", "kin", "k0", "k2")
    records: list[dict[str, object]] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                "The pair table lacks the IBD probabilities this diagnostic needs "
                f"({', '.join(missing)}); it must be produced with ibd.probs=TRUE"
            )
        for row in reader:
            kinship = float(row["kin"])
            if kinship < threshold:
                continue
            records.append(
                {
                    "ID1": row["ID1"],
                    "ID2": row["ID2"],
                    "kin": kinship,
                    "k0": float(row["k0"]),
                    "k2": float(row["k2"]),
                }
            )
    return records


def structural_ceiling(
    pairs: list[dict],
    nodes: list[str],
    strata: dict[str, dict[str, str]],
    call_rate: dict[str, float],
    column: str,
    min_members: int,
    max_component_nodes: int,
) -> dict[str, object]:
    """Exact alpha(G[P]) per population, against what the global selection retained."""
    edges = {
        (row["ID1"], row["ID2"]) if row["ID1"] <= row["ID2"] else (row["ID2"], row["ID1"])
        for row in pairs
    }
    retained = set(maximal_independent_set(nodes, edges, call_rate))
    by_label: dict[str, list[str]] = defaultdict(list)
    for sample in nodes:
        row = strata.get(sample)
        label = (str(row.get(column, "")).strip() if row else "") or "(unlabelled)"
        by_label[label].append(sample)

    populations = {}
    ceiling_total = 0
    retained_total = 0
    for label, members in sorted(by_label.items()):
        if len(members) < min_members:
            continue
        member_set = set(members)
        induced = {edge for edge in edges if edge[0] in member_set and edge[1] in member_set}
        if not induced:
            continue
        exact = maximum_independent_set_size(
            members, induced, max_component_nodes=max_component_nodes
        )
        kept = len(member_set & retained)
        possible = len(members) * (len(members) - 1) / 2
        ceiling_total += exact["size"]
        retained_total += kept
        populations[label] = {
            "n_members": len(members),
            "n_internal_edges": len(induced),
            "edge_density": round(len(induced) / possible, 6) if possible else None,
            "alpha_exact": exact["size"],
            "alpha_is_exact": exact["exact"],
            "n_retained_by_global_selection": kept,
            # A population can retain fewer than its own ceiling because of edges to other
            # populations, so the shortfall is an upper bound on what the order costs here.
            "shortfall_vs_local_ceiling": exact["size"] - kept,
            "is_complete_clique": len(induced) == possible,
        }
    cliques = sorted(label for label, row in populations.items() if row["is_complete_clique"])
    return {
        "column": column,
        "min_members_reported": min_members,
        "n_populations_evaluated": len(populations),
        "sum_alpha_exact": ceiling_total,
        "sum_retained": retained_total,
        "total_shortfall_vs_local_ceilings": ceiling_total - retained_total,
        "populations_that_are_complete_cliques": cliques,
        "n_populations_that_are_complete_cliques": len(cliques),
        "by_population": populations,
    }


def pedigree_locus(
    pairs: list[dict],
    strata: dict[str, dict[str, str]],
    lower: float,
    upper: float,
) -> dict[str, object]:
    """Residual against k0 = 1 - 4*phi, the no-IBD2 pedigree line.

    Restricted to a kinship band because the line only holds where IBD2 is negligible; a
    first-degree pair legitimately sits off it and would be read as inflation.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pairs:
        if not lower <= row["kin"] < upper:
            continue
        left, right = strata.get(row["ID1"]), strata.get(row["ID2"])
        if left is None or right is None:
            continue
        same = str(left.get("Population", "")).strip() == str(right.get("Population", "")).strip()
        groups["within_population" if same else "between_populations"].append(row)

    report = {}
    for label, rows in sorted(groups.items()):
        residuals = sorted(row["k0"] - (1.0 - 4.0 * row["kin"]) for row in rows)
        if not residuals:
            continue
        report[label] = {
            "n_edges": len(rows),
            "residual_median": round(statistics.median(residuals), 6),
            "residual_q1": round(residuals[len(residuals) // 4], 6),
            "residual_q3": round(residuals[3 * len(residuals) // 4], 6),
            "fraction_residual_above_0_10": round(
                sum(1 for value in residuals if value > 0.10) / len(residuals), 6
            ),
            "fraction_with_ibd2_above_0_05": round(
                sum(1 for row in rows if row["k2"] > 0.05) / len(rows), 6
            ),
        }
    classified = sum(len(rows) for rows in groups.values())
    return {
        "kinship_band": [lower, upper],
        "n_edges_in_band": classified,
        "n_edges_outside_band": len([row for row in pairs if not lower <= row["kin"] < upper]),
        "expected_k0_without_ibd2": "1 - 4*phi",
        "by_group": report,
        "what_a_positive_residual_would_mean": (
            "phi raised without the matching IBS0 deficit, which is the signature of a "
            "mis-specified allele frequency rather than of shared descent"
        ),
        "what_this_cannot_decide": (
            "Recent pedigree relatedness and old endogamy both produce a genuine IBS0 "
            "deficit and both sit on this line. Separating them needs IBD segment lengths."
        ),
    }


def axis_degeneracy(
    scores_path: Path,
    strata: dict[str, dict[str, str]],
    floor_fraction: float,
    column: str,
    top_labels: int,
) -> dict[str, object]:
    with open_text(scores_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        components = [name for name in (reader.fieldnames or []) if name.startswith("PC")]
        if not components:
            raise SystemExit("The score table has no PC columns")
        samples: list[str] = []
        columns: list[list[float]] = [[] for _ in components]
        for row in reader:
            samples.append(row["sample_id"])
            for index, name in enumerate(components):
                columns[index].append(float(row[name]))

    floor = floor_fraction * len(samples)
    axes = {}
    for name, values in zip(components, columns):
        total = sum(value * value for value in values)
        if total <= 0:
            axes[name] = {"participation_ratio": None, "status": "NOT_EVALUATED"}
            continue
        weights = [(value * value) / total for value in values]
        ratio = 1.0 / sum(weight * weight for weight in weights)
        carried: dict[str, float] = defaultdict(float)
        for sample, weight in zip(samples, weights):
            row = strata.get(sample)
            label = (str(row.get(column, "")).strip() if row else "") or "(unlabelled)"
            carried[label] += weight
        leaders = sorted(carried.items(), key=lambda item: -item[1])[:top_labels]
        axes[name] = {
            "participation_ratio": round(ratio, 3),
            "status": "PASS" if ratio >= floor else "FAIL",
            "carried_by": [
                {"label": label, "weight_fraction": round(weight, 6)} for label, weight in leaders
            ],
        }

    failing = sorted(name for name, row in axes.items() if row.get("status") == "FAIL")
    prefixes = {}
    for count in range(1, len(components) + 1):
        subset = components[:count]
        prefixes[count] = "FAIL" if any(axes[name]["status"] == "FAIL" for name in subset) else "PASS"
    return {
        "n_samples": len(samples),
        "floor_fraction": floor_fraction,
        "floor_effective_individuals": round(floor, 3),
        "by_axis": axes,
        "failing_axes": failing,
        "gate_status_by_leading_prefix": prefixes,
        "note": (
            "Evaluated on the score table supplied. Gate G2 in the audit is applied to the "
            "leading prefix a configuration uses, so a prefix marked FAIL here is a "
            "configuration that would abort before PC-Relate."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if contract.get("pcrelate", {}).get("king_allowed"):
        raise SystemExit("M27D forbids KING")
    threshold = float(contract["pcrelate"]["primary_phi_threshold"])
    floor_fraction = float(
        contract["pca_axis_contract"]["min_effective_individual_fraction_per_axis"]
    )

    nodes = read_sample_universe(args.samples)
    strata = read_strata_table(args.strata)
    call_rate = read_call_rates(args.call_rates)
    pairs = read_pairs_with_ibd(args.pairs, threshold)

    summary = {
        "stage": "M27D_KINSHIP_ATTRIBUTION",
        "scientific_result": False,
        "derived_from_existing_artifacts_only": True,
        "new_pcrelate_pass_executed": False,
        "king_executed": False,
        "pcair_used": False,
        "phi_threshold": threshold,
        "structural_ceiling": structural_ceiling(
            pairs, nodes, strata, call_rate, args.population_column,
            args.min_population_members, args.max_component_nodes,
        ),
        "pedigree_locus": pedigree_locus(pairs, strata, threshold, args.ibd2_band_upper),
        "sample_ids_emitted": False,
        "interpretation_limits": [
            "The structural ceiling exonerates or convicts the selection order only; it "
            "says nothing about why the edges exist.",
            "The pedigree locus separates real sharing from an estimation artifact, never "
            "recent pedigree from old endogamy.",
            "Axis degeneracy is a property of the score table supplied, and the pass0 fit "
            "included relatives, so a refit on the training set can move it.",
        ],
    }
    if args.pca_scores is not None:
        summary["axis_degeneracy"] = axis_degeneracy(
            args.pca_scores, strata, floor_fraction, args.population_column, args.top_labels
        )

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_population_table(summary["structural_ceiling"], args.out_table)
    return summary


def write_population_table(ceiling: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["population", "n_members", "n_internal_edges", "edge_density", "alpha_exact",
             "n_retained", "shortfall_vs_local_ceiling", "is_complete_clique"]
        )
        for label, row in sorted(
            ceiling["by_population"].items(),
            key=lambda item: (item[1]["n_retained_by_global_selection"] / item[1]["n_members"]),
        ):
            writer.writerow([
                label, row["n_members"], row["n_internal_edges"], row["edge_density"],
                row["alpha_exact"], row["n_retained_by_global_selection"],
                row["shortfall_vs_local_ceiling"], row["is_complete_clique"],
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--call-rates", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--pca-scores", type=Path, default=None)
    parser.add_argument("--population-column", default="Population")
    parser.add_argument("--min-population-members", type=int, default=4)
    parser.add_argument("--max-component-nodes", type=int, default=60)
    parser.add_argument("--top-labels", type=int, default=3)
    parser.add_argument(
        "--ibd2-band-upper",
        type=float,
        default=0.177,
        help="upper edge of the band where k0 = 1 - 4*phi holds, i.e. below first degree",
    )
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-table", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
