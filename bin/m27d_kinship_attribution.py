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

``k0_algebraic_identity``
    An internal consistency check on the pair table, and **not** evidence about descent.
    Below first degree GENESIS does not estimate k0 independently: ``correctK0`` replaces
    it with ``1 - 4*kin + k2`` for every pair under the cutoff.  Comparing k0 against
    ``1 - 4*phi`` inside that band therefore recovers k2 and nothing else, so it cannot
    tell a real shared ancestor from an artifact of the fitted allele frequencies.  The
    block is kept because the report has to carry that warning where the numbers are, and
    because counting the edges that sit *above* the cutoff says how much of the table has
    an independently estimated k0 at all.

``axis_localization``
    The participation ratio of every principal component, next to the populations that
    carry it and the share carried by samples with no metadata row.  This is the
    calibration the preregistration demands before any donor is certified.  It measures
    how concentrated an axis is and stops there: a localised axis may be a family, a small
    differentiated population, an isolate, a technical group or a bookkeeping gap, and
    PC-Relate needs the axes that describe small differentiated populations.  Below the
    contract fraction an axis is marked ``REVIEW``, never ``FAIL``.
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
    is_interpretable,
    maximal_independent_set,
    maximum_independent_set_size,
    open_text,
    read_call_rates,
    read_sample_universe,
    read_strata_table,
)

# GENESIS substitutes k0 below first degree instead of estimating it: correctK0() runs
# `k0 := 1 - 4*kin + k2` for every pair with kin < 2^(-5/2), and pcrelate() calls it
# whenever ibd.probs=TRUE.  Read in the source of the exact version this module runs, and
# confirmed on the pass0 table, where the identity holds to 3e-15 over 2422 pairs.
GENESIS_K0_SUBSTITUTION_CUTOFF = 2.0**-2.5


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


def k0_algebraic_identity(
    pairs: list[dict],
    strata: dict[str, dict[str, str]],
    lower: float,
    upper: float,
) -> dict[str, object]:
    """Check the substitution GENESIS imposes on k0, and label it as non-independent.

    ``correctK0`` sets ``k0 := 1 - 4*kin + k2`` for every pair below the cutoff, and
    ``pcrelate`` calls it whenever ``ibd.probs=TRUE``.  Everything reported here is that
    identity or a restatement of k2; none of it is an independent observation about
    descent.  Pairs at or above the cutoff keep the estimator's own k0 and are counted
    apart, because those are the only rows where k0 carries information the identity did
    not put there.
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
        substituted = [row for row in rows if row["kin"] < GENESIS_K0_SUBSTITUTION_CUTOFF]
        k2_values = sorted(row["k2"] for row in rows)
        report[label] = {
            "n_edges": len(rows),
            "n_edges_with_k0_substituted_by_genesis": len(substituted),
            "n_edges_with_independently_estimated_k0": len(rows) - len(substituted),
            # k2 is reported under its own name.  Reporting it as a "residual against the
            # pedigree line" was the error this block now documents: inside the band the
            # two are the same number.
            "k2_median": round(statistics.median(k2_values), 6),
            "k2_q1": round(k2_values[len(k2_values) // 4], 6),
            "k2_q3": round(k2_values[3 * len(k2_values) // 4], 6),
            "fraction_k2_above_0_05": round(
                sum(1 for value in k2_values if value > 0.05) / len(k2_values), 6
            ),
        }

    in_band = [row for row in pairs if lower <= row["kin"] < upper]
    substituted = [row for row in in_band if row["kin"] < GENESIS_K0_SUBSTITUTION_CUTOFF]
    deviations = [
        abs(row["k0"] - (1.0 - 4.0 * row["kin"] + row["k2"])) for row in substituted
    ]
    worst = max(deviations) if deviations else None
    return {
        "diagnostic_class": "NON_INDEPENDENT_DIAGNOSTIC",
        "evidence_status": "NOT_EVIDENCE_OF_IBD",
        "warning": (
            "Inside this band k0 was not estimated independently: GENESIS derived it from "
            "kin and k2 through correctK0. Any comparison of k0 against 1 - 4*phi here "
            "returns k2 by construction and says nothing about identity by descent."
        ),
        "genesis_rule": "k0 := 1 - 4*kin + k2 for kin < 2^(-5/2)",
        "genesis_rule_cutoff": GENESIS_K0_SUBSTITUTION_CUTOFF,
        "genesis_rule_source": "GENESIS R/pcrelate.R, correctK0(), called when ibd.probs=TRUE",
        "kinship_band": [lower, upper],
        "n_edges_in_band": len(in_band),
        "n_edges_outside_band": len(pairs) - len(in_band),
        "n_edges_in_band_with_k0_substituted": len(substituted),
        "n_edges_in_band_with_independently_estimated_k0": len(in_band) - len(substituted),
        # A pair table that does not reproduce the substitution did not come from this
        # code path, and the warning above would then be describing something else.
        "max_abs_deviation_from_substitution": None if worst is None else float(f"{worst:.3e}"),
        "substitution_reproduced": None if worst is None else worst < 1e-9,
        "by_group": report,
        "what_this_cannot_decide": (
            "Nothing here separates recent pedigree relatedness from old endogamy, and "
            "nothing here separates either from a mis-specified individual-specific allele "
            "frequency. Segment lengths are the evidence that could."
        ),
    }


def axis_localization(
    scores_path: Path,
    strata: dict[str, dict[str, str]],
    review_fraction: float,
    column: str,
    top_labels: int,
) -> dict[str, object]:
    """How concentrated each axis is, and on whom. A measurement, not a verdict."""
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

    bound = review_fraction * len(samples)
    axes = {}
    for name, values in zip(components, columns):
        total = sum(value * value for value in values)
        if total <= 0:
            axes[name] = {"participation_ratio": None, "status": "NOT_EVALUATED"}
            continue
        weights = [(value * value) / total for value in values]
        ratio = 1.0 / sum(weight * weight for weight in weights)
        carried: dict[str, float] = defaultdict(float)
        unlabelled = 0.0
        for sample, weight in zip(samples, weights):
            row = strata.get(sample)
            label = (str(row.get(column, "")).strip() if row else "") or "(unlabelled)"
            carried[label] += weight
            # The resolver emits a row for every panel member, so absence from the table is
            # not the test: an unresolved sample has a row with empty fields. Using `row is
            # None` made this field identically zero, including for the axis that is 90 per
            # cent carried by exactly those samples. Same criterion the R gate uses.
            if not is_interpretable(row):
                unlabelled += weight
        leaders = sorted(carried.items(), key=lambda item: -item[1])[:top_labels]
        axes[name] = {
            "participation_ratio": round(ratio, 3),
            "status": "PASS" if ratio >= bound else "REVIEW",
            "fraction_carried_by_samples_without_metadata": round(unlabelled, 6),
            "carried_by": [
                {"label": label, "weight_fraction": round(weight, 6)} for label, weight in leaders
            ],
        }

    flagged = sorted(name for name, row in axes.items() if row.get("status") == "REVIEW")
    prefixes = {}
    for count in range(1, len(components) + 1):
        subset = components[:count]
        prefixes[count] = (
            "REVIEW" if any(axes[name]["status"] == "REVIEW" for name in subset) else "PASS"
        )
    return {
        "n_samples": len(samples),
        "review_fraction": review_fraction,
        "review_bound_effective_individuals": round(bound, 3),
        "by_axis": axes,
        "axes_under_review": flagged,
        "status_by_leading_prefix": prefixes,
        "enforcement": "REPORT_ONLY_REVIEW_DOES_NOT_ABORT",
        "note": (
            "Evaluated on the score table supplied, and reported per leading prefix because "
            "a configuration reads only the components it asks for. REVIEW means the axis is "
            "concentrated on few individuals; it does not say why, and it does not abort a "
            "run. Separating a family from a small differentiated population, an isolate, a "
            "technical group or a metadata gap needs evidence this ratio does not carry."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if contract.get("pcrelate", {}).get("king_allowed"):
        raise SystemExit("M27D forbids KING")
    threshold = float(contract["pcrelate"]["primary_phi_threshold"])
    # One authority for the bound: the R gate and this report must not drift apart.
    review_fraction = float(
        contract["pca_axis_contract"]["g2b_axis_localization"]["review_fraction_per_axis"]
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
        "k0_algebraic_identity": k0_algebraic_identity(
            pairs, strata, threshold, args.ibd2_band_upper
        ),
        "sample_ids_emitted": False,
        "interpretation_limits": [
            "The structural ceiling exonerates or convicts the selection order only; it "
            "says nothing about why the edges exist.",
            "The k0 identity check is algebra imposed by GENESIS, not evidence: it cannot "
            "support or refute identity by descent, and this module offers nothing that "
            "can. The cause of the edges is unidentified.",
            "Axis localisation is a property of the score table supplied, and the pass0 fit "
            "included relatives, so a refit on the training set can move it. A localised "
            "axis is not by itself a family axis.",
        ],
    }
    if args.pca_scores is not None:
        summary["axis_localization"] = axis_localization(
            args.pca_scores, strata, review_fraction, args.population_column, args.top_labels
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
        help=(
            "upper edge of the reported band. It sits just above the 2^(-5/2) cutoff where "
            "GENESIS stops substituting k0, so edges on either side are counted separately"
        ),
    )
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-table", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
