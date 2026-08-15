#!/usr/bin/env python3
"""Whether the set PC-Relate fits on still represents the small populations.

Independence and representativeness are different properties, and the M27D design only
ever enforced the first.  The training set is built by removing individuals until no
kinship edge is left, and that construction is blind to what it empties: a population
whose members are all related to one another survives as a single person, or as nobody,
and the components fitted on the survivors then have nothing left to describe it with.
PC-Relate uses those components to estimate individual-specific allele frequencies, so a
population that has been emptied out of the fit is exactly where the estimate is least
trustworthy — and it is the same population whose members were removed for being related.

This reads only artifacts an earlier pass already produced and answers, per population:
how many were eligible, how many the training set keeps, whether the internal graph is a
complete clique, what the exact ceiling alpha(G[P]) is, and how close the internal edges
sit to first degree.  That last one matters: a clique whose edges sit near phi=0.5 and a
clique whose edges sit near phi=0.05 are very different objects, and the count alone hides
the difference.  It does not adjudicate G2C.  The contract leaves that open on purpose,
and choosing a floor after seeing which value keeps more donors is what stop rule 5
forbids.

Samples with no metadata row are reported as their own group rather than dropped.  They
cannot be attributed to a population, so they are not evidence about one; leaving them out
of the table would silently shrink the denominator instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m27d_kinship_graph import (  # noqa: E402
    is_interpretable,
    maximum_independent_set_size,
    open_text,
    read_sample_universe,
    read_strata_table,
)

UNLABELLED = "(no metadata row)"
# Degree bands are named after the pedigree relationship whose expected phi they bracket,
# cut at the half-powers of two the field uses and GENESIS itself branches on.  They say
# where an edge sits, not what produced it.  The duplicate band is kept separate from first
# degree on purpose: a cohort where the close edges are duplicated samples and one where
# they are siblings need different answers, and merging the two hides that.
DEGREE_BANDS = (
    ("duplicate_or_monozygotic", 2.0**-1.5, 1.01),
    ("first_degree", 2.0**-2.5, 2.0**-1.5),
    ("second_degree", 2.0**-3.5, 2.0**-2.5),
    ("third_degree", 2.0**-4.5, 2.0**-3.5),
    ("below_third_degree", 0.0, 2.0**-4.5),
)


def read_edges_with_kinship(path: Path, threshold: float) -> list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for column in ("ID1", "ID2", "kin"):
            if column not in (reader.fieldnames or []):
                raise SystemExit(f"{path.name} is missing the {column!r} column")
        for row in reader:
            kinship = float(row["kin"])
            if kinship < threshold:
                continue
            left, right = row["ID1"], row["ID2"]
            edges.append((left, right, kinship) if left <= right else (right, left, kinship))
    return edges


def read_membership(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_of(strata: dict[str, dict[str, str]], sample: str, column: str) -> str:
    row = strata.get(sample)
    if row is None:
        return UNLABELLED
    return (str(row.get(column, "")).strip() or UNLABELLED)


def degree_profile(values: list[float]) -> dict[str, object]:
    counts = {name: 0 for name, _, _ in DEGREE_BANDS}
    for value in values:
        for name, low, high in DEGREE_BANDS:
            if low <= value < high:
                counts[name] += 1
                break
    return {
        "phi_median": round(statistics.median(values), 6),
        "phi_min": round(min(values), 6),
        "phi_max": round(max(values), 6),
        "edges_by_degree_band": counts,
    }


def survey(
    universe: list[str],
    training: set[str],
    strata: dict[str, dict[str, str]],
    edges: list[tuple[str, str, float]],
    column: str,
    max_component_nodes: int,
    suppress_below: int,
) -> dict[str, object]:
    members: dict[str, list[str]] = defaultdict(list)
    for sample in universe:
        members[label_of(strata, sample, column)].append(sample)

    rows: dict[str, dict[str, object]] = {}
    suppressed: list[str] = []
    for label, group in sorted(members.items()):
        # The rest of the module suppresses strata below the same floor before publishing.
        # "Guajajara: 1 eligible, 0 retained" is a cell of size one about a named indigenous
        # population, and it says nothing the aggregate counts do not already say.
        if len(group) < suppress_below:
            suppressed.append(label)
            continue
        group_set = set(group)
        internal = [edge for edge in edges if edge[0] in group_set and edge[1] in group_set]
        kept = sorted(group_set & training)
        possible = len(group) * (len(group) - 1) / 2
        ceiling = (
            maximum_independent_set_size(
                group,
                {(left, right) for left, right, _ in internal},
                max_component_nodes=max_component_nodes,
            )
            if internal
            else {"size": len(group), "exact": True}
        )
        row: dict[str, object] = {
            "n_eligible": len(group),
            "n_in_training_set": len(kept),
            "retained_fraction": round(len(kept) / len(group), 6),
            "n_internal_edges": len(internal),
            "edge_density": round(len(internal) / possible, 6) if possible else None,
            "is_complete_clique": bool(internal) and len(internal) == possible,
            "alpha_exact": ceiling["size"],
            "alpha_is_exact": ceiling["exact"],
            # A population can fall below its own ceiling because of edges to other
            # populations, so this is an upper bound on what the tie-break costs here.
            "shortfall_vs_local_ceiling": ceiling["size"] - len(kept),
            "status": (
                "ABSENT" if not kept
                else "SINGLETON" if len(kept) == 1
                else "REDUCED" if len(kept) < len(group)
                else "INTACT"
            ),
        }
        if internal:
            row["internal_edges"] = degree_profile([value for _, _, value in internal])
        rows[label] = row

    def collect(status: str) -> list[str]:
        return sorted(label for label, row in rows.items() if row["status"] == status)

    absent, singleton = collect("ABSENT"), collect("SINGLETON")
    # A population represented by one person contributes no within-population variation to
    # the fit, so it is grouped with the empty ones when judging what the axes can describe.
    at_or_below_one = sorted(set(absent) | set(singleton))
    # Suppressed groups are counted, never named, and their aggregate is reported so the
    # denominator does not quietly shrink to the groups large enough to publish.
    suppressed_members = sum(len(members[label]) for label in suppressed)
    suppressed_retained = sum(len(set(members[label]) & training) for label in suppressed)
    return {
        "column": column,
        "n_groups_published": len(rows),
        "n_groups_total": len(rows) + len(suppressed),
        "n_eligible": len(universe),
        "n_in_training_set": len(training & set(universe)),
        "suppression": {
            "suppress_below_n_eligible": suppress_below,
            "n_groups_suppressed": len(suppressed),
            "n_eligible_in_suppressed_groups": suppressed_members,
            "n_in_training_set_in_suppressed_groups": suppressed_retained,
            "n_suppressed_groups_absent_from_training_set": sum(
                1 for label in suppressed if not (set(members[label]) & training)
            ),
            "rationale": (
                "Groups this small are counted but not named. A row saying one eligible and "
                "zero retained for a named population is a cell of size one, and the "
                "aggregate below carries the same information about the loss."
            ),
        },
        "groups_absent_from_training_set": absent,
        "groups_reduced_to_a_single_member": singleton,
        "n_groups_absent": len(absent),
        "n_groups_reduced_to_a_single_member": len(singleton),
        "n_groups_at_or_below_one_member": len(at_or_below_one),
        "n_eligible_in_groups_at_or_below_one_member": sum(
            rows[label]["n_eligible"] for label in at_or_below_one
        ),
        "groups_that_are_complete_cliques": sorted(
            label for label, row in rows.items() if row["is_complete_clique"]
        ),
        "n_published_groups_that_are_complete_cliques": sum(
            1 for row in rows.values() if row["is_complete_clique"]
        ),
        "by_group": rows,
    }


def unlabelled_group(
    universe: list[str],
    training: set[str],
    strata: dict[str, dict[str, str]],
    edges: list[tuple[str, str, float]],
    scores_path: Path | None,
) -> dict[str, object]:
    """The samples with no metadata row, described without publishing an identifier.

    Their origin is documented operationally but their population is not, so nothing here
    assigns them one.  The identifier prefix is the only provenance signal available and it
    is reported as a count per prefix, which names no individual.
    """
    orphans = [sample for sample in universe if not is_interpretable(strata.get(sample))]
    if not orphans:
        return {"n_samples": 0, "status": "NONE_PRESENT"}
    orphan_set = set(orphans)
    internal = [edge for edge in edges if edge[0] in orphan_set and edge[1] in orphan_set]
    external = [
        edge for edge in edges
        if (edge[0] in orphan_set) != (edge[1] in orphan_set)
    ]
    prefixes: dict[str, int] = defaultdict(int)
    for sample in orphans:
        match = re.match(r"^([A-Za-z]+)", sample)
        prefixes[match.group(1) if match else "(none)"] += 1
    possible = len(orphans) * (len(orphans) - 1) / 2
    block: dict[str, object] = {
        "n_samples": len(orphans),
        "n_in_training_set": len(orphan_set & training),
        "identifier_prefix_counts": dict(sorted(prefixes.items())),
        "n_internal_edges": len(internal),
        "n_edges_to_the_rest_of_the_panel": len(external),
        "is_complete_clique": bool(internal) and len(internal) == possible,
        "interpretation_status": "BLIND_SPOT_POPULATION_UNRESOLVED",
        "why_blind": (
            "No metadata row exists for these samples, so no population, country or source "
            "can be attributed to them. The identifier prefix is an operational trace, not "
            "an authoritative annotation, and a prefix is not a population."
        ),
        "not_excluded": (
            "They stay in the PCA and in PC-Relate. Removing people from a kinship audit "
            "because of a bookkeeping gap would bias the very estimate the audit produces."
        ),
    }
    if internal:
        block["internal_edges"] = degree_profile([value for _, _, value in internal])
    if scores_path is not None:
        block["axis_share"] = axis_share(scores_path, orphan_set)
    return block


def axis_share(scores_path: Path, subset: set[str]) -> dict[str, float]:
    """Fraction of each axis carried by a subset, on the same weights the gate uses."""
    with open_text(scores_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        components = [name for name in (reader.fieldnames or []) if name.startswith("PC")]
        totals = {name: 0.0 for name in components}
        inside = {name: 0.0 for name in components}
        for row in reader:
            member = row["sample_id"] in subset
            for name in components:
                weight = float(row[name]) ** 2
                totals[name] += weight
                if member:
                    inside[name] += weight
    return {
        name: round(inside[name] / totals[name], 6) if totals[name] > 0 else None
        for name in components
    }


def axis_support(scores_path: Path, training: set[str]) -> dict[str, object]:
    """How much of each axis the refit actually gets to see.

    Counting populations answers a question PC-Relate never asks. What it consumes is a
    set of axes, and the refit estimates them from the training set alone: an axis whose
    mass lives mostly outside that set is being fitted on a fraction of itself and
    evaluated everywhere else by extrapolation. This is the same weight the localisation
    gate uses, split by membership instead of by label, so the two are directly comparable.
    """
    retained = axis_share(scores_path, training)
    with open_text(scores_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        components = [name for name in (reader.fieldnames or []) if name.startswith("PC")]
        columns = {name: [] for name in components}
        for row in reader:
            for name in components:
                columns[name].append(float(row[name]) ** 2)
    axes = {}
    for name in components:
        total = sum(columns[name])
        if total <= 0:
            axes[name] = {"mass_retained_by_training_set": None, "participation_ratio": None}
            continue
        weights = [value / total for value in columns[name]]
        axes[name] = {
            "mass_retained_by_training_set": retained[name],
            "mass_projected_not_fitted": round(1.0 - (retained[name] or 0.0), 6),
            "participation_ratio": round(1.0 / sum(w * w for w in weights), 3),
        }
    return {
        "by_axis": axes,
        "note": (
            "Mass retained is the fraction of an axis carried by individuals the refit is "
            "fitted on. It is reported, not gated: no operating value for it has been "
            "justified against a real distribution, and inventing one here would be "
            "choosing a number before measuring."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if contract.get("pcrelate", {}).get("king_allowed"):
        raise SystemExit("M27D forbids KING")
    threshold = float(contract["pcrelate"]["primary_phi_threshold"])
    representativeness = contract["pca_axis_contract"]["g2c_ancestry_representativeness"]

    universe = read_sample_universe(args.universe)
    strata = read_strata_table(args.strata)
    training = set(read_membership(args.training_set))
    missing = training - set(universe)
    if missing:
        raise SystemExit(f"{len(missing)} training-set members are outside the eligible universe")
    edges = read_edges_with_kinship(args.pairs, threshold)

    summary: dict[str, object] = {
        "stage": "M27D_TRAINING_SET_REPRESENTATIVENESS",
        "scientific_result": False,
        "derived_from_existing_artifacts_only": True,
        "new_pcrelate_pass_executed": False,
        "king_executed": False,
        "pcair_used": False,
        "phi_threshold": threshold,
        "g2c_status": representativeness["status"],
        "g2c_adjudicated_here": False,
        "by_population": survey(
            universe, training, strata, edges, args.population_column,
            args.max_component_nodes, args.suppress_below,
        ),
        # Ancestry groups are continental and never small enough to need suppression, so
        # the floor is set to one rather than reusing the population value by accident.
        "by_ancestry": survey(
            universe, training, strata, edges, args.ancestry_column,
            args.max_component_nodes, 1,
        ),
        "samples_without_metadata": unlabelled_group(
            universe, training, strata, edges, args.pca_scores
        ),
        "axis_support": (
            axis_support(args.pca_scores, training)
            if args.pca_scores is not None
            else {"status": "NOT_EVALUATED_NO_SCORES_SUPPLIED"}
        ),
        "sample_ids_emitted": False,
        "interpretation_limits": [
            "Being independent under the kinship graph and being representative of a "
            "population are different properties. This table measures the second; nothing "
            "here re-opens the first.",
            "A degree band names where an edge sits, not what produced it. A clique of "
            "close edges and a clique of distant ones are different objects, but neither "
            "band separates recent pedigree from old endogamy on its own.",
            "The training set must not be changed after reading which variant of it keeps "
            "more donors. If it fails, the finding is a design limitation to record and an "
            "alternative to propose before running anything.",
        ],
    }
    if args.alternate_training_set is not None:
        alternate = set(read_membership(args.alternate_training_set))
        summary["alternate_order"] = {
            "n_in_training_set": len(alternate),
            "n_shared_with_primary": len(alternate & training),
            "n_symmetric_difference": len(alternate ^ training),
            "groups_absent_only_under_primary": sorted(
                label
                for label, row in summary["by_population"]["by_group"].items()
                if row["status"] == "ABSENT"
                and any(
                    label_of(strata, sample, args.population_column) == label
                    for sample in alternate
                )
            ),
        }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_table(summary["by_population"], args.out_table)
    return summary


def write_table(survey_block: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["population", "n_eligible", "n_in_training_set", "retained_fraction",
             "n_internal_edges", "edge_density", "is_complete_clique", "alpha_exact",
             "shortfall_vs_local_ceiling", "status", "internal_phi_median"]
        )
        for label, row in sorted(
            survey_block["by_group"].items(),
            key=lambda item: (item[1]["retained_fraction"], -item[1]["n_eligible"]),
        ):
            edges = row.get("internal_edges") or {}
            writer.writerow([
                label, row["n_eligible"], row["n_in_training_set"], row["retained_fraction"],
                row["n_internal_edges"], row["edge_density"], row["is_complete_clique"],
                row["alpha_exact"], row["shortfall_vs_local_ceiling"], row["status"],
                edges.get("phi_median", ""),
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--training-set", type=Path, required=True)
    parser.add_argument("--alternate-training-set", type=Path, default=None)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--pca-scores", type=Path, default=None)
    parser.add_argument("--population-column", default="Population")
    parser.add_argument("--ancestry-column", default="Ancestry")
    parser.add_argument("--max-component-nodes", type=int, default=60)
    # Same default as bin/m27d_prepare_sample_strata.py, so the module has one floor.
    parser.add_argument("--suppress-below", type=int, default=5)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-table", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
