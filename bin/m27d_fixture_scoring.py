#!/usr/bin/env python3
"""Score a pipeline run against a cohort whose relationships are known in advance.

The question is not whether phi is estimated accurately.  It is whether the design puts
recent pedigree and deme-level coancestry on *opposite sides* of the threshold: relatives
retained, look-alikes released.  So the headline numbers are a sensitivity per true degree
and a false-positive rate among pairs whose only similarity is drift.

Two things this deliberately refuses to do.

It does not report the edge count alone.  A total can stay flat while a real first-degree
pair falls from 0.25 to 0.05 and survives only because the threshold happens to sit below
it; the same total also hides a coancestry pair that crossed upward.  Every pair is
tracked by its own true class instead.

It does not treat a missing pair as a zero without saying so.  The stage writes only pairs
at or above the reporting threshold, so a relative that vanished from the table is the
single most important observation available and it is counted explicitly.
"""

from __future__ import annotations

import csv
import gzip
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m27d_kinship_graph import maximum_independent_set_size  # noqa: E402
from m27d_training_set_representativeness import axis_support  # noqa: E402


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def read_pairs(path: Path) -> dict[tuple[str, str], float]:
    pairs: dict[tuple[str, str], float] = {}
    with open_text(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pairs[tuple(sorted((row["ID1"], row["ID2"])))] = float(row["kin"])
    return pairs


def read_marker_counts(path: Path) -> dict[tuple[str, str], int]:
    """Markers behind each estimate.

    GENESIS drops a marker for a pair when either individual-specific frequency falls
    under the bound, and a drifted deme loses more of them.  A phi computed from a small
    fraction of the panel is not the same observation as one computed from all of it, so
    the count travels with the estimate instead of being assumed constant.
    """
    counts: dict[tuple[str, str], int] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "nsnp" not in (reader.fieldnames or []):
            return counts
        for row in reader:
            counts[tuple(sorted((row["ID1"], row["ID2"])))] = int(float(row["nsnp"]))
    return counts


def read_truth(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def truth_class(row: dict[str, str]) -> str:
    """One label per pair, so every pair is counted once and in the right denominator.

    Pedigree pairs are split by where they sit.  A relative lost in the panmictic
    background is a failure of the estimator; a relative lost inside a small drifted deme
    may be a failure of the deme's representation instead, and merging the two would make
    the distinction unrecoverable from the output.
    """
    location = row.get("pedigree_location", "none")
    if row["true_relationship"] == "parent_offspring":
        return f"first_degree_in_{location}"
    if row["true_relationship"] == "half_sibling":
        return f"second_degree_in_{location}"
    if row["coancestry_class"] == "within_deme":
        return "coancestry_within_deme"
    if row["coancestry_class"] == "between_demes":
        return "coancestry_between_demes"
    if row["coancestry_class"] == "within_background_group":
        # A large group with the same drift the method is asked to remove from a small one.
        # It is the positive control for absorption, and it only works if it is named.
        return "coancestry_within_background_group"
    return "unrelated_no_coancestry"


# Classes that must be retained, and classes that must not be.  Naming them here rather
# than inside the arithmetic keeps the decision rule readable and reviewable.
POSITIVE_CLASSES = (
    "first_degree_in_background", "first_degree_in_deme",
    "second_degree_in_background", "second_degree_in_deme",
)
NEGATIVE_CLASSES = (
    "coancestry_within_deme",
    "coancestry_between_demes",
    "coancestry_within_background_group",
    "unrelated_no_coancestry",
)


def score_pass(
    observed: dict[tuple[str, str], float],
    truth: list[dict[str, str]],
    threshold: float,
    report_threshold: float,
    marker_counts: dict[tuple[str, str], int] | None = None,
    training: set[str] | None = None,
) -> dict[str, object]:
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in truth:
        key = (row["ID1"], row["ID2"])
        phi = observed.get(key)
        if training is None:
            membership = "not_evaluated"
        else:
            inside = (row["ID1"] in training) + (row["ID2"] in training)
            membership = ("both_in_training", "one_outside", "both_outside")[2 - inside]
        by_class[truth_class(row)].append(
            {
                "phi": phi,
                "present": phi is not None,
                "retained": phi is not None and phi >= threshold,
                "pedigree_phi": float(row["pedigree_phi"]),
                "coancestry_phi": float(row["coancestry_phi"]),
                "total_phi": float(row.get("total_phi", row["pedigree_phi"])),
                "nsnp": (marker_counts or {}).get(key),
                "membership": membership,
            }
        )

    summary: dict[str, object] = {}
    for name, rows in sorted(by_class.items()):
        retained = sum(1 for row in rows if row["retained"])
        present = [row["phi"] for row in rows if row["present"]]
        block: dict[str, object] = {
            "n_pairs": len(rows),
            "n_retained_at_threshold": retained,
            "fraction_retained": round(retained / len(rows), 6) if rows else None,
            # A pair absent from the table fell below the reporting threshold entirely.
            # For a true relative that is the loudest possible result, so it is named.
            "n_absent_from_pair_table": sum(1 for row in rows if not row["present"]),
        }
        if present:
            # The production stage suppresses every pair below report_threshold.  These
            # summaries therefore describe the published upper tail, not the full class.
            block["phi_median_among_reported_pairs"] = round(statistics.median(present), 6)
            block["phi_min_among_reported_pairs"] = round(min(present), 6)
            block["phi_max_among_reported_pairs"] = round(max(present), 6)
            block["phi_summary_scope"] = "reported_pairs_only_left_truncated"
        counts = [row["nsnp"] for row in rows if row["nsnp"] is not None]
        if counts:
            block["nsnp_median"] = int(statistics.median(counts))
            block["nsnp_min"] = min(counts)
        block["membership"] = {
            key: sum(1 for row in rows if row["membership"] == key)
            for key in ("both_in_training", "one_outside", "both_outside", "not_evaluated")
            if any(row["membership"] == key for row in rows)
        }
        if name in POSITIVE_CLASSES:
            # Two truths, because the estimator is not trying to recover the same one at
            # every component count: the pedigree component is the target and survives
            # conditioning, while the total includes coancestry the components may remove
            # on purpose. Reporting one alone would score correct behaviour as error.
            pedigree_expected = rows[0]["pedigree_phi"]
            total_expected = rows[0]["total_phi"]
            present = [row["phi"] for row in rows if row["present"]]
            block["expected_pedigree_phi"] = pedigree_expected
            block["expected_total_phi"] = total_expected
            block["absolute_error_vs_pedigree_median_among_reported_pairs"] = round(
                statistics.median([abs(v - pedigree_expected) for v in present]), 6
            ) if present else None
            block["absolute_error_vs_total_median_among_reported_pairs"] = round(
                statistics.median([abs(v - total_expected) for v in present]), 6
            ) if present else None
            block["sensitivity"] = block["fraction_retained"]
        else:
            block["false_positive_rate"] = block["fraction_retained"]
            block["expected_coancestry_phi"] = rows[0]["coancestry_phi"]
        summary[name] = block

    negatives = [row for name in NEGATIVE_CLASSES for row in by_class.get(name, [])]
    positives = [row for name in POSITIVE_CLASSES for row in by_class.get(name, [])]
    # A global specificity is diluted by thousands of trivially unrelated pairs: it reads
    # 0.99 in the same cells where three quarters of the coancestry-only pairs are called
    # related. It is reported under a name that says what it is, next to the per-class
    # rates that actually decide, so nobody can quote it as if it meant the method worked.
    trivial = by_class.get("unrelated_no_coancestry", [])
    informative = [
        row for name in NEGATIVE_CLASSES if name != "unrelated_no_coancestry"
        for row in by_class.get(name, [])
    ]
    summary["overall"] = {
        "report_threshold": report_threshold,
        "primary_threshold": threshold,
        "n_pairs_in_table": len(observed),
        "n_true_positives": sum(1 for row in positives if row["retained"]),
        "n_false_negatives": sum(1 for row in positives if not row["retained"]),
        "n_false_positives": sum(1 for row in negatives if row["retained"]),
        "n_true_negatives": sum(1 for row in negatives if not row["retained"]),
        "sensitivity": round(
            sum(1 for row in positives if row["retained"]) / len(positives), 6
        ) if positives else None,
        "specificity_diluted_by_trivial_negatives": round(
            sum(1 for row in negatives if not row["retained"]) / len(negatives), 6
        ) if negatives else None,
        "n_trivially_unrelated_pairs_in_that_denominator": len(trivial),
        "specificity_over_informative_negatives_only": round(
            sum(1 for row in informative if not row["retained"]) / len(informative), 6
        ) if informative else None,
        # The design is only safe if no truly related pair ends up entirely inside the set
        # the frequencies are fitted on. It was never aggregated, so the property held by
        # luck rather than by measurement.
        "n_related_pairs_with_both_members_in_training": sum(
            1 for row in positives if row["membership"] == "both_in_training"
        ),
    }
    return summary


def delta_versus_pass0(
    pass0: dict[tuple[str, str], float],
    final: dict[tuple[str, str], float],
    truth: list[dict[str, str]],
    threshold: float,
) -> dict[str, object]:
    """How each true class moved between the two passes, and which pairs crossed.

    Stratifying by true class is what separates "the refit removed structure" from "the
    refit ate real relatedness": both shrink phi, and only the class tells them apart.
    """
    moves: dict[str, list[float]] = defaultdict(list)
    crossings: dict[str, dict[str, int]] = defaultdict(lambda: {"down": 0, "up": 0, "stayed_related": 0})
    for row in truth:
        key = (row["ID1"], row["ID2"])
        name = truth_class(row)
        before, after = pass0.get(key), final.get(key)
        if before is not None and after is not None:
            moves[name].append(after - before)
        was = before is not None and before >= threshold
        now = after is not None and after >= threshold
        if was and not now:
            crossings[name]["down"] += 1
        elif now and not was:
            crossings[name]["up"] += 1
        elif was and now:
            crossings[name]["stayed_related"] += 1
    report: dict[str, object] = {}
    for name in sorted(set(moves) | set(crossings)):
        values = moves.get(name, [])
        report[name] = {
            "n_pairs_in_both_tables": len(values),
            "delta_phi_median": round(statistics.median(values), 6) if values else None,
            "delta_phi_min": round(min(values), 6) if values else None,
            "delta_phi_max": round(max(values), 6) if values else None,
            "crossed_below_threshold": crossings[name]["down"],
            "crossed_above_threshold": crossings[name]["up"],
            "stayed_related": crossings[name]["stayed_related"],
        }
    return report


def representation(
    truth_json: dict,
    training: set[str],
    alternate: set[str] | None,
    pca_summary: dict,
    pass0_scores: Path | None = None,
    universe: list[str] | None = None,
    pass0_edges: set[tuple[str, str]] | None = None,
    training_is_pass0_independent_set: bool = True,
) -> dict[str, object]:
    """Whether the fitting set still holds the demes it is supposed to describe."""
    demes = truth_json["demes"]
    per_deme = {
        deme: {
            "n_members": len(members),
            "n_in_training_set": len(set(members) & training),
            "n_in_alternate_set": (
                len(set(members) & alternate) if alternate is not None else None
            ),
        }
        for deme, members in demes.items()
    }
    ratios = pca_summary.get("g2b_effective_individuals_by_axis") or []
    # The mass of the pass0 axes that survives into the fitting set is the quantity the
    # real panel is failing on, so the fixture has to measure the same thing rather than
    # a proxy: an axis refitted from a fraction of itself is extrapolated everywhere else.
    axis_mass = None
    if pass0_scores is not None and pass0_scores.exists():
        axis_mass = axis_support(pass0_scores, training)["by_axis"]
    # Size stability is not identity stability, and the exact ceiling separates what the
    # tie-break costs from what the graph makes impossible.
    algorithmic = None
    if universe is not None and pass0_edges is not None and training_is_pass0_independent_set:
        exact = maximum_independent_set_size(universe, pass0_edges, max_component_nodes=60)
        algorithmic = {
            "n_greedy_primary": len(training),
            "n_greedy_alternate": len(alternate) if alternate is not None else None,
            "alpha_exact": exact["size"],
            "alpha_is_exact": exact["exact"],
            "greedy_shortfall_vs_exact": exact["size"] - len(training),
        }
    return {
        "by_deme": per_deme,
        "axis_mass_retained_by_training_set": axis_mass,
        "algorithmic_control": algorithmic,
        "algorithmic_control_applicable": training_is_pass0_independent_set,
        "n_demes_absent_from_training_set": sum(
            1 for row in per_deme.values() if row["n_in_training_set"] == 0
        ),
        "n_demes_with_a_single_member": sum(
            1 for row in per_deme.values() if row["n_in_training_set"] == 1
        ),
        "participation_ratio_by_axis": [round(float(value), 3) for value in ratios],
        "g2a_status": pca_summary.get("g2a_technical_integrity_status"),
        "g2b_status_by_prefix": {
            key: entry.get("status")
            for key, entry in (pca_summary.get("g2b_status_by_preregistered_n_pcs") or {}).items()
        },
        "g2c_status": pca_summary.get("g2c_ancestry_representativeness_status"),
        # Size stability is not identity stability: two orders can retain the same count
        # and disagree on who, which is the thing that changes who becomes a donor.
        "training_set_identity_jaccard_vs_alternate": round(
            len(training & alternate) / len(training | alternate), 6
        ) if alternate is not None and (training | alternate) else None,
        "n_training_primary": len(training),
        "n_training_alternate": len(alternate) if alternate is not None else None,
        "n_shared": len(training & alternate) if alternate is not None else None,
    }
