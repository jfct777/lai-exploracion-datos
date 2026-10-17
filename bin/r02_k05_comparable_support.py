#!/usr/bin/env python3
"""Audit comparable K05 supports and close-kin sensitivity without refitting.

This is a metadata synthesis.  It authenticates the completed K05 table and the
per-person partition assignments, then reports exact support overlap and paired
within-partition descriptive contrasts.  It never selects or ranks partitions.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics


ANCESTRY_FIELDS = (
    "Autosomes_African_anc",
    "Autosomes_European_anc",
    "Autosomes_Indigenous_anc",
    "Autosomes_EastAsian_anc",
)
ADJUSTED_MODEL = "ANCESTRY_COLLECTION_GT_AVAILABILITY"
FULL_CALLSET = "COMPLETE_REPORTED_CALLSET"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tsv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows, fieldnames=None):
    rows = list(rows)
    require(rows or fieldnames, "Cannot infer columns for an empty table")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames or list(rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_supports(individuals, partition_ids, ancestry_fields=ANCESTRY_FIELDS):
    sample_ids = [row["sample_id"] for row in individuals]
    require(len(sample_ids) == len(set(sample_ids)), "Duplicate sample_id")
    complete = {
        row["sample_id"]
        for row in individuals
        if all(row[field] != "NA" for field in ancestry_fields)
    }
    supports = {}
    for partition_id in partition_ids:
        supports[partition_id] = frozenset(
            row["sample_id"]
            for row in individuals
            if row["sample_id"] in complete and int(row[partition_id]) >= 0
        )
    return supports


def support_pair_rows(partitions, supports):
    rows = []
    for left, right in itertools.combinations(partitions, 2):
        a, b = left["partition_id"], right["partition_id"]
        sa, sb = supports[a], supports[b]
        shared = len(sa & sb)
        union = len(sa | sb)
        exact = sa == sb
        rows.append(
            {
                "partition_a": a,
                "partition_b": b,
                "family_a": left["family"],
                "family_b": right["family"],
                "support_a": len(sa),
                "support_b": len(sb),
                "shared_people": shared,
                "a_only": len(sa - sb),
                "b_only": len(sb - sa),
                "union_people": union,
                "jaccard": shared / union,
                "containment_a": shared / len(sa),
                "containment_b": shared / len(sb),
                "exact_same_people": exact,
                "comparison_scope": (
                    "MATCHED_PEOPLE_DESCRIPTIVE_EXPOSURES_STILL_DIFFER"
                    if exact
                    else "DIFFERENT_PEOPLE_NO_CROSS_PARTITION_RANKING"
                ),
            }
        )
    return rows


def index_fits(fits):
    keyed = {}
    for row in fits:
        key = tuple(
            row[name]
            for name in (
                "partition_id",
                "kinship_threshold",
                "call_scope",
                "model",
                "relation",
            )
        )
        require(key not in keyed, "Duplicate completed fit: " + repr(key))
        keyed[key] = row
    return keyed


def number(row, field):
    value = row[field]
    return None if value == "NA" else float(value)


def closekin_rows(partitions, thresholds, fits):
    keyed = index_fits(fits)
    rows = []
    for item in partitions:
        partition_id = item["partition_id"]
        for threshold in thresholds:
            base = (partition_id, str(threshold), FULL_CALLSET, ADJUSTED_MODEL)
            all_pairs = keyed[base + ("ALL_KNOWN_COMPONENT",)]
            different = keyed[base + ("DIFFERENT_COMPONENT",)]
            require(
                all_pairs["support_people"] == different["support_people"],
                "Within-partition support differs by relation",
            )
            n_all, n_different = int(all_pairs["n_pairs"]), int(different["n_pairs"])
            a, d = number(all_pairs, "coefficient"), number(different, "coefficient")
            rows.append(
                {
                    "partition_id": partition_id,
                    "family": item["family"],
                    "graph_id": item["graph_id"],
                    "resolution": item["resolution"],
                    "kinship_threshold": threshold,
                    "support_people": all_pairs["support_people"],
                    "all_known_pairs": n_all,
                    "different_component_pairs": n_different,
                    "different_component_pair_fraction": n_different / n_all,
                    "all_known_status": all_pairs["status"],
                    "different_component_status": different["status"],
                    "all_known_coefficient_Mb": all_pairs["coefficient"],
                    "different_component_coefficient_Mb": different["coefficient"],
                    "descriptive_coefficient_change_Mb": (
                        "NA" if a is None or d is None else d - a
                    ),
                    "same_sign_when_estimable": (
                        "NA" if a is None or d is None else (a > 0) == (d > 0)
                    ),
                    "different_component_interval_low_Mb": different["interval_low"],
                    "different_component_interval_high_Mb": different["interval_high"],
                    "different_component_partial_r2": different["partial_r2"],
                    "different_component_loo_sign_changes": different["loo_sign_changes"],
                    "different_component_max_component_score_fraction": different[
                        "max_component_score_fraction"
                    ],
                    "comparison_scope": (
                        "WITHIN_PARTITION_NESTED_PAIR_SET_DESCRIPTIVE_SENSITIVITY"
                    ),
                }
            )
    return rows


def exact_support_groups(partition_ids, supports):
    groups = defaultdict(list)
    for partition_id in partition_ids:
        groups[supports[partition_id]].append(partition_id)
    return [
        {
            "support_people": len(support),
            "partition_count": len(ids),
            "partition_ids": ids,
        }
        for support, ids in sorted(
            groups.items(), key=lambda item: (-len(item[1]), -len(item[0]), item[1])
        )
    ]


def summarize_closekin(rows, thresholds):
    answer = {}
    for threshold in thresholds:
        selected = [row for row in rows if str(row["kinship_threshold"]) == str(threshold)]
        estimable = [
            row
            for row in selected
            if row["different_component_status"] == "CONDITIONAL_ASSOCIATION"
        ]
        intervals = [
            row
            for row in estimable
            if row["different_component_interval_low_Mb"] != "NA"
            and row["different_component_interval_high_Mb"] != "NA"
        ]
        fractions = [row["different_component_pair_fraction"] for row in selected]
        answer[str(threshold)] = {
            "partitions": len(selected),
            "estimable_after_excluding_same_component_pairs": len(estimable),
            "not_estimable_after_excluding_same_component_pairs": len(selected) - len(estimable),
            "positive_coefficients": sum(
                float(row["different_component_coefficient_Mb"]) > 0 for row in estimable
            ),
            "negative_coefficients": sum(
                float(row["different_component_coefficient_Mb"]) < 0 for row in estimable
            ),
            "conditional_intervals_excluding_zero_positive": sum(
                float(row["different_component_interval_low_Mb"]) > 0 for row in intervals
            ),
            "conditional_intervals_excluding_zero_negative": sum(
                float(row["different_component_interval_high_Mb"]) < 0 for row in intervals
            ),
            "same_sign_before_after_pair_exclusion": sum(
                row["same_sign_when_estimable"] is True for row in selected
            ),
            "sign_changed_before_after_pair_exclusion": sum(
                row["same_sign_when_estimable"] is False for row in selected
            ),
            "different_component_pair_fraction_range": [min(fractions), max(fractions)],
            "different_component_pair_fraction_median": statistics.median(fractions),
        }
    return answer


def synthesize(plan_path, plan_sha256, output):
    require(sha(plan_path) == plan_sha256, "Plan changed")
    plan = json.loads(Path(plan_path).read_text())
    require(
        plan["scope"] == "K05_EXISTING_RESULTS_COMPARABLE_SUPPORT_NO_REFIT_NO_RANKING",
        "Invalid scope",
    )
    for name, descriptor in plan["inputs"].items():
        require(sha(descriptor["path"]) == descriptor["sha256"], "Input changed: " + name)
    require(sha(__file__) == plan["code_sha256"], "Implementation changed")

    inputs = {name: descriptor["path"] for name, descriptor in plan["inputs"].items()}
    summary = json.loads(Path(inputs["community_summary"]).read_text())
    catalogue = json.loads(Path(inputs["catalogue"]).read_text())
    coverage = read_tsv(inputs["coverage"])
    individuals = read_tsv(inputs["individuals"])
    fits = read_tsv(inputs["associations"])
    partitions = summary["partition_ledger"]
    partition_ids = [item["partition_id"] for item in partitions]

    require(summary["n_partitions"] == 57 == len(partitions), "Expected 57 partitions")
    require(len(fits) == 1368, "Expected all 1,368 completed fits")
    require(catalogue["schema"] == "m165_family_catalogue_v1", "Invalid catalogue")
    require(not catalogue["evaluate_opened"], "EVALUATE was opened")
    require(not catalogue["human_fit_admitted"], "Unexpected human fit admission")
    require(
        [item["partition_id"] for item in catalogue["partitions"]] == partition_ids,
        "Catalogue and community partition order differ",
    )
    require(
        Counter(item["family"] for item in partitions) == Counter(catalogue["families"]),
        "Catalogue family counts differ",
    )

    supports = build_supports(individuals, partition_ids, summary["ancestry_fields"])
    coverage_by_id = {row["partition_id"]: row for row in coverage}
    require(set(coverage_by_id) == set(partition_ids), "Coverage partitions differ")
    for partition_id in partition_ids:
        require(
            len(supports[partition_id]) == int(coverage_by_id[partition_id]["assigned_n"]),
            "Authenticated support count differs: " + partition_id,
        )

    pairwise = support_pair_rows(partitions, supports)
    require(len(pairwise) == 1596, "Expected 57 choose 2 support pairs")
    closekin = closekin_rows(partitions, summary["kinship_thresholds"], fits)
    require(len(closekin) == 114, "Expected two close-kin rows per partition")
    groups = exact_support_groups(partition_ids, supports)
    common = set.intersection(*(set(supports[partition_id]) for partition_id in partition_ids))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_tsv(output / "support_pairwise.tsv", pairwise)
    write_tsv(output / "within_partition_closekin_sensitivity.tsv", closekin)
    matrix = []
    for left in partition_ids:
        row = {"partition_id": left}
        row.update({right: len(supports[left] & supports[right]) for right in partition_ids})
        matrix.append(row)
    write_tsv(output / "support_overlap_matrix.tsv", matrix)

    jaccards = [float(row["jaccard"]) for row in pairwise]
    exact_pairs = [row for row in pairwise if row["exact_same_people"]]
    exact_multi = [group for group in groups if group["partition_count"] > 1]
    result = {
        "schema": "r02_k05_comparable_support_v1",
        "status": "COMPLETE_EXISTING_RESULTS_COMPARABILITY_AUDIT_NO_REFIT_NO_RANKING",
        "partitions": len(partitions),
        "families": dict(Counter(item["family"] for item in partitions)),
        "source_completed_fits": len(fits),
        "fits_rerun": 0,
        "support_people_range": [
            min(map(len, supports.values())),
            max(map(len, supports.values())),
        ],
        "universal_intersection_people": len(common),
        "universal_intersection_filter_applied": False,
        "pairwise_partition_comparisons": len(pairwise),
        "pairwise_intersection_people_range": [
            min(row["shared_people"] for row in pairwise),
            max(row["shared_people"] for row in pairwise),
        ],
        "pairwise_jaccard_range": [min(jaccards), max(jaccards)],
        "unique_person_supports": len(groups),
        "exact_same_people_partition_pairs": len(exact_pairs),
        "exact_same_people_groups": exact_multi,
        "within_partition_closekin": summarize_closekin(
            closekin, summary["kinship_thresholds"]
        ),
        "can_conclude": [
            "The 57 completed descriptions use 32 distinct person supports; 128 partition pairs use exactly the same people and the other pairs do not.",
            "For each partition, the table measures whether the descriptive association persists after excluding pairs in the same PC-Relate component under each pre-existing threshold.",
            "Positive conditional coefficients after that exclusion describe residual community-associated IBD length on the observed eligible dyads.",
        ],
        "cannot_conclude": [
            "A larger coefficient identifies a better or more biological partition across different person supports.",
            "Excluding same PC-Relate components makes remaining dyads statistically or biologically independent.",
            "Residual association proves population structure, a founder event, or causal demography beyond close kin.",
            "Conditional intervals provide selection-adjusted uncertainty across 57 globally learned partitions.",
        ],
        "missing_dependency_for_selection_or_beyond_closekin_claim": (
            "Partitions learned only in FIT, fixed projection to SELECT and untouched EVALUATE, "
            "a prespecified shared-support direct contrast, and uncertainty that accounts for "
            "nested/shared dyads, partition selection and multiple configurations."
        ),
        "winner_selected": False,
        "evaluate_opened": False,
        "source_pins": plan["inputs"],
        "code_sha256": sha(__file__),
    }
    for name in (
        "support_pairwise.tsv",
        "within_partition_closekin_sensitivity.tsv",
        "support_overlap_matrix.tsv",
    ):
        result.setdefault("outputs_sha256", {})[name] = sha(output / name)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(synthesize(args.plan, args.plan_sha256, args.output)["status"])
