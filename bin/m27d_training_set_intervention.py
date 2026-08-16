#!/usr/bin/env python3
"""Build a size-matched PCA fitting set that represents every synthetic deme.

This is a diagnostic intervention, not a donor-selection policy.  It starts from the
strict independent set produced by M27D and changes only who is allowed to fit the PCA.
Membership is derived from the fixture's known pedigree and deme labels; no estimated
kinship value from the final PC-Relate pass is read.

The represented set includes all pedigree-unrelated members of the two pure-coancestry
demes.  In the pedigree deme it includes founders and the unrelated spare individual,
but not the two offspring.  It remains the same size as the strict set by removing a
proportional, deterministically selected number of non-pedigree background samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate identifiers in {path}")
    return values


def read_truth_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_population(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    mapping = {row["IID"]: row["Population"] for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("Metadata contains duplicate IID values")
    return mapping


def is_recent_pair(row: dict[str, str]) -> bool:
    value = row.get("has_recent_kinship", "")
    return value if isinstance(value, bool) else str(value).strip().lower() == "true"


def recent_pairs(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {
        tuple(sorted((row["ID1"], row["ID2"])))
        for row in rows
        if is_recent_pair(row)
    }


def count_internal_pairs(
    members: set[str], pairs: set[tuple[str, str]]
) -> list[tuple[str, str]]:
    return sorted(pair for pair in pairs if pair[0] in members and pair[1] in members)


def stable_rank(sample: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"m27d-representation|{seed}|{sample}".encode()).hexdigest()
    return digest, sample


def proportional_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate removals by largest remainder while preserving group proportions."""
    if total < 0:
        raise ValueError("Removal total cannot be negative")
    available = sum(counts.values())
    if total > available:
        raise ValueError(f"Need {total} removals but only {available} candidates exist")
    if total == 0:
        return {group: 0 for group in sorted(counts)}

    exact = {group: total * count / available for group, count in counts.items()}
    quota = {group: int(value) for group, value in exact.items()}
    remainder = total - sum(quota.values())
    order = sorted(counts, key=lambda group: (-(exact[group] - quota[group]), group))
    for group in order[:remainder]:
        quota[group] += 1
    if any(quota[group] > counts[group] for group in quota):
        raise ValueError("Proportional allocation exceeded a group capacity")
    return quota


def represented_training_set(
    strict_ids: list[str],
    truth: dict,
    truth_rows: list[dict[str, str]],
    population: dict[str, str],
    seed: int,
) -> tuple[list[str], dict]:
    strict = set(strict_ids)
    universe = set(population)
    if not strict <= universe:
        raise ValueError("Strict set contains samples outside fixture metadata")

    pedigree = recent_pairs(truth_rows)
    offspring = {
        person
        for unit in truth["pedigree_units"]
        for person in (unit["child"], unit["half_sibling"])
    }
    required: set[str] = set()
    for deme, members in truth["demes"].items():
        candidates = set(members) - offspring
        if deme != "DEME_C" and candidates != set(members):
            raise ValueError(f"Pure-coancestry deme {deme} unexpectedly contains offspring")
        required.update(candidates)

    required_conflicts = count_internal_pairs(required, pedigree)
    if required_conflicts:
        raise ValueError(f"Required represented members contain pedigree pairs: {required_conflicts}")

    represented = set(strict)
    # Adding a founder can complete a known pair whose offspring survived the strict set.
    # Remove the non-required endpoint before adding the required members.
    represented.update(required)
    while count_internal_pairs(represented, pedigree):
        left, right = count_internal_pairs(represented, pedigree)[0]
        removable = [sample for sample in (left, right) if sample not in required]
        if not removable:
            raise ValueError(f"Cannot break pedigree pair {(left, right)} without removing a required member")
        represented.remove(sorted(removable, key=lambda value: stable_rank(value, seed))[0])

    target_size = len(strict)
    n_remove = len(represented) - target_size
    if n_remove < 0:
        raise ValueError("Represented set became smaller than the strict set before size matching")

    pedigree_people = {sample for pair in pedigree for sample in pair}
    removable_by_group: dict[str, list[str]] = defaultdict(list)
    for sample in represented:
        group = population[sample]
        if not group.startswith("POP_BG"):
            continue
        if sample in pedigree_people:
            continue
        removable_by_group[group].append(sample)
    for samples in removable_by_group.values():
        samples.sort(key=lambda value: stable_rank(value, seed))

    quota = proportional_quotas(
        {group: len(samples) for group, samples in removable_by_group.items()}, n_remove
    )
    removed_for_size = {
        sample
        for group, amount in quota.items()
        for sample in removable_by_group[group][:amount]
    }
    represented.difference_update(removed_for_size)

    if len(represented) != target_size:
        raise AssertionError(f"Size matching failed: expected {target_size}, got {len(represented)}")
    remaining_conflicts = count_internal_pairs(represented, pedigree)
    if remaining_conflicts:
        raise AssertionError(f"Represented set retains pedigree pairs: {remaining_conflicts}")
    if not required <= represented:
        raise AssertionError("A required deme representative was removed during size matching")

    ordered = sorted(represented)
    strict_counts = Counter(population[sample] for sample in strict)
    represented_counts = Counter(population[sample] for sample in represented)
    summary = {
        "stage": "M27D_SYNTHETIC_TRAINING_SET_INTERVENTION",
        "diagnostic_only": True,
        "uses_final_pcrelate_estimates": False,
        "selection_inputs": ["strict_training_set", "synthetic_deme_labels", "synthetic_pedigree_truth"],
        "seed": seed,
        "n_strict": len(strict),
        "n_represented": len(represented),
        "same_size_as_strict": len(represented) == len(strict),
        "required_deme_members": sorted(required),
        "added_to_strict": sorted(represented - strict),
        "removed_from_strict": sorted(strict - represented),
        "removed_for_size_matching": sorted(removed_for_size),
        "removal_quota_by_background_group": dict(sorted(quota.items())),
        "population_counts_strict": dict(sorted(strict_counts.items())),
        "population_counts_represented": dict(sorted(represented_counts.items())),
        "n_recent_pairs_both_in_strict": len(count_internal_pairs(strict, pedigree)),
        "n_recent_pairs_both_in_represented": len(remaining_conflicts),
        "represented_set_sha256": hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest(),
    }
    return ordered, summary


def write_ids(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-set", type=Path, required=True)
    parser.add_argument("--truth-json", type=Path, required=True)
    parser.add_argument("--truth-pairs", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-set", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    truth = json.loads(args.truth_json.read_text(encoding="utf-8"))
    values, summary = represented_training_set(
        read_ids(args.strict_set),
        truth,
        read_truth_pairs(args.truth_pairs),
        read_population(args.metadata),
        args.seed,
    )
    args.out_set.parent.mkdir(parents=True, exist_ok=True)
    write_ids(args.out_set, values)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
