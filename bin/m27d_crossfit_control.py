#!/usr/bin/env python3
"""Cross-fitted known-answer control for the M27D PC-Relate training set.

Each of the 15 pairs in the two six-person pure-coancestry demes is evaluated once.
Both endpoints are excluded from the PCA fit and from PC-Relate's ``training.set``;
the other four members of the same deme remain as representatives.  First-cousin
controls are generated from latent ancestors and are excluded from every fitting set.

This is a local synthetic diagnostic.  It uses oracle deme labels and cannot define a
donor-selection policy for the real panel.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m27d_coancestry_screen import SCENARIOS, load  # noqa: E402
from m27d_fixture_scoring import read_pairs  # noqa: E402
from m27d_pipeline_chain import IMAGE, image_available  # noqa: E402
from m27d_representation_control import (  # noqa: E402
    PRIMARY_CONFIGURATION,
    PRIMARY_DEMES,
    PointContext,
    git_receipt,
    prepare_context,
    run_training_arm,
    sha256_file,
)
from m27d_synthetic_cohort import CohortLayout, first_cousin_units  # noqa: E402
from m27d_training_set_intervention import (  # noqa: E402
    count_internal_pairs,
    read_ids,
    read_population,
    read_truth_pairs,
    recent_pairs,
    stable_rank,
)


PREREGISTRATION = (
    Path(__file__).resolve().parents[1]
    / "conf"
    / "m27d_crossfit_control_preregistration.json"
)
CONTROL = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
ARM_STRICT_SAFE = "strict_without_cousin_endpoints"
ARM_REPRESENTED_IN_SAMPLE = "represented_in_sample"
ARM_LPO_PREFIX = "leave_pair_out"


def _pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def safe_run_id(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in value
    ):
        raise ValueError(f"Unsafe run identifier: {value!r}")
    return value


def _bool(value: object) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() == "true"


def _relationship_pairs(
    truth_rows: list[dict[str, str]], degrees: set[int]
) -> list[tuple[str, str]]:
    return sorted(
        {
            _pair(row["ID1"], row["ID2"])
            for row in truth_rows
            if int(row["true_degree"]) in degrees and _bool(row["has_recent_kinship"])
        }
    )


def _safe_background_candidates(
    universe: set[str],
    population: dict[str, str],
    pedigree_people: set[str],
    excluded: set[str],
    seed: int,
) -> list[str]:
    return sorted(
        (
            sample
            for sample in universe - excluded
            if population[sample].startswith("POP_BG") and sample not in pedigree_people
        ),
        key=lambda sample: stable_rank(sample, seed),
    )


def size_matched_fit_set(
    base_ids: list[str],
    target_size: int,
    truth_rows: list[dict[str, str]],
    population: dict[str, str],
    required: set[str],
    forbidden: set[str],
    seed: int,
    label: str,
) -> tuple[list[str], dict[str, object]]:
    """Apply a deterministic, auditable intervention to one fitting set."""
    universe = set(population)
    if not set(base_ids) <= universe:
        raise ValueError(f"{label}: base set contains samples outside metadata")
    if required & forbidden:
        raise ValueError(f"{label}: required and forbidden overlap: {sorted(required & forbidden)}")
    if not required <= universe or not forbidden <= universe:
        raise ValueError(f"{label}: required/forbidden samples fall outside metadata")

    pedigree = recent_pairs(truth_rows)
    pedigree_people = {sample for pair in pedigree for sample in pair}
    selected = (set(base_ids) - forbidden) | required

    conflicts = count_internal_pairs(selected, pedigree)
    while conflicts:
        left, right = conflicts[0]
        removable = [sample for sample in (left, right) if sample not in required]
        if not removable:
            raise ValueError(f"{label}: required members complete recent pair {(left, right)}")
        selected.remove(sorted(removable, key=lambda sample: stable_rank(sample, seed))[0])
        conflicts = count_internal_pairs(selected, pedigree)

    candidates = _safe_background_candidates(
        universe, population, pedigree_people, forbidden | selected, seed
    )
    if len(selected) < target_size:
        need = target_size - len(selected)
        if len(candidates) < need:
            raise ValueError(
                f"{label}: need {need} safe background replacements, have {len(candidates)}"
            )
        selected.update(candidates[:need])
    elif len(selected) > target_size:
        removable = _safe_background_candidates(
            selected, population, pedigree_people, required | forbidden, seed
        )
        need = len(selected) - target_size
        if len(removable) < need:
            raise ValueError(f"{label}: cannot remove {need} safe background members")
        selected.difference_update(removable[:need])

    if len(selected) != target_size:
        raise AssertionError(f"{label}: expected size {target_size}, got {len(selected)}")
    if required - selected:
        raise AssertionError(f"{label}: required members were lost")
    if forbidden & selected:
        raise AssertionError(f"{label}: forbidden members entered the fit")
    remaining_conflicts = count_internal_pairs(selected, pedigree)
    if remaining_conflicts:
        raise AssertionError(f"{label}: recent pairs remain: {remaining_conflicts}")

    ordered = sorted(selected)
    receipt = {
        "label": label,
        "target_size": target_size,
        "n_selected": len(ordered),
        "required": sorted(required),
        "forbidden": sorted(forbidden),
        "added_vs_base": sorted(selected - set(base_ids)),
        "removed_vs_base": sorted(set(base_ids) - selected),
        "n_recent_pairs_both_in_set": 0,
        "set_sha256": hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest(),
    }
    return ordered, receipt


def _pair_record(
    observed: dict[tuple[str, str], float],
    pair: tuple[str, str],
    threshold: float,
) -> dict[str, object]:
    value = observed.get(pair)
    return {
        "id1": pair[0],
        "id2": pair[1],
        "kinship": None if value is None else round(float(value), 8),
        "reported": value is not None,
        "detected_at_primary_threshold": bool(value is not None and value >= threshold),
    }


def _positive_control_summary(
    observed: dict[tuple[str, str], float],
    truth_rows: list[dict[str, str]],
    threshold: float,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name, degrees in (("first_degree", {1}), ("second_degree", {2})):
        pairs = _relationship_pairs(truth_rows, degrees)
        detected = sum(observed.get(pair, float("-inf")) >= threshold for pair in pairs)
        summary[name] = {
            "n_pairs": len(pairs),
            "n_detected": detected,
            "all_detected": detected == len(pairs),
        }
    return summary


def _cousin_records(
    observed: dict[tuple[str, str], float],
    truth: dict,
    threshold: float,
) -> list[dict[str, object]]:
    records = []
    for unit in truth["first_cousin_units"]:
        record = _pair_record(
            observed, _pair(unit["cousin_1"], unit["cousin_2"]), threshold
        )
        record.update(
            {
                "pedigree_location": unit["location"],
                "pedigree_group": unit["group"],
                "pedigree_phi": 0.0625,
            }
        )
        records.append(record)
    return records


def _arm_observations(
    context: PointContext, arm: str, threshold: float
) -> tuple[dict[tuple[str, str], float], Path]:
    pair_path = (
        context.workspace
        / arm
        / f"m27d_pcrelate_{PRIMARY_CONFIGURATION}_pairs.private.tsv.gz"
    )
    return read_pairs(pair_path), pair_path


def _persist_arm(context: PointContext, arm: str, destination: Path) -> None:
    source = context.workspace / arm
    target = destination / arm
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "pca_training_set.txt",
        "pcrelate_training_set.txt",
        "training_set_intervention.json",
        "m27d_pca_anchor.json",
        f"m27d_pcrelate_{PRIMARY_CONFIGURATION}.json",
        f"m27d_pcrelate_{PRIMARY_CONFIGURATION}_pairs.private.tsv.gz",
    ):
        shutil.copy2(source / name, target / name)


def run_seed(
    seed: int,
    scenario,
    layout: CohortLayout,
    repo: Path,
    base_preregistration: Path,
    threads: int,
    point_timeout: int,
    deadline: float,
) -> tuple[dict[str, object], PointContext]:
    if time.monotonic() >= deadline:
        raise TimeoutError("Wall budget exhausted before seed preparation")
    context = prepare_context(
        scenario,
        seed,
        layout,
        repo,
        base_preregistration,
        threads,
        point_timeout,
        pass0_excluded=set(
            sample
            for unit in first_cousin_units(layout)
            for sample in (unit["cousin_1"], unit["cousin_2"])
        ),
    )
    truth = load(context.fixture_dir / "truth.json")
    truth_rows = read_truth_pairs(context.fixture_dir / "truth_pairs.tsv")
    population = read_population(context.fixture_dir / "metadata.tsv")
    strict = read_ids(context.base_out / "training_set.txt")
    target_size = len(strict)
    cousin_endpoints = set(truth["always_excluded_from_training"])
    primary_members = {
        deme: list(truth["demes"][deme]) for deme in PRIMARY_DEMES
    }
    if any(len(members) != 6 for members in primary_members.values()):
        raise ValueError("Primary demes must each contain exactly six members")

    strict_safe, strict_receipt = size_matched_fit_set(
        strict,
        target_size,
        truth_rows,
        population,
        required=set(),
        forbidden=cousin_endpoints,
        seed=seed,
        label=ARM_STRICT_SAFE,
    )
    represented_required = {
        sample for members in primary_members.values() for sample in members
    }
    represented, represented_receipt = size_matched_fit_set(
        context.represented_set,
        target_size,
        truth_rows,
        population,
        required=represented_required,
        forbidden=cousin_endpoints,
        seed=seed,
        label=ARM_REPRESENTED_IN_SAMPLE,
    )

    strict_result = run_training_arm(
        context, ARM_STRICT_SAFE, repo, threads, point_timeout,
        strict_safe, strict_safe,
        intervention_summary=strict_receipt,
    )
    represented_result = run_training_arm(
        context, ARM_REPRESENTED_IN_SAMPLE, repo, threads, point_timeout,
        represented, represented,
        intervention_summary=represented_receipt,
    )
    threshold = float(CONTROL["estimand"]["primary_threshold"])
    strict_observed, _ = _arm_observations(context, ARM_STRICT_SAFE, threshold)
    represented_observed, _ = _arm_observations(
        context, ARM_REPRESENTED_IN_SAMPLE, threshold
    )
    baseline = {
        ARM_STRICT_SAFE: {
            "fit_set": strict_receipt,
            "primary_units": strict_result["primary_units"],
            "first_cousins": _cousin_records(strict_observed, truth, threshold),
            "other_positive_controls": _positive_control_summary(
                strict_observed, truth_rows, threshold
            ),
        },
        ARM_REPRESENTED_IN_SAMPLE: {
            "fit_set": represented_receipt,
            "primary_units": represented_result["primary_units"],
            "first_cousins": _cousin_records(represented_observed, truth, threshold),
            "other_positive_controls": _positive_control_summary(
                represented_observed, truth_rows, threshold
            ),
        },
    }

    folds = []
    for fold_index, (left_index, right_index) in enumerate(itertools.combinations(range(6), 2)):
        if time.monotonic() >= deadline:
            raise TimeoutError("Wall budget exhausted before the next leave-pair-out fold")
        held_out = {
            primary_members[deme][index]
            for deme in PRIMARY_DEMES
            for index in (left_index, right_index)
        }
        required = represented_required - held_out
        forbidden = cousin_endpoints | held_out
        arm = f"{ARM_LPO_PREFIX}_{left_index}_{right_index}"
        fit_set, receipt = size_matched_fit_set(
            represented,
            target_size,
            truth_rows,
            population,
            required=required,
            forbidden=forbidden,
            seed=seed,
            label=arm,
        )
        run_training_arm(
            context,
            arm,
            repo,
            threads,
            point_timeout,
            fit_set,
            fit_set,
            intervention_summary=receipt,
        )
        observed, _ = _arm_observations(context, arm, threshold)
        evaluated = {}
        for deme in PRIMARY_DEMES:
            pair = _pair(
                primary_members[deme][left_index],
                primary_members[deme][right_index],
            )
            evaluated[deme] = _pair_record(observed, pair, threshold)
        folds.append(
            {
                "fold_index": fold_index,
                "pair_indices": [left_index, right_index],
                "arm": arm,
                "fit_set": receipt,
                "evaluated_primary_pairs": evaluated,
                "first_cousins": _cousin_records(observed, truth, threshold),
                "other_positive_controls": _positive_control_summary(
                    observed, truth_rows, threshold
                ),
            }
        )

    seen = {
        deme: {
            _pair(row["id1"], row["id2"])
            for fold in folds
            for name, row in fold["evaluated_primary_pairs"].items()
            if name == deme
        }
        for deme in PRIMARY_DEMES
    }
    expected = {
        deme: {
            _pair(left, right)
            for left, right in itertools.combinations(primary_members[deme], 2)
        }
        for deme in PRIMARY_DEMES
    }
    if seen != expected:
        raise AssertionError("Leave-pair-out folds did not cover every primary pair exactly once")

    crossfit_units = {}
    for deme in PRIMARY_DEMES:
        detected = sum(
            fold["evaluated_primary_pairs"][deme]["detected_at_primary_threshold"]
            for fold in folds
        )
        crossfit_units[deme] = {
            "repeated_subunit": f"deme={deme}",
            "n_pairs_descriptive_only": 15,
            "n_recent_pedigree_false_positives": detected,
            "false_positive_fraction": round(detected / 15, 6),
        }

    cousin_stability = []
    for unit in truth["first_cousin_units"]:
        pair = _pair(unit["cousin_1"], unit["cousin_2"])
        records = [
            next(
                row for row in fold["first_cousins"]
                if _pair(row["id1"], row["id2"]) == pair
            )
            for fold in folds
        ]
        values = [row["kinship"] for row in records if row["kinship"] is not None]
        cousin_stability.append(
            {
                "id1": pair[0],
                "id2": pair[1],
                "pedigree_location": unit["location"],
                "pedigree_phi": 0.0625,
                "n_folds": len(records),
                "n_detected": sum(row["detected_at_primary_threshold"] for row in records),
                "detected_in_every_fold": all(
                    row["detected_at_primary_threshold"] for row in records
                ),
                "kinship_min": None if not values else round(min(values), 8),
                "kinship_max": None if not values else round(max(values), 8),
            }
        )

    return (
        {
            "seed": seed,
            "baseline_arms": baseline,
            "leave_pair_out": {
                "n_folds": len(folds),
                "primary_units": crossfit_units,
                "first_cousin_stability": cousin_stability,
                "folds": folds,
            },
        },
        context,
    )


def adjudicate(seed_results: list[dict[str, object]]) -> dict[str, object]:
    maximum = int(
        CONTROL["analysis"]["primary_gate"]
        ["maximum_false_positives_per_15_pair_deme_subunit"]
    )
    fixture_rows = []
    primary_rows = []
    cousin_rows = []
    regression_rows = []
    expected_seeds = {int(value) for value in CONTROL["fixture"]["pilot_seeds"]}
    observed_seeds = {int(point["seed"]) for point in seed_results}
    all_seeds_present_once = (
        observed_seeds == expected_seeds and len(seed_results) == len(expected_seeds)
    )
    for point in seed_results:
        seed = point["seed"]
        baselines = point["baseline_arms"]
        strict = baselines[ARM_STRICT_SAFE]
        represented = baselines[ARM_REPRESENTED_IN_SAMPLE]
        crossfit = point["leave_pair_out"]
        for deme in PRIMARY_DEMES:
            strict_count = strict["primary_units"][deme][
                "n_recent_pedigree_false_positives"
            ]
            represented_count = represented["primary_units"][deme][
                "n_recent_pedigree_false_positives"
            ]
            crossfit_count = crossfit["primary_units"][deme][
                "n_recent_pedigree_false_positives"
            ]
            fixture_rows.append(
                {
                    "seed": seed,
                    "deme": deme,
                    "strict_count": strict_count,
                    "represented_in_sample_count": represented_count,
                    "strict_failure_reproduced": (
                        strict_count > maximum if deme == "DEME_A" else True
                    ),
                    "in_sample_rescue_reproduced": represented_count <= maximum,
                }
            )
            primary_rows.append(
                {
                    "seed": seed,
                    "deme": deme,
                    "strict_count": strict_count,
                    "crossfit_count": crossfit_count,
                    "near_floor": crossfit_count <= maximum,
                    "not_worse": crossfit_count <= strict_count,
                    "improved_when_possible": strict_count == 0 or crossfit_count < strict_count,
                }
            )
        strict_cousins = {
            _pair(row["id1"], row["id2"]): row
            for row in strict["first_cousins"]
        }
        for row in crossfit["first_cousin_stability"]:
            pair = _pair(row["id1"], row["id2"])
            cousin_rows.append(
                {
                    "seed": seed,
                    "pedigree_location": row["pedigree_location"],
                    "strict_detected": strict_cousins[pair][
                        "detected_at_primary_threshold"
                    ],
                    "crossfit_detected_in_every_fold": row["detected_in_every_fold"],
                    "crossfit_n_detected": row["n_detected"],
                    "crossfit_n_folds": row["n_folds"],
                }
            )
        for fold in crossfit["folds"]:
            regression_rows.append(
                {
                    "seed": seed,
                    "fold_index": fold["fold_index"],
                    **{
                        f"{degree}_all_detected": block["all_detected"]
                        for degree, block in fold["other_positive_controls"].items()
                    },
                }
            )

    fixture_valid = all_seeds_present_once and all(
        row["strict_failure_reproduced"] and row["in_sample_rescue_reproduced"]
        for row in fixture_rows
    )
    cousins_valid = all(row["strict_detected"] for row in cousin_rows)
    primary_pass = all(
        row["near_floor"] and row["not_worse"] and row["improved_when_possible"]
        for row in primary_rows
    )
    cousins_pass = cousins_valid and all(
        row["crossfit_detected_in_every_fold"] for row in cousin_rows
    )
    regression_pass = all(
        row["first_degree_all_detected"] and row["second_degree_all_detected"]
        for row in regression_rows
    )
    if not fixture_valid or not cousins_valid:
        verdict = "INCONCLUSIVE_FIXTURE_CONTROL_FAILED"
    elif primary_pass and cousins_pass and regression_pass:
        verdict = "PASS_SYNTHETIC_CROSSFIT_ONLY"
    else:
        verdict = "STOP_REPRESENTATION_AS_SOLUTION"
    return {
        "verdict": verdict,
        "fixture_reproduction_pass": fixture_valid,
        "all_preregistered_seeds_present_once": all_seeds_present_once,
        "strict_first_cousin_control_pass": cousins_valid,
        "primary_crossfit_pass": primary_pass,
        "first_cousin_crossfit_pass": cousins_pass,
        "first_second_degree_regression_pass": regression_pass,
        "fixture_rows": fixture_rows,
        "primary_rows": primary_rows,
        "first_cousin_rows": cousin_rows,
        "n_independent_seed_replicates": len(observed_seeds),
        "pair_fold_or_deme_counts_are_replicates": False,
    }


def persist_seed_details(
    context: PointContext, point: dict[str, object], destination: Path
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("truth.json", "truth_pairs.tsv", "metadata.tsv", "prereg.json"):
        shutil.copy2(context.fixture_dir / name, destination / name)
    for name in (
        "training_set.txt",
        "training_set_alt.txt",
        "training_set.json",
        "m27d_pass0_pcrelate.json",
        "m27d_pass0_sample_universe.private.txt",
        "m27d_pass0_related_pairs.private.tsv.gz",
        "crossfit_pass0_exclusion.json",
    ):
        shutil.copy2(context.base_out / name, destination / name)
    arms = [ARM_STRICT_SAFE, ARM_REPRESENTED_IN_SAMPLE] + [
        fold["arm"] for fold in point["leave_pair_out"]["folds"]
    ]
    for arm in arms:
        _persist_arm(context, arm, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--base-preregistration",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "conf"
        / "m27d_donor_kinship_preregistration.json",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    safe_run_id(args.run_id)
    manifest_path = args.out.parent / f"{args.run_id}.manifest.json"
    resolved_outputs = {
        args.out.resolve(), args.details_dir.resolve(), manifest_path.resolve()
    }
    if len(resolved_outputs) != 3:
        raise SystemExit("Result, details and manifest paths must be distinct")
    if not image_available():
        raise SystemExit("Pinned M27D analysis image is unavailable")
    receipt = git_receipt(repo)
    if not receipt["working_tree_clean"]:
        raise SystemExit(f"Refusing to run from a dirty tree: {receipt['dirty_paths']}")
    if args.out.exists() or args.details_dir.exists() or manifest_path.exists():
        raise SystemExit("Output path already exists; cross-fit controls never overwrite")
    for output_path in (args.out.resolve(), args.details_dir.resolve(), manifest_path.resolve()):
        if output_path.is_relative_to(repo):
            raise SystemExit("Private cross-fit outputs must remain outside the repository")
    parent = repo / CONTROL["parent_control"]["path"]
    parent_hash = sha256_file(parent)
    if parent_hash != CONTROL["parent_control"]["sha256"]:
        raise SystemExit(
            f"Parent preregistration hash mismatch: expected "
            f"{CONTROL['parent_control']['sha256']}, observed {parent_hash}"
        )
    base = load(args.base_preregistration)
    parent_control = load(parent)
    base_hash = sha256_file(args.base_preregistration)
    expected_base_hash = parent_control["base_preregistration"]["sha256"]
    if CONTROL["base_preregistration"]["sha256"] != expected_base_hash:
        raise SystemExit("Cross-fit and parent controls pin different base contracts")
    if base_hash != expected_base_hash:
        raise SystemExit(
            f"Base preregistration hash mismatch: expected {expected_base_hash}, "
            f"observed {base_hash}"
        )
    threshold = float(base["pcrelate"]["primary_phi_threshold"])
    if threshold != float(CONTROL["estimand"]["primary_threshold"]):
        raise SystemExit("Primary threshold differs between base and cross-fit contracts")
    matching_configurations = [
        row for row in base["configurations"]
        if row["id"] == CONTROL["analysis"]["configuration_id"]
    ]
    if len(matching_configurations) != 1:
        raise SystemExit("Primary configuration is absent or duplicated in the base contract")
    primary_configuration = matching_configurations[0]
    if (
        int(primary_configuration["n_pcs"]) != 8
        or abs(float(primary_configuration["ld_r2_max"]) - 0.20) > 1e-12
    ):
        raise SystemExit("Primary configuration is not the frozen PC8 / LD r2=0.20 setting")
    if tuple(CONTROL["estimand"]["primary_demes"]) != PRIMARY_DEMES:
        raise SystemExit("Primary deme contract differs from the parent control")
    if CONTROL["analysis"]["configuration_id"] != PRIMARY_CONFIGURATION:
        raise SystemExit("Primary configuration differs from the parent control")
    fixture = CONTROL["fixture"]
    if [int(value) for value in fixture["pilot_seeds"]] != [
        int(value) for value in parent_control["fixture"]["pilot_seeds"]
    ]:
        raise SystemExit("Pilot seeds differ from the parent representation control")

    resources = CONTROL["resources"]
    layout = CohortLayout(
        n_deme_members=int(fixture["pure_deme_members"]),
        n_pedigree_deme_members=int(fixture["pedigree_deme_members"]),
        n_first_cousin_pairs_in_deme=int(
            fixture["first_cousin_pairs_per_seed"]["deme"]
        ),
        n_first_cousin_pairs_in_background=int(
            fixture["first_cousin_pairs_per_seed"]["background"]
        ),
        n_markers_per_chromosome=int(fixture["markers_per_chromosome"]),
        n_chromosomes=int(fixture["chromosomes"]),
    )
    scenario = next(
        value for value in SCENARIOS if value.name == str(fixture["scenario"])
    )
    expected_cousins = sum(fixture["first_cousin_pairs_per_seed"].values())
    if len(first_cousin_units(layout)) != expected_cousins:
        raise SystemExit("Fixture does not contain the preregistered cousin controls")

    started = time.monotonic()
    deadline = started + int(resources["max_wall_seconds"])
    seed_results: list[dict[str, object]] = []
    contexts: list[PointContext] = []
    staging_result: Path | None = None
    staging_manifest: Path | None = None
    staging_details: Path | None = None
    try:
        for seed in [int(value) for value in fixture["pilot_seeds"]]:
            point, context = run_seed(
                seed,
                scenario,
                layout,
                repo,
                args.base_preregistration,
                int(resources["threads"]),
                int(resources["point_timeout_seconds"]),
                deadline,
            )
            seed_results.append(point)
            contexts.append(context)
            print(f"[OK] seed={seed} baseline=2 lpo=15", flush=True)

        elapsed = time.monotonic() - started
        if elapsed > int(resources["max_wall_seconds"]):
            raise TimeoutError("Cross-fit control exceeded its preregistered wall budget")
        decision = adjudicate(seed_results)

        args.details_dir.parent.mkdir(parents=True, exist_ok=True)
        token = hashlib.sha256(args.run_id.encode()).hexdigest()[:12]
        staging_details = args.details_dir.parent / f".{args.details_dir.name}.partial-{token}"
        staging_result = args.out.parent / f".{args.out.name}.partial"
        staging_manifest = args.out.parent / f".{args.run_id}.manifest.json.partial"
        if any(path.exists() for path in (staging_details, staging_result, staging_manifest)):
            raise SystemExit("Staged output already exists; inspect it before retrying")
        staging_details.mkdir()
        for point, context in zip(seed_results, contexts):
            persist_seed_details(
                context, point, staging_details / f"isolate-seed{point['seed']}"
            )

        code_paths = [
            repo / "bin" / name
            for name in (
                "m27d_crossfit_control.py",
                "m27d_representation_control.py",
                "m27d_training_set_intervention.py",
                "m27d_synthetic_cohort.py",
                "m27d_fixture_scoring.py",
                "m27d_pipeline_chain.py",
                "m27d_pca_projection.R",
                "m27d_pcrelate_configuration.R",
            )
        ]
        result = {
            "stage": CONTROL["stage"],
            "run_id": args.run_id,
            "synthetic_only": True,
            "scientific_result": False,
            "cloud_executed": False,
            "king_executed": False,
            "pcair_used": False,
            "gnomix_accessed": False,
            "primary_configuration": PRIMARY_CONFIGURATION,
            "primary_experimental_unit": "independently_generated_seed",
            "repeated_subunits": CONTROL["estimand"]["repeated_subunits"],
            "pair_or_fold_counts_are_replicates": False,
            "results": seed_results,
            "decision": decision,
            "elapsed_seconds": round(elapsed, 1),
            "wall_clock_budget_seconds": int(resources["max_wall_seconds"]),
            "container": IMAGE,
            "crossfit_preregistration": {
                "path": str(PREREGISTRATION.relative_to(repo)),
                "sha256": sha256_file(PREREGISTRATION),
            },
            "parent_control_preregistration": {
                "path": str(parent.relative_to(repo)),
                "sha256": parent_hash,
            },
            "base_preregistration": {
                "path": str(args.base_preregistration.resolve()),
                "sha256": base_hash,
            },
            "git": receipt,
            "code_sha256": {
                str(path.relative_to(repo)): sha256_file(path) for path in code_paths
            },
            "command_argv": sys.argv,
            "declared_limitations": CONTROL["declared_limitations"],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        staging_result.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        details_files = [
            {
                "path": str(path.relative_to(staging_details)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging_details.rglob("*"))
            if path.is_file()
        ]
        manifest = {
            "run_id": args.run_id,
            "result": str(args.out),
            "result_sha256": sha256_file(staging_result),
            "details_dir": str(args.details_dir),
            "details_file_count": len(details_files),
            "details_files": details_files,
            "git": receipt,
            "container": IMAGE,
            "elapsed_seconds": round(elapsed, 1),
        }
        staging_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging_details.replace(args.details_dir)
        staging_result.replace(args.out)
        staging_manifest.replace(manifest_path)
    finally:
        for context in contexts:
            shutil.rmtree(context.workspace, ignore_errors=True)
        if staging_details is not None:
            shutil.rmtree(staging_details, ignore_errors=True)
        for path in (staging_result, staging_manifest):
            if path is not None:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
