#!/usr/bin/env python3
"""Identify whether small-deme loss enters M27D through PCA representation.

The control changes one object at a time on the same synthetic genotypes:

* ``strict_strict`` fits PCA and PC-Relate on the pass0 independent set;
* ``represented_strict`` fits PCA on a size-matched set that represents the demes,
  while PC-Relate still estimates frequencies on the strict set;
* ``represented_represented`` is run only when the first intervention fails.  It is a
  diagnostic of the PC-Relate training set, not a proposed donor policy.

The independently generated seed is the primary replicate.  DEME_A and DEME_B share an
intermediate ancestral-frequency draw inside a seed, so they are repeated subunits, not
two independent observations.  Pair counts are descriptive within each deme.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m27d_coancestry_screen import SCENARIOS, load, score_run  # noqa: E402
from m27d_fixture_scoring import read_pairs, read_truth  # noqa: E402
from m27d_pipeline_chain import (  # noqa: E402
    IMAGE,
    REFIT_CONFIGURATIONS,
    THROUGH_TRAINING_SET,
    image_available,
    run as run_chain,
)
from m27d_synthetic_cohort import CohortLayout, Scenario, build  # noqa: E402
from m27d_training_set_intervention import (  # noqa: E402
    read_ids,
    read_population,
    read_truth_pairs,
    represented_training_set,
    write_ids,
)


CONTROL_PREREGISTRATION = (
    Path(__file__).resolve().parents[1] / "conf" / "m27d_representation_control_preregistration.json"
)
CONTROL = json.loads(CONTROL_PREREGISTRATION.read_text(encoding="utf-8"))
PRIMARY_CONFIGURATION = CONTROL["analysis"]["configuration_id"]
PRIMARY_DEMES = tuple(CONTROL["estimand"]["primary_demes"])
PILOT_SEEDS = tuple(int(value) for value in CONTROL["fixture"]["pilot_seeds"])
CONFIRMATION_SEED_COUNT = int(CONTROL["fixture"]["confirmation_seed_count"])
MAX_FALSE_POSITIVES_PER_SUBUNIT = int(
    CONTROL["analysis"]["primary_gate"]["maximum_false_positives_per_15_pair_deme_subunit"]
)
ARM_STRICT = "strict_strict"
ARM_PCA_REPRESENTED = "represented_strict"
ARM_BOTH_REPRESENTED = "represented_represented"


@dataclass
class PointContext:
    workspace: Path
    fixture_dir: Path
    base_out: Path
    represented_set: list[str]
    intervention_summary: dict
    scenario: Scenario
    seed: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_receipt(repo: Path) -> dict[str, object]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip().splitlines()
    return {"commit": head, "working_tree_clean": not dirty, "dirty_paths": dirty}


def clone_base(source: Path, destination: Path) -> None:
    """Clone immutable preparation outputs; later stages only add new files."""
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def pair_false_positive_units(
    pairs_path: Path,
    truth_path: Path,
    threshold: float,
) -> dict[str, dict[str, object]]:
    """Classify recent-pedigree false positives by deme, without pair pseudoreplication."""
    observed = read_pairs(pairs_path)
    truth = read_truth(truth_path)
    result: dict[str, dict[str, object]] = {}
    for deme in PRIMARY_DEMES:
        rows = [
            row
            for row in truth
            if row["true_relationship"] == "unrelated"
            and row["deme_1"] == deme
            and row["deme_2"] == deme
        ]
        if not rows:
            raise ValueError(f"No pedigree-unrelated within-deme pairs for {deme}")
        positives = 0
        absent = 0
        for row in rows:
            key = tuple(sorted((row["ID1"], row["ID2"])))
            value = observed.get(key)
            absent += value is None
            positives += value is not None and value >= threshold
        result[deme] = {
            "repeated_subunit": f"deme={deme}",
            "independent_replicate_declared_by_caller": "seed",
            "n_pairs_descriptive_only": len(rows),
            "n_recent_pedigree_false_positives": positives,
            "false_positive_fraction": round(positives / len(rows), 6),
            "n_below_reporting_threshold": absent,
            "threshold": threshold,
        }
    return result


def derive_confirmation_seeds(run_id: str, count: int, excluded: set[int]) -> list[int]:
    """Derive seeds before data exist, so confirmation cannot cherry-pick them."""
    seeds: list[int] = []
    index = 0
    while len(seeds) < count:
        raw = hashlib.sha256(f"{run_id}|confirmation-seed|{index}".encode()).digest()
        value = int.from_bytes(raw[:4], "big") % 2_147_483_647
        index += 1
        if value and value not in excluded and value not in seeds:
            seeds.append(value)
    return seeds


def intervention_supported(
    point_results: list[dict], intervention_arm: str = ARM_PCA_REPRESENTED
) -> dict[str, object]:
    units = []
    for point in point_results:
        control = point["arms"][ARM_STRICT]["primary_units"]
        intervention = point["arms"][intervention_arm]["primary_units"]
        for deme in PRIMARY_DEMES:
            before_fraction = control[deme]["false_positive_fraction"]
            after_fraction = intervention[deme]["false_positive_fraction"]
            before_count = control[deme]["n_recent_pedigree_false_positives"]
            after_count = intervention[deme]["n_recent_pedigree_false_positives"]
            units.append(
                {
                    "seed": point["seed"],
                    "deme": deme,
                    "control_arm": ARM_STRICT,
                    "intervention_arm": intervention_arm,
                    "control_false_positive_count": before_count,
                    "intervention_false_positive_count": after_count,
                    "n_pairs_descriptive_only": control[deme]["n_pairs_descriptive_only"],
                    "control_false_positive_fraction": before_fraction,
                    "intervention_false_positive_fraction": after_fraction,
                    "paired_delta_fraction": round(after_fraction - before_fraction, 6),
                    "near_floor": (
                        after_count <= MAX_FALSE_POSITIVES_PER_SUBUNIT
                    ),
                    "not_worse": after_count <= before_count,
                    "improved_when_possible": before_count == 0 or after_count < before_count,
                }
            )
    supported = bool(units) and all(
        row["near_floor"] and row["not_worse"] and row["improved_when_possible"]
        for row in units
    )
    return {
        "criterion": (
            "Every DEME_A/DEME_B x seed unit has at most one false positive among 15 pairs, "
            "never worsens, and improves whenever the strict arm was above zero"
        ),
        "supported": supported,
        "intervention_arm": intervention_arm,
        "n_independent_seed_units": len({row["seed"] for row in units}),
        "n_deme_subunits": len(units),
        "deme_subunits": units,
    }


def prepare_context(
    scenario: Scenario,
    seed: int,
    layout: CohortLayout,
    repo: Path,
    preregistration: Path,
    threads: int,
    point_timeout: int,
) -> PointContext:
    workspace = Path(tempfile.mkdtemp(prefix=f"m27d-representation-{scenario.name}-{seed}-"))
    try:
        fixture_dir = workspace / "fixture"
        base_out = workspace / "base"
        build(fixture_dir, preregistration, scenario, layout, seed)
        completed = run_chain(
            fixture_dir,
            base_out,
            repo,
            THROUGH_TRAINING_SET,
            threads=threads,
            timeout=point_timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Base chain failed for {scenario.name} seed={seed}:\n{completed.stderr[-4000:]}"
            )
        represented, summary = represented_training_set(
            read_ids(base_out / "training_set.txt"),
            load(fixture_dir / "truth.json"),
            read_truth_pairs(fixture_dir / "truth_pairs.tsv"),
            read_population(fixture_dir / "metadata.tsv"),
            seed,
        )
        return PointContext(
            workspace, fixture_dir, base_out, represented, summary, scenario, seed
        )
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


def run_arm(
    context: PointContext,
    arm: str,
    repo: Path,
    threads: int,
    point_timeout: int,
) -> dict[str, object]:
    arm_out = context.workspace / arm
    clone_base(context.base_out, arm_out)
    strict = read_ids(arm_out / "training_set.txt")
    if arm == ARM_STRICT:
        pca_set, pcrelate_set = strict, strict
    elif arm == ARM_PCA_REPRESENTED:
        pca_set, pcrelate_set = context.represented_set, strict
    elif arm == ARM_BOTH_REPRESENTED:
        pca_set, pcrelate_set = context.represented_set, context.represented_set
    else:
        raise ValueError(f"Unknown arm: {arm}")
    if len(pca_set) != len(strict) or len(pcrelate_set) != len(strict):
        raise ValueError(
            f"Arm {arm} is not size matched: strict={len(strict)}, "
            f"PCA={len(pca_set)}, PC-Relate={len(pcrelate_set)}"
        )

    pca_name = "pca_training_set.txt"
    pcrelate_name = "pcrelate_training_set.txt"
    write_ids(arm_out / pca_name, pca_set)
    write_ids(arm_out / pcrelate_name, pcrelate_set)
    passed_pca_name = "training_set.txt" if arm == ARM_STRICT else pca_name
    passed_pcrelate_name = (
        "pcrelate_training_set.txt" if arm == ARM_BOTH_REPRESENTED else "training_set.txt"
    )
    (arm_out / "training_set_intervention.json").write_text(
        json.dumps(context.intervention_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    completed = run_chain(
        context.fixture_dir,
        arm_out,
        repo,
        REFIT_CONFIGURATIONS,
        threads=threads,
        timeout=point_timeout,
        pca_training_set=passed_pca_name,
        pcrelate_training_set=passed_pcrelate_name,
        configuration_ids=(PRIMARY_CONFIGURATION,),
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Arm {arm} failed for {context.scenario.name} seed={context.seed}:\n"
            f"{completed.stderr[-4000:]}"
        )
    contract = load(context.fixture_dir / "prereg.json")
    threshold = float(contract["pcrelate"]["primary_phi_threshold"])
    pair_path = arm_out / f"m27d_pcrelate_{PRIMARY_CONFIGURATION}_pairs.private.tsv.gz"
    scored = score_run(
        arm_out,
        context.fixture_dir,
        contract,
        pca_training_set_path=arm_out / pca_name,
        pcrelate_training_set_path=arm_out / pcrelate_name,
    )
    return {
        "arm": arm,
        "elapsed_seconds": round(elapsed, 1),
        "pca_training_set_sha256": sha256_file(arm_out / pca_name),
        "pcrelate_training_set_sha256": sha256_file(arm_out / pcrelate_name),
        "pca_training_set_input_basename": passed_pca_name,
        "pcrelate_training_set_input_basename": passed_pcrelate_name,
        "primary_units": pair_false_positive_units(
            pair_path, context.fixture_dir / "truth_pairs.tsv", threshold
        ),
        "scored": scored,
    }


def persist_point_details(context: PointContext, destination: Path, arms: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("truth.json", "truth_pairs.tsv", "metadata.tsv", "prereg.json"):
        shutil.copy2(context.fixture_dir / name, destination / name)
    for name in ("training_set.txt", "training_set_alt.txt", "training_set.json"):
        shutil.copy2(context.base_out / name, destination / name)
    (destination / "represented_training_set.txt").write_text(
        "\n".join(context.represented_set) + "\n", encoding="utf-8"
    )
    (destination / "training_set_intervention.json").write_text(
        json.dumps(context.intervention_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for arm in arms:
        arm_source = context.workspace / arm
        arm_destination = destination / arm
        arm_destination.mkdir(exist_ok=True)
        for name in (
            "pca_training_set.txt",
            "pcrelate_training_set.txt",
            f"m27d_pcrelate_{PRIMARY_CONFIGURATION}.json",
            f"m27d_pcrelate_{PRIMARY_CONFIGURATION}_pairs.private.tsv.gz",
            "m27d_pca_anchor.json",
        ):
            shutil.copy2(arm_source / name, arm_destination / name)


def execute_points(
    seeds: list[int],
    scenario: Scenario,
    layout: CohortLayout,
    repo: Path,
    preregistration: Path,
    threads: int,
    point_timeout: int,
    arms: list[str],
    deadline: float,
) -> tuple[list[dict], list[PointContext]]:
    results: list[dict] = []
    contexts: list[PointContext] = []
    try:
        for seed in seeds:
            if time.monotonic() >= deadline:
                raise TimeoutError("Wall-clock budget exhausted before the next seed")
            context = prepare_context(
                scenario, seed, layout, repo, preregistration, threads, point_timeout
            )
            contexts.append(context)
            point = {"scenario": scenario.name, "seed": seed, "arms": {}}
            for arm in arms:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Wall-clock budget exhausted before the next arm")
                point["arms"][arm] = run_arm(context, arm, repo, threads, point_timeout)
            results.append(point)
            print(f"[OK] {scenario.name} seed={seed} arms={','.join(arms)}", flush=True)
    except Exception:
        for context in contexts:
            shutil.rmtree(context.workspace, ignore_errors=True)
        raise
    return results, contexts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "conf" / "m27d_donor_kinship_preregistration.json",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    args = parser.parse_args()

    resources = CONTROL["resources"]
    threads = int(resources["threads"])
    point_timeout_seconds = int(resources["point_timeout_seconds"])
    max_wall_seconds = int(resources["max_wall_seconds"])
    markers_per_chromosome = int(CONTROL["fixture"]["markers_per_chromosome"])
    pilot_seeds = list(PILOT_SEEDS)
    confirmation_seed_count = CONFIRMATION_SEED_COUNT

    if not image_available():
        raise SystemExit("Pinned M27D analysis image is unavailable")
    receipt = git_receipt(args.repo)
    if not receipt["working_tree_clean"]:
        raise SystemExit(f"Refusing to run from a dirty tree: {receipt['dirty_paths']}")
    if args.out.exists() or args.details_dir.exists():
        raise SystemExit("Output path already exists; representation controls never overwrite")
    expected_base_hash = CONTROL["base_preregistration"]["sha256"]
    observed_base_hash = sha256_file(args.preregistration)
    if observed_base_hash != expected_base_hash:
        raise SystemExit(
            f"Base preregistration hash mismatch: expected {expected_base_hash}, "
            f"observed {observed_base_hash}"
        )
    base_contract = load(args.preregistration)
    base_threshold = float(base_contract["pcrelate"]["primary_phi_threshold"])
    control_threshold = float(CONTROL["estimand"]["primary_threshold"])
    if base_threshold != control_threshold:
        raise SystemExit(
            f"Threshold mismatch: base={base_threshold}, control={control_threshold}"
        )

    scenario_name = str(CONTROL["fixture"]["scenario"])
    try:
        scenario = next(value for value in SCENARIOS if value.name == scenario_name)
    except StopIteration as error:
        raise SystemExit(f"Unknown preregistered scenario: {scenario_name}") from error
    layout = CohortLayout(n_markers_per_chromosome=markers_per_chromosome)
    started = time.monotonic()
    deadline = started + max_wall_seconds
    all_results: list[dict] = []
    all_contexts: list[PointContext] = []
    arms_run = [ARM_STRICT, ARM_PCA_REPRESENTED]
    decision: dict[str, object]
    staging_details: Path | None = None
    staging_result: Path | None = None
    staging_manifest: Path | None = None
    try:
        pilot, contexts = execute_points(
            pilot_seeds, scenario, layout, args.repo, args.preregistration,
            threads, point_timeout_seconds, arms_run, deadline,
        )
        all_results.extend(pilot)
        all_contexts.extend(contexts)
        pilot_gate = intervention_supported(pilot)

        if pilot_gate["supported"]:
            confirmation = derive_confirmation_seeds(
                args.run_id, confirmation_seed_count, set(pilot_seeds)
            )
            confirm_results, confirm_contexts = execute_points(
                confirmation, scenario, layout, args.repo, args.preregistration,
                threads, point_timeout_seconds, arms_run, deadline,
            )
            all_results.extend(confirm_results)
            all_contexts.extend(confirm_contexts)
            final_gate = intervention_supported(all_results)
            decision = {
                "branch": "PCA_REPRESENTATION_CONFIRMATION",
                "pilot_gate": pilot_gate,
                "confirmation_seeds": confirmation,
                "final_gate": final_gate,
                "interpretation": (
                    "PCA representation mechanism supported"
                    if final_gate["supported"]
                    else "Pilot improvement did not survive deterministic confirmation"
                ),
            }
        else:
            arms_run.append(ARM_BOTH_REPRESENTED)
            for point, context in zip(all_results, all_contexts):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Wall-clock budget exhausted before represented/represented arm")
                point["arms"][ARM_BOTH_REPRESENTED] = run_arm(
                    context, ARM_BOTH_REPRESENTED, args.repo, threads,
                    point_timeout_seconds,
                )
            both_gate = intervention_supported(all_results, ARM_BOTH_REPRESENTED)
            if both_gate["supported"]:
                confirmation = derive_confirmation_seeds(
                    args.run_id, confirmation_seed_count, set(pilot_seeds)
                )
                confirm_results, confirm_contexts = execute_points(
                    confirmation, scenario, layout, args.repo, args.preregistration,
                    threads, point_timeout_seconds, arms_run, deadline,
                )
                all_results.extend(confirm_results)
                all_contexts.extend(confirm_contexts)
                final_both_gate = intervention_supported(
                    all_results, ARM_BOTH_REPRESENTED
                )
                decision = {
                    "branch": "PCRELATE_TRAINING_CONFIRMATION",
                    "pilot_gate": pilot_gate,
                    "both_represented_pilot_gate": both_gate,
                    "confirmation_seeds": confirmation,
                    "both_represented_final_gate": final_both_gate,
                    "interpretation": (
                        "PC-Relate training-set effect supported within the fixture"
                        if final_both_gate["supported"]
                        else "The represented PC-Relate pilot did not survive deterministic confirmation"
                    ),
                }
            else:
                decision = {
                    "branch": "PCRELATE_TRAINING_DIAGNOSTIC",
                    "pilot_gate": pilot_gate,
                    "both_represented_gate": both_gate,
                    "interpretation": (
                        "Neither represented PCA nor represented PC-Relate training reached the preregistered floor"
                    ),
                }

        elapsed = time.monotonic() - started
        if elapsed > max_wall_seconds:
            raise TimeoutError(
                f"Wall budget exceeded: {elapsed:.1f}s > {max_wall_seconds}s"
            )

        args.details_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_details = args.details_dir.parent / (
            f".{args.details_dir.name}.partial-"
            f"{hashlib.sha256(args.run_id.encode()).hexdigest()[:12]}"
        )
        if staging_details.exists():
            raise SystemExit(f"Staging path already exists: {staging_details}")
        staging_details.mkdir()
        for point, context in zip(all_results, all_contexts):
            persist_point_details(
                context,
                staging_details / f"{point['scenario']}-seed{point['seed']}",
                list(point["arms"]),
            )

        code_paths = [
            args.repo / "bin" / name
            for name in (
                "m27d_representation_control.py",
                "m27d_training_set_intervention.py",
                "m27d_pipeline_chain.py",
                "m27d_coancestry_screen.py",
                "m27d_fixture_scoring.py",
                "m27d_synthetic_cohort.py",
                "m27d_pca_projection.R",
                "m27d_pcrelate_configuration.R",
            )
        ]
        summary = {
            "stage": "M27D_SYNTHETIC_REPRESENTATION_CONTROL",
            "run_id": args.run_id,
            "scientific_result": False,
            "synthetic_only": True,
            "cloud_executed": False,
            "king_executed": False,
            "pcair_used": False,
            "gnomix_accessed": False,
            "primary_configuration": PRIMARY_CONFIGURATION,
            "primary_experimental_unit": "independently_generated_seed",
            "repeated_subunit": "deme_within_seed",
            "pair_counts_are_replicates": False,
            "pilot_seeds": pilot_seeds,
            "decision": decision,
            "results": all_results,
            "elapsed_seconds": round(elapsed, 1),
            "wall_clock_budget_seconds": max_wall_seconds,
            "container": IMAGE,
            "control_preregistration": {
                "path": str(CONTROL_PREREGISTRATION.relative_to(args.repo)),
                "sha256": sha256_file(CONTROL_PREREGISTRATION),
            },
            "base_preregistration": {
                "path": str(args.preregistration.relative_to(args.repo)),
                "sha256": observed_base_hash,
            },
            "git": receipt,
            "code_sha256": {str(path.relative_to(args.repo)): sha256_file(path) for path in code_paths},
            "command_argv": sys.argv,
            "interpretation_limits": [
                "No physical linkage or IBD-segment length is simulated",
                "Positive controls only cover true pedigree phi >= 0.125",
                "Only the anchor PC8/r2=0.20 configuration is evaluated",
                "Missing pairs are known only to fall below the reporting threshold",
                "The represented PC-Relate arm is diagnostic and not a donor policy",
                "The intervention tests representation as a mechanism; it does not estimate an optimal number of representatives",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        staging_result = args.out.parent / f".{args.out.name}.partial"
        staging_manifest = args.out.parent / f".{args.run_id}.manifest.json.partial"
        if staging_result.exists() or staging_manifest.exists():
            raise SystemExit("A staged result already exists; inspect it before retrying")
        staging_result.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "run_id": args.run_id,
            "result": str(args.out),
            "result_sha256": sha256_file(staging_result),
            "details_dir": str(args.details_dir),
            "git": receipt,
            "container": IMAGE,
            "elapsed_seconds": round(elapsed, 1),
        }
        staging_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # The manifest is renamed last and is therefore the completion marker.
        staging_details.replace(args.details_dir)
        staging_result.replace(args.out)
        staging_manifest.replace(args.out.parent / f"{args.run_id}.manifest.json")
    finally:
        for context in all_contexts:
            shutil.rmtree(context.workspace, ignore_errors=True)
        if staging_details is not None:
            shutil.rmtree(staging_details, ignore_errors=True)
        for path in (staging_result, staging_manifest):
            if path is not None:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
