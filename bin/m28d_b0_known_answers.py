#!/usr/bin/env python3
"""Run synthetic known-answer gates for the M28D scorer without real truth."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("m28d_b0_scorer_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scorer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def one_hot(module, ancestry: str) -> tuple[float, float, float]:
    return tuple(float(label == ancestry) for label in module.ANCESTRIES)


def window(module, left, right, first, second, marker_start, marker_end):
    return module.PredictionWindow(
        left=left,
        right=right,
        marker_start=marker_start,
        marker_end=marker_end,
        n_markers=marker_end - marker_start,
        probabilities={"T000": (one_hot(module, first), one_hot(module, second))},
        hard_labels={"T000": (first, second)},
    )


def core_boundary(summary: dict) -> dict:
    return {
        key: value
        for key, value in summary.items()
        if key != "global_phase_permutation_counts"
    }


def run_checks(module) -> dict[str, bool]:
    checks: dict[str, bool] = {}

    def raises(expected_exception, function) -> bool:
        try:
            function()
        except expected_exception:
            return True
        return False

    genetic_map = module.GeneticMap([module.MapPoint(0, 0.0), module.MapPoint(100, 1.0)])

    cells = module.discrete_voronoi([10, 20, 31])
    checks["voronoi_midpoint_tie_left_and_exact_weight"] = (
        cells == [(10, 16), (16, 26), (26, 32)]
        and sum(right - left for left, right in cells) == 22
    )

    markers = [10, 20, 30]
    direct = [
        window(module, 10, 16, "AFR", "ASIA", 0, 1),
        window(module, 16, 26, "EUR", "ASIA", 1, 2),
        window(module, 26, 31, "EUR", "ASIA", 2, 3),
    ]
    swapped = [
        window(module, 10, 16, "ASIA", "AFR", 0, 1),
        window(module, 16, 26, "ASIA", "EUR", 1, 2),
        window(module, 26, 31, "ASIA", "EUR", 2, 3),
    ]
    truth = {
        "T000": (
            [
                module.TruthSegment(10, 16, "AFR"),
                module.TruthSegment(16, 31, "EUR"),
            ],
            [module.TruthSegment(10, 31, "ASIA")],
        )
    }
    perfect = module.score_objects(markers, direct, truth, genetic_map, [0.1, 0.2, 0.5])
    checks["perfect_probabilities_dosage_and_hard_state"] = (
        perfect["primary"]["macro"] == 0.0
        and perfect["secondary"]["phase_aligned_haplotype_brier"] == 0.0
        and perfect["secondary"]["hard_unordered_diploid_state"]["accuracy"] == 1.0
    )
    checks["perfect_boundary"] = all(
        summary["truth_boundaries"] == 1
        and summary["predicted_boundaries"] == 1
        and summary["matched_boundaries"] == 1
        and summary["matched_distance_median_cm"] == 0.0
        for summary in perfect["secondary"]["boundaries"].values()
    )

    phase_swapped = module.score_objects(markers, swapped, truth, genetic_map, [0.1, 0.2, 0.5])
    checks["global_haplotype_swap_invariance"] = (
        perfect["primary"] == phase_swapped["primary"]
        and perfect["secondary"]["phase_aligned_haplotype_brier"]
        == phase_swapped["secondary"]["phase_aligned_haplotype_brier"]
        and perfect["secondary"]["hard_unordered_diploid_state"]
        == phase_swapped["secondary"]["hard_unordered_diploid_state"]
        and all(
            core_boundary(perfect["secondary"]["boundaries"][key])
            == core_boundary(phase_swapped["secondary"]["boundaries"][key])
            for key in perfect["secondary"]["boundaries"]
        )
    )
    checks["A_B_scientific_payload_exact"] = perfect == module.score_objects(
        markers, direct, truth, genetic_map, [0.1, 0.2, 0.5]
    )

    maximum_truth = {
        "T000": (
            [module.TruthSegment(10, 31, "AFR")],
            [module.TruthSegment(10, 31, "AFR")],
        )
    }
    maximum_prediction = [window(module, 10, 31, "EUR", "EUR", 0, 3)]
    maximum = module.score_objects(
        markers, maximum_prediction, maximum_truth, genetic_map, [0.2]
    )
    checks["primary_theoretical_maximum_two_thirds"] = math.isclose(
        maximum["primary"]["macro"], 2.0 / 3.0, rel_tol=0.0, abs_tol=1e-15
    )

    constant_truth = {
        "T000": (
            [module.TruthSegment(10, 31, "AFR")],
            [module.TruthSegment(10, 31, "ASIA")],
        )
    }
    constant = module.score_objects(
        markers,
        [window(module, 10, 31, "AFR", "ASIA", 0, 3)],
        constant_truth,
        genetic_map,
        [0.2],
    )
    no_boundary = constant["secondary"]["boundaries"]["0.2"]
    checks["no_boundary_zero_denominator_is_null"] = (
        no_boundary["truth_boundaries"] == 0
        and no_boundary["predicted_boundaries"] == 0
        and no_boundary["precision"] is None
        and no_boundary["recall"] is None
        and no_boundary["f1"] is None
    )

    incompatible_truth = {
        "T000": (
            [
                module.TruthSegment(10, 16, "AFR"),
                module.TruthSegment(16, 21, "EUR"),
            ],
            [module.TruthSegment(10, 21, "ASIA")],
        )
    }
    incompatible_prediction = [
        window(module, 10, 16, "AFR", "ASIA", 0, 1),
        window(module, 16, 21, "ASIA", "ASIA", 1, 2),
    ]
    no_matches = module.score_objects(
        [10, 20], incompatible_prediction, incompatible_truth, genetic_map, [0.2]
    )["secondary"]["boundaries"]["0.2"]
    checks["defined_zero_boundary_matches_has_zero_f1"] = (
        no_matches["truth_boundaries"] == 1
        and no_matches["predicted_boundaries"] == 1
        and no_matches["matched_boundaries"] == 0
        and no_matches["precision"] == 0.0
        and no_matches["recall"] == 0.0
        and no_matches["f1"] == 0.0
        and no_matches["per_directed_transition"]["AFR->EUR"]["recall"] == 0.0
    )

    crossed_truth = {
        "T000": (
            [
                module.TruthSegment(10, 18, "AFR"),
                module.TruthSegment(18, 21, "EUR"),
            ],
            [module.TruthSegment(10, 21, "ASIA")],
        )
    }
    crossed = module.score_objects(
        [10, 20],
        [window(module, 10, 21, "AFR", "ASIA", 0, 2)],
        crossed_truth,
        genetic_map,
        [0.2],
    )
    checks["truth_boundary_inside_prediction_cell_exact_overlap"] = math.isclose(
        crossed["primary"]["macro"], 1.0 / 11.0, rel_tol=0.0, abs_tol=1e-15
    )

    label_truth = [
        module.Boundary(0.20, "AFR", "EUR"),
        module.Boundary(0.25, "AFR", "EUR"),
    ]
    label_prediction = [
        module.Boundary(0.21, "EUR", "AFR"),
        module.Boundary(0.24, "AFR", "EUR"),
    ]
    distances = module.ordered_boundary_match(label_truth, label_prediction, 0.1)
    checks["boundary_label_aware_one_to_one"] = len(distances) == 1 and math.isclose(
        distances[0], 0.01, abs_tol=1e-15
    )
    crossing_truth = [
        module.Boundary(0.20, "AFR", "EUR"),
        module.Boundary(0.30, "EUR", "AFR"),
    ]
    crossing_prediction = [
        module.Boundary(0.21, "EUR", "AFR"),
        module.Boundary(0.29, "AFR", "EUR"),
    ]
    checks["boundary_matching_preserves_global_order"] = (
        len(module.ordered_boundary_match(crossing_truth, crossing_prediction, 0.2)) == 1
    )
    checks["missing_and_extra_boundaries_retained"] = (
        module.ordered_boundary_match(label_truth, [], 0.5) == []
        and len(label_prediction) - len(distances) == 1
    )

    shifted_truth = [module.Boundary(1.0, "AFR", "EUR")]
    shifted_prediction = [module.Boundary(1.2, "AFR", "EUR")]
    checks["boundary_tolerance_curve_fixed"] = (
        len(module.ordered_boundary_match(shifted_truth, shifted_prediction, 0.1)) == 0
        and len(module.ordered_boundary_match(shifted_truth, shifted_prediction, 0.2)) == 1
        and len(module.ordered_boundary_match(shifted_truth, shifted_prediction, 0.5)) == 1
    )

    plateau = module.GeneticMap(
        [
            module.MapPoint(0, 0.0),
            module.MapPoint(10, 0.1),
            module.MapPoint(20, 0.1),
            module.MapPoint(30, 0.2),
        ]
    )
    checks["genetic_map_plateau_nonnegative"] = plateau.cm_at(18) - plateau.cm_at(12) == 0.0

    fb_positions = [10, 25]
    fb_probabilities = [
        {"T000": (one_hot(module, "AFR"), one_hot(module, "ASIA"))},
        {"T000": (one_hot(module, "EUR"), one_hot(module, "ASIA"))},
    ]
    msp_metadata = [
        {"spos": 10, "epos": 10, "sgpos": 0.10, "egpos": 0.10, "n_snps": 1},
        {"spos": 20, "epos": 30, "sgpos": 0.20, "egpos": 0.30, "n_snps": 2},
    ]
    msp_labels = [
        {"T000": ("AFR", "ASIA")},
        {"T000": ("EUR", "ASIA")},
    ]
    module.validate_genetic_coordinates(
        markers,
        [0.10, 0.20, 0.30],
        fb_positions,
        [0.10, 0.25],
        msp_metadata,
        genetic_map,
        1e-8,
        1e-5,
    )
    checks["genetic_coordinate_authority_and_rounding"] = True
    bad_cm_metadata = [dict(item) for item in msp_metadata]
    bad_cm_metadata[1]["egpos"] = 0.31
    checks["genetic_coordinate_misalignment_fails_closed"] = raises(
        ValueError,
        lambda: module.validate_genetic_coordinates(
            markers,
            [0.10, 0.20, 0.30],
            fb_positions,
            [0.10, 0.25],
            bad_cm_metadata,
            genetic_map,
            1e-8,
            1e-5,
        ),
    )
    built = module.build_prediction_windows(
        markers,
        fb_positions,
        ["T000"],
        fb_probabilities,
        msp_metadata,
        ["T000"],
        msp_labels,
    )
    checks["terminal_window_and_domain_clipping"] = (
        built[0].left == 10
        and built[-1].right == 31
        and built[-1].n_markers == 2
        and sum(item.n_markers for item in built) == 3
    )
    bad_metadata = [dict(item) for item in msp_metadata]
    bad_metadata[1]["spos"] = 21
    checks["coordinate_misalignment_fails_closed"] = raises(
        ValueError,
        lambda: module.build_prediction_windows(
            markers,
            fb_positions,
            ["T000"],
            fb_probabilities,
            bad_metadata,
            ["T000"],
            msp_labels,
        ),
    )
    checks["sample_misalignment_fails_closed"] = raises(
        ValueError,
        lambda: module.score_objects(markers, direct, {}, genetic_map, [0.2]),
    )
    bad_probability_window = module.PredictionWindow(
        left=10,
        right=31,
        marker_start=0,
        marker_end=3,
        n_markers=3,
        probabilities={"T000": ((math.nan, 0.0, 1.0), one_hot(module, "ASIA"))},
        hard_labels={"T000": ("AFR", "ASIA")},
    )
    checks["nonfinite_probability_fails_closed"] = raises(
        ValueError,
        lambda: module.score_objects(
            markers, [bad_probability_window], constant_truth, genetic_map, [0.2]
        ),
    )
    non_normalized_window = module.PredictionWindow(
        left=10,
        right=31,
        marker_start=0,
        marker_end=3,
        n_markers=3,
        probabilities={"T000": ((0.8, 0.3, 0.0), one_hot(module, "ASIA"))},
        hard_labels={"T000": ("AFR", "ASIA")},
    )
    checks["finite_non_normalized_probability_fails_closed"] = raises(
        ValueError,
        lambda: module.score_objects(
            markers, [non_normalized_window], constant_truth, genetic_map, [0.2]
        ),
    )
    invalid_class_truth = {
        "T000": (
            [module.TruthSegment(10, 31, "UNKNOWN")],
            [module.TruthSegment(10, 31, "ASIA")],
        )
    }
    checks["invalid_ancestry_class_fails_closed"] = raises(
        ValueError,
        lambda: module.score_objects(
            markers,
            [window(module, 10, 31, "AFR", "ASIA", 0, 3)],
            invalid_class_truth,
            genetic_map,
            [0.2],
        ),
    )
    checks["zero_genetic_span_fails_closed"] = raises(
        ValueError,
        lambda: module.score_objects(
            markers,
            [window(module, 10, 31, "AFR", "ASIA", 0, 3)],
            constant_truth,
            module.GeneticMap([module.MapPoint(0, 0.1), module.MapPoint(100, 0.1)]),
            [0.2],
        ),
    )
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        fixture = Path(directory_name) / "fixture.txt"
        fixture.write_text("known answer\n", encoding="utf-8")
        checks["hash_mismatch_fails_closed"] = raises(
            ValueError, lambda: module.require_hash(fixture, "0" * 64, "fixture")
        )
        clipped_truth = directory / "clipped_truth.tsv"
        clipped_truth.write_text(
            "target_haplotype\tchrom\tstart_bp\tend_bp_exclusive\tancestry\n"
            "T000_h0\tchr22\t0\t40\tAFR\n"
            "T000_h1\tchr22\t0\t40\tASIA\n",
            encoding="utf-8",
        )
        clipped = module.load_truth(clipped_truth, ["T000"], "chr22", 10, 31)
        checks["truth_domain_clipping_exact"] = (
            clipped["T000"][0] == [module.TruthSegment(10, 31, "AFR")]
            and clipped["T000"][1] == [module.TruthSegment(10, 31, "ASIA")]
        )
        malformed_truth = directory / "malformed_truth.tsv"
        malformed_truth.write_text(
            "target_haplotype\tchrom\tstart_bp\tend_bp_exclusive\tancestry\n"
            "T000_h2\tchr22\t10\t31\tAFR\n",
            encoding="utf-8",
        )
        checks["malformed_haplotype_fails_closed"] = raises(
            ValueError,
            lambda: module.load_truth(malformed_truth, ["T000"], "chr22", 10, 31),
        )
        valid_score = {
            "schema_version": 1,
            "stage": "M28D_B0_DESCRIPTIVE_SCORING",
            "decision": "B0_DESCRIBED_NO_CEILING_INFERENCE",
            "replicate": "A",
            "scope": "descriptive_scoring_and_scorer_validation_only",
            "seed": 20260818,
            "scientific_inference_authorized": False,
            "BR_BS_authorized": False,
            "SESOI_fixed": False,
            "independent_seed_count_fixed": False,
            "validation_seed_consumed": True,
            "metrics": perfect,
        }
        invalid_score = dict(valid_score)
        invalid_score["BR_BS_authorized"] = True
        checks["invalid_score_scope_fails_closed"] = raises(
            ValueError, lambda: module.validate_score_document(invalid_score, "A")
        )

    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"known-answer failure: {failed}")
    return checks


def run_unit_suite(test_path: Path, scorer_path: Path) -> dict[str, object]:
    previous = os.environ.get("M28D_SCORER_PATH")
    os.environ["M28D_SCORER_PATH"] = str(scorer_path.resolve())
    try:
        spec = importlib.util.spec_from_file_location("m28d_b0_scorer_unit_suite", test_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load M28D unit-test module")
        test_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = test_module
        spec.loader.exec_module(test_module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
        result = unittest.TestResult()
        suite.run(result)
    finally:
        if previous is None:
            os.environ.pop("M28D_SCORER_PATH", None)
        else:
            os.environ["M28D_SCORER_PATH"] = previous
    if not result.wasSuccessful():
        details = [str(error) for _, error in [*result.failures, *result.errors]]
        raise RuntimeError(f"M28D unit suite failed: {details}")
    return {"tests_run": result.testsRun, "failures": 0, "errors": 0, "passed": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--unit-test-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_module(args.scorer)
    contract = module.load_contract(args.contract)
    if contract["seed_policy"]["role"] != "protected_validation_seed_descriptive_only":
        raise ValueError("unexpected seed policy")
    implementation = contract["authenticated_implementation"]
    scorer_hash = module.sha256_file(args.scorer)
    runner_hash = module.sha256_file(Path(__file__))
    if scorer_hash != implementation["scorer_sha256"]:
        raise ValueError("scorer hash does not match the frozen implementation")
    if runner_hash != implementation["known_answer_runner_sha256"]:
        raise ValueError("known-answer runner hash does not match the frozen implementation")
    unit_test_hash = module.sha256_file(args.unit_test_file)
    if unit_test_hash != implementation["unit_test_sha256"]:
        raise ValueError("unit-test hash does not match the frozen implementation")
    checks = run_checks(module)
    unit_suite = run_unit_suite(args.unit_test_file, args.scorer)
    result = {
        "schema_version": 1,
        "stage": "M28D_B0_SCORER_KNOWN_ANSWERS",
        "decision": "PASS_M28D_SCORER_KNOWN_ANSWERS",
        "real_truth_accessed": False,
        "checks": checks,
        "unit_suite": unit_suite,
        "unit_test_file_sha256": unit_test_hash,
        "scorer_sha256": scorer_hash,
        "contract_sha256": module.sha256_file(args.contract),
        "known_answer_runner_sha256": runner_hash,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
