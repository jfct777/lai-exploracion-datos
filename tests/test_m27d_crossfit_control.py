"""Unit tests for the preregistered M27D leave-pair-out control."""

from __future__ import annotations

import itertools
import json
import shutil
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m27d_crossfit_control import (  # noqa: E402
    ARM_REPRESENTED_IN_SAMPLE,
    ARM_STRICT_SAFE,
    adjudicate,
    safe_run_id,
    size_matched_fit_set,
)
from m27d_coancestry_screen import SCENARIOS, load  # noqa: E402
from m27d_pipeline_chain import image_available  # noqa: E402
from m27d_representation_control import (  # noqa: E402
    prepare_context,
    run_training_arm,
)
from m27d_synthetic_cohort import CohortLayout, first_cousin_units  # noqa: E402
from m27d_training_set_intervention import (  # noqa: E402
    read_ids,
    read_population,
    read_truth_pairs,
)


class TestSizeMatchedFitSet(unittest.TestCase):
    def test_required_and_forbidden_members_are_handled_without_recent_pairs(self):
        population = {
            **{f"BG{i}": "POP_BG1" for i in range(8)},
            "A": "DEME_A",
            "B": "DEME_A",
            "C": "DEME_A",
        }
        truth = [
            {"ID1": "A", "ID2": "B", "has_recent_kinship": "true"},
            {"ID1": "BG0", "ID2": "BG1", "has_recent_kinship": "true"},
        ]
        values, receipt = size_matched_fit_set(
            ["BG0", "BG2", "BG3", "BG4", "C"],
            5,
            truth,
            population,
            required={"A", "C"},
            forbidden={"B", "BG0"},
            seed=11,
            label="test",
        )
        self.assertEqual(len(values), 5)
        self.assertTrue({"A", "C"} <= set(values))
        self.assertFalse({"B", "BG0"} & set(values))
        self.assertEqual(receipt["n_recent_pairs_both_in_set"], 0)

    def test_a_required_recent_pair_is_rejected(self):
        population = {"A": "DEME_A", "B": "DEME_A", "BG0": "POP_BG1"}
        truth = [{"ID1": "A", "ID2": "B", "has_recent_kinship": "true"}]
        with self.assertRaises(ValueError):
            size_matched_fit_set(
                ["BG0", "A"], 2, truth, population,
                required={"A", "B"}, forbidden=set(), seed=11, label="test"
            )

    def test_run_identifier_cannot_escape_or_name_a_path(self):
        self.assertEqual(safe_run_id("m27d-crossfit-20260816a"), "m27d-crossfit-20260816a")
        for value in ("", ".", "..", "../run", "nested/run", "run manifest", "run;touch"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_run_id(value)

    def test_base_contract_hash_matches_both_frozen_controls(self):
        import hashlib
        import json

        crossfit = json.loads(
            (ROOT / "conf" / "m27d_crossfit_control_preregistration.json").read_text()
        )
        parent = json.loads(
            (ROOT / "conf" / "m27d_representation_control_preregistration.json").read_text()
        )
        base = ROOT / crossfit["base_preregistration"]["path"]
        observed = hashlib.sha256(base.read_bytes()).hexdigest()
        self.assertEqual(observed, crossfit["base_preregistration"]["sha256"])
        self.assertEqual(observed, parent["base_preregistration"]["sha256"])


def _controls(all_detected: bool = True) -> dict[str, dict[str, object]]:
    return {
        "first_degree": {"all_detected": all_detected},
        "second_degree": {"all_detected": all_detected},
    }


def result_point(
    seed: int,
    strict_a: int = 15,
    strict_b: int = 0,
    represented: int = 0,
    crossfit_a: int = 0,
    crossfit_b: int = 0,
    cousin_strict: bool = True,
    cousin_crossfit: bool = True,
    other_controls: bool = True,
) -> dict[str, object]:
    cousin_pairs = [
        {"id1": "BG0", "id2": "BG1", "detected_at_primary_threshold": cousin_strict},
        {"id1": "DC5", "id2": "DC6", "detected_at_primary_threshold": cousin_strict},
    ]
    folds = [
        {"fold_index": index, "other_positive_controls": _controls(other_controls)}
        for index in range(15)
    ]
    stability = [
        {
            "id1": row["id1"],
            "id2": row["id2"],
            "pedigree_location": location,
            "detected_in_every_fold": cousin_crossfit,
            "n_detected": 15 if cousin_crossfit else 14,
            "n_folds": 15,
        }
        for row, location in zip(cousin_pairs, ("background", "deme"))
    ]
    primary = {
        "DEME_A": {"n_recent_pedigree_false_positives": crossfit_a},
        "DEME_B": {"n_recent_pedigree_false_positives": crossfit_b},
    }
    return {
        "seed": seed,
        "baseline_arms": {
            ARM_STRICT_SAFE: {
                "primary_units": {
                    "DEME_A": {"n_recent_pedigree_false_positives": strict_a},
                    "DEME_B": {"n_recent_pedigree_false_positives": strict_b},
                },
                "first_cousins": cousin_pairs,
            },
            ARM_REPRESENTED_IN_SAMPLE: {
                "primary_units": {
                    deme: {"n_recent_pedigree_false_positives": represented}
                    for deme in ("DEME_A", "DEME_B")
                }
            },
        },
        "leave_pair_out": {
            "primary_units": primary,
            "first_cousin_stability": stability,
            "folds": folds,
        },
    }


class TestAdjudication(unittest.TestCase):
    def test_pass_requires_all_independent_seeds(self):
        decision = adjudicate([result_point(seed) for seed in (11, 23, 37)])
        self.assertEqual(decision["verdict"], "PASS_SYNTHETIC_CROSSFIT_ONLY")
        self.assertEqual(decision["n_independent_seed_replicates"], 3)
        self.assertFalse(decision["pair_fold_or_deme_counts_are_replicates"])

    def test_a_subset_of_preregistered_seeds_is_inconclusive(self):
        decision = adjudicate([result_point(11)])
        self.assertEqual(decision["verdict"], "INCONCLUSIVE_FIXTURE_CONTROL_FAILED")
        self.assertFalse(decision["all_preregistered_seeds_present_once"])

    def test_loss_of_a_third_degree_control_stops_the_proposal(self):
        points = [result_point(seed) for seed in (11, 23, 37)]
        points[1] = result_point(23, cousin_crossfit=False)
        self.assertEqual(adjudicate(points)["verdict"], "STOP_REPRESENTATION_AS_SOLUTION")

    def test_failure_to_reproduce_the_fixture_is_inconclusive(self):
        points = [result_point(seed, strict_a=0) for seed in (11, 23, 37)]
        self.assertEqual(
            adjudicate(points)["verdict"], "INCONCLUSIVE_FIXTURE_CONTROL_FAILED"
        )

    def test_fifteen_index_pairs_cover_a_six_person_deme_once(self):
        pairs = list(itertools.combinations(range(6), 2))
        self.assertEqual(len(pairs), 15)
        self.assertEqual(len(set(pairs)), 15)


@unittest.skipUnless(image_available(), "pinned M27D analysis container is not available")
class TestOneFoldIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = CohortLayout(
            n_background_per_group=20,
            n_deme_members=6,
            n_pedigree_deme_members=8,
            n_pedigree_units_in_background=2,
            n_first_cousin_pairs_in_deme=1,
            n_first_cousin_pairs_in_background=1,
            n_markers_per_chromosome=200,
        )
        cls.cousins = {
            sample
            for unit in first_cousin_units(cls.layout)
            for sample in (unit["cousin_1"], unit["cousin_2"])
        }
        scenario = next(value for value in SCENARIOS if value.name == "isolate")
        cls.context = prepare_context(
            scenario,
            11,
            cls.layout,
            ROOT,
            ROOT / "conf" / "m27d_donor_kinship_preregistration.json",
            threads=2,
            point_timeout=600,
            pass0_excluded=cls.cousins,
        )
        truth = load(cls.context.fixture_dir / "truth.json")
        rows = read_truth_pairs(cls.context.fixture_dir / "truth_pairs.tsv")
        population = read_population(cls.context.fixture_dir / "metadata.tsv")
        strict = read_ids(cls.context.base_out / "training_set.txt")
        members = {deme: truth["demes"][deme] for deme in ("DEME_A", "DEME_B")}
        held_out = {
            members[deme][index]
            for deme in members
            for index in (0, 1)
        }
        required = {
            sample for deme in members for sample in members[deme]
        } - held_out
        cls.fit_set, cls.fit_receipt = size_matched_fit_set(
            cls.context.represented_set,
            len(strict),
            rows,
            population,
            required=required,
            forbidden=held_out | cls.cousins,
            seed=11,
            label="integration_leave_pair_out_0_1",
        )
        cls.held_out = held_out
        cls.result = run_training_arm(
            cls.context,
            "integration_leave_pair_out_0_1",
            ROOT,
            2,
            600,
            cls.fit_set,
            cls.fit_set,
            intervention_summary=cls.fit_receipt,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.context.workspace, ignore_errors=True)

    def test_pass0_controls_are_absent_and_full_contract_is_restored(self):
        receipt = json.loads(
            (self.context.base_out / "crossfit_pass0_exclusion.json").read_text()
        )
        universe = set(
            read_ids(self.context.base_out / "m27d_pass0_sample_universe.private.txt")
        )
        self.assertFalse(self.cousins & universe)
        self.assertEqual(set(receipt["excluded_sample_ids"]), self.cousins)
        self.assertTrue(receipt["full_strata_and_contract_restored_for_final_evaluation"])

    def test_one_fold_has_no_endpoint_leakage_and_r_consumes_explicit_sets(self):
        selected = set(self.fit_set)
        self.assertFalse((self.held_out | self.cousins) & selected)
        self.assertEqual(self.result["pca_training_set_sha256"],
                         self.result["pcrelate_training_set_sha256"])
        self.assertEqual(self.result["pca_training_set_input_basename"],
                         "pca_training_set.txt")
        self.assertEqual(self.result["pcrelate_training_set_input_basename"],
                         "pcrelate_training_set.txt")


if __name__ == "__main__":
    unittest.main()
