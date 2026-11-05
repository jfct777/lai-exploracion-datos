#!/usr/bin/env python3
"""Short adversarial tests for the optional whole-person dose-view binding."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import r02_context_train as trainer
import r02_lai_dose_control_training as binding
from r02_lai_control_views import array_sha256, sha256
from r02_lai_dose_controls import REMOVED, PERMUTED
from r02_lai_person_microbatch import slice_people
from r02_lai_query_burden import build_prefix


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": sha256(path)}


class DoseBindingFixture:
    def __init__(self, root, condition=REMOVED):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.reference = {name: (str(index) * 64)[:64] for index, name in enumerate((
            "event_catalogue_sha256", "ref_train_receipt_sha256",
            "ref_support_by_provenance_sha256", "reference_vcf_sha256",
            "reference_roles_sha256"), 1)}
        self.people = np.asarray(["p0", "p1"])
        self.cm = np.asarray([0.0, 0.5, 1.0, 1.5], np.float64)
        self.chrom = np.ones(4, np.int64)
        self.valid = np.ones((2, 4), bool)
        self.original = np.asarray([[0, 1, 2, 0], [1, 0, 2, 1]], np.int8)
        self.permuted = self.original[::-1].copy()
        self.cases = [f"case-{index}" for index in range(1, 4)]
        products = {case: self._product(case) for case in self.cases}
        self.entries = [self._entry(case) for case in self.cases]
        self.contract = {
            "objective": "lai", "information_condition": "NONE",
            "lai_chromosome": "1",
            "rare_allele": binding.SOURCE_MINOR,
            "reference_identity_by_case": {
                case: self.reference for case in self.cases + ["select-1", "select-2", "select-3"]},
            "limits": {"max_batch_bytes": 2_000_000},
            "batches": {"FIT": self.entries, "SELECT": []},
            binding.BLOCK: {
                "schema": binding.BINDING_SCHEMA, "base_condition": "NONE",
                "effective_condition": condition, "role": "FIT",
                "source_minor": binding.SOURCE_MINOR,
                "personal_dose_available": condition == PERMUTED,
                "products_by_case": products,
            },
        }
        self.resolved = {"rare_radius_cm": 1.0}
        self.entry = self.entries[0]
        self.base = self._regional()

    def _product(self, case):
        directory = self.root / case
        directory.mkdir()
        arrays = {
            "person_id": self.people, "event_valid": self.valid,
            "event_cm": self.cm, "event_global_index": np.arange(4, dtype=np.int64),
            "event_chromosome": self.chrom,
            "event_position_bp": np.asarray([10, 20, 30, 40], np.int64),
            "event_ref": np.asarray(["A", "C", "G", "T"]),
            "event_alt": np.asarray(["G", "T", "A", "C"]),
            "event_counted_allele": np.asarray(["G", "T", "G", "C"]),
            "event_is_alt": np.asarray([True, True, False, True]),
            "event_source_index": np.arange(100, 104, dtype=np.int64),
        }
        for condition, dose in ((REMOVED, np.zeros_like(self.original)),
                                (PERMUTED, self.permuted)):
            prefix = build_prefix(dose, self.valid, self.cm, self.chrom, "NONE", np)
            arrays[condition + "_event_dosage"] = dose
            arrays[condition + "_person_id"] = self.people
            for name, value in prefix.items():
                arrays[condition + "_" + name] = value
        batch = directory / "personal-dose-views.npz"
        np.savez_compressed(batch, **arrays)
        factor = write_json(directory / "factor.json", {
            "schema": "r02_lai_factorized_predictors_v1",
            "status": "PREDICTOR_INDEX_COMPLETE_NOT_TRAINING", "case_id": case,
            "role": "FIT", "chromosome": "1", "direction": "A_TO_B",
            "training_authorized": False, "truth_arrays_loaded": False,
            "evaluate_opened": False, "reference_identity": self.reference,
            "outputs_sha256": {"indices.npz": "a" * 64}, "task_id": case + "-task",
        })
        link = write_json(directory / "link.json", {
            "schema": "r02_lai_regional_truth_join_request_v2", "case_id": case,
            "role": "FIT", "chromosome": "1", "condition": "NONE",
            "direction": "A_TO_B", "training_authorized": False,
            "selection_authorized": False, "evaluate_opened": False,
        })
        hashes = {name: array_sha256(value, np) for name, value in arrays.items()}
        metrics = {
            "algorithm": "OBSERVED_PERSON_DOSE_SHA256_PCG64_V1", "role": "FIT",
            "seed": 7,
            "people": len(self.people), "loci": len(self.cm),
            "universe_sha256": hashlib.sha256(
                json.dumps(sorted(self.people.tolist()), separators=(",", ":")).encode()).hexdigest(),
            "original_dose_sha256": array_sha256(self.original, np),
            "observation_sha256": array_sha256(self.valid, np),
            "statistical_null_certified": False, "training_authorized": False,
            "evaluate_opened": False,
        }
        receipt = {
            "schema": binding.PRODUCT_SCHEMA, "status": binding.PRODUCT_STATUS,
            "training_authorized": False, "truth_loaded": False, "evaluate_opened": False,
            "batch_sha256": sha256(batch), "bytes": batch.stat().st_size,
            "arrays_sha256": hashes, "metrics": metrics,
            "code_sha256": {name: str(index) * 64 for index, name in enumerate(sorted(
                binding._PRODUCER_CODE), 4)},
            "source_link": link, "source_factorized_receipt": factor,
            "source_index_sha256": "a" * 64, "reference_identity": self.reference,
        }
        receipt_spec = write_json(directory / "receipt.json", receipt)
        return {"receipt": receipt_spec,
                "batch": {"path": str(batch), "sha256": sha256(batch)}}

    def _entry(self, case):
        source = write_json(self.root / (case + "-prepare.json"), {
            "schema": "r02_lai_regional_truth_factorized_link_receipt_v1",
            "status": "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION",
            "condition": "NONE", "role": "FIT", "case_id": case,
            "chromosome": "1", "direction": "A_TO_B", "rare_radius_cm": 1.0,
            "region_id": "region-00", "training_authorized": False,
            "selection_authorized": False, "evaluate_opened": False,
        })
        receipt = write_json(self.root / (case + "-training.json"), {
            "schema": "r02_lai_factorized_training_link_receipt_v1",
            "status": "FACTORIZED_LINK_BOUND_TO_AUTHENTICATED_TRAINING_PROTOCOL",
            "condition": "NONE", "role": "FIT", "case_id": case,
            "chromosome": "1", "direction": "A_TO_B", "region_id": "region-00",
            "source_prepare_receipt": source,
        })
        return {"path": "base.npz", "sha256": "c" * 64,
                "receipt_path": receipt["path"], "receipt_sha256": receipt["sha256"]}

    def _regional(self):
        cols = np.asarray([1, 3])
        return {
            "person_id": self.people.copy(), "event_global_index": cols.copy(),
            "event_cm": np.broadcast_to(self.cm[cols], (2, 2)).copy(),
            "event_chrom": np.ones((2, 2), np.int64),
            "event_pos_bp": np.broadcast_to(np.asarray([20, 40]), (2, 2)).copy(),
            "event_ref": np.asarray(["C", "T"]), "event_alt": np.asarray(["T", "C"]),
            "event_counted_allele": np.asarray(["T", "C"]),
            "event_is_alt": np.asarray([True, True]),
            "event_valid": self.valid[:, cols].copy(),
            "event_dosage": self.original[:, cols].copy(),
            "query_cm": np.broadcast_to(np.asarray([0.5, 1.5]), (2, 2)).copy(),
            "query_chrom": np.ones((2, 2), np.int64),
            "query_burden": np.full((2, 2, 3), 99, np.int64),
        }


def microbatch_data():
    people, common, event, query = 2, 2, 2, 2
    data = {
        "person_id": np.asarray(["p0", "p1"]),
        "component_id": np.asarray(["c0", "c1"]), "role": np.asarray(["FIT"] * people),
        "common_features": np.zeros((people, common, 1)),
        "common_cm": np.zeros((people, common)), "common_chrom": np.ones((people, common), int),
        "common_valid": np.ones((people, common), bool), "common_core": np.ones((people, common), bool),
        "event_metadata": np.zeros((people, event, 1)),
        "event_dosage": np.zeros((people, event), int),
        "event_cm": np.zeros((people, event)), "event_chrom": np.ones((people, event), int),
        "event_valid": np.ones((people, event), bool), "event_core": np.ones((people, event), bool),
        "event_reference_features": np.zeros((people, event, 15)),
        "query_features": np.zeros((people, query, 1)),
        "query_cm": np.zeros((people, query)), "query_chrom": np.ones((people, query), int),
        "query_valid": np.ones((people, query), bool), "query_core": np.ones((people, query), bool),
        "baseline_probabilities": np.zeros((people, query, 6)),
        "query_burden": np.zeros((people, query, 3), int),
        "truth_state": np.zeros((people, query), int),
        "personal_dose_intervention": REMOVED, "personal_dose_available": False,
    }
    return data


class DoseControlTrainingTest(unittest.TestCase):
    def test_absent_binding_is_exact_noop(self):
        base = {"sentinel": object()}
        self.assertIs(binding.apply_product(base, {}, {}, "FIT", None), base)
        self.assertIsNone(binding.checkpoint_binding_identity({}))

    def test_case_inventory_comes_from_fit_wrappers_not_a_fixed_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DoseBindingFixture(Path(temporary), REMOVED)
            two_cases = copy.deepcopy(fixture.contract)
            removed_case = fixture.cases[-1]
            two_cases["batches"]["FIT"] = two_cases["batches"]["FIT"][:-1]
            del two_cases[binding.BLOCK]["products_by_case"][removed_case]
            self.assertIsNotNone(binding.validate_binding_spec(two_cases))
            missing = copy.deepcopy(fixture.contract)
            del missing[binding.BLOCK]["products_by_case"][removed_case]
            with self.assertRaisesRegex(ValueError, "exact declared NONE FIT cases"):
                binding.validate_binding_spec(missing)

    def test_removed_uses_stored_whole_chromosome_counts_and_marks_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DoseBindingFixture(Path(temporary), REMOVED)
            original = fixture.base["event_dosage"].copy()
            result = binding.apply_product(fixture.base, fixture.entry, fixture.contract,
                                           "FIT", fixture.resolved)
            self.assertFalse(result["event_dosage"].any())
            self.assertFalse(result["personal_dose_available"])
            self.assertEqual(result["personal_dose_intervention"], REMOVED)
            self.assertFalse(result["query_burden"][..., :2].any())
            self.assertTrue((result["query_burden"][..., 2] > 0).all())
            np.testing.assert_array_equal(fixture.base["event_dosage"], original)
            self.assertFalse(any("original" in key for key in result))

    def test_permuted_view_keeps_geometry_and_allows_fixed_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DoseBindingFixture(Path(temporary), PERMUTED)
            result = binding.apply_product(fixture.base, fixture.entry, fixture.contract,
                                           "FIT", fixture.resolved)
            np.testing.assert_array_equal(result["event_dosage"], fixture.permuted[:, [1, 3]])
            np.testing.assert_array_equal(result["event_valid"], fixture.base["event_valid"])
            self.assertTrue(result["personal_dose_available"])
            # Locus 2 is deliberately a fixed point for every person; this is valid.
            self.assertTrue(np.array_equal(fixture.permuted[:, 2], fixture.original[:, 2]))

    def test_adversarial_person_source_and_stage_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DoseBindingFixture(Path(temporary), REMOVED)
            wrong_people = dict(fixture.base)
            wrong_people["person_id"] = wrong_people["person_id"][::-1]
            with self.assertRaisesRegex(ValueError, "person order"):
                binding.apply_product(wrong_people, fixture.entry, fixture.contract,
                                      "FIT", fixture.resolved)
            wrong_allele = dict(fixture.base)
            wrong_allele["event_ref"] = np.asarray(["A", "T"])
            with self.assertRaisesRegex(ValueError, "locus, allele"):
                binding.apply_product(wrong_allele, fixture.entry, fixture.contract,
                                      "FIT", fixture.resolved)
            with self.assertRaisesRegex(ValueError, "FIT-only"):
                binding.apply_product(fixture.base, fixture.entry, fixture.contract,
                                      "SELECT", fixture.resolved)
            wrong_hash = copy.deepcopy(fixture.contract)
            wrong_hash[binding.BLOCK]["products_by_case"][fixture.cases[0]]["receipt"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                binding.apply_product(fixture.base, fixture.entry, wrong_hash,
                                      "FIT", fixture.resolved)

    def test_regional_axis_length_and_declared_chromosome_are_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DoseBindingFixture(Path(temporary), REMOVED)
            wrong_axis = dict(fixture.base)
            wrong_axis["event_cm"] = np.zeros(3, np.float64)
            with self.assertRaisesRegex(ValueError, "axis shape differs"):
                binding.apply_product(wrong_axis, fixture.entry, fixture.contract,
                                      "FIT", fixture.resolved)
            wrong_chromosome = copy.deepcopy(fixture.contract)
            wrong_chromosome["lai_chromosome"] = "chr1"
            with self.assertRaisesRegex(ValueError, "integer autosome"):
                binding.apply_product(fixture.base, fixture.entry, wrong_chromosome,
                                      "FIT", fixture.resolved)

    def test_shifted_prefix_origin_is_rejected_even_when_increments_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DoseBindingFixture(Path(temporary), REMOVED)
            product = fixture.contract[binding.BLOCK]["products_by_case"][fixture.cases[0]]
            batch_path = Path(product["batch"]["path"])
            with np.load(batch_path, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            key = REMOVED + "_prefix_counts"
            arrays[key] = arrays[key] + 1
            np.savez_compressed(batch_path, **arrays)
            receipt_path = Path(product["receipt"]["path"])
            receipt = json.loads(receipt_path.read_text())
            receipt["batch_sha256"] = sha256(batch_path)
            receipt["bytes"] = batch_path.stat().st_size
            receipt["arrays_sha256"][key] = array_sha256(arrays[key], np)
            receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            product["batch"]["sha256"] = receipt["batch_sha256"]
            product["receipt"]["sha256"] = sha256(receipt_path)
            with self.assertRaisesRegex(ValueError, "prefix origin is not zero"):
                binding.apply_product(fixture.base, fixture.entry, fixture.contract,
                                      "FIT", fixture.resolved)

    def test_checkpoint_identity_separates_effective_condition(self):
        with tempfile.TemporaryDirectory() as temporary:
            removed = DoseBindingFixture(Path(temporary) / "removed", REMOVED)
            permuted = DoseBindingFixture(Path(temporary) / "permuted", PERMUTED)
            left = binding.checkpoint_binding_identity(removed.contract)
            right = binding.checkpoint_binding_identity(permuted.contract)
            self.assertNotEqual(left["effective_condition"], right["effective_condition"])
            self.assertIn("original_dose_sha256_ledger_only",
                          left["products_by_case"][removed.cases[0]])

    def test_context_entry_applies_before_return_and_revalidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DoseBindingFixture(Path(temporary), REMOVED)
            with mock.patch.object(trainer, "load_batch", return_value=fixture.base), \
                    mock.patch.object(trainer, "validate_batch") as checked:
                result, dataset = trainer.validate_training_entry(
                    fixture.entry, fixture.contract, "FIT", fixture.resolved)
            self.assertIsNone(dataset)
            self.assertFalse(result["personal_dose_available"])
            checked.assert_called_once_with(result, fixture.contract, "FIT", action="TRAIN")
            with mock.patch.object(trainer, "load_batch") as unopened:
                with self.assertRaisesRegex(ValueError, "SELECT/EVALUATE remain closed"):
                    trainer.validate_training_entry(
                        fixture.entry, fixture.contract, "SELECT", fixture.resolved)
                unopened.assert_not_called()

    def test_microbatch_preserves_authenticated_scalar_metadata(self):
        data = microbatch_data()
        result = slice_people(data, 0, 1)
        self.assertEqual(result["personal_dose_intervention"], REMOVED)
        self.assertFalse(result["personal_dose_available"])
        broken = dict(data); broken["personal_dose_available"] = True
        with self.assertRaisesRegex(ValueError, "condition/availability"):
            slice_people(broken, 0, 1)
        unknown = dict(data); unknown["dose_counts_original"] = np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, "unclassified"):
            slice_people(unknown, 0, 1)


if __name__ == "__main__":
    unittest.main()
