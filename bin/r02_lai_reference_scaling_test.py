#!/usr/bin/env python3
"""Focused tests for versioned LAI AC/AN scaling; no training is run."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from r02_lai_reference_scaling import (
    AC_AN_COLUMNS, COUNT_FIELD, PANEL_RECEIPT_SCHEMA, LOG1P_MODE,
    PRESERVED_COLUMNS, RAW_MODE, REFERENCE_FEATURE_NAMES, SCHEMA, contract_scaling,
    scale_reference_features,
)
from r02_context_train import TaskModel, checkpoint_identity, lai_events
import torch
from torch import nn


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def object_identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def contract(specification=None):
    value = {
        "reference_identity": {
            "reference_vcf_sha256": DIGEST_A,
            "ref_train_receipt_sha256": DIGEST_B,
        }
    }
    if specification is not None:
        value["lai_reference_scaling"] = specification
        if "sha256" in specification["provenance"]:
            value["reference_identity"]["ref_train_receipt_sha256"] = specification["provenance"]["sha256"]
    return value


def scaled_spec(receipt_path, count=753):
    return {
        "schema": SCHEMA,
        "mode": LOG1P_MODE,
        "reference_panel_people": count,
        "provenance": {
            "schema": PANEL_RECEIPT_SCHEMA,
            "path": str(receipt_path / "receipt.json"),
            "sha256": DIGEST_B,
            "count_field": COUNT_FIELD,
            "reference_vcf_sha256": DIGEST_A,
            "roles_path": str(receipt_path / "roles.tsv"),
            "roles_sha256": "c" * 64,
            "reference_people_by_ancestry": {
                "African": 341, "European": 387, "Native_American": 25
            },
        },
    }


class ReferenceScalingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.receipt_dir = Path(self.temporary.name)
        self.receipt_path = self.receipt_dir / "receipt.json"
        self.roles_path = self.receipt_dir / "roles.tsv"
        rows = ["sample_id\trole\tancestry"]
        for ancestry, number in (("African", 341), ("European", 387), ("Native_American", 25)):
            rows.extend(f"{ancestry}-{index}\tREF_TRAIN\t{ancestry}" for index in range(number))
        self.roles_path.write_text("\n".join(rows) + "\n")
        roles_sha256 = hashlib.sha256(self.roles_path.read_bytes()).hexdigest()
        receipt = {
            "schema": PANEL_RECEIPT_SCHEMA,
            "status": "COMPLETE_COUNTS_AND_CELL_PROVENANCE_NOT_TRAINING",
            "role": "REF_TRAIN",
            "people": 753,
            "input_sha256": {str(self.roles_path): roles_sha256},
        }
        self.receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        self.receipt_sha256 = hashlib.sha256(self.receipt_path.read_bytes()).hexdigest()
        self.specification_value = scaled_spec(self.receipt_dir)
        self.specification_value["provenance"]["sha256"] = self.receipt_sha256
        self.specification_value["provenance"]["roles_sha256"] = roles_sha256
        self.features = np.zeros((2, 3, 15), dtype=np.float32)
        base = np.arange(6, dtype=np.float32).reshape(2, 3) + 1
        for offset, addition in zip((0, 5, 10), (0, 10, 20)):
            ac = base + addition
            an = ac + 10
            self.features[..., offset] = ac
            self.features[..., offset + 1] = an
            self.features[..., offset + 2] = ac / an
            self.features[..., offset + 3] = 1
            self.features[..., offset + 4] = 1
        self.original_features = self.features.copy()

    def tearDown(self):
        self.temporary.cleanup()

    def specification(self, count=753):
        value = json.loads(json.dumps(self.specification_value))
        value["reference_panel_people"] = count
        return value

    def test_absent_configuration_is_exact_raw_mode(self):
        specification = contract_scaling(contract())
        actual = scale_reference_features(self.features, specification, np)
        self.assertEqual(specification, {"schema": SCHEMA, "mode": RAW_MODE})
        self.assertIs(actual, self.features)
        self.assertEqual(actual.tobytes(), self.features.tobytes())

    def test_only_six_ac_an_columns_are_scaled(self):
        specification = contract_scaling(contract(self.specification()))
        actual = scale_reference_features(self.features, specification, np)
        expected = np.log1p(self.features[..., AC_AN_COLUMNS].astype(np.float64)) / np.log1p(1506)
        np.testing.assert_allclose(actual[..., AC_AN_COLUMNS], expected.astype(np.float32), rtol=0, atol=0)
        np.testing.assert_array_equal(actual[..., PRESERVED_COLUMNS], self.features[..., PRESERVED_COLUMNS])
        np.testing.assert_array_equal(self.features, self.original_features)

    def test_lai_tensor_preserves_metadata_dosage_mask_and_coordinates(self):
        reference_loci = self.features[0]
        references = np.broadcast_to(reference_loci, self.features.shape).copy()
        mask = np.array([[True, False, True], [True, True, False]])
        dose = np.array([[2, 1, 0], [1, 2, 2]], dtype=np.int64)
        metadata = np.arange(72, dtype=np.float32).reshape(2, 3, 12)
        data = {
            "event_reference_features": references,
            "event_valid": mask,
            "event_dosage": dose,
            "event_metadata": metadata,
            "event_cm": np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
            "event_chrom": np.ones((2, 3), dtype=np.int64),
        }
        legacy = lai_events(data)
        explicit_raw = lai_events(data, contract_scaling({}))
        self.assertEqual(legacy.features.numpy().tobytes(), explicit_raw.features.numpy().tobytes())
        self.assertEqual(legacy.valid.numpy().tobytes(), explicit_raw.valid.numpy().tobytes())
        actual = lai_events(data, contract_scaling(contract(self.specification())))
        np.testing.assert_array_equal(actual.features[..., :12].numpy(), metadata)
        np.testing.assert_array_equal(actual.features[..., 12].numpy(), np.where(mask, dose, 0))
        np.testing.assert_array_equal(actual.valid.numpy(), mask)
        np.testing.assert_array_equal(actual.cm.numpy(), data["event_cm"])
        np.testing.assert_array_equal(actual.chrom.numpy(), data["event_chrom"])
        np.testing.assert_array_equal(
            actual.features[..., 13:][..., PRESERVED_COLUMNS].numpy(),
            references[..., PRESERVED_COLUMNS])

    def test_panel_count_changes_values_and_configuration_identity(self):
        first = contract_scaling(contract(self.specification(753)))
        one = scale_reference_features(self.features, first, np)
        second_specification = dict(first)
        second_specification["reference_panel_people"] = 800
        two = scale_reference_features(self.features, second_specification, np)
        self.assertFalse(np.array_equal(one[..., AC_AN_COLUMNS], two[..., AC_AN_COLUMNS]))
        self.assertNotEqual(object_identity(first), object_identity(second_specification))

    def test_scaled_checkpoint_identity_is_explicitly_distinct(self):
        base = contract()
        base.update({
            "objective": "lai", "steps": 96,
            "batches": {
                "FIT": [{"path": "/fit", "sha256": DIGEST_A}],
                "SELECT": [{"path": "/select", "sha256": DIGEST_B}],
            },
        })
        scaled = contract(self.specification())
        scaled.update({key: value for key, value in base.items() if key != "reference_identity"})
        raw_identity = checkpoint_identity(base, "c" * 64, "d" * 64, "RARE", 7)
        null_identity = checkpoint_identity(dict(base, lai_reference_scaling=None),
                                            "c" * 64, "d" * 64, "RARE", 7)
        self.assertEqual(raw_identity, null_identity)
        scaled_identity = checkpoint_identity(scaled, "c" * 64, "d" * 64, "RARE", 7)
        self.assertNotIn("lai_reference_scaling", raw_identity)
        self.assertNotIn("r02_lai_reference_scaling.py", raw_identity["code_sha256"])
        self.assertEqual(scaled_identity["lai_reference_scaling"], self.specification())
        self.assertIn("r02_lai_reference_scaling.py", scaled_identity["code_sha256"])
        self.assertNotEqual(object_identity(raw_identity), object_identity(scaled_identity))

    def test_zero_and_diploid_panel_bound(self):
        features = np.zeros((1, 1, 15), dtype=np.float32)
        bounds = np.array((682, 682, 774, 774, 50, 50), dtype=np.float32)
        features[..., AC_AN_COLUMNS] = bounds
        features[..., (3, 8, 13)] = 1
        actual = scale_reference_features(features, contract_scaling(contract(self.specification())), np)
        expected = (np.log1p(bounds.astype(np.float64)) / np.log1p(1506)).astype(np.float32)
        np.testing.assert_array_equal(actual[..., AC_AN_COLUMNS], expected.reshape(1, 1, 6))
        features[..., AC_AN_COLUMNS[0]] = 683
        with self.assertRaisesRegex(ValueError, "diploid panel bound"):
            scale_reference_features(features, contract_scaling(contract(self.specification())), np)
        features[..., AC_AN_COLUMNS] = bounds
        features[..., 11] = 51
        with self.assertRaisesRegex(ValueError, "ancestry diploid panel bound"):
            scale_reference_features(features, contract_scaling(contract(self.specification())), np)
        features = np.zeros((1, 1, 15), dtype=np.float32)
        features[..., (3, 8, 13)] = 1
        features[..., 0], features[..., 1] = 2, 1
        with self.assertRaisesRegex(ValueError, "contradicts knownness"):
            scale_reference_features(features, contract_scaling(contract(self.specification())), np)
        features[..., 0] = -1
        with self.assertRaisesRegex(ValueError, "contradicts knownness"):
            scale_reference_features(features, contract_scaling(contract(self.specification())), np)

    def test_provenance_and_panel_identity_are_required(self):
        wrong = self.specification()
        wrong["provenance"]["reference_vcf_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "receipt or VCF"):
            contract_scaling(contract(wrong))
        missing = self.specification()
        del missing["provenance"]["sha256"]
        with self.assertRaisesRegex(ValueError, "provenance fields differ"):
            contract_scaling(contract(missing))

        invented = self.specification(754)
        with self.assertRaisesRegex(ValueError, "authenticated REF_TRAIN receipt"):
            contract_scaling(contract(invented))

    def test_semantic_columns_unknown_and_padding_zero(self):
        self.assertEqual(
            tuple(REFERENCE_FEATURE_NAMES[index] for index in AC_AN_COLUMNS),
            ("African.AC_0", "African.AN_0", "European.AC_0", "European.AN_0",
             "Native_American.AC_0", "Native_American.AN_0"))
        features = np.zeros((1, 2, 15), dtype=np.float32)
        actual = scale_reference_features(
            features, contract_scaling(contract(self.specification())), np)
        np.testing.assert_array_equal(actual, features)
        self.assertEqual(contract_scaling({}), {"schema": SCHEMA, "mode": RAW_MODE})

    def test_task_model_forward_receives_scaled_reference_columns(self):
        class CapturePredictor(nn.Module):
            def predict_lai(self, common, queries, baseline, events, include_rare, query_burden):
                return events.features

        model = TaskModel.__new__(TaskModel)
        nn.Module.__init__(model)
        model.objective = "lai"
        model.lai_reference_scaling = contract_scaling(contract(self.specification()))
        model.encoder = CapturePredictor()
        references = np.broadcast_to(self.features[:1, :2], (1, 2, 15)).copy()
        data = {
            "common_features": np.zeros((1, 1, 2), np.float32),
            "common_cm": np.zeros((1, 1), np.float64),
            "common_chrom": np.ones((1, 1), np.int64),
            "common_valid": np.ones((1, 1), bool),
            "query_features": np.zeros((1, 1, 1), np.float32),
            "query_cm": np.zeros((1, 1), np.float64),
            "query_chrom": np.ones((1, 1), np.int64),
            "query_valid": np.ones((1, 1), bool),
            "event_metadata": np.zeros((1, 2, 12), np.float32),
            "event_reference_features": references,
            "event_dosage": np.zeros((1, 2), np.int64),
            "event_cm": np.zeros((1, 2), np.float64),
            "event_chrom": np.ones((1, 2), np.int64),
            "event_valid": np.ones((1, 2), bool),
            "baseline_probabilities": np.zeros((1, 1), np.float32),
            "query_burden": np.zeros((1, 1, 3), np.float32),
        }
        output = model(data, "RARE").detach().numpy()[..., 13:]
        expected = scale_reference_features(references, model.lai_reference_scaling, np)
        np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    unittest.main()
