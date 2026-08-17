"""Known-answer tests for the M28C exact-loader ingest audit."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_m28c_gnomix_ingest", REPO / "bin" / "audit_m28c_gnomix_ingest.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def loaded_fixture(samples: int) -> dict:
    return {
        "calldata/GT": np.zeros((2, samples, 2), dtype=np.int8),
        "variants/POS": np.array([100, 200]),
        "variants/REF": np.array(["A", "A"]),
        "variants/ALT": np.array([["C"], ["C"]]),
    }


class TestLoadedPair(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "markers": 2,
            "reference_samples": 3,
            "target_samples": 1,
            "ploidy": 2,
            "ref": "A",
            "alt": "C",
        }

    def test_accepts_exact_shapes_and_marker_parity(self):
        audit = MODULE.audit_loaded_pair(loaded_fixture(3), loaded_fixture(1), self.expected)
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(audit["reference_shape"], [2, 3, 2])

    def test_rejects_missing_genotype(self):
        target = loaded_fixture(1)
        target["calldata/GT"][0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "loader parity failed"):
            MODULE.audit_loaded_pair(loaded_fixture(3), target, self.expected)

    def test_rejects_position_mismatch(self):
        target = loaded_fixture(1)
        target["variants/POS"][1] = 201
        with self.assertRaisesRegex(ValueError, "loader parity failed"):
            MODULE.audit_loaded_pair(loaded_fixture(3), target, self.expected)


class TestUpstreamAuthentication(unittest.TestCase):
    def test_rejects_nonpassing_upstream(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            reference = root / "reference.vcf.gz"
            target = root / "target.vcf.gz"
            report = root / "report.json"
            reference.write_bytes(b"reference")
            target.write_bytes(b"target")
            report.write_text(json.dumps({"decision": "STOP"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not pass"):
                MODULE.authenticate_upstream(reference, target, report)


class TestContract(unittest.TestCase):
    def test_scope_forbids_training_and_truth(self):
        contract = MODULE.load_contract(
            REPO / "conf" / "m28c_gnomix_ingest_preregistration.json"
        )
        self.assertEqual(contract["root_seed"], 20260818)
        self.assertIn("no_training", contract["scope"])
        self.assertIn("no_truth", contract["scope"])


if __name__ == "__main__":
    unittest.main()
