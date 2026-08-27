#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_validate_experiment_contract as subject  # noqa: E402


CONTRACT = ROOT / "conf/m34_nam_experiment_contract.json"
EXPECTED_SHA256 = "dff5442ff413dd5b2cd901b2407082cf7f0629eb02d927c2942154054993c3ff"


class M34ValidateExperimentContractTests(unittest.TestCase):
    def test_exact_contract_passes_and_registers_hash(self):
        receipt = subject.validate(CONTRACT, EXPECTED_SHA256)
        self.assertEqual(receipt["status"], "PASS_EXACT_SELECTED_ROOT_AND_SIZE_CONTRACT")
        self.assertEqual(receipt["experiment_contract_sha256"], EXPECTED_SHA256)
        self.assertEqual(receipt["splits"]["FIT"]["donor_role"], "SOURCE_VALID")
        self.assertEqual(receipt["splits"]["VALID"]["donor_role"], "SOURCE_TEST")
        self.assertEqual(receipt["splits"]["FIT"]["people"], 24)
        self.assertEqual(receipt["splits"]["VALID"]["people"], 8)

    def test_pilot_128_and_each_root_are_selected_exactly(self):
        for root, fit_seed, valid_seed in (
            ("R0", 1439610605, 1702577247),
            ("R1", 667875703, 513710823),
            ("R2", 348301061, 1179260632),
        ):
            with self.subTest(root=root):
                receipt = subject.validate(CONTRACT, EXPECTED_SHA256, root, "pilot_128")
                self.assertEqual(receipt["splits"]["FIT"]["people"], 96)
                self.assertEqual(receipt["splits"]["VALID"]["people"], 32)
                self.assertEqual(receipt["splits"]["FIT"]["seed"], fit_seed)
                self.assertEqual(receipt["splits"]["VALID"]["seed"], valid_seed)

    def test_hash_or_scientific_parameter_drift_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            subject.validate(CONTRACT, "0" * 64)
        original = json.loads(CONTRACT.read_text(encoding="utf-8"))
        for path, value, expected in (
            (("mosaics", "primary_admixture_generations"), 13,
             "admixture generations differ"),
            (("mosaics", "seeds", "R0_VALID"), 1, "mosaic seeds differ"),
            (("roles", "mosaic_valid_donors"), "SOURCE_VALID", "role mapping differs"),
            (("rare_definition", "minimum_mac"), 3, "definition differs"),
        ):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as raw:
                payload = copy.deepcopy(original)
                cursor = payload
                for member in path[:-1]:
                    cursor = cursor[member]
                cursor[path[-1]] = value
                candidate = Path(raw) / "contract.json"
                candidate.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    subject.validate(candidate, subject.sha256_file(candidate))


if __name__ == "__main__":
    unittest.main()
