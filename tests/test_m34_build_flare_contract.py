#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_build_flare_contract as subject


class M34BuildFlareContractTests(unittest.TestCase):
    def test_contract_is_derived_from_frozen_experiment_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = {}
            for index, name in enumerate(subject.INPUT_ARGUMENTS):
                path = base / name
                path.write_bytes(f"input-{index}".encode())
                paths[name] = path
            payload = subject.build(
                ROOT / "conf" / "m34_nam_experiment_contract.json", paths)
            self.assertEqual(payload["ancestry_names"], ["AFR", "EUR", "NAM"])
            self.assertEqual(payload["parameters"], {
                "array": False, "probs": True, "em": True,
                "min-mac": 1, "min-maf": 0.0, "gen": 12.0,
                "update-p": False, "panel-probs": False,
                "seed": 3401103, "nthreads": 4,
            })
            for name, path in paths.items():
                self.assertEqual(payload["expected_sha256"][name],
                                 subject.sha256_file(path))
            self.assertEqual(
                payload["status"], "EXPLORATORY_CONTRACT_BLINDED_TO_LABELS"
            )
            self.assertTrue(all("truth" not in key.lower() for key in payload))

    def test_missing_or_linked_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = {}
            for name in subject.INPUT_ARGUMENTS:
                path = base / name
                path.write_bytes(name.encode())
                paths[name] = path
            paths["target_tbi"].unlink()
            with self.assertRaisesRegex(ValueError, "target_tbi"):
                subject.build(ROOT / "conf" / "m34_nam_experiment_contract.json", paths)


if __name__ == "__main__":
    unittest.main()
