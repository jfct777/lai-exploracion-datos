#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m33_m0_factorized_lazy_source_auth",
    ROOT / "bin" / "m33_m0_factorized_lazy_source_auth.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FactorizedLazySourceAuthTests(unittest.TestCase):
    def test_inventory_covers_runtime_contract_nextflow_and_tests(self):
        expected = {
            "bin/m33_materialize.py",
            "bin/m33_m0_contract.py",
            "bin/m31_ordered_linear.py",
            "bin/m33_m0_factorized_lazy_amendment.py",
            "bin/m33_m0_factorized_lazy_source_auth.py",
            "bin/m33_m0_factorized_lazy_technical_kat.py",
            "conf/m33_m0_factorized_lazy_amendment_contract.json",
            "conf/m33_m0_factorized_lazy_contract.config",
            "modules/33_M0_FACTORIZED_LAZY_CONTRACT.nf",
            "modules/33_M0_FACTORIZED_LAZY_TECHNICAL_KAT.nf",
            "workflows/m33_m0_factorized_lazy_contract.nf",
            "workflows/m33_m0_factorized_lazy_technical_kat.nf",
            "tests/test_m33_materialize.py",
            "tests/test_m33_m0_factorized_lazy_amendment.py",
            "tests/test_m33_m0_factorized_lazy_nextflow.py",
            "tests/test_m33_m0_factorized_lazy_source_auth.py",
            "tests/test_m33_m0_factorized_lazy_technical_kat.py",
            "conf/m33_m0_factorized_lazy_technical_kat.config",
        }
        self.assertEqual(MODULE.REQUIRED_SOURCES, expected)

    def test_exclusive_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "source_auth.json"
            payload = {"status": "PASS"}
            MODULE.write_exclusive(path, payload)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
            with self.assertRaises(FileExistsError):
                MODULE.write_exclusive(path, payload)


if __name__ == "__main__":
    unittest.main()
