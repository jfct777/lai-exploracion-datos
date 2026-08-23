#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m33_m0_factorized_lazy_technical_kat",
    ROOT / "bin" / "m33_m0_factorized_lazy_technical_kat.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FactorizedLazyTechnicalKatTests(unittest.TestCase):
    def test_stratified_markers_are_deterministic_unique_and_cover_extremes(self):
        rare = np.linspace(0.0, 10.0, 100, dtype="<f8")
        markers = np.linspace(-1.0, 11.0, 1000, dtype="<f8")
        first = MODULE.stratified_markers(rare, markers, 128)
        second = MODULE.stratified_markers(rare, markers, 128)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, 128)
        self.assertEqual(np.unique(first).size, 128)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 999)

    def test_nextflow_is_two_root_read_only_nontraining_kat(self):
        module = (ROOT / "modules/33_M0_FACTORIZED_LAZY_TECHNICAL_KAT.nf").read_text()
        workflow = (ROOT / "workflows/m33_m0_factorized_lazy_technical_kat.nf").read_text()
        config = (ROOT / "conf/m33_m0_factorized_lazy_technical_kat.config").read_text()
        combined = "\n".join((module, workflow, config)).lower()
        self.assertIn("root17", workflow)
        self.assertIn("root18", workflow)
        self.assertIn("maxforks = 2", config.lower())
        self.assertIn("--network none", config)
        for forbidden in ("truth", "optimizer", "train_model", "gsutil", "gcloud"):
            self.assertNotIn(forbidden, combined)

    def test_canonical_marker_count_is_fixed(self):
        MODULE.validate_marker_count(512)
        for value in (1, 16, 511, 513):
            with self.assertRaisesRegex(ValueError, "exactly 512"):
                MODULE.validate_marker_count(value)


if __name__ == "__main__":
    unittest.main()
