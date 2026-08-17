"""Known-answer tests for the M28C Gnomix smoke resource audit."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_m28c_gnomix_smoke_resources",
    REPO / "bin" / "audit_m28c_gnomix_smoke_resources.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestUnits(unittest.TestCase):
    def test_parse_sizes(self):
        self.assertAlmostEqual(MODULE.parse_size_gib("1024 MiB"), 1.0)
        self.assertAlmostEqual(MODULE.parse_size_gib("1 GiB"), 1.0)
        self.assertAlmostEqual(MODULE.parse_size_gib("1 GB"), 1e9 / 1024**3)

    def test_parse_composite_durations(self):
        self.assertAlmostEqual(MODULE.parse_duration_seconds("1m 2.5s"), 62.5)
        self.assertAlmostEqual(MODULE.parse_duration_seconds("751ms"), 0.751)

    def test_rejects_unknown_units(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Nextflow size"):
            MODULE.parse_size_gib("12 widgets")
        with self.assertRaisesRegex(ValueError, "Unsupported Nextflow duration"):
            MODULE.parse_duration_seconds("soon")


if __name__ == "__main__":
    unittest.main()
