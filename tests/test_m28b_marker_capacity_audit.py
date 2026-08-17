"""Known-answer and invariant tests for the M28B marker-capacity audit."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))
SPEC = importlib.util.spec_from_file_location(
    "m28b_marker_capacity_audit", BIN / "m28b_marker_capacity_audit.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_m28b_reproducibility", BIN / "verify_m28b_reproducibility.py"
)
VERIFY = importlib.util.module_from_spec(VERIFY_SPEC)
sys.modules[VERIFY_SPEC.name] = VERIFY
VERIFY_SPEC.loader.exec_module(VERIFY)


def marker(site_id: int, bp: int, cm: float, maf: float = 0.1, ref: int = 3):
    return MODULE.Marker(site_id, bp, cm, 1, 10, 100, maf, ref, ref, 0, 0)


class TestContract(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            (REPO / "conf" / "m28b_lai_marker_capacity_preregistration.json").read_text()
        )

    def test_primary_rules_are_frozen_before_lai(self):
        self.assertEqual(self.contract["primary_nonrare_screen"], "nonrare_ge_0_01")
        self.assertEqual(self.contract["rare_selector"]["primary_reference_minor_copy_threshold"], 1)
        self.assertEqual(self.contract["bs_matching"]["bin_widths_cm"], [0.05, 0.1, 0.25, 0.5, 1.0])
        self.assertIn("no_lai", self.contract["scope"])

    def test_cli_rejects_truth_target_and_donor_inputs(self):
        required = [
            "audit.py", "--tree-sequence", "a", "--pool-manifest", "b",
            "--genetic-map", "c", "--baseline-template", "d",
            "--m28-preregistration", "e", "--preregistration", "f",
            "--outdir", "g", "--lai-truth", "forbidden",
        ]
        with mock.patch.object(sys, "argv", required), self.assertRaises(SystemExit):
            MODULE.parse_args()


class TestMapAndSelection(unittest.TestCase):
    def test_bp_to_cm_interpolates_without_extrapolation(self):
        genetic_map = MODULE.GeneticMap("chr22", (100, 200, 300), (1.0, 2.0, 4.0))
        self.assertAlmostEqual(MODULE.bp_to_cm(genetic_map, 250), 3.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.bp_to_cm(genetic_map, 99)

    def test_monotonic_mapping_is_unique_and_deterministic(self):
        queries = [MODULE.TemplatePosition(100, 0.10), MODULE.TemplatePosition(200, 0.20)]
        candidates = [
            marker(1, 99, 0.099), marker(2, 101, 0.101),
            marker(3, 199, 0.199), marker(4, 201, 0.201),
        ]
        left = MODULE.nearest_monotonic_pairs(queries, candidates)
        right = MODULE.nearest_monotonic_pairs(queries, list(reversed(candidates)))
        self.assertEqual(left, right)
        self.assertEqual(len({pair.control.site_id for pair in left}), 2)
        self.assertEqual([pair.control.site_id for pair in left], [1, 3])

    def test_mapping_fails_when_candidates_are_insufficient(self):
        queries = [MODULE.TemplatePosition(100, 0.1), MODULE.TemplatePosition(200, 0.2)]
        self.assertIsNone(MODULE.nearest_monotonic_pairs(queries, [marker(1, 100, 0.1)]))


class TestCapacityMatching(unittest.TestCase):
    def test_exact_per_bin_matching_passes(self):
        rare = [marker(1, 100, 0.01, 0.005), marker(2, 200, 0.06, 0.005)]
        reserve = [marker(3, 101, 0.011), marker(4, 199, 0.061)]
        pairs, diagnostics = MODULE.match_controls_by_bin(rare, reserve, 0.0, 0.05)
        self.assertIsNotNone(pairs)
        self.assertEqual(diagnostics["matched"], 2)
        self.assertEqual(diagnostics["bins_without_capacity"], 0)

    def test_capacity_fails_instead_of_borrowing_across_bins(self):
        rare = [marker(1, 100, 0.01, 0.005), marker(2, 200, 0.06, 0.005)]
        reserve = [marker(3, 101, 0.011), marker(4, 102, 0.012)]
        pairs, diagnostics = MODULE.match_controls_by_bin(rare, reserve, 0.0, 0.05)
        self.assertIsNone(pairs)
        self.assertEqual(diagnostics["unmatched_rare_count"], 1)
        self.assertEqual(diagnostics["bins_without_capacity"], 1)

    def test_nonrare_reference_rule_requires_both_alleles(self):
        markers = [marker(1, 100, 0.1, 0.01, ref=0), marker(2, 200, 0.2, 0.05, ref=3)]
        selected = MODULE.nonrare_candidates(markers, 0.01, ref_total_haplotypes=10)
        self.assertEqual([value.site_id for value in selected], [2])


class TestDeterministicOutput(unittest.TestCase):
    def test_marker_manifest_is_byte_identical(self):
        values = [marker(2, 200, 0.2), marker(1, 100, 0.1)]
        with tempfile.TemporaryDirectory() as name:
            left = Path(name) / "left.tsv.gz"
            right = Path(name) / "right.tsv.gz"
            MODULE.write_marker_manifest(left, "B0", values)
            MODULE.write_marker_manifest(right, "B0", values)
            self.assertEqual(left.read_bytes(), right.read_bytes())

    def test_reproducibility_verifier_detects_a_changed_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run1, run2 = root / "run1", root / "run2"
            run1.mkdir()
            run2.mkdir()
            for filename in VERIFY.SCIENTIFIC_FILES:
                (run1 / filename).write_bytes(b"same")
                (run2 / filename).write_bytes(b"same")
            self.assertEqual(VERIFY.verify(run1, run2)["gate"], "PASS")
            (run2 / VERIFY.SCIENTIFIC_FILES[-1]).write_bytes(b"changed")
            self.assertEqual(VERIFY.verify(run1, run2)["decision"], "STOP_REPRODUCIBILITY")


if __name__ == "__main__":
    unittest.main()
