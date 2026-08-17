"""Known-answer tests for M28C B0 input materialization."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_m28c_b0_inputs", REPO / "bin" / "materialize_m28c_b0_inputs.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestReferencePairing(unittest.TestCase):
    def test_pairs_every_haplotype_once_within_ancestry(self):
        nodes = {"AFR": [5, 2, 9, 1], "EUR": [14, 11, 13, 12]}
        pairs = MODULE.pair_reference_haplotypes(nodes, expected_haplotypes=4)
        self.assertEqual(pairs, [
            ("REF_AFR_000", "AFR", 1, 2),
            ("REF_AFR_001", "AFR", 5, 9),
            ("REF_EUR_000", "EUR", 11, 12),
            ("REF_EUR_001", "EUR", 13, 14),
        ])

    def test_rejects_wrong_haplotype_count(self):
        with self.assertRaisesRegex(ValueError, "Expected 4"):
            MODULE.pair_reference_haplotypes({"AFR": [1, 2]}, expected_haplotypes=4)


class TestMosaicLookup(unittest.TestCase):
    def setUp(self):
        self.segments = [
            MODULE.Segment(100, 200, "AFR", 7),
            MODULE.Segment(200, 300, "EUR", 9),
        ]

    def test_half_open_boundaries_are_respected(self):
        self.assertEqual(MODULE.segment_node_at(self.segments, 199, 0), (7, 0))
        self.assertEqual(MODULE.segment_node_at(self.segments, 200, 0), (9, 1))

    def test_rejects_position_outside_mosaic(self):
        with self.assertRaisesRegex(ValueError, "not covered"):
            MODULE.segment_node_at(self.segments, 99, 0)


class TestLeakageBoundary(unittest.TestCase):
    def test_cli_has_no_truth_or_prediction_argument(self):
        source = (REPO / "bin" / "materialize_m28c_b0_inputs.py").read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--truth', source)
        self.assertNotIn('add_argument("--prediction', source)

    def test_contract_keeps_smoke_seed_out_of_inference(self):
        contract = MODULE.load_contract(REPO / "conf" / "m28c_b0_input_preregistration.json")
        self.assertEqual(contract["root_seed"], 20260818)
        self.assertEqual(contract["seed_role"], "technical_smoke_excluded_from_inference")
        self.assertEqual(contract["expected"]["b0_markers"], 79791)


class TestVcfAudit(unittest.TestCase):
    def test_audits_phased_binary_calls(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "tiny.vcf.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("##fileformat=VCFv4.2\n")
                handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\n")
                handle.write("22\t100\tm28s3\tA\tC\t.\tPASS\tTSID=3\tGT\t0|1\n")
            audit = MODULE.audit_vcf(path, expected_samples=1)
            self.assertEqual(audit["record_count"], 1)
            self.assertTrue(audit["ordered_unique"])
            self.assertTrue(audit["all_phased_binary"])

    def test_rejects_unexpected_sample_count(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "tiny.vcf.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("##fileformat=VCFv4.2\n")
                handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\n")
            with self.assertRaisesRegex(ValueError, "Expected 2 samples"):
                MODULE.audit_vcf(path, expected_samples=2)


if __name__ == "__main__":
    unittest.main()
