"""Known-answer tests for the M28 simulation preflight."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m28_simulation_preflight", REPO / "bin" / "m28_simulation_preflight.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CONTRACT_PATH = REPO / "conf" / "m28_lai_simulation_preflight_preregistration.json"


class TestRareDefinition(unittest.TestCase):
    def setUp(self):
        self.contract = MODULE.load_contract(CONTRACT_PATH)

    def test_preregistered_frequency_pool_selects_exactly_mac_2_to_5(self):
        observed = []
        for mac in range(0, 9):
            genotypes = [1] * mac + [0] * (600 - mac)
            stats = MODULE.minor_allele_stats(genotypes)
            if MODULE.rare_under_contract(stats, self.contract):
                observed.append(mac)
        self.assertEqual(observed, [2, 3, 4, 5])

    def test_minor_allele_can_be_the_encoded_zero_allele(self):
        stats = MODULE.minor_allele_stats([1] * 598 + [0, 0])
        self.assertEqual(stats["minor_code"], 0)
        self.assertEqual(stats["mac"], 2)
        self.assertAlmostEqual(stats["maf"], 2 / 600)

    def test_source_counts_preserve_roles_and_published_weights(self):
        counts = MODULE.source_diploid_counts(self.contract)
        self.assertEqual(counts, {"AFR": 336, "EUR": 386, "ASIA": 436})


class TestGeneticMap(unittest.TestCase):
    def setUp(self):
        self.contract = MODULE.load_contract(CONTRACT_PATH)

    def test_rejects_non_increasing_positions(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "bad.map"
            path.write_text("22\t10\t0\n22\t10\t1\n", encoding="utf-8")
            contract = copy.deepcopy(self.contract)
            contract["region"].update({
                "start_bp": 10,
                "end_bp": 10,
                "length_bp_inclusive": 1,
                "span_cm": 1,
            })
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                MODULE.read_genetic_map(path, contract, verify_hash=False)

    def test_cm_interpolation_keeps_half_open_coordinates(self):
        genetic_map = MODULE.GeneticMap("chr22", (100, 200, 300), (1.0, 2.0, 4.0))
        self.assertEqual(genetic_map.length_bp, 201)
        self.assertEqual(genetic_map.cm_to_offset(1.0), 0)
        self.assertEqual(genetic_map.cm_to_offset(2.0), 100)
        self.assertEqual(genetic_map.cm_to_offset(4.0), 200)
        self.assertEqual(genetic_map.cm_to_offset(3.0), 150)


class TestMosaics(unittest.TestCase):
    def setUp(self):
        self.contract = MODULE.load_contract(CONTRACT_PATH)
        self.contract = copy.deepcopy(self.contract)
        self.contract["pools"]["target_haplotypes"] = 4
        self.contract["source_populations"]["pulse_generations"] = 2
        self.map = MODULE.GeneticMap("chr22", (100, 1100), (0.0, 20.0))
        self.donors = {
            "AFR": list(range(0, 100)),
            "EUR": list(range(100, 200)),
            "ASIA": list(range(200, 300)),
        }

    def test_mosaics_are_reproducible_disjoint_and_fully_covered(self):
        left = MODULE.draw_mosaics(self.map, self.donors, self.contract, 17)
        right = MODULE.draw_mosaics(self.map, self.donors, self.contract, 17)
        self.assertEqual(left, right)
        donors = [segment.donor_node for segments in left for segment in segments]
        self.assertEqual(len(donors), len(set(donors)))
        for segments in left:
            MODULE.validate_segment_cover(segments, self.map.length_bp)
            MODULE.validate_segment_cover(MODULE.merge_truth(segments), self.map.length_bp)

    def test_truth_merges_silent_recombinations_only(self):
        segments = [
            MODULE.MosaicSegment("T000_h0", 0, 10, "EUR", 1),
            MODULE.MosaicSegment("T000_h0", 10, 20, "EUR", 2),
            MODULE.MosaicSegment("T000_h0", 20, 30, "AFR", 3),
        ]
        truth = MODULE.merge_truth(segments)
        self.assertEqual([(row.start, row.end, row.ancestry) for row in truth], [
            (0, 20, "EUR"),
            (20, 30, "AFR"),
        ])


class TestTinyIntegration(unittest.TestCase):
    def test_msprime_and_stdpopsim_generate_disjoint_source_nodes(self):
        contract = copy.deepcopy(MODULE.load_contract(CONTRACT_PATH))
        contract["pools"]["frequency_diploids"].update({
            "AFR": 1, "EUR": 1, "ASIA": 1, "total": 3,
        })
        contract["pools"]["lai_reference_diploids_per_ancestry"] = 1
        contract["pools"]["mosaic_donor_haplotypes_per_ancestry"] = 4
        genetic_map = MODULE.GeneticMap("chr22", (100, 10_099), (0.0, 0.01))
        seeds = MODULE.derive_seeds(20260817)
        ts = MODULE.simulate_sources(genetic_map, contract, seeds)
        pools = MODULE.allocate_pools(ts, contract, seeds["pool"])
        nodes = [node for role in pools.values() for values in role.values() for node in values]
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertEqual(ts.sequence_length, genetic_map.length_bp)


if __name__ == "__main__":
    unittest.main()
