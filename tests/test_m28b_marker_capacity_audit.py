"""Known-answer and invariant tests for the M28B marker-capacity audit."""

from __future__ import annotations

import importlib.util
import gzip
import hashlib
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

    def test_pool_loader_rejects_one_individual_crossing_roles(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "pools.tsv"
            rows = [
                ("FREQ", "AFR", 10, 1),
                ("REF_LAI", "AFR", 10, 2),
            ]
            lines = ["role\tancestry\tindividual_id\tnode_id\tnode_identity_sha256"]
            for role, ancestry, individual, node in rows:
                identity = hashlib.sha256(f"source-node:{node}".encode()).hexdigest()
                lines.append(f"{role}\t{ancestry}\t{individual}\t{node}\t{identity}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crosses roles"):
                MODULE.load_allowed_pools(path, ["AFR"])

    def test_pool_loader_accepts_two_nodes_per_individual_in_one_role(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "pools.tsv"
            rows = [
                ("FREQ", "AFR", 10, 1),
                ("FREQ", "AFR", 10, 2),
                ("REF_LAI", "AFR", 11, 3),
                ("REF_LAI", "AFR", 11, 4),
            ]
            lines = ["role\tancestry\tindividual_id\tnode_id\tnode_identity_sha256"]
            for role, ancestry, individual, node in rows:
                identity = hashlib.sha256(f"source-node:{node}".encode()).hexdigest()
                lines.append(f"{role}\t{ancestry}\t{individual}\t{node}\t{identity}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            pools = MODULE.load_allowed_pools(path, ["AFR"])
            self.assertEqual(pools["FREQ"]["AFR"], [1, 2])
            self.assertEqual(pools["REF_LAI"]["AFR"], [3, 4])

    def test_individual_safe_loader_rejects_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "pools.tsv"
            identity = hashlib.sha256(b"source-haplotype:1").hexdigest()
            path.write_text(
                "role\tancestry\tnode_id\thaplotype_sha256\n"
                f"FREQ\tAFR\t1\t{identity}\n"
                f"REF_LAI\tAFR\t2\t{identity}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "v2 pool-manifest schema"):
                MODULE.load_allowed_pools(
                    path, ["AFR"], require_individual_schema=True
                )

    def test_pool_loader_rejects_manifest_tree_individual_mismatch(self):
        class Node:
            def __init__(self, individual):
                self.individual = individual

        class Tree:
            def node(self, node):
                return Node(99 if node == 2 else 10)

        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "pools.tsv"
            lines = ["role\tancestry\tindividual_id\tnode_id\tnode_identity_sha256"]
            for node in (1, 2):
                identity = hashlib.sha256(f"source-node:{node}".encode()).hexdigest()
                lines.append(f"FREQ\tAFR\t10\t{node}\t{identity}")
            for node in (3, 4):
                identity = hashlib.sha256(f"source-node:{node}".encode()).hexdigest()
                lines.append(f"REF_LAI\tAFR\t11\t{node}\t{identity}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match tree individual"):
                MODULE.load_allowed_pools(
                    path,
                    ["AFR"],
                    tree_sequence=Tree(),
                    require_individual_schema=True,
                )

    def test_inventory_checks_manifest_individuals_against_tree(self):
        class Node:
            def __init__(self, individual):
                self.individual = individual

        class Tree:
            def samples(self):
                return [1, 2]

            def node(self, node):
                return Node(10)

        pools = {
            "FREQ": {"AFR": [1]},
            "REF_LAI": {"AFR": [2]},
        }
        contract = {
            "version": 2,
            "source_populations": {"labels": ["AFR"]},
        }
        genetic_map = MODULE.GeneticMap("chr22", (100, 200), (0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "crosses allowed roles"):
            MODULE.inventory_markers(Tree(), pools, genetic_map, contract)

    def test_homozygous_minor_is_one_carrier_not_two(self):
        class Node:
            def __init__(self, individual):
                self.individual = individual

        class Tree:
            def node(self, node):
                return Node(10 if node in (1, 2) else 11)

        self.assertEqual(
            MODULE.count_minor_carrier_individuals(
                Tree(), [1, 2, 3, 4], {1: 1, 2: 1, 3: 0, 4: 0}, 1
            ),
            1,
        )
        self.assertEqual(
            MODULE.count_minor_carrier_individuals(
                Tree(), [1, 2, 3, 4], {1: 1, 2: 0, 3: 1, 4: 0}, 1
            ),
            2,
        )


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
            MODULE.write_marker_manifest(
                left, "B0", values, include_carrier_individuals=True
            )
            MODULE.write_marker_manifest(
                right, "B0", values, include_carrier_individuals=True
            )
            self.assertEqual(left.read_bytes(), right.read_bytes())
            with gzip.open(left, "rt", encoding="utf-8") as handle:
                header = handle.readline().rstrip("\n").split("\t")
            self.assertIn("freq_minor_carrier_individuals", header)

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
