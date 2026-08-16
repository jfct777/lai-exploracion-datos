#!/usr/bin/env python3
"""Focused contracts for the M27E read-only feasibility audit."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))

import audit_m27e_ibd_rare_transfer as m27e  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestM27EContracts(unittest.TestCase):
    def test_refined_ibd_uses_lod_column_eight_and_cm_column_nine(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ibd_chr_1.ibd"
            path.write_text("A\t1\tB\t2\t1\t100\t200\t4.75\t12.5\n", encoding="utf-8")
            pairs, endpoints, receipt = m27e.read_ibd(
                {1: path},
                {"A": "A", "B": "B"},
                {"reported_segment_min_lod": 3.0, "reported_segment_min_cm": 2.0},
            )
        self.assertEqual(pairs[("A", "B")].total_cm, 12.5)
        self.assertEqual(pairs[("A", "B")].max_cm, 12.5)
        self.assertEqual(receipt["minimum_observed_lod"], 4.75)
        self.assertEqual(receipt["minimum_observed_length_cm"], 12.5)
        self.assertEqual(endpoints[1], (100, 200))

    def test_refined_ibd_log_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ibd_chr_22.log"
            path.write_text(
                "java -jar refined-ibd.17Jan20.102.jar\nSamples: 3,685\n  length=2\n  lod=3.0\n",
                encoding="utf-8",
            )
            receipt = m27e.parse_refined_ibd_log(path)
        self.assertEqual(receipt["version"], "refined-ibd.17Jan20.102.jar")
        self.assertEqual(receipt["n_samples"], 3685)
        self.assertEqual(receipt["minimum_length_cm"], 2.0)
        self.assertEqual(receipt["minimum_lod"], 3.0)

    def test_map_interpolation(self):
        self.assertAlmostEqual(m27e.interpolate_cm(150, [100, 200], [1.0, 3.0]), 2.0)
        self.assertAlmostEqual(m27e.interpolate_cm(50, [100, 200], [1.0, 3.0]), 0.0)

    def test_genome_span_follows_official_ibd_endpoint_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            maps = {}
            endpoints = {}
            for chromosome in range(1, 23):
                path = root / f"genetic.map.chr{chromosome}"
                path.write_text(
                    f"{chromosome}\t100\t1.0\n{chromosome}\t200\t3.0\n",
                    encoding="utf-8",
                )
                maps[chromosome] = path
                endpoints[chromosome] = (125, 175)
            total, per_chromosome = m27e.autosomal_span_cm(endpoints, maps)
        self.assertAlmostEqual(total, 22.0)
        self.assertTrue(all(value == 1.0 for value in per_chromosome.values()))

    def test_hash_pinned_strata_preserve_unmatched_without_population_union(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = root / "m27d_sample_strata.private.tsv"
            fields = [
                "sample_id",
                "match_status",
                "population_interpretable",
                "Source",
                "Ancestry",
                "Population",
            ]
            rows = [
                ["A", "MATCHED", "TRUE", "S1", "Native_American", "P"],
                ["B", "MATCHED", "TRUE", "S2", "Native_American", "P"],
                ["U", "UNMATCHED", "FALSE", "", "", ""],
            ]
            with table.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(fields)
                writer.writerows(rows)
            manifest = root / "m27d_sample_strata.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "stage": "M27D_SAMPLE_STRATA_RESOLUTION",
                        "sha256": {table.name: sha256(table)},
                    }
                ),
                encoding="utf-8",
            )
            upstream = {
                "resolved_strata_sha256": sha256(table),
                "resolved_strata_manifest_sha256": sha256(manifest),
                "expected_population_interpretable_samples": 2,
                "expected_population_unresolved_samples": 1,
            }
            metadata, receipt = m27e.read_resolved_strata(
                table, manifest, ["A", "B", "U"], upstream
            )
            roots, summary = m27e.build_blocks(
                ["A", "B", "U"], metadata, {}, 100.0, 10.0, 0.044, True
            )
        self.assertEqual(receipt["n_population_interpretable"], 2)
        self.assertEqual(receipt["n_population_unresolved"], 1)
        self.assertEqual(roots["A"], roots["B"])
        self.assertNotEqual(roots["A"], roots["U"])
        self.assertEqual(summary["n_blocks_total"], 2)

    def test_unphased_heterozygote_is_not_a_usable_haplotype_carrier(self):
        self.assertFalse(m27e.is_usable_minor_carrier(1, False, True, True))
        self.assertTrue(m27e.is_usable_minor_carrier(1, True, True, True))
        self.assertTrue(m27e.is_usable_minor_carrier(2, False, True, True))
        self.assertFalse(m27e.is_usable_minor_carrier(0, True, True, True))

    def test_effective_number_reports_concentration(self):
        self.assertEqual(m27e.effective_number([5, 5]), 2.0)
        self.assertAlmostEqual(m27e.effective_number([9, 1]), 100 / 82)
        self.assertEqual(m27e.effective_number([]), 0.0)

    def test_joint_baseline_and_leave_population_out_counts(self):
        samples = ["D1", "D2", "E1", "E2", "E3"]
        populations = {"D1": "D1", "D2": "D2", "E1": "P1", "E2": "P2", "E3": "P3"}
        metadata = {
            sample: {
                "Ancestry": "Native_American",
                "Source": "NatWGS" if sample.startswith("D") else "External",
                "Population": populations[sample],
                "_population_interpretable": "True",
            }
            for sample in samples
        }
        roots = {sample: sample for sample in samples}
        key = ("22", 100, "A", "G")
        with tempfile.TemporaryDirectory() as tmp:
            panel = Path(tmp) / "panel.vcf"
            panel.write_text(
                "##fileformat=VCFv4.2\n"
                + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(samples)
                + "\n22\t100\t.\tA\tG\t.\tPASS\t.\tGT\t"
                + "\t".join(["0|1"] * len(samples))
                + "\n",
                encoding="utf-8",
            )
            summary = m27e.summarize_panel_bridge(
                panel,
                samples,
                samples,
                ["D1", "D2"],
                {key: m27e.RawRareSite(True, bytes([1, 1]))},
                metadata,
                {"primary": roots},
                set(),
                "primary",
            )
        self.assertEqual(summary["n_direct_phase_bridge_sites"], 1)
        self.assertEqual(
            summary["primary_native_american_transferable_sites_outside_frozen_baseline"], 1
        )
        self.assertEqual(summary["primary_native_american_baseline_disjoint_lopo_robust_sites"], 1)
        concentration = summary["primary_native_american_baseline_disjoint_concentration"]
        self.assertEqual(concentration["n_contributing_external_populations"], 3)
        self.assertEqual(concentration["effective_external_populations_by_site_support"], 3.0)

    def test_recent_kinship_requires_anchor_segment_and_total_ibd(self):
        metadata = {
            sample: {
                "Ancestry": "Native_American",
                "Source": sample,
                "Population": sample,
                "_population_interpretable": "True",
            }
            for sample in ("A", "B", "C")
        }
        pairs = {
            ("A", "B"): m27e.PairIbd(total_cm=20.0, max_cm=9.9, n_segments=3),
            ("A", "C"): m27e.PairIbd(total_cm=20.0, max_cm=10.0, n_segments=3),
        }
        roots, summary = m27e.build_blocks(
            ["A", "B", "C"], metadata, pairs, 100.0, 10.0, 0.05, False
        )
        self.assertNotEqual(roots["A"], roots["B"])
        self.assertEqual(roots["A"], roots["C"])
        self.assertEqual(summary["n_kinship_edges"], 1)

    def test_raw_reported_segment_components_are_only_a_percolation_diagnostic(self):
        metadata = {
            sample: {
                "Ancestry": "Native_American",
                "Source": sample,
                "Population": sample,
                "_population_interpretable": "True",
            }
            for sample in ("A", "B", "C")
        }
        pairs = {
            ("A", "B"): m27e.PairIbd(total_cm=2.0, max_cm=2.0, n_segments=1),
            ("B", "C"): m27e.PairIbd(total_cm=2.0, max_cm=2.0, n_segments=1),
        }
        roots, summary = m27e.build_blocks(
            ["A", "B", "C"], metadata, pairs, 100.0, 0.0, 0.0, False
        )
        self.assertEqual(len(set(roots.values())), 1)
        self.assertEqual(summary["largest_block_samples"], 3)

    def test_workflow_is_nextflow_first_and_cloud_label_is_inherited(self):
        workflow = (REPO / "workflows/m27e_ibd_rare_transfer_feasibility.nf").read_text()
        module = (REPO / "modules/27E_IBD_RARE_TRANSFER_FEASIBILITY.nf").read_text()
        cloud = (REPO / "conf/google_batch.config").read_text()
        self.assertIn("AUDIT_IBD_RARE_TRANSFER_FEASIBILITY", workflow)
        self.assertIn("--resolved-strata ${resolved_strata}", module)
        self.assertIn("(?:[._]|$)", workflow)
        self.assertIn("resourceLabels = [team: 'frank']", cloud)
        self.assertNotIn("\\n+", module)
        self.assertNotIn("sbatch", workflow + module)


if __name__ == "__main__":
    unittest.main()
