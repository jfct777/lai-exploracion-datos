"""Synthetic aggregate bundles only; no real subjects or production execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
import r02_community_summary as summary
from test_r02_community_evidence import SyntheticFixture, read_rows, write_json, write_tsv


class SummaryFixture:
    """Factory for Nextflow smoke/resume; all individuals are fictional.

    `fixture = SummaryFixture(root)` exposes manifest_path and manifest_hash.
    """
    def __init__(self, root, all_unassigned=False):
        self.root = Path(root).absolute()
        self.integration = SyntheticFixture(self.root, all_unassigned=all_unassigned)
        self.integration.run()
        self.bundle = self.root / "output"
        self.manifest_path = self.bundle / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())

    @property
    def manifest_hash(self):
        return summary.sha256(self.manifest_path)

    def save(self):
        write_json(self.manifest_path, self.manifest)

    def rewrite_table(self, name, rows):
        path = self.bundle / (name + ".tsv")
        write_tsv(path, rows)
        self.manifest["outputs_sha256"][path.name] = summary.sha256(path)
        self.manifest["table_rows"][name] = len(rows)
        self.save()

    def run(self, name="summary"):
        return summary.run(self.manifest_path, self.manifest_hash, self.root / name, 512)


class CommunitySummaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.fx = SummaryFixture(temp.name)

    def output_json(self):
        return json.loads((self.fx.root / "summary/summary.json").read_text())

    def test_preserves_every_partition_and_all_communities(self):
        result = self.fx.run()
        self.assertEqual(result["status"], summary.STATUS)
        self.assertEqual((result["n_graphs"], result["n_partitions"], result["n_cohort"]), (4, 5, 6))
        data = self.output_json()
        self.assertEqual(len(data["partitions"]), 5)
        self.assertEqual(len(read_rows(self.fx.root / "summary/partitions.tsv")), 10)
        self.assertEqual(len(read_rows(self.fx.root / "summary/community_components.tsv")), 20)
        self.assertTrue(all(len(p["communities"]) == 2 for p in data["partitions"]))
        self.assertEqual(result["incremental_rare_utility"], "NOT_ESTIMATED")
        self.assertTrue(result["no_winner_selection"])
        self.assertFalse(result["public_distribution_allowed"])
        self.assertEqual(result["sample_ids_sha256"], self.fx.manifest["sample_ids_sha256"])

    def test_concentration_two_denominators_and_both_phi(self):
        self.fx.run()
        p = next(p for p in self.output_json()["partitions"] if p["graph_id"] == "primary/R")
        loose, strict = p["concentration"]["0.0221"], p["concentration"]["0.0442"]
        self.assertEqual((loose["sum_community_maximum_observed_n"], loose["n_assigned"], loose["n_component_known"]), (2, 4, 2))
        self.assertEqual((loose["sum_maximum_fraction_assigned"], loose["sum_maximum_fraction_known"]), (.5, 1.))
        self.assertEqual((strict["sum_maximum_fraction_assigned"], strict["sum_maximum_fraction_known"]), (.25, .5))
        self.assertEqual(loose["status"], "DESCRIPTIVE_PARTIAL")
        unknown = p["communities"][1]["components"]["0.0221"]
        self.assertIsNone(unknown["maximum_observed_n"])
        self.assertEqual(unknown["status"], "NO_EVALUABLE")

    def test_four_ancestries_partial_missingness_and_category_ties(self):
        self.fx.run()
        p = next(p for p in self.output_json()["partitions"] if p["graph_id"] == "primary/R")
        annotations = p["scopes"]["unassigned"]["ancestry"]
        self.assertEqual(set(annotations), set(summary.evidence.ANCESTRY))
        item = annotations[summary.evidence.ANCESTRY[0]]
        self.assertEqual(item["field_observed"]["n_observed"], 1)
        self.assertEqual(item["field_observed"]["mean_fraction"], .4)
        self.assertEqual(item["complete_vector"]["field_n_missing"], 1)
        self.assertEqual(item["complete_vector"]["n_excluded_incomplete_vector"], 1)
        self.assertIsNone(item["complete_vector"]["mean_fraction"])
        category = p["communities"][0]["categorical"]["Cohort"]
        self.assertEqual(category["dominant_observed_categories"], ["Collection_0", "Collection_1"])
        self.assertEqual(category["dominant_observed_n"], 1)
        self.assertEqual(p["scopes"]["cohort"]["categorical"]["Region"]["n_missing"], 1)

    def test_reused_ari_not_recalculated_or_truth(self):
        self.fx.run()
        rows = self.output_json()["comparisons"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["ari"], -.5)
        self.assertEqual((rows[0]["n_both_assigned"], rows[0]["n_neither_assigned"]), (4, 2))

    def test_report_spanish_private_complete_and_reproducible(self):
        first = self.fx.run()
        second = self.fx.run("second")
        self.assertEqual(first["outputs_sha256"], second["outputs_sha256"])
        report = (self.fx.root / "summary/report.md").read_text()
        self.assertIn("todas las 5 particiones", report)
        self.assertIn("No es una tasa familiar", report)
        self.assertIn("No usarla para ordenar resoluciones", report)
        self.assertIn("Componente conocido", report)
        self.assertIn("NO ESTIMADA", report)
        self.assertIn("Indígena americana", report)
        self.assertIn("Asiática oriental", report)
        for name, digest in first["outputs_sha256"].items():
            path = self.fx.root / "summary" / name
            self.assertEqual(summary.sha256(path), digest)
            text = path.read_text()
            self.assertNotIn("PRIVATE_CITY_SENTINEL", text)
            self.assertNotIn("PRIVATE_CLINICAL_SENTINEL", text)
            for sid in self.fx.integration.samples:
                self.assertNotIn(sid, text)
        for path, entry in first["input_files"].items():
            self.assertEqual(summary.sha256(path), entry["sha256"])

    def test_all_unassigned_keeps_cohort_and_unknowns(self):
        empty = SummaryFixture(self.fx.root / "empty", all_unassigned=True)
        empty.run()
        data = json.loads((empty.root / "summary/summary.json").read_text())
        for p in data["partitions"]:
            self.assertEqual((p["n_communities"], p["scopes"]["unassigned"]["n_samples"]), (0, 6))
            self.assertIsNone(p["concentration"]["0.0221"]["sum_maximum_fraction_assigned"])
            self.assertEqual(p["concentration"]["0.0221"]["status"], "NO_EVALUABLE")
        self.assertTrue(all(row["ari"] is None for row in data["comparisons"]))

    def test_hash_tampering_and_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            summary.run(self.fx.manifest_path, "0" * 64, self.fx.root / "wrong", 512)
        self.fx.run()
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.fx.run()
        path = self.fx.bundle / "components.tsv"
        path.write_text(path.read_text() + "tampering\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.fx.run("tampered")
        self.assertFalse((self.fx.root / "tampered").exists())

    def test_unknown_version_and_wrong_cohort_rejected(self):
        self.fx.manifest["schema"] = "r02_community_evidence_v2"
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "version/status"):
            self.fx.run()
        self.fx.manifest["schema"] = "r02_community_evidence_v1"
        self.fx.manifest["n_cohort"] = 7
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Cohort universe"):
            self.fx.run()

    def test_source_claim_upgrade_rejected(self):
        self.fx.manifest["incremental_rare_utility"] = "CONFIRMED"
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "interpretation/privacy"):
            self.fx.run()

    def test_missing_and_duplicate_scope_rejected(self):
        rows = read_rows(self.fx.bundle / "scopes.tsv")
        self.fx.rewrite_table("scopes", rows[1:])
        with self.assertRaisesRegex(ValueError, "Missing required scope"):
            self.fx.run()
        self.fx.rewrite_table("scopes", rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "Unknown/duplicate scope"):
            self.fx.run()

    def test_missing_component_cutoff_rejected(self):
        rows = read_rows(self.fx.bundle / "components.tsv")
        self.fx.rewrite_table("components", rows[1:])
        with self.assertRaisesRegex(ValueError, "Incomplete annotation grid"):
            self.fx.run()

    def test_missing_component_not_accepted_as_zero(self):
        rows = read_rows(self.fx.bundle / "components.tsv")
        index = next(i for i, row in enumerate(rows) if row["max_component_n"] == "NA")
        rows[index]["max_component_n"] = "0"
        self.fx.rewrite_table("components", rows)
        with self.assertRaisesRegex(ValueError, "maximum component"):
            self.fx.run()

    def test_missing_ancestry_field_and_wrong_units_rejected(self):
        rows = read_rows(self.fx.bundle / "ancestry.tsv")
        self.fx.rewrite_table("ancestry", [r for r in rows if r["field"] != summary.evidence.ANCESTRY[3]])
        with self.assertRaisesRegex(ValueError, "Incomplete annotation grid"):
            self.fx.run()
        rows[0]["unit"] = "percent"
        self.fx.rewrite_table("ancestry", rows)
        with self.assertRaisesRegex(ValueError, "units/std differ"):
            self.fx.run()

    def test_wrong_ancestry_scope_denominator_rejected(self):
        rows = read_rows(self.fx.bundle / "ancestry.tsv")
        index = next(i for i, row in enumerate(rows) if row["n_excluded_incomplete_vector"] == "1")
        rows[index]["n_excluded_incomplete_vector"] = "0"
        self.fx.rewrite_table("ancestry", rows)
        with self.assertRaisesRegex(ValueError, "incomplete-vector exclusion"):
            self.fx.run()

    def test_category_universe_and_prohibited_column_rejected(self):
        rows = read_rows(self.fx.bundle / "categorical.tsv")
        rows[0]["n_cohort"] = "7"
        self.fx.rewrite_table("categorical", rows)
        with self.assertRaisesRegex(ValueError, "scope/universe differs"):
            self.fx.run()
        rows[0]["n_cohort"] = "6"
        rows[0]["field"] = "City"
        self.fx.rewrite_table("categorical", rows)
        with self.assertRaisesRegex(ValueError, "Unknown categorical field"):
            self.fx.run()

    def test_category_counts_must_reconcile_disjoint_scopes(self):
        rows = read_rows(self.fx.bundle / "categorical.tsv")
        index = next(i for i, row in enumerate(rows) if row["scope"] == "community" and row["is_missing"] == "False")
        rows[index]["category"] = "different_category"
        self.fx.rewrite_table("categorical", rows)
        with self.assertRaisesRegex(ValueError, "Category counts do not match disjoint scopes"):
            self.fx.run()

    def test_wrong_comparison_denominators_rejected(self):
        rows = read_rows(self.fx.bundle / "comparisons.tsv")
        rows[0]["n_both_assigned"] = "5"
        rows[0]["n_neither_assigned"] = "1"
        self.fx.rewrite_table("comparisons", rows)
        with self.assertRaisesRegex(ValueError, "assigned support differs"):
            self.fx.run()

    def test_manifest_row_counts_and_ledger_rejected(self):
        self.fx.manifest["table_rows"]["scopes"] += 1
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "row count differs"):
            self.fx.run()
        self.fx.manifest["table_rows"]["scopes"] -= 1
        self.fx.manifest["n_partitions"] += 1
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Incomplete graph/partition ledger"):
            self.fx.run()

    def test_nonnumeric_and_infinite_values_rejected(self):
        rows = read_rows(self.fx.bundle / "ancestry.tsv")
        rows[0]["mean_fraction"] = "inf"
        self.fx.rewrite_table("ancestry", rows)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.fx.run()

    def test_memory_guard(self):
        with self.assertRaisesRegex(ValueError, "RSS exceeds"):
            summary.run(self.fx.manifest_path, self.fx.manifest_hash, self.fx.root / "tiny", 1)

    def test_code_change_detected_without_mutating_files(self):
        original = summary.sha256
        reads = 0

        def changed(path):
            nonlocal reads
            if Path(path).resolve() == Path(summary.__file__).resolve():
                reads += 1
                if reads > 1:
                    return "0" * 64
            return original(path)

        with mock.patch.object(summary, "sha256", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Source changed during summary"):
                self.fx.run()
        self.assertFalse((self.fx.root / "summary/manifest.json").exists())

    def test_cli(self):
        result = subprocess.run([sys.executable, str(ROOT / "bin/r02_community_summary.py"),
            "--manifest", str(self.fx.manifest_path), "--expected-manifest-sha256", self.fx.manifest_hash,
            "--output-dir", str(self.fx.root / "cli"), "--max-memory-mb", "512"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["n_partitions"], 5)


if __name__ == "__main__":
    unittest.main()
