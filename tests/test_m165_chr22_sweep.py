"""Synthetic-only contracts for the private saved-summary M16.5 adapter."""
import copy
import csv
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m165_chr22_adapter", ROOT / "bin/m165_chr22_sweep.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def settings():
    return dict(expected_samples=5, resolutions=[.5, 1., 2.], n_seeds=3, seed=42,
                min_community_size=3, consensus_resolution=1., configurations=[
                    dict(length_bp=length, gap_bp=50000, min_shared=20,
                         min_edge_bp=edge, min_max_segment_bp=0)
                    for length, edges in [(250000, (250000, 500000, 1000000)),
                                          (500000, (500000, 750000, 1000000))]
                    for edge in edges])


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pairs = self.root / "pairs.tsv.gz"
        self.summary = self.root / "summary.tsv"
        self.samples = self.root / "samples.txt"
        self.samples.write_text("SYN_A\nSYN_B\nSYN_C\nSYN_D\nSYN_E\n")
        self.rows = []
        for length in (250000, 500000):
            for a, b, bp in [("SYN_A", "SYN_B", 600000),
                             ("SYN_A", "SYN_C", 700000),
                             ("SYN_B", "SYN_C", 800000)]:
                self.rows.append(dict(config_id=f"L{length}_G50000_N20", sample_a=a,
                                      sample_b=b, n_segments=1, total_shared_bp=bp,
                                      n_shared_variants_total=30, max_segment_bp=bp))
        self.save_fixture()

    def save_fixture(self):
        with gzip.open(self.pairs, "wt") as handle:
            writer = csv.DictWriter(handle, fieldnames=["config_id", *MOD.PAIR_FIELDS], delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows)
        with self.summary.open("w") as handle:
            fields = ["config_id", "n_pairs", "n_segments", "total_shared_bp", "n_shared_variants_total"]
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for length in (250000, 500000):
                rows = [r for r in self.rows if r["config_id"] == f"L{length}_G50000_N20"]
                writer.writerow(dict(config_id=f"L{length}_G50000_N20", n_pairs=len(rows),
                                     **{k: sum(r[k] for r in rows) for k in fields[2:]}))

    def prepare(self, output="prepared", config=None):
        return MOD.prepare(self.pairs, self.summary, self.samples, self.root / output,
                           settings() if config is None else config)

    def test_six_plans_and_single_pass_digest(self):
        receipt = self.prepare()
        self.assertEqual(receipt["n_configurations"], 6)
        self.assertEqual(receipt["n_source_configurations"], 2)
        self.assertEqual(receipt["rows_scanned"], len(self.rows))
        self.assertEqual(receipt["source_sha256"]["pair_configuration_summary"], MOD.sha256(self.pairs))
        folders = list((self.root / "prepared").glob("L*"))
        self.assertEqual(len(folders), 6)
        for folder in folders:
            payload = json.loads((folder / "input_manifest.json").read_text())
            self.assertTrue(payload["contains_individual_identifiers"])
            self.assertFalse(payload["public_distribution_allowed"])
            self.assertEqual((folder / "samples.txt").read_text(), self.samples.read_text())

    def test_no_overwrite(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()

    def test_aggregate_mismatch_rejected_before_creating_output(self):
        self.summary.write_text(self.summary.read_text().replace("2100000", "2100001"))
        with self.assertRaisesRegex(ValueError, "aggregate parity"):
            self.prepare()
        self.assertFalse((self.root / "prepared").exists())

    def test_reversed_duplicate_rejected(self):
        row = dict(self.rows[0], sample_a="SYN_B", sample_b="SYN_A")
        self.rows.append(row)
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate or reversed"):
            self.prepare()

    def test_self_pair_rejected(self):
        self.rows[0]["sample_b"] = "SYN_A"
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "self-pair"):
            self.prepare()

    def test_outside_sample_rejected(self):
        self.rows[0]["sample_b"] = "SYN_OUTSIDE"
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "outside cohort"):
            self.prepare()

    def test_wrong_cohort_size_rejected(self):
        cfg = settings()
        cfg["expected_samples"] = 4
        with self.assertRaisesRegex(ValueError, "count/uniqueness"):
            self.prepare(config=cfg)

    def test_unsupported_longest_segment_filter_rejected(self):
        cfg = settings()
        cfg["configurations"][0]["min_max_segment_bp"] = 500000
        with self.assertRaisesRegex(ValueError, "U=0"):
            self.prepare(config=cfg)

    def test_duplicate_graph_plan_rejected(self):
        cfg = settings()
        cfg["configurations"].append(copy.deepcopy(cfg["configurations"][0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            self.prepare(config=cfg)

    def test_invalid_resolution_rejected(self):
        for resolution in (0, float("nan"), float("inf")):
            cfg = settings()
            cfg["resolutions"].append(resolution)
            with self.assertRaisesRegex(ValueError, "Resolutions"):
                self.prepare(config=cfg)

    def test_unchosen_source_rows_are_not_accumulated(self):
        self.rows.append(dict(self.rows[0], config_id="UNUSED", sample_a="not_in_keep"))
        self.save_fixture()
        receipt = self.prepare()
        self.assertEqual(receipt["rows_scanned"], 7)
        self.assertEqual(receipt["source_totals"]["L250000_G50000_N20"]["n_pairs"], 3)

    def test_prepared_tampering_rejected(self):
        self.prepare()
        folder = self.root / "prepared/L250000_G50000_N20_T250000_U0"
        (folder / "samples.txt").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            MOD.run(folder, self.root / "result", ROOT / "bin/ibd_community_enhanced.py")

    def test_real_core_preserves_isolates_and_fixed_edge_rule(self):
        self.prepare()
        folder = self.root / "prepared/L250000_G50000_N20_T250000_U0"
        result = MOD.run(folder, self.root / "result", ROOT / "bin/ibd_community_enhanced.py")
        self.assertEqual((result["n_cohort"], result["n_active"], result["n_edges"]), (5, 3, 3))
        self.assertTrue(result["no_nmf"])
        with (self.root / "result/leiden_assignments.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 5)
        for row in rows[3:]:
            self.assertTrue(all(row[k] == "-1" for k in row if k.startswith("community_res_")))
        with self.assertRaisesRegex(ValueError, "already exists"):
            MOD.run(folder, self.root / "result", ROOT / "bin/ibd_community_enhanced.py")

    def test_prepared_configuration_identity_rejected(self):
        self.prepare()
        folder = self.root / "prepared/L250000_G50000_N20_T250000_U0"
        path = folder / "input_manifest.json"
        data = json.loads(path.read_text())
        data["configuration"]["config_id"] = "wrong_identity"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "configuration identity"):
            MOD.run(folder, self.root / "result", ROOT / "bin/ibd_community_enhanced.py")

    def test_no_edges_is_complete_descriptive_not_fake_clusters(self):
        self.prepare()
        folder = self.root / "prepared/L500000_G50000_N20_T1000000_U0"
        result = MOD.run(folder, self.root / "empty", ROOT / "bin/ibd_community_enhanced.py")
        self.assertEqual(result["status"], "COMPLETE_NO_EDGES")
        self.assertEqual(result["n_active"], 0)


if __name__ == "__main__":
    unittest.main()
