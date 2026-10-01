"""Synthetic autosomal provenance, aggregation and display contracts."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(Path(__file__).parent))
import m165_autosome_sweep as aggregation
import m165_chr22_sweep as sweep
import m165_sweep_figures as figures
from test_m165_chr22_sweep import settings
from test_m165_sweep_figures import fixture

try:
    import numpy
    import pandas
    import igraph
    import leidenalg
    HAVE_CORE = True
except ImportError:
    HAVE_CORE = False


class AggregationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.samples = self.base / "samples.txt"
        self.samples.write_text("SYN_A\nSYN_B\nSYN_C\nSYN_D\nSYN_E\n")
        self.settings = self.base / "settings.json"
        self.settings.write_text(json.dumps(settings()))
        entries = []
        for chrom in aggregation.AUTOSOMES:
            folder = self.base / chrom
            folder.mkdir()
            pairs, configurations, summary = [folder / n for n in ("pairs.tsv.gz", "configurations.tsv", "summary.json")]
            rows = [dict(config_id=f"L{length}_G50000_N20", sample_a="SYN_A", sample_b="SYN_B",
                         n_segments=1, total_shared_bp=600000, n_shared_variants_total=30, max_segment_bp=600000)
                    for length in (250000, 500000)]
            with gzip.open(pairs, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=aggregation.FIELDS, delimiter="\t")
                writer.writeheader(); writer.writerows(rows)
            with configurations.open("w", newline="") as handle:
                table = [dict(config_id=r["config_id"], min_length_bp=length, max_gap_bp=50000,
                              min_shared_effective=20, n_pairs=1, **{k: r[k] for k in aggregation.SUM_FIELDS})
                         for length, r in zip((250000, 500000), rows)]
                writer = csv.DictWriter(handle, fieldnames=list(table[0]), delimiter="\t")
                writer.writeheader(); writer.writerows(table)
            summary.write_text(json.dumps(dict(chrom=f"chr{chrom}", status="COMPLETE_EXPLORATORY_NOT_VALIDATED",
                n_samples=5, selected_samples_order_sha256=sweep.sha256(self.samples), carrier_allele_mode="source_minor",
                source_rare_contract=dict(contract="minor_v1", cohort_sha256="0"*64, cohort_n_samples=5))))
            entries.append(dict(chromosome=chrom, pair_summary=str(pairs), configuration_summary=str(configurations),
                                summary=str(summary), sha256=dict(pair_summary=sweep.sha256(pairs),
                                    configuration_summary=sweep.sha256(configurations), summary=sweep.sha256(summary))))
        self.manifest = self.base / "manifest.json"
        self.manifest.write_text(json.dumps(dict(schema_version=1, chromosomes=entries)))

    def execute(self):
        return aggregation.aggregate(self.manifest, self.settings, self.samples, self.base / "aggregate")

    def alter(self, fn):
        manifest = json.loads(self.manifest.read_text())
        fn(manifest)
        self.manifest.write_text(json.dumps(manifest))

    def test_all_22_sum_once_and_preserve_maximum(self):
        result = self.execute()
        self.assertEqual(result["chromosomes"], aggregation.AUTOSOMES)
        self.assertFalse(result["interval_union_computed"])
        with gzip.open(self.base / "aggregate/pair_configuration_summary.tsv.gz", "rt") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(int(row["n_segments"]), 22)
            self.assertEqual(int(row["total_shared_bp"]), 13200000)
            self.assertEqual(int(row["max_segment_bp"]), 600000)
        sweep.prepare(self.base / "aggregate/pair_configuration_summary.tsv.gz",
                      self.base / "aggregate/configuration_summary.tsv", self.samples,
                      self.base / "prepared", settings(), aggregation.AUTOSOMES,
                      self.base / "aggregate/aggregation.json")
        manifest = json.loads(next((self.base / "prepared").glob("L*/input_manifest.json")).read_text())
        self.assertEqual(manifest["chromosome"], "autosomes_1_22")
        self.assertEqual(manifest["source_sha256"]["autosomal_aggregation"],
                         sweep.sha256(self.base / "aggregate/aggregation.json"))

    def test_missing_chromosome_rejected(self):
        self.alter(lambda m: m["chromosomes"].pop())
        with self.assertRaisesRegex(ValueError, "each of the 22"):
            self.execute()
        self.assertFalse((self.base / "aggregate").exists())

    def test_duplicate_chromosome_rejected(self):
        self.alter(lambda m: m["chromosomes"].__setitem__(1, m["chromosomes"][0]))
        with self.assertRaisesRegex(ValueError, "each of the 22"):
            self.execute()

    def test_changed_genotype_cohort_rejected(self):
        def change(m):
            entry = m["chromosomes"][2]
            path = Path(entry["summary"])
            data = json.loads(path.read_text()); data["source_rare_contract"]["cohort_sha256"] = "1"*64
            path.write_text(json.dumps(data)); entry["sha256"]["summary"] = sweep.sha256(path)
        self.alter(change)
        with self.assertRaisesRegex(ValueError, "frequency cohort"):
            self.execute()

    def test_analytical_sample_order_rejected(self):
        self.samples.write_text("SYN_B\nSYN_A\nSYN_C\nSYN_D\nSYN_E\n")
        with self.assertRaisesRegex(ValueError, "cohort, orientation"):
            self.execute()

    def test_wrong_numeric_config_rejected(self):
        def change(m):
            entry = m["chromosomes"][0]; path = Path(entry["configuration_summary"])
            path.write_text(path.read_text().replace("250000\t50000\t20", "250000\t25000\t20"))
            entry["sha256"]["configuration_summary"] = sweep.sha256(path)
        self.alter(change)
        with self.assertRaisesRegex(ValueError, "numeric thresholds"):
            self.execute()

    def test_pair_file_hash_rejected(self):
        self.alter(lambda m: m["chromosomes"][0]["sha256"].__setitem__("pair_summary", "f"*64))
        with self.assertRaisesRegex(ValueError, "compressed-byte hash"):
            self.execute()
        self.assertFalse((self.base / "aggregate/aggregation.json").exists())

    def test_reversed_duplicate_in_one_chromosome_rejected(self):
        def change(m):
            entry = m["chromosomes"][0]; path = Path(entry["pair_summary"])
            with gzip.open(path, "at") as handle:
                handle.write("L250000_G50000_N20\tSYN_B\tSYN_A\t1\t600000\t30\t600000\n")
            entry["sha256"]["pair_summary"] = sweep.sha256(path)
        self.alter(change)
        with self.assertRaisesRegex(ValueError, "Duplicate or reversed"):
            self.execute()

    def test_no_overwrite(self):
        self.execute()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.execute()

    def test_autosomal_scope_requires_authenticated_aggregation(self):
        with self.assertRaisesRegex(ValueError, "aggregation receipt"):
            sweep.prepare("absent", "absent", self.samples, self.base / "new", settings(), aggregation.AUTOSOMES)

    @unittest.skipUnless(HAVE_CORE, "requires pinned M16.5 runtime")
    def test_autosomal_aggregate_runs_actual_existing_core(self):
        self.execute()
        sweep.prepare(self.base / "aggregate/pair_configuration_summary.tsv.gz",
                      self.base / "aggregate/configuration_summary.tsv", self.samples,
                      self.base / "prepared", settings(), aggregation.AUTOSOMES,
                      self.base / "aggregate/aggregation.json")
        folder = self.base / "prepared/L250000_G50000_N20_T250000_U0"
        result = sweep.run(folder, self.base / "result", ROOT / "bin/ibd_community_enhanced.py")
        self.assertEqual(result["chromosomes"], aggregation.AUTOSOMES)
        self.assertEqual(result["chromosome"], "autosomes_1_22")
        self.assertEqual(result["n_edges"], 1)
        self.assertEqual(result["n_active"], 2)
        self.assertIn("autosomal_aggregation", result["source_sha256"])


class FigureScopeTests(unittest.TestCase):
    def test_all_22_scope_and_reject_mixed_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(Path(tmp) / "source")
            for path in root.glob("*/manifest.json"):
                data = json.loads(path.read_text())
                data.update(chromosome="autosomes_1_22", chromosomes=aggregation.AUTOSOMES,
                            aggregation_receipt_sha256="3"*64)
                data["source_sha256"]["autosomal_aggregation"] = "3"*64
                path.write_text(json.dumps(data))
            result = figures.load_results(root)
            self.assertEqual(result["chromosomes"], aggregation.AUTOSOMES)
            self.assertEqual(len(result["rows"]), 42)
            self.assertIn("22 autosomas", figures.chromosome_label(result))
            path = next(root.glob("*/manifest.json"))
            data = json.loads(path.read_text()); data["chromosomes"] = ["22"]; data["chromosome"] = "22"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "chromosome coverage"):
                figures.load_results(root)

    def test_incomplete_scope_cannot_be_called_whole_genome(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            figures.chromosome_label(dict(chromosomes=["1", "22"]))


if __name__ == "__main__":
    unittest.main()
