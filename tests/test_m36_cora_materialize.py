from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m36_cora_materialize", ROOT / "bin/m36_cora_materialize.py")
MATERIALIZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATERIALIZE
SPEC.loader.exec_module(MATERIALIZE)


class M36CoraMaterializeTests(unittest.TestCase):
    def fixture(self, name: str) -> Path:
        return ROOT / "tests/fixtures" / name

    def run_fixture(self, outdir: Path):
        return MATERIALIZE.run(type("Args", (), {
            "rare_vcf": self.fixture("m36_cora_factorized_rare.vcf"),
            "locus_metadata": self.fixture("m36_cora_factorized_loci.tab"),
            "genetic_map": self.fixture("m36_cora_factorized_map.tab"),
            "sample_metadata": self.fixture("m36_cora_factorized_metadata.tab"),
            "pcrelate_components": self.fixture("m36_cora_factorized_components.tab"),
            "asibd_manifest": self.fixture("m36_cora_factorized_asibd_manifest.tab"),
            "asibd_segments": [self.fixture("m36_cora_factorized_anc1.gapfilled_ibd")],
            "feature_chrom": "chr22", "zero_negative_ratio": 1, "seed": 1701, "outdir": outdir,
        })())

    def test_factorized_output_orients_minor_and_keeps_missing_sparse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            receipt = self.run_fixture(Path(tmpdir))
            with (Path(tmpdir) / "m36_cora_loci.tsv").open() as handle:
                loci = list(csv.DictReader(handle, delimiter="\t"))
            with (Path(tmpdir) / "m36_cora_carriers.tsv").open() as handle:
                carriers = list(csv.DictReader(handle, delimiter="\t"))
            with (Path(tmpdir) / "m36_cora_missing.tsv").open() as handle:
                missing = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(loci), 2)
            self.assertEqual(len(carriers), 4)  # S1/S2 at AC2; S3/S4 after REF-minor orientation.
            self.assertEqual(missing, [{"sample_id": "S4", "event_id": loci[0]["event_id"]}])
            second = loci[1]
            self.assertEqual(second["minor_allele"], "C")
            self.assertEqual(loci[0]["callability"], "0.75")
            self.assertEqual(receipt["feature_schema"], "m36_factorized_sparse_v1")
            self.assertIn("not orthogonal truth", receipt["target_interpretation"])

    def test_targets_have_log1p_positive_and_stratified_zero_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_fixture(Path(tmpdir))
            with (Path(tmpdir) / "m36_cora_external_targets.tsv").open() as handle:
                targets = list(csv.DictReader(handle, delimiter="\t"))
            positives = [row for row in targets if row["target_positive"] == "1"]
            negatives = [row for row in targets if row["target_positive"] == "0"]
            self.assertEqual(len(positives), len(negatives))
            self.assertTrue(all(float(row["target"]) > 0 and float(row["target_cm"]) > 0 for row in positives))
            self.assertTrue(all(row["target"] == "0" and row["target_cm"] == "0" for row in negatives))
            self.assertEqual({row["target_chrom"] for row in targets}, {"outside_chr22_total"})
            self.assertEqual(len({tuple(sorted((row["sample_i"], row["sample_j"]))) for row in targets}), len(targets))

    def test_within_component_zero_saturation_is_recorded_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.tsv"
            segments = root / "anc1.ibd"
            manifest.write_text("gnomix_ancestry\tsegment_file\nAFR\tanc1.ibd\n")
            segments.write_text(
                "S1 1 S2 1 21 1 10 1.0\n"
                "S1 1 S3 1 21 20 30 1.0\n"
            )
            rows = MATERIALIZE.materialize_targets(
                manifest, [segments], ["S1", "S2", "S3"],
                {"S1": "S1", "S2": "S2", "S3": "S3"},
                [{"sample_id": sample, "pcrelate_component": "C1"} for sample in ("S1", "S2", "S3")],
                "chr22", 2, 1701, 10,
            )
            summary = MATERIALIZE.target_balance(rows, 2)["within_component"]
            self.assertEqual(summary["positive"], 2)
            self.assertEqual(summary["zero"], 1)
            self.assertTrue(summary["zero_universe_saturated"])

    def test_missing_optional_context_is_masked_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = type("Args", (), {
                "rare_vcf": self.fixture("m36_cora_factorized_rare.vcf"),
                "locus_metadata": None, "genetic_map": self.fixture("m36_cora_factorized_map.tab"),
                "sample_metadata": self.fixture("m36_cora_factorized_metadata.tab"),
                "pcrelate_components": self.fixture("m36_cora_factorized_components.tab"),
                "asibd_manifest": self.fixture("m36_cora_factorized_asibd_manifest.tab"),
                "asibd_segments": [self.fixture("m36_cora_factorized_anc1.gapfilled_ibd")],
                "feature_chrom": "chr22", "zero_negative_ratio": 1, "seed": 1701, "outdir": Path(tmpdir),
            })()
            MATERIALIZE.run(args)
            with (Path(tmpdir) / "m36_cora_loci.tsv").open() as handle:
                loci = list(csv.DictReader(handle, delimiter="\t"))
            self.assertTrue(all(row["mutation_context"] == "UNAVAILABLE" for row in loci))
            self.assertTrue(all(row["mutation_context_available"] == "0" for row in loci))
            self.assertTrue(all(row["common_copying_context_available"] == "0" for row in loci))

    def test_subsets_2723_style_vcf_universe_before_mac_and_reads_headerless_map(self) -> None:
        """The M20/master cohort is a strict 2619 subset of the VCF columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "subset.vcf").write_text(
                "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tOUT\n"
                "22\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t0/1\t0/0\t./.\n"
            )
            (root / "map.tab").write_text("22\t1\t0.0\n22\t200\t2.0\n")
            (root / "metadata.tab").write_text(
                "sample_id\tcohort\trare_callability\tQ_AFR\tQ_EUR\tQ_NAM\tQ_EAS\n"
                "S1\tC\t1\t0.2\t0.5\t0.3\t0\nS2\tC\t1\t0.2\t0.5\t0.3\t0\nS3\tC\t1\t0.2\t0.5\t0.3\t0\n"
            )
            locus_meta = MATERIALIZE.load_optional_locus_metadata(None, "chr22")
            points = MATERIALIZE.load_genetic_map(root / "map.tab", "chr22")
            self.assertAlmostEqual(MATERIALIZE.interpolate_cm(points, 1), 0.0)
            self.assertAlmostEqual(MATERIALIZE.interpolate_cm(points, 100), 99 * 2.0 / 199.0)
            self.assertAlmostEqual(MATERIALIZE.interpolate_cm(points, 200), 2.0)
            samples, loci, carriers, missing = MATERIALIZE.factorize_vcf(
                root / "subset.vcf", locus_meta, points, "chr22", ["S1", "S2", "S3"]
            )
            self.assertEqual(samples, ["S1", "S2", "S3"])
            self.assertEqual(len(loci), 1)
            self.assertEqual(loci[0]["an_called"], 6)
            self.assertEqual(loci[0]["mac"], 2)
            self.assertEqual({row["sample_id"] for row in carriers}, {"S1", "S2"})
            self.assertEqual(missing, [])

    def test_excludes_loci_outside_observed_map_support_without_extrapolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "outside.vcf").write_text(
                "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
                "22\t50\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t0/1\n"
                "22\t100\t.\tC\tT\t.\tPASS\t.\tGT\t0/1\t0/1\n"
            )
            (root / "map.tab").write_text("22\t100\t0.0\n22\t200\t2.0\n")
            samples, loci, _, _ = MATERIALIZE.factorize_vcf(
                root / "outside.vcf", {}, MATERIALIZE.load_genetic_map(root / "map.tab", "chr22"), "chr22", ["S1", "S2"]
            )
            self.assertEqual(samples, ["S1", "S2"])
            self.assertEqual([row["position"] for row in loci], ["100"])


    def test_receipt_is_hash_bound_and_chainable_before_publication_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.run_fixture(Path(tmpdir))
            receipt = json.loads((Path(tmpdir) / "m36_cora_materialization_receipt.json").read_text())
            self.assertEqual(receipt["status"], "MATERIALIZED_PASS")
            self.assertTrue(all(value["generation"] == "LOCAL_CHAIN" for value in receipt["input_descriptors"].values()))

    def test_canonical_adapter_derives_callability_and_pcrelate_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metadata.tsv").write_text("ID\tCohort\nS1\tC\nS2\tC\n")
            (root / "m20.tsv").write_text(
                "sample_id\tcohort\tQ_AFR\tQ_EUR\tQ_NAM\tQ_EAS\trare_gt_nonmissing_sites\trare_missing_sites\n"
                "S1\tC\t0.2\t0.5\t0.3\t0\t90\t10\nS2\tC\t0.2\t0.5\t0.3\t0\t100\t0\n"
            )
            (root / "master.tsv").write_text("sample_id\tkinship_group_id_phi0442\nS1\tG1\nS2\tG2\n")
            result = subprocess.run([
                sys.executable, str(ROOT / "bin/m36_cora_canonical_adapter.py"), "--metadata", str(root / "metadata.tsv"),
                "--m20-feature-store", str(root / "m20.tsv"), "--modeling-master", str(root / "master.tsv"), "--outdir", str(root / "out"),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "out/m36_cora_sample_metadata.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rows[0]["rare_callability"], "0.9")
            self.assertEqual(rows[0]["asibd_id"], "C_S1")


if __name__ == "__main__":
    unittest.main()
