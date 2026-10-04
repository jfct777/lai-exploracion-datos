"""Synthetic-only equivalence and adversarial persistent checkpoint contracts."""
import gzip
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r02_biological_evaluation import BiologicalEvaluationTests
import r02_biological_checkpoint as checkpoint
import r02_biological_evaluation as evaluation


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = BiologicalEvaluationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def prepare(self, source=None, name="checkpoint22"):
        f = self.fixture
        source = source or f.manifest
        output = self.root/name
        checkpoint.prepare(source, evaluation.sha256(source), f.samples, 4, 3,
                           output, self.root, 16, 0)
        return output/"checkpoint.json"

    def aggregate(self, paths, **changes):
        f = self.fixture
        options = dict(checkpoint_manifests=paths,
            expected_checkpoint_sha256s=[evaluation.sha256(p) for p in paths],
            chromosomes=[json.loads(p.read_text())["chrom"] for p in paths],
            sample_ids=f.samples, expected_samples=4, pcrelate_file=f.kin,
            expected_pcrelate_sha256=evaluation.sha256(f.kin), output_dir=self.root/"combined",
            expected_configurations=3, edge_thresholds_bp=(0, 50),
            scratch_dir=self.root, min_free_disk_mb=0)
        options.update(changes)
        return checkpoint.aggregate(**options)

    def rewrite(self, path, edit):
        record = json.loads(path.read_text())
        edit(record)
        path.write_text(json.dumps(record))

    def test_two_chromosome_scientific_equivalence_and_global_filter(self):
        f = self.fixture
        other = f.other_chromosome()
        study = f.study_fixture()
        direct = f.run_evaluation(segment_manifests=[f.manifest, other],
            study_contract=study, edge_thresholds_bp=(0, 50))
        paths = [self.prepare(), self.prepare(other, "checkpoint21")]
        result = self.aggregate(paths, study_contract=study)
        self.assertEqual(result["outputs_sha256"], direct["outputs_sha256"])
        self.assertEqual(result["chromosomes"], ["21", "22"])
        # Each chromosome contributes 30 bp for this pair, below 50; summed 60 passes.
        rows = list(evaluation.read_table(self.root/"combined/configuration_metrics.tsv", evaluation.METRIC_FIELDS))
        self.assertTrue(any(r["min_edge_bp"] == "50" and r["config_id"] == f.full and
                            r["status"] == "OBSERVED" for r in rows))
        for path in paths:
            self.assertTrue((path.parent/json.loads(path.read_text())["streams"]["identifiers"]).exists())
        reverse = self.aggregate(list(reversed(paths)), study_contract=study, output_dir=self.root/"reverse")
        self.assertEqual(reverse["outputs_sha256"], direct["outputs_sha256"])

    def test_duplicate_global_ids_rejected(self):
        other = self.fixture.other_chromosome(preserve_ids=True)
        paths = [self.prepare(), self.prepare(other, "checkpoint21")]
        with self.assertRaisesRegex(ValueError, "Duplicate chain ID"):
            self.aggregate(paths)
        self.assertFalse((self.root/"combined").exists())

    def test_checkpoint_hash_mismatch(self):
        path = self.prepare()
        with self.assertRaisesRegex(ValueError, "Checkpoint SHA256"):
            self.aggregate([path], expected_checkpoint_sha256s=["a"*64])

    def test_stream_corruption_rejected(self):
        path = self.prepare()
        row = json.loads(path.read_text())
        stream = path.parent/row["streams"]["segments"]
        stream.write_bytes(stream.read_bytes()+b"corruption")
        with self.assertRaisesRegex(ValueError, "stream SHA256/size"):
            self.aggregate([path])

    def test_authenticated_but_unsorted_stream_rejected(self):
        path = self.prepare()
        record = json.loads(path.read_text())
        stream = path.parent/record["streams"]["segments"]
        with gzip.open(stream, "rt") as handle:
            rows = handle.readlines()
        with gzip.open(stream, "wt") as handle:
            handle.writelines([rows[0], *reversed(rows[1:])])
        self.rewrite(path, lambda r: r["outputs"][stream.name].update(
            sha256=evaluation.sha256(stream), bytes=stream.stat().st_size))
        with self.assertRaisesRegex(ValueError, "unsorted"):
            self.aggregate([path])

    def test_wrong_row_width_rejected_even_with_rehashed_stream(self):
        path = self.prepare()
        record = json.loads(path.read_text())
        stream = path.parent/record["streams"]["segments"]
        with gzip.open(stream, "rt") as handle:
            rows = handle.readlines()
        rows[1] = json.dumps(json.loads(rows[1])[:-1])+"\n"
        with gzip.open(stream, "wt") as handle:
            handle.writelines(rows)
        self.rewrite(path, lambda r: r["outputs"][stream.name].update(
            sha256=evaluation.sha256(stream), bytes=stream.stat().st_size))
        with self.assertRaisesRegex(ValueError, "row width"):
            self.aggregate([path])

    def test_code_mismatch_and_incomplete_checkpoint_rejected(self):
        path = self.prepare()
        self.rewrite(path, lambda r: r["code_sha256"].update({"r02_biological_evaluation.py": "a"*64}))
        with self.assertRaisesRegex(ValueError, "source code mismatch"):
            self.aggregate([path])
        self.rewrite(path, lambda r: r.update(status="RUNNING"))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.aggregate([path])

    def test_missing_chromosome_and_cohort_mismatch(self):
        path = self.prepare()
        with self.assertRaisesRegex(ValueError, "Missing checkpoint"):
            self.aggregate([path], chromosomes=["21", "22"])
        self.rewrite(path, lambda r: r.update(sample_ids_sha256="a"*64))
        with self.assertRaisesRegex(ValueError, "sample cohort/order"):
            self.aggregate([path])

    def test_identity_config_and_optional_criteria_mismatch(self):
        other = self.fixture.other_chromosome()
        paths = [self.prepare(), self.prepare(other, "checkpoint21")]
        original = paths[1].read_text()
        for field in ("identity", "configs", "optional_criteria"):
            paths[1].write_text(original)
            def change(record):
                if field == "identity":
                    record["registry"][field]["genome_build"] = "OTHER"
                elif field == "configs":
                    record["registry"][field][self.fixture.full]["min_length_bp"] += 1
                else:
                    record["registry"][field]["common_criteria"] = {"different": True}
            self.rewrite(paths[1], change)
            with self.assertRaisesRegex(ValueError, "Incompatible|Configuration ID"):
                self.aggregate(paths)

    def test_no_overwrite_and_failed_prepare_has_no_complete_marker(self):
        path = self.prepare()
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            self.prepare()
        f = self.fixture
        with self.assertRaises(ValueError):
            checkpoint.prepare(f.manifest, evaluation.sha256(f.manifest), f.samples,
                4, 3, self.root/"failed", self.root, .00001, 0)
        self.assertFalse((self.root/"failed/checkpoint.json").exists())

    def test_nonfinite_values_overlap_and_duplicate_geometry_rejected(self):
        path = self.prepare()
        manifest = path.read_text()
        record = json.loads(manifest)
        stream = path.parent/record["streams"]["segments"]
        original = stream.read_bytes()
        for defect in ("nan", "overlap", "duplicate"):
            path.write_text(manifest)
            stream.write_bytes(original)
            with gzip.open(stream, "rt") as handle:
                header, *rows = map(json.loads, handle)
            if defect == "nan":
                rows[0][header.index("length_cm")] = float("nan")
            elif defect == "duplicate":
                rows[1] = rows[0]
            else:
                rows[1][header.index("start_pos")] = 6
                rows[1][header.index("end_pos")] = 25
            with gzip.open(stream, "wt") as handle:
                for row in (header, *rows):
                    handle.write(json.dumps(row)+"\n")
            self.rewrite(path, lambda r: r["outputs"][stream.name].update(
                sha256=evaluation.sha256(stream), bytes=stream.stat().st_size))
            with self.assertRaisesRegex(ValueError, "Nonfinite|Overlapping|duplicated"):
                self.aggregate([path])

    def test_prepare_rejects_wrong_chromosome_before_output(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, "chromosome/manifest"):
            checkpoint.prepare(f.manifest, evaluation.sha256(f.manifest), f.samples,
                4, 3, self.root/"wrong", self.root, 16, 0, expected_chromosome="21")
        self.assertFalse((self.root/"wrong").exists())

    def test_code_drift_during_aggregate_leaves_no_manifest(self):
        path = self.prepare()
        original = checkpoint.code_hashes()
        changed = dict(original, drift="x")
        state = [False]
        compute = evaluation.aggregate_pairs
        def compute_then_drift(*args, **kwargs):
            result = compute(*args, **kwargs)
            state[0] = True
            return result
        with mock.patch.object(checkpoint, "code_hashes", side_effect=lambda: changed if state[0] else original), \
             mock.patch.object(evaluation, "aggregate_pairs", side_effect=compute_then_drift):
            with self.assertRaisesRegex(ValueError, "Code changed"):
                self.aggregate([path])
        self.assertFalse((self.root/"combined/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
