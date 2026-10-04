"""Import-only provenance tests use synthetic bundles; never real evaluation."""
import copy
import json
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bin"))
sys.path.insert(0, str(ROOT/"tests"))
import r02_biological_import as importer
import test_r02_biological_evaluation as fixtures


class BiologicalImportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BiologicalEvaluationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        t = self.fixture
        self.study = t.study_fixture()
        self.manifests = [t.other_chromosome(), t.manifest]
        self.contract = t.root/"import.json"
        self.record = dict(schema=importer.SCHEMA, chromosomes=["21", "22"],
            expected_samples=4, expected_source_samples=5, expected_configurations=3,
            genome_build="GRCh38", sample_ids_sha256=importer.sha256(t.samples),
            source_cohort_sha256="a"*64, kinship_sha256=importer.sha256(t.kin),
            study=dict(path=str(self.study), sha256=importer.sha256(self.study)),
            segments=[dict(chrom=c, path=str(p), sha256=importer.sha256(p))
                      for c, p in zip(("21", "22"), self.manifests)])
        self.write_contract()

    def write_contract(self):
        self.contract.write_text(json.dumps(self.record))

    def args(self):
        t = self.fixture
        return dict(import_contract=self.contract, expected_import_sha256=importer.sha256(self.contract),
            segment_manifests=self.manifests, study_contract=self.study, sample_ids=t.samples,
            expected_samples=4, expected_source_samples=5, expected_configurations=3,
            pcrelate_file=t.kin, kinship_sha256=importer.sha256(t.kin), genome_build="GRCh38",
            scope=["21", "22"], thresholds=(.0221, .0442), edge_thresholds_bp=(0, 25))

    def validate(self, **changes):
        args = self.args()
        args.update(changes)
        return importer.validate(**args)

    def test_existing_bundles_metadata_and_byte_inventory_only(self):
        result = self.validate()
        self.assertEqual(result["chromosomes"], ["21", "22"])
        self.assertEqual(result["status"], "PREFLIGHT_METADATA_VERIFIED")
        self.assertEqual(len(result["input_inventory"]), 8)
        self.assertEqual(result["segment_table_bytes"], sum(v["bytes"] for v in result["input_inventory"]))
        self.assertFalse((self.fixture.root/"result").exists())
        self.assertIn("no biological validation", result["interpretation"])

    def test_single_chromosome_scope_is_explicit_not_autosomal(self):
        self.record["chromosomes"] = ["22"]
        self.record["segments"] = self.record["segments"][1:]
        self.write_contract()
        result = self.validate(segment_manifests=[self.fixture.manifest], scope=["22"])
        self.assertEqual(result["chromosomes"], ["22"])

    def test_changed_import_contract_is_rejected(self):
        args = self.args()
        self.record["expected_samples"] = 3
        self.write_contract()
        with self.assertRaisesRegex(ValueError, "Import contract SHA256"):
            importer.validate(**args)

    def test_duplicate_json_keys_are_rejected(self):
        self.contract.write_text(self.contract.read_text().replace('"expected_samples": 4',
            '"expected_samples": 3, "expected_samples": 4'))
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.validate()

    def test_bad_or_missing_manifests_fail_closed(self):
        for field in ("study", "segments"):
            with self.subTest(field=field):
                prior = copy.deepcopy(self.record)
                target = self.record[field] if field == "study" else self.record[field][0]
                target["sha256"] = "0"*64
                self.write_contract()
                with self.assertRaisesRegex(ValueError, "[Mm]anifest SHA256|Study contract SHA256"):
                    self.validate()
                self.record = prior
                self.write_contract()

    def test_missing_duplicate_or_out_of_scope_chromosomes(self):
        cases = [["22"], ["21", "21"], ["21", "23"], [21, "22"], []]
        for scope in cases:
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                self.validate(scope=scope)
        with self.assertRaisesRegex(ValueError, "Staged manifest count"):
            self.validate(segment_manifests=self.manifests[:1])
        with self.assertRaisesRegex(ValueError, "duplicate staged chromosome"):
            self.validate(segment_manifests=[self.manifests[0]]*2)

    def test_wrong_cohort_build_count_configuration_and_kinship_expectations(self):
        for name, value in (("expected_samples", 3), ("expected_source_samples", 4),
                            ("expected_configurations", 2), ("genome_build", "GRCh37"),
                            ("kinship_sha256", "0"*64)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(**{name: value})

    def test_manifest_source_minor_build_and_analytical_order(self):
        original = copy.deepcopy(self.fixture.record)
        cases = [dict(status="IN_PROGRESS"), dict(genome_build="GRCh37"),
                 dict(sample_ids_sha256="0"*64), dict(n_samples=5),
                 dict(source_rare_contract=dict(original["source_rare_contract"], dnabr_rare_cohort_n_samples=4)),
                 dict(source_rare_contract=dict(original["source_rare_contract"], dnabr_rare_cohort_sha256="0"*64)),
                 dict(source_rare_contract=dict(original["source_rare_contract"], dnabr_rare_contract="alt_v1"))]
        for change in cases:
            with self.subTest(change=change):
                self.fixture.record = dict(original, **change)
                self.fixture.write_manifest()
                self.record["segments"][1]["sha256"] = importer.sha256(self.fixture.manifest)
                self.write_contract()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_missing_staged_tables(self):
        path = self.manifests[0].parent/importer.evaluation.FILES[1]
        path.rename(path.with_suffix(".saved"))
        with self.assertRaisesRegex(ValueError, "Missing staged"):
            self.validate()

    def test_configuration_bytes_cannot_change(self):
        path = self.manifests[0].parent/importer.evaluation.FILES[2]
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "Configuration ledger SHA256"):
            self.validate()

    def test_full_large_table_hashes_remain_evaluator_responsibility(self):
        path = self.manifests[0].parent/importer.evaluation.FILES[0]
        path.write_bytes(path.read_bytes()+b"changed")
        receipt = self.validate()
        self.assertIn("Delegated", receipt["table_validation"])
        args = self.fixture.args()
        args["segment_manifests"] = self.manifests
        with self.assertRaisesRegex(ValueError, "output SHA256 mismatch"):
            importer.evaluation.run(**args)

    def test_pcrelate_and_study_people_mutations_are_rejected(self):
        t = self.fixture
        t.kin.write_text(t.kin.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "expectation mismatch"):
            self.validate()
        t.kin.write_text("ID1\tID2\tkin\nB\tA\t0.05\nA\tC\tNA\nD\tC\t0.01\n")
        people = self.study.parent/"persons.private.tsv"
        people.write_text(people.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validate()

    def test_paths_reject_shell_globs_relative_and_traversal(self):
        for path in ("manifest.json", "/x/../manifest.json", "/tmp/*/manifest.json", "/tmp/x;id/manifest.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                importer.manifest_entry(dict(path=path, sha256="a"*64), "manifest.json")

    def test_invalid_thresholds_and_missing_baseline_zero(self):
        for changes in (dict(thresholds=(float("nan"),)), dict(thresholds=(.0221, .0221)),
                        dict(edge_thresholds_bp=(25,)), dict(edge_thresholds_bp=(0, 0))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(**changes)

    def test_different_staged_basenames_preserve_authenticated_identity(self):
        staged = self.fixture.root/"staged"
        staged.mkdir()
        paths = []
        for i, path in enumerate(self.manifests):
            dest = staged/f"segment{i+1:02d}"
            shutil.copytree(path.parent, dest)
            paths.append(dest/"manifest.json")
        result = self.validate(segment_manifests=paths)
        self.assertEqual(result["chromosomes"], ["21", "22"])


if __name__ == "__main__":
    unittest.main()
