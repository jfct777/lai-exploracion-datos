"""Cohort/metadata audit tests; not biological validation."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import r02_study_design as study


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = ["a", "b", "c", "d"]
        self.ids = self.root / "samples.ids"
        self.ids.write_text("\n".join(self.samples) + "\n")
        self.metadata = self.root / "metadata.tsv"
        self.metadata.write_text("ID\tCohort\tRegion\na\tA\tNORTH\nb\tA\tNORTH\nc\tB\tSOUTH\nd\tB\tUNKNOWN\na\tA\tNORTH\n")
        self.kin = self.root / "kin.tsv"
        self.kin.write_text("ID1\tID2\tkin\na\tb\t0.1\na\tc\t0\na\td\t0\nb\tc\t0.03\nb\td\t0\nc\td\tNA\n")

    def args(self):
        return argparse.Namespace(sample_ids=self.ids, expected_samples=4, max_matrix_mb=10,
                                  metadata=self.metadata, expected_metadata_sha256=study.evidence.sha256(self.metadata),
                                  id_column="ID", cohort_column="Cohort", region_column="Region", metadata_column=["Batch"],
                                  pcrelate_file=self.kin, expected_pcrelate_sha256=study.evidence.sha256(self.kin),
                                  phi=[.0221, .0442], roles=None, output_dir=self.root / "out")

    def test_audit_exact_duplicates_missingness_and_components(self):
        result = study.run(self.args())
        self.assertEqual(result["metadata_audit"]["exact_duplicate_rows"], 1)
        self.assertEqual(result["kinship_audit"]["n_retained_missing_nonfinite"], 1)
        self.assertEqual(result["training_status"], "NOT_AUTHORIZED")
        with (self.root / "out/components_summary.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual([r["n_components"] for r in rows], ["2", "3"])
        self.assertFalse(result["roles_supplied"])
        self.assertIn("COLUMN_ABSENT", (self.root / "out/metadata_coverage.tsv").read_text())
        for name, digest in result["outputs_sha256"].items():
            self.assertEqual(digest, study.evidence.sha256(self.root / "out" / name))

    def test_conflicting_duplicate_fails(self):
        with self.metadata.open("a") as h:
            h.write("a\tA\tSOUTH\n")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            study.run(self.args())

    def test_duplicate_kinship_pair_fails(self):
        with self.kin.open("a") as h:
            h.write("b\ta\t0.1\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            study.run(self.args())

    def test_role_crossing_is_explicit_fail_not_new_roles(self):
        args = self.args()
        args.roles = self.root / "roles.tsv"
        args.roles.write_text("sample_id\trole\na\tFIT\nb\tEVALUATE\nc\tSELECT\nd\tEXCLUDE\n")
        result = study.run(args)
        self.assertTrue(any(c["status"] == "FAIL" for c in result["checks"]))
        self.assertEqual(result["training_status"], "NOT_AUTHORIZED")
        self.assertIn("a\tFIT", (args.output_dir / "persons.private.tsv").read_text())

    def test_absent_pair_not_treated_as_unrelated(self):
        self.kin.write_text("ID1\tID2\tkin\na\tb\t0.1\n")
        result = study.run(self.args())
        self.assertEqual(result["kinship_audit"]["n_retained_missing_absent"], 5)
        self.assertTrue(any(c["check"] == "pairwise_kinship_coverage" and c["status"] == "INCOMPLETE" for c in result["checks"]))

    def test_hash_mismatch_fails(self):
        args = self.args()
        args.expected_metadata_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            study.run(args)

    def test_output_never_overwritten(self):
        args = self.args()
        study.run(args)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            study.run(args)

    def test_budget_guard_and_invalid_phi(self):
        args = self.args()
        args.max_matrix_mb = 0
        with self.assertRaises(ValueError):
            study.run(args)
        args = self.args()
        args.phi = [float("nan")]
        with self.assertRaisesRegex(ValueError, "phi"):
            study.run(args)

    def test_roles_require_all_people(self):
        args = self.args()
        args.roles = self.root / "roles.tsv"
        args.roles.write_text("sample_id\trole\na\tFIT\n")
        with self.assertRaisesRegex(ValueError, "exactly"):
            study.run(args)

    def test_samples_changing_after_read_never_receive_completion(self):
        original = study.read_roles
        def mutate(path, samples):
            result = original(path, samples)
            self.ids.write_text("d\nc\nb\na\n")
            return result
        with mock.patch.object(study, "read_roles", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "Sample IDs changed during audit"):
                study.run(self.args())
        self.assertFalse((self.root/"out").exists())

    def test_roles_changing_after_read_never_receive_completion(self):
        args = self.args()
        args.roles = self.root/"roles.tsv"
        args.roles.write_text("sample_id\trole\na\tUNASSIGNED\nb\tUNASSIGNED\nc\tUNASSIGNED\nd\tUNASSIGNED\n")
        original = study.read_roles
        def mutate(path, samples):
            result = original(path, samples)
            path.write_text(path.read_text().replace("UNASSIGNED", "FIT"))
            return result
        with mock.patch.object(study, "read_roles", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "Roles changed during audit"):
                study.run(args)
        self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
