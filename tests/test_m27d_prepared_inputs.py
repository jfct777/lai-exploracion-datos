import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import verify_m27d_prepared_inputs as verifier  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestM27DPreparedInputs(unittest.TestCase):
    def make_fixture(self, root: Path):
        paths = [
            root / "m27d_official_panel_autosomes.gds",
            root / "m27d_ld_pruned_anchor_snp_ids.rds",
            root / "m27d_ld_pruned_strict_snp_ids.rds",
            root / "m27d_sample_strata.private.tsv",
        ]
        for index, path in enumerate(paths):
            path.write_bytes(f"fixture-{index}\n".encode("utf-8"))
        manifest = root / "m27d_marker_preparation.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "stage": verifier.EXPECTED_STAGE,
                    "params": {
                        "scope": "m27d_marker_preparation",
                        "scientific_result": False,
                        "full_run_authorized": False,
                    },
                    "sha256": {path.name: digest(path) for path in paths},
                }
            ),
            encoding="utf-8",
        )
        return manifest, paths

    def test_verifies_exact_prepared_files_without_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, paths = self.make_fixture(Path(tmp))
            result = verifier.verify_prepared_inputs(manifest, paths, digest(manifest))

            self.assertTrue(result["verified"])
            self.assertFalse(result["sample_ids_emitted"])
            self.assertEqual(set(result["verified_files"]), {p.name for p in paths})

    def test_rejects_a_corrupted_prepared_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, paths = self.make_fixture(Path(tmp))
            paths[0].write_bytes(b"changed\n")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verifier.verify_prepared_inputs(manifest, paths, digest(manifest))

    def test_rejects_a_substituted_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, paths = self.make_fixture(Path(tmp))

            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                verifier.verify_prepared_inputs(manifest, paths, "0" * 64)

    def test_rejects_manifest_that_authorizes_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, paths = self.make_fixture(Path(tmp))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["params"]["full_run_authorized"] = True
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "claims an authorized full run"):
                verifier.verify_prepared_inputs(manifest, paths, digest(manifest))

    def test_accepts_a_manifest_without_the_legacy_run_level_flag(self):
        """Authorization moved to run_provenance.json; a per-stage copy would drift."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest, paths = self.make_fixture(Path(tmp))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["params"].pop("full_run_authorized", None)
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            result = verifier.verify_prepared_inputs(manifest, paths, digest(manifest))
            self.assertTrue(result["verified"])

    def test_rejects_a_manifest_that_claims_to_be_a_scientific_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, paths = self.make_fixture(Path(tmp))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["params"]["scientific_result"] = True
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-scientific"):
                verifier.verify_prepared_inputs(manifest, paths, digest(manifest))


if __name__ == "__main__":
    unittest.main()
