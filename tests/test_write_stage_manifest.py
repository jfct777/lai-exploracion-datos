import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "bin" / "write_stage_manifest.py"
SPEC = importlib.util.spec_from_file_location("write_stage_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WriteStageManifestTest(unittest.TestCase):
    def test_checksums_keep_duplicate_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            historical = root / "historical"
            minor = root / "minor"
            historical.mkdir()
            minor.mkdir()
            historical_file = historical / "summary.json"
            minor_file = minor / "summary.json"
            historical_file.write_text("historical")
            minor_file.write_text("minor")

            checksums = MODULE._checksums_by_unambiguous_name(
                [historical_file, minor_file]
            )

            self.assertEqual(
                set(checksums), {str(historical_file), str(minor_file)}
            )
            self.assertNotEqual(
                checksums[str(historical_file)], checksums[str(minor_file)]
            )

    def test_checksums_preserve_basename_for_unique_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            unique_file = Path(tmp) / "unique.json"
            unique_file.write_text("unique")

            checksums = MODULE._checksums_by_unambiguous_name([unique_file])

            self.assertEqual(set(checksums), {"unique.json"})


if __name__ == "__main__":
    unittest.main()
