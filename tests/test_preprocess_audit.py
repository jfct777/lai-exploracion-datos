"""Small provenance fixtures: no real genotypes, Nextflow jobs or cloud calls."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("preprocess_audit", ROOT / "bin/preprocess_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
RUNNER_SPEC = importlib.util.spec_from_file_location("audit_test_runner", ROOT / "bin/r02_autosome_pipeline.py")
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


class PreprocessAuditTests(unittest.TestCase):
    def fixture(self, folder):
        rows = ["task_id\thash\tname\tstatus\texit"]
        tasks = []
        for index, (process, suffixes) in enumerate(audit.TASK_FILES.items(), 1):
            prefix, suffix = f"{index:02x}", str(index) * 30
            task = folder / "work" / prefix / suffix
            task.mkdir(parents=True)
            (task / ".exitcode").write_text("0\n")
            for extension in suffixes:
                (task / f"dnabr.hg38.2723.chr21.{extension}").write_text("small evidence " + extension)
            (task / "must-not-copy.bcf").write_text("synthetic genotype sentinel")
            rows.append(f"{index}\t{prefix}/{suffix[:6]}\tPREPROCESS_NORM_LEFTALIGN_CHECKPOINTED:{process} (chr21)\tCOMPLETED\t0")
            tasks.append(task)
        (folder / "trace.tsv").write_text("\n".join(rows) + "\n")
        return tasks

    def test_collects_four_files_into_publishable_folder_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            report = audit.collect_preprocess_audit(folder, 21)
            self.assertEqual(report["files"], 4)
            self.assertEqual(audit.collect_preprocess_audit(folder, 21), report)
            manifest = json.loads(Path(report["manifest"]).read_text())
            published = runner.publication_files(folder)
            self.assertTrue(all(folder / "preprocess/audit" / f["relative_path"] in published for f in manifest["files"]))
            self.assertTrue(Path(report["manifest"]) in published)
            self.assertFalse(any(p.suffix == ".bcf" or "work" in p.relative_to(folder).parts for p in published))
            for entry in manifest["files"]:
                self.assertEqual(audit.digest(Path(entry["source"]).read_bytes()), entry["sha256"])

    def test_cached_observation_preserves_first_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            original = audit.collect_preprocess_audit(folder, 21)
            trace = folder / "trace.tsv"
            trace.write_text(trace.read_text().replace("COMPLETED", "CACHED"))
            cached = audit.collect_preprocess_audit(folder, 21)
            self.assertNotEqual(original["manifest"], cached["manifest"])
            self.assertTrue(Path(original["manifest"]).exists())
            self.assertEqual(len(list((folder / "preprocess/audit").glob("*/*.log"))), 2)

    def test_missing_or_failed_task_is_rejected_before_copy(self):
        for mode in ("missing", "failed", "badexit"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                tasks = self.fixture(folder)
                if mode == "missing":
                    (tasks[0] / "dnabr.hg38.2723.chr21.annotation.log").unlink()
                elif mode == "failed":
                    trace = folder / "trace.tsv"
                    trace.write_text(trace.read_text().replace("COMPLETED", "FAILED", 1))
                else:
                    (tasks[0] / ".exitcode").write_text("1\n")
                with self.assertRaises(ValueError):
                    audit.collect_preprocess_audit(folder, 21)
                self.assertFalse((folder / "preprocess/audit").exists())

    def test_ambiguous_work_hash_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            tasks = self.fixture(folder)
            (tasks[0].parent / (tasks[0].name[:6] + "f" * 24)).mkdir()
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                audit.collect_preprocess_audit(folder, 21)

    def test_changed_evidence_and_symlink_destinations_are_not_overwritten(self):
        for mode in ("changed", "symlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                tasks = self.fixture(folder)
                report = audit.collect_preprocess_audit(folder, 21)
                manifest = json.loads(Path(report["manifest"]).read_text())
                target = folder / "preprocess/audit" / manifest["files"][0]["relative_path"]
                original = target.read_bytes()
                if mode == "changed":
                    (tasks[0] / "dnabr.hg38.2723.chr21.annotation.log").write_text("changed")
                else:
                    target.unlink()
                    target.symlink_to(tasks[0] / "dnabr.hg38.2723.chr21.annotation.log")
                with self.assertRaises(ValueError):
                    audit.collect_preprocess_audit(folder, 21)
                self.assertEqual(target.read_bytes(), original)

    def test_cli_is_read_only_outside_audit_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            tasks = self.fixture(folder)
            trace_bytes = (folder / "trace.tsv").read_bytes()
            result = subprocess.run([sys.executable, str(ROOT / "bin/preprocess_audit.py"),
                                     "--folder", str(folder), "--chrom", "21"],
                                    check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(json.loads(result.stdout)["files"], 4)
            self.assertEqual((folder / "trace.tsv").read_bytes(), trace_bytes)
            self.assertTrue((tasks[0] / "must-not-copy.bcf").exists())


if __name__ == "__main__":
    unittest.main()
