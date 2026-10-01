"""Opt-in real Nextflow/container execution contract; no genomic data.

RUN_R02_ANALYSIS_NEXTFLOW=1 python3 -m unittest discover -s tests \
  -p test_r02_analysis_task_nextflow.py -v

Uses the actual allowed pair-evidence tool's --help entry point and a frozen
source copy. Small logs and receipts remain under .claude/runs for inspection.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sha256:76a0fddd300d3634363b5a13a1d28983122644a42e7d6c18339f9c8ac08548a7"


@unittest.skipUnless(os.environ.get("RUN_R02_ANALYSIS_NEXTFLOW") == "1" and shutil.which("nextflow"),
                     "opt-in real Nextflow/container smoke without genomic data")
class AnalysisTaskNextflowTests(unittest.TestCase):
    def execute(self, wrong_hash=False):
        base = Path(tempfile.mkdtemp(prefix="r02-analysis-task-test-", dir=ROOT / ".claude/runs"))
        source = base / "frozen_bin"
        source.mkdir()
        tool = source / "r02_genomic_pair_evidence.py"
        shutil.copy2(ROOT / "bin" / tool.name, tool)
        executor = source / "r02_exec_task.py"
        shutil.copy2(ROOT / "bin" / executor.name, executor)
        command = ["python3", str(tool), "--help"]
        digest = hashlib.sha256(tool.read_bytes()).hexdigest()
        payload = dict(command=command, tool_sha256="0" * 64 if wrong_hash else digest)
        command_file = base / "command.json"
        command_file.write_text(json.dumps(payload))
        config = base / "resources.config"
        config.write_text(f"""
process.executor = 'local'
process.container = '{IMAGE}'
process.cpus = 1
process.memory = '2 GB'
process.time = '3m'
process.maxForks = 1
process.errorStrategy = 'terminate'
executor.queueSize = 1
docker.enabled = true
docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()} -v {ROOT}:{ROOT}'
env.OMP_NUM_THREADS = '1'
env.OPENBLAS_NUM_THREADS = '1'
params.r02_stage = 'synthetic_help_no_genotypes'
params.r02_command_file = '{command_file}'
params.r02_exec_tool = '{executor}'
params.r02_source_bin = '{source}'
""")
        workflow = ROOT / "workflows/r02_analysis_task.nf"
        result = subprocess.run([
            "nextflow", "-log", str(base / "nextflow.log"), "-C", str(config),
            "run", str(workflow), "-work-dir", str(base / "work"),
            "-ansi-log", "false", "-with-trace", str(base / "trace.tsv")],
            cwd=base, capture_output=True, text=True, timeout=180,
            env={**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true",
                 "NXF_SYNTAX_PARSER": "v1", "NXF_OPTS": "-Xms64m -Xmx512m"})
        (base / "captured.log").write_text(result.stdout + result.stderr)
        print(f"Synthetic Nextflow task evidence: {base}", flush=True)
        self.assertEqual(hashlib.sha256(tool.read_bytes()).hexdigest(), digest)
        return base, command_file, payload, result

    def test_absolute_frozen_tool_executes_in_real_container_and_records_hashes(self):
        base, command_file, payload, result = self.execute()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipts = list((base / "work").glob("*/*/task_receipt.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text())
        self.assertEqual(receipt["command"], payload["command"])
        self.assertEqual(receipt["tool_sha256"], payload["tool_sha256"])
        self.assertEqual(receipt["command_sha256"], hashlib.sha256(command_file.read_bytes()).hexdigest())
        self.assertEqual(receipt["status"], "COMMAND_COMPLETED_OUTPUT_VALIDATION_BY_MODULE_AND_SUPERVISOR")
        work = receipts[0].parent
        self.assertIn("usage:", (work / ".command.out").read_text())
        self.assertIn(IMAGE, (work / ".command.run").read_text())
        self.assertIn("--network none", (work / ".command.run").read_text())
        self.assertEqual((work / ".exitcode").read_text().strip(), "0")

    def test_hash_mismatch_fails_without_success_receipt(self):
        base, _, _, result = self.execute(wrong_hash=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Scientific tool differs from frozen command", result.stdout + result.stderr)
        self.assertFalse(list((base / "work").glob("*/*/task_receipt.json")))


if __name__ == "__main__":
    unittest.main()
