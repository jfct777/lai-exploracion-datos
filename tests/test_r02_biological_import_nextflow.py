"""Opt-in import-only Nextflow, deep-cache and rejection tests; synthetic only."""
import csv
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


@unittest.skipUnless(os.environ.get("RUN_R02_BIOLOGICAL_NEXTFLOW") == "1", "opt-in synthetic Nextflow integration")
class BiologicalImportNextflowTests(unittest.TestCase):
    def test_only_evaluator_runs_resume_and_byte_change_invalidates_cache(self):
        base = Path(tempfile.mkdtemp(prefix="r02-biological-import-nextflow-", dir=ROOT/".claude/runs"))
        frozen = base/"frozen_bin"
        shutil.copytree(ROOT/"bin", frozen, ignore=shutil.ignore_patterns("__pycache__"))
        prepare = r'''
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'tests'))
from test_r02_biological_import import BiologicalImportTests
t = BiologicalImportTests()
t.setUp()
destination = Path(sys.argv[1])
shutil.copytree(t.fixture.root, destination)
record = t.record
record['study']['path'] = str(destination/'study/study_contract.json')
for chrom, entry in zip(('21','22'), record['segments']):
    folder = destination/('campaign'+chrom)
    folder.mkdir()
    shutil.move(str(destination/('chr'+chrom)), folder/'segment_evidence')
    entry['path'] = str(folder/'segment_evidence/manifest.json')
(destination/'import.json').write_text(json.dumps(record))
'''
        prep = subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OPENBLAS_NUM_THREADS=1", "-v", f"{ROOT}:{ROOT}",
            "-w", str(ROOT), IMAGE, "python3", "-c", prepare, str(base/"input")],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(prep.returncode, 0, prep.stdout+prep.stderr)
        inputs = base/"input"
        contract = inputs/"import.json"
        params = base/"params.json"
        parameters = dict(source_bin=str(frozen), sample_ids=str(inputs/"samples.txt"),
            pcrelate_file=str(inputs/"kin.tsv"), kinship_sha256=hashlib.sha256((inputs/"kin.tsv").read_bytes()).hexdigest(),
            import_contract=str(contract), import_contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
            chromosomes="21,22", expected_samples=4, expected_source_samples=5, expected_configurations=3,
            genome_build="GRCh38", min_free_gib=0, max_database_mb=16, min_free_disk_mb=0,
            kinship_thresholds="0.0221,0.0442", edge_thresholds_bp="0,25")
        params.write_text(json.dumps(parameters))
        config = base/"resources.config"
        config.write_text(f"""
process.executor = 'local'
process.container = '{IMAGE}'
process.memory = '2 GB'
process.time = '3m'
process.maxForks = 1
process.errorStrategy = 'terminate'
executor.queueSize = 1
docker.enabled = true
docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()} -v {ROOT}:{ROOT}'
env.PYTHONDONTWRITEBYTECODE = '1'
env.OMP_NUM_THREADS = '1'
env.OPENBLAS_NUM_THREADS = '1'
env.MKL_NUM_THREADS = '1'
""")
        common = ["nextflow", "-C", str(config), "run", str(ROOT/"workflows/r02_biological_import.nf"),
                  "-params-file", str(params), "-work-dir", str(base/"work"), "-ansi-log", "false"]
        env = {**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true",
               "NXF_SYNTAX_PARSER": "v1", "NXF_OPTS": "-Xms64m -Xmx512m"}
        print(f"Synthetic import-only Nextflow evidence: {base}", flush=True)

        def run(index, expected_status, success=True):
            result = subprocess.run([*common, "-with-trace", str(base/f"trace{index}.tsv"),
                                    *(["-resume"] if index else [])], cwd=base, env=env,
                                    capture_output=True, text=True, timeout=180)
            (base/f"captured{index}.log").write_text(result.stdout+result.stderr)
            self.assertEqual(result.returncode == 0, success, result.stdout+result.stderr)
            with (base/f"trace{index}.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1, rows)
            self.assertTrue(rows[0]["name"].startswith("R02_BIOLOGICAL_EVALUATION"), rows)
            self.assertEqual(rows[0]["status"], expected_status, rows)
            return result

        run(0, "COMPLETED")
        run(1, "CACHED")
        outputs = list((base/"work").glob("*/*/evaluation/manifest.json"))
        self.assertEqual(len(outputs), 1)
        manifest = json.loads(outputs[0].read_text())
        self.assertEqual(manifest["chromosomes"], ["21", "22"])
        self.assertEqual(manifest["n_configurations"], 3)
        self.assertTrue(manifest["no_winner_selected"])
        receipts = list((base/"work").glob("*/*/import_receipt.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(json.loads(receipts[0].read_text())["chromosomes"], ["21", "22"])
        script = (outputs[0].parent.parent/".command.sh").read_text()
        for forbidden in ("r02_study_design.py", "r02_segment_evidence.py", "m165_chr22_sweep.py"):
            self.assertNotIn(forbidden, script)
        # Same path and size, restored timestamp: deep content hashing must
        # invalidate the task even when normal metadata-based caching would not.
        target = inputs/"campaign21/segment_evidence/segment_evidence.tsv.gz"
        original = target.read_bytes()
        original_stat = target.stat()
        target.write_bytes(bytes([original[0]^1])+original[1:])
        os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        failed = run(2, "FAILED", success=False)
        self.assertIn("output SHA256 mismatch", failed.stdout+failed.stderr)
        target.write_bytes(original)
        run(3, "CACHED")
        # An effective calculation parameter also invalidates the cache.
        parameters["edge_thresholds_bp"] = "0,26"
        params.write_text(json.dumps(parameters))
        run(4, "COMPLETED")
        manifests = [json.loads(p.read_text()) for p in (base/"work").glob("*/*/evaluation/manifest.json")]
        self.assertEqual(len(manifests), 2)
        self.assertEqual(sorted(m["parameters"]["edge_thresholds_bp"] for m in manifests), [[0, 25], [0, 26]])
        helper = frozen/"r02_biological_import.py"
        before = helper.stat()
        helper.write_text(helper.read_text().replace("Metadata preflight", "Metadata Preflight", 1))
        self.assertEqual(before.st_size, helper.stat().st_size)
        os.utime(helper, ns=(before.st_atime_ns, before.st_mtime_ns))
        run(5, "COMPLETED")
        self.assertEqual(len(list((base/"work").glob("*/*/evaluation/manifest.json"))), 3)


if __name__ == "__main__":
    unittest.main()
