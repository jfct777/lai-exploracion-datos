"""Opt-in, tiny Nextflow checkpoint completion/cache/failure regression."""
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


@unittest.skipUnless(os.environ.get("RUN_R02_BIOLOGICAL_NEXTFLOW") == "1", "opt-in synthetic Nextflow")
class CheckpointNextflowTests(unittest.TestCase):
    def test_prepare_resume_failed_aggregate_and_localized_invalidation(self):
        base = Path(tempfile.mkdtemp(prefix="r02-checkpoint-nextflow-", dir=ROOT/".claude/runs"))
        frozen = base/"frozen_bin"
        shutil.copytree(ROOT/"bin", frozen, ignore=shutil.ignore_patterns("__pycache__"))
        prepare = r'''
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'tests'))
from test_r02_biological_evaluation import BiologicalEvaluationTests
t = BiologicalEvaluationTests(); t.setUp(); t.other_chromosome(); t.study_fixture()
shutil.copytree(t.root, Path(sys.argv[1]))
'''
        prep = subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OPENBLAS_NUM_THREADS=1", "-v", f"{ROOT}:{ROOT}",
            "-w", str(ROOT), IMAGE, "python3", "-c", prepare, str(base/"input")],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(prep.returncode, 0, prep.stdout+prep.stderr)
        inputs = base/"input"
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        contract = base/"prepare.json"
        record = dict(schema="r02_biological_checkpoint_prepare_v1", expected_samples=4,
            expected_configurations=3, sample_ids_sha256=sha(inputs/"samples.txt"),
            segments=[dict(chrom=c, path=str(inputs/f"chr{c}/manifest.json"),
                           sha256=sha(inputs/f"chr{c}/manifest.json")) for c in ("21", "22")])
        contract.write_text(json.dumps(record))
        params = base/"prepare.params.json"
        common_params = dict(source_bin=str(frozen), sample_ids=str(inputs/"samples.txt"),
            expected_samples=4, expected_configurations=3, chromosomes="21,22",
            min_free_gib=0, min_free_disk_mb=0)
        params.write_text(json.dumps(dict(common_params, prepare_contract=str(contract),
            prepare_contract_sha256=sha(contract), max_database_mb=16)))
        config = base/"resources.config"
        config.write_text(f"""
process.executor='local'
process.container='{IMAGE}'
process.memory='2 GB'
process.time='3m'
process.maxForks=1
process.errorStrategy='finish'
executor.queueSize=1
docker.enabled=true
docker.runOptions='--network none --user {os.getuid()}:{os.getgid()} -v {ROOT}:{ROOT}'
env.PYTHONDONTWRITEBYTECODE='1'
env.OPENBLAS_NUM_THREADS='1'
env.OMP_NUM_THREADS='1'
env.MKL_NUM_THREADS='1'
""")
        env = {**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true",
               "NXF_SYNTAX_PARSER": "v1", "NXF_OPTS": "-Xms64m -Xmx512m"}
        print(f"Checkpoint Nextflow evidence: {base}", flush=True)

        def run(kind, index, params_file, statuses):
            command = ["nextflow", "-C", str(config), "run",
                str(ROOT/f"workflows/r02_biological_checkpoint_{kind}.nf"),
                "-params-file", str(params_file), "-work-dir", str(base/"work"),
                "-with-trace", str(base/f"{kind}{index}.tsv"), "-ansi-log", "false"]
            # Explicit session IDs keep the two entrypoint caches independent.
            if index:
                command += ["-resume", sessions[kind]]
            result = subprocess.run(command, cwd=base, env=env, capture_output=True, text=True, timeout=180)
            (base/f"{kind}{index}.log").write_text(result.stdout+result.stderr)
            self.assertEqual(result.returncode == 0, "FAILED" not in statuses, result.stdout+result.stderr)
            with (base/f"{kind}{index}.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(sorted(r["status"] for r in rows), sorted(statuses), result.stdout+result.stderr)
            if not index:
                # The previous session identifier is recorded by Nextflow in its history.
                history = (base/".nextflow/history").read_text().strip().splitlines()[-1].split("\t")
                sessions[kind] = next(x for x in history if len(x)==36 and x.count('-')==4)
            return rows

        sessions = {}
        run("prepare", 0, params, ["COMPLETED", "COMPLETED"])
        run("prepare", 1, params, ["CACHED", "CACHED"])
        checkpoints = sorted((base/"work").glob("*/*/checkpoint/checkpoint.json"))
        self.assertEqual(len(checkpoints), 2)
        aggregate_contract = base/"aggregate.json"
        aggregate_record = dict(schema="r02_biological_checkpoint_aggregate_v1",
            expected_samples=4, expected_source_samples=5, expected_configurations=3,
            genome_build="GRCh38", source_cohort_sha256="a"*64,
            kinship_thresholds="0.0221,0.0442", edge_thresholds_bp="1",
            sample_ids_sha256=sha(inputs/"samples.txt"), kinship_sha256=sha(inputs/"kin.tsv"),
            study=dict(path=str(inputs/"study/study_contract.json"), sha256=sha(inputs/"study/study_contract.json")),
            checkpoints=[dict(chrom=json.loads(p.read_text())["chrom"], path=str(p), sha256=sha(p)) for p in checkpoints])
        aggregate_contract.write_text(json.dumps(aggregate_record))
        aggregate_params = base/"aggregate.params.json"
        ap = dict(common_params, expected_source_samples=5, genome_build="GRCh38",
            source_cohort_sha256="a"*64, aggregate_contract=str(aggregate_contract),
            aggregate_contract_sha256=sha(aggregate_contract), pcrelate_file=str(inputs/"kin.tsv"),
            kinship_sha256=sha(inputs/"kin.tsv"), kinship_thresholds="0.0221,0.0442", edge_thresholds_bp="1")
        aggregate_params.write_text(json.dumps(ap))
        run("aggregate", 0, aggregate_params, ["FAILED"])
        # A global failure never consumes/deletes the per-chromosome artifacts.
        self.assertTrue(all(p.exists() for p in checkpoints))
        run("prepare", 2, params, ["CACHED", "CACHED"])
        ap["edge_thresholds_bp"]="0,50"
        aggregate_record["edge_thresholds_bp"]="0,50"
        aggregate_contract.write_text(json.dumps(aggregate_record))
        ap["aggregate_contract_sha256"]=sha(aggregate_contract)
        aggregate_params.write_text(json.dumps(ap))
        run("aggregate", 1, aggregate_params, ["COMPLETED"])
        run("aggregate", 2, aggregate_params, ["CACHED"])
        outputs = list((base/"work").glob("*/*/evaluation/manifest.json"))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(json.loads(outputs[0].read_text())["chromosomes"], ["21", "22"])
        target = inputs/"chr21/segment_evidence.tsv.gz"
        data, stat = target.read_bytes(), target.stat()
        target.write_bytes(bytes([data[0]^1])+data[1:])
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        run("prepare", 3, params, ["FAILED", "CACHED"])
        target.write_bytes(data)
        run("prepare", 4, params, ["CACHED", "CACHED"])


if __name__ == "__main__":
    unittest.main()
