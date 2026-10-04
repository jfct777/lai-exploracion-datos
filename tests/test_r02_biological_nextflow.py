"""Opt-in real data-flow and resume test on synthetic genotypes, never DNABR."""
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
PREP_IMAGE = "sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9"


@unittest.skipUnless(os.environ.get("RUN_R02_BIOLOGICAL_NEXTFLOW") == "1", "opt-in synthetic Nextflow integration")
class BiologicalNextflowTests(unittest.TestCase):
    def test_all_three_stages_optional_evidence_and_resume(self):
        base = Path(tempfile.mkdtemp(prefix="r02-biological-nextflow-", dir=ROOT / ".claude/runs"))
        frozen = base / "frozen_bin"
        shutil.copytree(ROOT / "bin", frozen, ignore=shutil.ignore_patterns("__pycache__"))
        prepare = r'''
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_r02_segment_evidence import SegmentEvidenceTests
t = SegmentEvidenceTests()
t.setUp()
t.spanning_chain()
t.common_fixture()
t.map_fixture()
t.ibd_fixture()
t.write_table(t.configurations, ('config_id','max_gap_bp','min_length_bp','min_shared_effective','min_shared_nominal_aliases','n_geo'), [
    ['L401_G500_N2',500,401,2,'2',2], ['L1000_G500_N3',500,1000,3,'3',3]])
destination = Path(sys.argv[1])
shutil.copytree(t.root, destination)
ibd = destination / 'ibd_bundle'
ibd.mkdir()
for name in ['ibd.tsv','callable.tsv']:
    shutil.copy2(destination / name, ibd / name)
shutil.copy2(destination / 'ibd.json', ibd / 'ibd_contract.json')
'''
        command = ["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
                   "-e", "NUMBA_CACHE_DIR=/tmp/numba", "-e", "MPLCONFIGDIR=/tmp/mpl", "-e", "OPENBLAS_NUM_THREADS=1",
                   "-v", f"{ROOT}:{ROOT}", "-w", str(ROOT), PREP_IMAGE, "python3", "-c", prepare, str(base / "input")]
        prepared = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        inputs = base / "input"
        metadata = inputs / "metadata.tsv"
        metadata.write_text("ID\tCohort\tRegion\na\tA\tNORTH\nb\tA\tNORTH\nc\tB\tSOUTH\n")
        kinship = inputs / "kin.tsv"
        kinship.write_text("ID1\tID2\tkin\na\tb\t0.1\na\tc\t0\nb\tc\t0\n")
        entry = dict(chrom="22", chains=str(inputs / "candidate_chains.tsv.gz"),
                     configurations=str(inputs / "configuration_summary.tsv"), rare_vcf=str(inputs / "spanning.vcf.gz"),
                     rare_index=str(inputs / "spanning.vcf.gz.tbi"), common_vcf=str(inputs / "common.vcf.gz"),
                     common_index=str(inputs / "common.vcf.gz.tbi"), common_contract=str(inputs / "common.json"),
                     genetic_map=str(inputs / "map.tsv"), map_contract=str(inputs / "map.json"), ibd_bundle=str(inputs / "ibd_bundle"))
        (base / "inputs.json").write_text(json.dumps([entry]))
        parameters = dict(source_bin=str(frozen), sample_ids=str(inputs / "samples.ids"), metadata=str(metadata),
                          metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(), metadata_id_column="ID",
                          pcrelate_file=str(kinship), kinship_sha256=hashlib.sha256(kinship.read_bytes()).hexdigest(),
                          chromosome_inputs=str(base / "inputs.json"), chromosomes="22", expected_samples=3,
                          expected_source_samples=101, expected_configurations=2, genome_build="GRCh38",
                          block_bp=150, chunk_sites=1, max_active_intervals=1000, min_free_gib=0,
                          max_matrix_mb=16, max_database_mb=16, min_free_disk_mb=0,
                          kinship_thresholds="0.0221,0.0442", edge_thresholds_bp="0,500", roles=None,
                          metadata_columns=["Batch"])
        params = base / "params.json"
        params.write_text(json.dumps(parameters))
        config = base / "resources.config"
        config.write_text(f"""
process.executor = 'local'
process.container = '{IMAGE}'
process {{ withName: R02_SEGMENT_EVIDENCE {{ container = '{PREP_IMAGE}' }} }}
process.memory = '2 GB'
process.time = '3m'
process.maxForks = 1
process.errorStrategy = 'terminate'
executor.queueSize = 1
docker.enabled = true
docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()} -v {ROOT}:{ROOT}'
env.OMP_NUM_THREADS = '1'
env.OPENBLAS_NUM_THREADS = '1'
env.NUMBA_CACHE_DIR = '/tmp/numba'
env.MPLCONFIGDIR = '/tmp/mpl'
""")
        common = ["nextflow", "-C", str(config), "run", str(ROOT / "workflows/r02_biological_evidence.nf"),
                  "-params-file", str(params), "-work-dir", str(base / "work"), "-ansi-log", "false"]
        environment = {**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true",
                       "NXF_SYNTAX_PARSER": "v1", "NXF_OPTS": "-Xms64m -Xmx512m"}
        print(f"Synthetic biology-workflow test evidence: {base}", flush=True)
        for i, suffix in enumerate([[], ["-resume"]]):
            result = subprocess.run([*common, "-with-trace", str(base / f"trace{i}.tsv"), *suffix],
                                    cwd=base, env=environment, capture_output=True, text=True, timeout=240)
            (base / f"captured{i}.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (base / f"trace{i}.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r["status"] == ("COMPLETED" if i == 0 else "CACHED") for r in rows), rows)
        evaluation_paths = list((base / "work").glob("*/*/evaluation/manifest.json"))
        self.assertEqual(len(evaluation_paths), 1)
        manifest = json.loads(evaluation_paths[0].read_text())
        self.assertEqual(manifest["n_configurations"], 2)
        self.assertEqual(manifest["chromosomes"], ["22"])
        self.assertTrue(manifest["no_winner_selected"])
        # The downstream task stages the same directory by symlink; count the
        # physical output once, not each consumer's reference to it.
        study_paths = sorted({p.resolve() for p in (base / "work").glob("*/*/study/metadata_coverage.tsv")})
        self.assertEqual(len(study_paths), 1)
        self.assertIn("Batch\t0\t3\tCOLUMN_ABSENT", study_paths[0].read_text())
        with (evaluation_paths[0].parent / "configuration_metrics.tsv").open() as h:
            measurements = list(csv.DictReader(h, delimiter="\t"))
        j = next(r for r in measurements if r["config_id"] == "L401_G500_N2" and r["metric"] == "rare_local_J" and r["min_edge_bp"] == "0")
        self.assertEqual((j["numerator"], j["denominator"]), ("2", "3"))


if __name__ == "__main__":
    unittest.main()
