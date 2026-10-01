"""Opt-in synthetic M01 checkpoint, equivalence and recovery integration.

RUN_M01_CHECKPOINT_TEST=1 enables local Nextflow/Docker only. No cohort inputs,
GCS publication, VM creation or active pipeline changes are performed.
"""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9"
ENV = {**os.environ, "NXF_SYNTAX_PARSER": "v1", "NXF_OFFLINE": "true",
       "NXF_DISABLE_CHECK_LATEST": "true"}


class CheckpointSourceContractTests(unittest.TestCase):
    def test_checkpoint_is_opt_in_at_both_entrypoints(self):
        for relative in ("main.nf", "workflows/r02_preprocess_autosome.nf"):
            source = (ROOT / relative).read_text()
            self.assertIn("if (params.preprocess_checkpointed_m01 == true)", source)
            self.assertIn("PREPROCESS_NORM_LEFTALIGN_CHECKPOINTED(", source)
            self.assertIn("PREPROCESS_NORM_LEFTALIGN(", source)

    def test_split_outputs_and_resource_contract(self):
        source = (ROOT / "modules/01_preprocess_checkpointed.nf").read_text()
        self.assertIn('path("dnabr.hg38.2723.chr${chr}.original.bcf"), emit: annotated', source)
        self.assertIn("--threads ${task.cpus}", source)
        self.assertIn("Math.max(0, (task.cpus as int) - 1)", source)
        self.assertIn("Math.min(maximum, requested as int)", source)
        self.assertEqual(source.count("path preprocess_storage_guard_py"), 2)
        self.assertNotIn("samtools faidx", source)
        self.assertNotIn("--samples", source)
        self.assertNotIn("bcftools view", source)
        annotation, normalization = source.split("process NORMALIZE_ANNOTATED_ALLELES", 1)
        self.assertIn("--required-free-bytes", annotation)
        self.assertNotIn("--required-free-bytes", normalization)
        workflow = source.split("workflow PREPROCESS_NORM_LEFTALIGN_CHECKPOINTED", 1)[1]
        self.assertRegex(workflow, r"emit:\s+norm = .*\.out\.norm\s+norm_log = .*\.out\.norm_log")


@unittest.skipUnless(os.environ.get("RUN_M01_CHECKPOINT_TEST") == "1" and shutil.which("nextflow"),
                     "opt-in synthetic local Nextflow/Docker integration")
class CheckpointIntegrationTests(unittest.TestCase):
    def setup_fixture(self, base, *, indexed=True):
        ref = base / "reference.fa"
        ref.write_text(">chr22\n" + "A" * 1000 + "\n")
        if indexed:
            Path(str(ref) + ".fai").write_text("chr22\t1000\t7\t1000\t1001\n")
        samples = [f"s{i}" for i in range(50)]
        header = ("##fileformat=VCFv4.2\n##contig=<ID=chr22,length=1000>\n"
                  '##FILTER=<ID=q10,Description="Rejected fixture">\n'
                  '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                  "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        rows = []
        for pos, alt, filt, genotypes in (
            (100, "C", "PASS", ["0/1"] * 2 + ["0/0"] * 48),
            (200, "C", "PASS", ["0/1"] * 2 + ["1/1"] * 48),
            (300, "C,G", "PASS", ["1/2"] + ["0/0"] * 49),
            (400, "C", "PASS", ["0/1"] * 2 + ["0/0"] * 48),
            (400, "G", "q10", ["0/1"] * 2 + ["0/0"] * 48),
            (500, "AT", "PASS", ["0/1"] * 2 + ["0/0"] * 48),
            (600, "C", "PASS", ["./.", "0/1"] + ["0/0"] * 48),
        ):
            rows.append(f"chr22\t{pos}\t.\tA\t{alt}\t.\t{filt}\t.\tGT\t" + "\t".join(genotypes))
        (base / "input.vcf").write_text(header + "\n".join(rows) + "\n")
        subprocess.run([
            "docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{base}:{base}", "-w", str(base), IMAGE, "bash", "-c",
            "bcftools view -Oz -o raw.vcf.gz input.vcf && bcftools index -t raw.vcf.gz"],
            check=True, capture_output=True, text=True, timeout=30)
        scripts = base / "scripts"
        scripts.mkdir()
        for name in ("mark_original_alleles.py", "preprocess_storage_guard.py", "select_rare_minor.py"):
            shutil.copy2(ROOT / "bin" / name, scripts / name)
        return ref, scripts

    def config(self, run, base, *, checkpoint, bulk):
        run.mkdir()
        if bulk:
            bulk.mkdir()
        settings = dict(
            outdir=str(run / "results"), r02_chrom=22, r02_raw_vcf=str(base / "raw.vcf.gz"),
            r02_bin_dir=str(base / "scripts"), ref_fasta=str(base / "reference.fa"),
            preprocess_large_temp_dir=str(bulk) if bulk else "", preprocess_checkpointed_m01=checkpoint,
            preprocess_minimum_free_gib=0, preprocess_required_free_bytes=1,
            cpus=1, memory="1 GB", time="5m",
            resources={"preprocess_norm_leftalign": {"threads": 8}},
            bcftools_min_alleles=2, plink_max_alleles=2, plink_snps_only=True,
            max_maf=None, keep_pass=True, lai_rare_max_maf=.03, lai_rare_min_mac=2,
            lai_rare_keep_format="GT", lai_rare_remove_info=False)
        (run / "parameters.json").write_text(json.dumps(settings))
        (run / "runtime.config").write_text(f"""
process.executor='local'
process.container='{IMAGE}'
process.stageInMode='symlink'
docker.enabled=true
docker.runOptions='--network none --user {os.getuid()}:{os.getgid()} -v {base}:{base}'
executor.queueSize=1
trace.enabled=true
trace.fields='task_id,hash,name,status,exit'
""")

    def run_nextflow(self, run, *, resume=False, trace="trace.tsv"):
        args = ["nextflow", "-log", str(run / ("resume.log" if resume else "nextflow.log")),
                "-C", str(run / "runtime.config"), "run", str(ROOT / "workflows/r02_preprocess_autosome.nf"),
                "-params-file", str(run / "parameters.json"), "-work-dir", str(run / "work"),
                "-ansi-log", "false", "-with-trace", str(run / trace)]
        if resume:
            args.append("-resume")
        return subprocess.run(args, cwd=run, env=ENV, capture_output=True, text=True, timeout=180)

    def rows(self, path):
        with gzip.open(path, "rt") as handle:
            return [line.rstrip() for line in handle if not line.startswith("#")]

    def trace(self, path):
        with path.open() as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def audit(self, run):
        result = subprocess.run(
            [sys.executable, str(ROOT / "bin/preprocess_audit.py"),
             "--folder", str(run), "--chrom", "22"],
            check=True, capture_output=True, text=True, timeout=15)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "PREPROCESS_AUDIT_COLLECTED")
        self.assertEqual(report["files"], 4)
        manifest_path = Path(report["manifest"])
        encoded = manifest_path.read_bytes()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), report["manifest_sha256"])
        manifest = json.loads(encoded)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["trace_sha256"], hashlib.sha256((run / "trace.tsv").read_bytes()).hexdigest())
        for task in manifest["tasks"]:
            self.assertRegex(task["work_hash"], r"^[0-9a-f]{32}$")
            self.assertTrue(Path(task["work_directory"]).is_dir())
        for entry in manifest["files"]:
            copied = manifest_path.parent / entry["relative_path"]
            self.assertEqual(copied.read_bytes(), Path(entry["source"]).read_bytes())
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), entry["sha256"])
        return manifest

    def test_checkpoint_preserves_chain_and_bounded_storage_outputs(self):
        with tempfile.TemporaryDirectory(prefix="m01-checkpoint-equivalence-") as tmp:
            base = Path(tmp)
            self.setup_fixture(base)
            runs = {}
            for mode, checkpoint, bounded in (("legacy", False, False), ("checkpoint", True, False),
                                                ("checkpoint-bulk", True, True)):
                run = base / mode
                self.config(run, base, checkpoint=checkpoint, bulk=base / "bulk" if bounded else None)
                result = self.run_nextflow(run)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                runs[mode] = run
                trace = self.trace(run / "trace.tsv")
                completed = [row for row in trace if row["status"] == "COMPLETED"]
                self.assertEqual(len(completed), 4 if checkpoint else 3)
                if checkpoint:
                    annotation = list((run / "work").glob("*/*/*.original.bcf"))
                    self.assertTrue(annotation, "An original-allele BCF must remain available for -resume")
                    commands = [p.read_text() for p in (run / "work").glob("*/*/.command.sh")]
                    annotate = next(x for x in commands if "marker_script" in x)
                    norm = next(x for x in commands if "bcftools norm -m -any" in x)
                    self.assertIn("--threads 1", annotate)
                    self.assertIn("--required-free-bytes '1'", annotate)
                    self.assertIn("--threads 0", norm)
                    manifest = self.audit(run)
                    self.assertEqual({x["trace_status"] for x in manifest["tasks"]}, {"COMPLETED"})
                    if bounded:
                        self.assertTrue(any(p.is_symlink() for p in annotation))
                        self.assertFalse((run / "results/01_norm").exists(), "Bounded mode must not duplicate a large norm VCF")
            for relative in ("02_filter/dnabr.hg38.2723.chr22.snv.bi.pass.vcf.gz",
                             "lai_rare/dnabr.hg38.2723.chr22.rare.minor.vcf.gz"):
                expected = self.rows(runs["legacy"] / "results" / relative)
                for mode in ("checkpoint", "checkpoint-bulk"):
                    self.assertEqual(self.rows(runs[mode] / "results" / relative), expected)
            rare = self.rows(runs["checkpoint"] / "results/lai_rare/dnabr.hg38.2723.chr22.rare.minor.vcf.gz")
            self.assertEqual([int(line.split("\t")[1]) for line in rare], [100, 200])
            self.assertIn("RARE_ALLELE=0", rare[1])
            for mode in ("checkpoint", "checkpoint-bulk"):
                contract = json.loads((runs[mode] / "results/lai_rare/dnabr.hg38.2723.chr22.rare.minor.contract.json").read_text())
                self.assertEqual(contract["cohort_samples_after"], 50)
                self.assertEqual(contract["counts"]["excluded_original_multiallelic"], 3)

    def test_resume_reuses_annotation_after_normalization_failure(self):
        with tempfile.TemporaryDirectory(prefix="m01-checkpoint-recovery-") as tmp:
            base = Path(tmp)
            ref, _ = self.setup_fixture(base, indexed=False)
            run = base / "recovery"
            self.config(run, base, checkpoint=True, bulk=base / "bulk")
            failed = self.run_nextflow(run)
            self.assertNotEqual(failed.returncode, 0)
            first = self.trace(run / "trace.tsv")
            annotation = next(x for x in first if "ANNOTATE_ORIGINAL_ALLELES" in x["name"])
            self.assertEqual(annotation["status"], "COMPLETED")
            self.assertTrue(any(x["status"] == "FAILED" for x in first))
            Path(str(ref) + ".fai").write_text("chr22\t1000\t7\t1000\t1001\n")
            resumed = self.run_nextflow(run, resume=True, trace="resume_trace.tsv")
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            second = self.trace(run / "resume_trace.tsv")
            cached = next(x for x in second if "ANNOTATE_ORIGINAL_ALLELES" in x["name"])
            self.assertEqual(cached["status"], "CACHED")
            self.assertEqual(cached["hash"], annotation["hash"])
            normalization = next(x for x in second if "NORMALIZE_ANNOTATED_ALLELES" in x["name"])
            self.assertEqual(normalization["status"], "COMPLETED")
            self.assertTrue((run / "results/lai_rare/dnabr.hg38.2723.chr22.rare.minor.vcf.gz").exists())
            # The collector reads the latest trace at the Runner's canonical
            # path. Preserve this fixture's failed trace separately first.
            shutil.copy2(run / "trace.tsv", run / "failed_trace.tsv")
            shutil.copy2(run / "resume_trace.tsv", run / "trace.tsv")
            manifest = self.audit(run)
            statuses = {x["process"]: x["trace_status"] for x in manifest["tasks"]}
            self.assertEqual(statuses["ANNOTATE_ORIGINAL_ALLELES"], "CACHED")
            self.assertEqual(statuses["NORMALIZE_ANNOTATED_ALLELES"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
