"""Synthetic M01/M02 storage equivalence; no human genotypes.

RUN_PREPROCESS_BOUNDED_TEST=1 enables two real local Nextflow/Docker runs.
Optional PREPROCESS_TEST_BULK_ROOT points to a pre-created, private GCSFuse
smoke directory. Its uniquely named test outputs are retained for inspection.
"""
import gzip
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9"


@unittest.skipUnless(os.environ.get("RUN_PREPROCESS_BOUNDED_TEST") == "1",
                     "opt-in synthetic local Nextflow/Docker integration")
class BoundedDiskTests(unittest.TestCase):
    def test_local_and_bulk_storage_preserve_variants_counts_and_source(self):
        with tempfile.TemporaryDirectory(prefix="r02-preprocess-storage-",
                                         dir=os.environ.get("PREPROCESS_TEST_LOCAL_ROOT")) as name:
            base = Path(name)
            bulk = Path(os.environ.get("PREPROCESS_TEST_BULK_ROOT", str(base / "bulk")))
            # A user-owned FUSE mount denies daemon-root traversal. Binding an
            # accessible parent exposes its child mount to the container user.
            # Production Nextflow already coalesces the inputs to this parent.
            docker_bulk_bind = Path(os.environ.get("PREPROCESS_TEST_BULK_BIND_ROOT", str(bulk)))
            if "PREPROCESS_TEST_BULK_ROOT" in os.environ:
                self.assertTrue(bulk.is_dir(), "Pre-create the private smoke directory explicitly")
            else:
                bulk.mkdir()
            ref = base / "ref.fa"
            ref.write_text(">chr22\n" + "A" * 1000 + "\n")
            ref.with_suffix(".fa.fai").write_text("chr22\t1000\t7\t1000\t1001\n")
            vcf = base / "input.vcf"
            vcf.write_text(
                "##fileformat=VCFv4.2\n##contig=<ID=chr22,length=1000>\n"
                '##FILTER=<ID=q10,Description="rejected">\n'
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n"
                "chr22\t100\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\n"
                "chr22\t200\t.\tA\tC,G\t.\tPASS\t.\tGT\t1/2\t0/0\n"
                "chr22\t300\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\n"
                "chr22\t300\t.\tA\tG\t.\tq10\t.\tGT\t0/1\t0/0\n"
                "chr22\t400\t.\tA\tAT\t.\tPASS\t.\tGT\t0/1\t0/0\n")
            source_bytes = vcf.read_bytes()
            workflow = base / "test.nf"
            workflow.write_text(f"""
nextflow.enable.dsl=2
include {{ PREPROCESS_NORM_LEFTALIGN }} from '{ROOT}/modules/01_preprocess_norm_leftalign'
include {{ PREPROCESS_FILTER_SNV_BIALLELIC_PASS }} from '{ROOT}/modules/02_preprocess_filter_snv_biallelic_pass'
process PACK {{
    input: path input_vcf
    output: tuple val('22'), path('input.vcf.gz'), path('input.vcf.gz.tbi')
    script:
    """ + '"""' + """
    bcftools view -Oz -o input.vcf.gz ${input_vcf}
    bcftools index -t input.vcf.gz
    """ + '"""' + f"""
}}
workflow {{
    raw = PACK(file('{vcf}'))
    norm = PREPROCESS_NORM_LEFTALIGN(raw.combine(Channel.value(file('{ref}'))), file('{ROOT}/bin/mark_original_alleles.py'))
    PREPROCESS_FILTER_SNV_BIALLELIC_PASS(raw.join(norm[0]))
}}
""")
            outputs = {}
            for mode in ("normal", "bounded"):
                run = base / mode
                run.mkdir()
                config = run / "test.config"
                config.write_text(f"""
process.executor = 'local'
process.container = '{IMAGE}'
process.stageInMode = 'symlink'
docker.enabled = true
docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()} -v {docker_bulk_bind}:{docker_bulk_bind}'
executor.queueSize = 1
params.outdir = '{run}/results'
params.cpus = 1
params.memory = '1 GB'
params.time = '5m'
params.resources = [:]
params.preprocess_large_temp_dir = '{bulk if mode == "bounded" else ""}'
params.bcftools_min_alleles = 2
params.plink_max_alleles = 2
params.plink_snps_only = true
params.max_maf = null
params.keep_pass = true
""")
                result = subprocess.run([
                    "nextflow", "-log", str(run / "nextflow.log"), "-C", str(config),
                    "run", str(workflow), "-work-dir", str(run / "work"), "-ansi-log", "false"],
                    cwd=run, capture_output=True, text=True, timeout=180,
                    env={**os.environ, "NXF_SYNTAX_PARSER": "v1", "NXF_OFFLINE": "true",
                         "NXF_DISABLE_CHECK_LATEST": "true"})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                outputs[mode] = run / "results"
                manifests = list((run / "work").glob("*/*/*.large_temp_dir.txt"))
                self.assertEqual(len(manifests), 2 if mode == "bounded" else 0)
                if mode == "bounded":
                    dirs = [Path(p.read_text().strip()) for p in manifests]
                    self.assertEqual(len(set(dirs)), 2)
                    self.assertTrue(all(p.parent == bulk and p.is_dir() for p in dirs))
                    norm_links = list((run / "work").glob("*/*/*.norm.vcf.gz"))
                    self.assertTrue(any(p.is_symlink() and p.resolve().parent in dirs for p in norm_links))
            for relative in ("01_norm/dnabr.hg38.2723.chr22.norm.vcf.gz",
                             "02_filter/dnabr.hg38.2723.chr22.snv.bi.pass.vcf.gz"):
                data = []
                for mode in ("normal", "bounded"):
                    with gzip.open(outputs[mode] / relative, "rt") as handle:
                        data.append([line for line in handle if not line.startswith("#")])
                    self.assertTrue(Path(str(outputs[mode] / relative) + ".tbi").exists())
                self.assertEqual(data[0], data[1])
                self.assertTrue(any("ORIG_NALLELES=3" in line for line in data[0]))
            counts = Path("02_filter/dnabr.hg38.2723.chr22.counts.tsv")
            self.assertEqual((outputs["normal"] / counts).read_bytes(),
                             (outputs["bounded"] / counts).read_bytes())
            self.assertEqual(vcf.read_bytes(), source_bytes)


if __name__ == "__main__":
    unittest.main()
