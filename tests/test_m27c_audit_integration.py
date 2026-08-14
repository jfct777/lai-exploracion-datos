import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import audit_targeted_gvcf_readiness as audit  # noqa: E402
from m27c_gvcf_core import STATE_EXPLICIT_EXACT, STATE_REFERENCE_BLOCK  # noqa: E402


class TestM27CAuditIntegration(unittest.TestCase):
    def test_entrypoint_imports_sibling_modules_from_nextflow_symlink_stage(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for name in (
                "audit_targeted_gvcf_readiness.py",
                "audit_rare_scaffold_bridge.py",
                "m27c_gvcf_core.py",
            ):
                (work / name).symlink_to(repo_root / "bin" / name)

            completed = subprocess.run(
                [sys.executable, "audit_targeted_gvcf_readiness.py", "--help"],
                cwd=work,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertIn("--gvcfs", completed.stdout)

    def test_tiny_panel_reaches_ready_only_with_phase_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.vcf"
            samples = [
                "African_A1", "African_A2", "European_E1", "European_E2",
                "Native-American_N1", "Native-American_N2",
            ]
            baseline.write_text(
                "##fileformat=VCFv4.2\n"
                + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(samples)
                + "\n"
                + "22\t2\t.\tA\tG\t.\t.\tCM=0.1\tGT\t0|0\t0|1\t0|0\t0|0\t0|1\t1|1\n"
                + "22\t4\t.\tC\tT\t.\t.\tCM=0.2\tGT\t0|0\t0|0\t0|1\t0|1\t0|0\t0|1\n",
                encoding="utf-8",
            )
            scaffold = root / "scaffold.vcf"
            scaffold.write_text(
                "##fileformat=VCFv4.2\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
                "22\t2\t.\tA\tG\t.\t.\t.\tGT\t0|1\t0|0\n"
                "22\t4\t.\tC\tT\t.\t.\t.\tGT\t0|0\t1|0\n",
                encoding="utf-8",
            )
            metadata = root / "metadata.tsv"
            metadata.write_text(
                "IID\tSample_ID(Aliases)\tIllumina_ID\toriginal_IID\n"
                "S1\tS1\tS1\tS1\nS2\tS2\tS2\tS2\n",
                encoding="utf-8",
            )
            config = root / "config.txt"
            config.write_text("A\t3\nC\t2\nM\t1\nS\t1\nW\t2\ncontext\t1\n", encoding="utf-8")
            prereg = root / "prereg.json"
            prereg.write_text(
                json.dumps(
                    {
                        "stage": "M27C_TARGETED_GVCF_READINESS",
                        "scope": "test",
                        "frozen_contract": {
                            "chromosome": "22",
                            "expected_gvcf_samples": 2,
                            "expected_model_markers": 2,
                            "expected_baseline_donors": 6,
                            "expected_donors_per_ancestry": 2,
                            "identity_min_jointly_called_markers": 2,
                            "identity_min_dosage_concordance": 0.99,
                            "ancestries": ["African", "European", "Native-American"],
                            "minimum_gnomix_ready_marker_fraction": 0.8,
                        },
                        "quality_policies": {
                            "primary": {
                                "minimum_effective_depth": 10,
                                "minimum_gq": 20,
                                "minimum_marker_call_rate": 1.0,
                            },
                            "one_factor_sensitivities": [
                                {
                                    "id": "same",
                                    "minimum_effective_depth": 10,
                                    "minimum_gq": 20,
                                    "minimum_marker_call_rate": 1.0,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            gvcfs = [root / "S1.g.vcf.gz", root / "S2.g.vcf.gz"]
            indexes = [root / "S1.g.vcf.gz.tbi", root / "S2.g.vcf.gz.tbi"]
            for path in gvcfs + indexes:
                path.touch()
            manifest = root / "manifest.tsv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["uri", "generation", "size_bytes", "crc32c", "md5_hash"],
                    delimiter="\t",
                )
                writer.writeheader()
                for path in gvcfs + indexes:
                    writer.writerow(
                        {
                            "uri": f"gs://bucket/{path.name}",
                            "generation": "1",
                            "size_bytes": "1",
                            "crc32c": "x",
                            "md5_hash": "",
                        }
                    )
            fasta = root / "reference.fa"
            fai = root / "reference.fa.fai"
            fasta.touch()
            fai.touch()
            outdir = root / "out"
            args = argparse.Namespace(
                gvcfs=gvcfs,
                gvcf_indexes=indexes,
                gcs_input_manifest=manifest,
                phased_scaffold_vcf=scaffold,
                gnomix_reference_vcf=baseline,
                gnomix_config=config,
                metadata=metadata,
                reference_fasta=fasta,
                reference_fai=fai,
                preregistration=prereg,
                bcftools="bcftools",
                samtools="samtools",
                readers=2,
                smoke=False,
                outdir=outdir,
            )

            def header_result(_bcftools, path):
                return {
                    "samples": [path.name.split(".", 1)[0]],
                    "source": "HaplotypeCaller",
                    "reference": "GRCh38",
                    "chr22_length": 50818468,
                    "format_ids": ["DP", "GQ", "GT", "MIN_DP"],
                    "has_required_fields": True,
                }

            def sample_result(index, *_args, **_kwargs):
                if index == 0:
                    dosages = np.asarray([1, 0], dtype=np.int8)
                else:
                    dosages = np.asarray([0, 1], dtype=np.int8)
                return audit.SampleResult(
                    sample_index=index,
                    states=np.asarray([STATE_EXPLICIT_EXACT, STATE_REFERENCE_BLOCK], dtype=np.int8),
                    dosages=dosages,
                    depths=np.asarray([20, 20], dtype=np.int32),
                    gqs=np.asarray([40, 40], dtype=np.int32),
                    n_records=2,
                    uncompressed_bytes=100,
                    elapsed_seconds=0.1,
                )

            with (
                mock.patch.object(audit, "read_header", side_effect=header_result),
                mock.patch.object(audit, "load_reference_contig", return_value="AACC"),
                mock.patch.object(audit, "parse_one_sample", side_effect=sample_result),
            ):
                summary = audit.run(args)

            input_contract = json.loads((outdir / "m27c_input_contract.json").read_text())
            self.assertEqual(
                summary["decision"],
                "READY_FOR_RARE_DONOR_AUDIT_ONLY",
                msg=json.dumps(input_contract, sort_keys=True),
            )
            self.assertEqual(summary["primary_candidate_panel_ready_fraction"], 1.0)
            identity = json.loads((outdir / "m27c_identity_control.json").read_text())
            self.assertEqual(identity["status"], "PASS")
            self.assertFalse(summary["final_donor_panel_certified"])


if __name__ == "__main__":
    unittest.main()
