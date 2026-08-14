import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import m27d_prepare_sample_strata as strata  # noqa: E402


class TestM27DSampleStrata(unittest.TestCase):
    def test_alias_resolution_and_public_redaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vcf = root / "panel.vcf"
            vcf.write_text(
                "##fileformat=VCFv4.2\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                "S1_S1\tS2\tUNKNOWN\n",
                encoding="utf-8",
            )
            metadata = root / "metadata.tsv"
            metadata.write_text(
                "IID\tSample_ID(Aliases)\tIllumina_ID\toriginal_IID\tSource\tAncestry\tPopulation\tCountry\n"
                "S1\tS1\tI1\tO1\tNatWGS\tNative_American\tP1\tBrazil\n"
                "S2\talias-S2\tS2\tO2\tHGDP\tAfrican\tP2\tKenya\n",
                encoding="utf-8",
            )
            private = root / "private.tsv"
            summary_path = root / "summary.json"
            args = type(
                "Args",
                (),
                {
                    "panel_vcf": vcf,
                    "metadata": metadata,
                    "private_out": private,
                    "summary_out": summary_path,
                    "suppress_below": 1,
                },
            )()

            summary = strata.run(args)

            self.assertEqual(summary["n_panel_samples"], 3)
            self.assertEqual(summary["n_matched"], 2)
            self.assertEqual(summary["n_unmatched"], 1)
            public_text = summary_path.read_text(encoding="utf-8")
            self.assertNotIn("S1_S1", public_text)
            with private.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rows[0]["match_status"], "MATCHED")
            self.assertEqual(rows[0]["Country"], "Brazil")

    def test_ambiguous_alias_is_not_silently_chosen(self):
        rows = [
            {"IID": "A", "Sample_ID(Aliases)": "X", "Illumina_ID": "", "original_IID": ""},
            {"IID": "B", "Sample_ID(Aliases)": "X", "Illumina_ID": "", "original_IID": ""},
        ]
        resolved = strata.resolve_rows(["X"], rows)
        self.assertEqual(resolved[0][1], "AMBIGUOUS")
        self.assertIsNone(resolved[0][2])


class TestM27DNoKing(unittest.TestCase):
    def test_new_scientific_scripts_do_not_call_king_or_pcair(self):
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "bin" / "m27d_prepare_genotype_resources.R",
            root / "bin" / "m27d_resource_smoke.R",
            root / "bin" / "m27d_prepare_sample_strata.py",
            root / "bin" / "verify_m27d_prepared_inputs.py",
        ]
        forbidden = ("snpgdsIBDKING", "pcair(", "kingToMatrix")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, msg=f"{token} found in {path.name}")

    def test_cloud_smoke_is_pinned_labeled_and_full_run_is_blocked(self):
        root = Path(__file__).resolve().parents[1]
        cloud = (root / "conf" / "google_batch.config").read_text(encoding="utf-8")
        workflow = (root / "workflows" / "m27d_donor_kinship_audit.nf").read_text(
            encoding="utf-8"
        )
        module = (root / "modules" / "27D_DONOR_KINSHIP_AUDIT.nf").read_text(
            encoding="utf-8"
        )
        digest = "3a4661e41f7e397e986472bb8039671f85b1e8f7b86fc26af83a9837ef83d954"
        self.assertIn("resourceLabels = [team: 'frank']", cloud)
        self.assertIn(f"dnabr-qc@sha256:{digest}", cloud)
        self.assertIn("full donor audit is not implemented or authorized", workflow)
        self.assertIn("PREPARE_DONOR_KINSHIP_RESOURCES", workflow)
        self.assertIn("phase in ['prepare', 'benchmark']", workflow)
        self.assertIn("M27D benchmark is missing prepared inputs", workflow)
        self.assertIn("donor_kinship_preparation_manifest_sha256", workflow)
        self.assertIn("m27d_prepared_input_verification.json", module)
        self.assertIn("time = '75m'", cloud)

    def test_batch_style_script_staging_resolves_helper_from_workdir(self):
        root = Path(__file__).resolve().parents[1]
        module = (root / "modules" / "27D_DONOR_KINSHIP_AUDIT.nf").read_text(
            encoding="utf-8"
        )
        self.assertIn("PYTHONPATH=. python3 ${sample_strata_py}", module)

        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            script_dir = temp_root / "batch-script"
            work_dir = temp_root / "work"
            script_dir.mkdir()
            work_dir.mkdir()
            script = script_dir / "m27d_prepare_sample_strata.py"
            script.write_bytes((root / "bin" / script.name).read_bytes())
            helper = work_dir / "audit_rare_scaffold_bridge.py"
            helper.write_bytes((root / "bin" / helper.name).read_bytes())
            (work_dir / "panel.vcf").write_text(
                "##fileformat=VCFv4.2\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n",
                encoding="utf-8",
            )
            (work_dir / "metadata.tsv").write_text(
                "IID\tSample_ID(Aliases)\tIllumina_ID\toriginal_IID\tSource\tAncestry\tPopulation\tCountry\n"
                "S1\tS1\tS1\tS1\tNatWGS\tNAM\tP1\tBrazil\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "."
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--panel-vcf",
                    "panel.vcf",
                    "--metadata",
                    "metadata.tsv",
                    "--private-out",
                    "private.tsv",
                    "--summary-out",
                    "summary.json",
                ],
                cwd=work_dir,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
