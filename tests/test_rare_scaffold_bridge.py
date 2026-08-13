from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_rare_scaffold_bridge", ROOT / "bin/audit_rare_scaffold_bridge.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def write_vcf(path: Path, samples: list[str], records: list[tuple]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##reference=GRCh38\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(samples) + "\n")
        for pos, ref, alt, genotypes in records:
            dosages = []
            for genotype in genotypes:
                gt = genotype.replace("|", "/").split("/")
                if len(gt) != 2 or "." in gt:
                    continue
                dosages.append(sum(int(allele) for allele in gt))
            ac = sum(dosages)
            an = 2 * len(dosages)
            handle.write(
                f"chr22\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\tAC={ac};AN={an}\tGT\t"
                + "\t".join(genotypes)
                + "\n"
            )


class RareScaffoldBridgeTest(unittest.TestCase):
    def make_inputs(self, root: Path, scaffold_phased: bool = True) -> dict[str, Path]:
        raw = root / "raw.vcf.gz"
        scaffold = root / "scaffold.vcf.gz"
        baseline = root / "baseline.vcf.gz"
        raw_records = [
            (100, "A", "G", ["0/1", "0/0", "0/0"]),
            (200, "C", "T", ["0/1", "0/1", "0/0"]),
            (300, "G", "A", ["0/1", "0/1", "1/1"]),
        ]
        separator = "|" if scaffold_phased else "/"
        scaffold_records = [
            (100, "A", "G", [f"0{separator}1", f"0{separator}0", f"0{separator}0", f"0{separator}0"]),
            (200, "C", "T", [f"0{separator}1", f"0{separator}1", f"0{separator}0", f"0{separator}0"]),
            (300, "G", "A", [f"0{separator}1", f"0{separator}1", f"1{separator}1", f"0{separator}0"]),
        ]
        baseline_records = [
            (100, "A", "G", ["0|0"]),
            (200, "C", "T", ["0|0"]),
            (300, "G", "A", ["0|0"]),
        ]
        write_vcf(raw, ["PSI0001", "PSI0002", "PSI0003"], raw_records)
        write_vcf(
            scaffold,
            ["PSI001_PSI001", "PSI002_PSI002", "PSI003_PSI003", "OTHER_OTHER"],
            scaffold_records,
        )
        write_vcf(baseline, ["Native-American_X"], baseline_records)

        prereg = json.loads(
            (ROOT / "conf/m27b_rare_scaffold_bridge_preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        prereg["frozen_contract"].update(
            {
                "expected_raw_samples": 3,
                "rare_maf_threshold": 0.4,
                "identity_min_jointly_called_markers": 2,
            }
        )
        preregistration = root / "prereg.json"
        preregistration.write_text(json.dumps(prereg), encoding="utf-8")
        return {
            "raw_wgs_vcf": raw,
            "phased_scaffold_vcf": scaffold,
            "gnomix_reference_vcf": baseline,
            "preregistration": preregistration,
            "outdir": root / "out",
        }

    def test_direct_bridge_recomputes_minor_allele_and_preserves_privacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = self.make_inputs(Path(tmpdir))
            summary = AUDIT.run(type("Args", (), inputs)())
            self.assertEqual(summary["decision"], "GO_PC_RELATE_DONOR_AUDIT_ONLY")
            self.assertEqual(summary["gates"], {"B0": "PASS", "B1": "PASS", "B2": "PASS", "B3": "PASS", "B4": "PASS"})

            rare = json.loads((inputs["outdir"] / "m27b_rare_support.json").read_text())
            self.assertEqual(rare["n_rare_sites"], 2)
            self.assertEqual(rare["n_rare_alt_major_sites"], 1)
            self.assertEqual(rare["definition"]["orientation"], "minor_allele_recomputed_from_GT")

            phase = json.loads((inputs["outdir"] / "m27b_phase_bridge.json").read_text())
            self.assertEqual(phase["n_direct_phase_bridge_sites"], 2)
            self.assertFalse(phase["phase_inferred_for_scaffold_absent_rare_sites"])

            emitted = "\n".join(
                path.read_text(encoding="utf-8")
                for path in inputs["outdir"].iterdir()
                if path.suffix in {".json", ".tsv"}
            )
            self.assertNotIn("PSI0001", emitted)
            self.assertNotIn("PSI001", emitted)

    def test_unphased_scaffold_stops_before_pcrelate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = self.make_inputs(Path(tmpdir), scaffold_phased=False)
            summary = AUDIT.run(type("Args", (), inputs)())
            self.assertEqual(summary["decision"], "STOP_NO_DIRECT_RARE_PHASE_BRIDGE")
            self.assertEqual(summary["gates"]["B1"], "PASS")
            self.assertEqual(summary["gates"]["B3"], "FAIL")
            self.assertFalse(summary["pcrelate_executed"])

    def test_nextflow_workflow_is_isolated_and_cloud_labeled(self) -> None:
        workflow = (ROOT / "workflows/m27b_rare_scaffold_bridge.nf").read_text(encoding="utf-8")
        module = (ROOT / "modules/27B_RARE_SCAFFOLD_BRIDGE.nf").read_text(encoding="utf-8")
        cloud = (ROOT / "conf/google_batch.config").read_text(encoding="utf-8")
        self.assertIn("AUDIT_RARE_SCAFFOLD_BRIDGE", workflow)
        self.assertNotIn("PCRelate", module)
        self.assertNotIn("gnomix.py", module)
        self.assertNotIn("msprime", module)
        self.assertIn("resourceLabels = [team: 'frank']", cloud)
        self.assertIn("withName: 'AUDIT_RARE_SCAFFOLD_BRIDGE'", cloud)
        self.assertIn("disk = '20 GB'", cloud)


if __name__ == "__main__":
    unittest.main()
