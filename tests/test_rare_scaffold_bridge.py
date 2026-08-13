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


def write_metadata(path: Path) -> None:
    path.write_text(
        "IID\tSample_ID(Aliases)\tIllumina_ID\toriginal_IID\n"
        "PSI0001\tPSI001_PSI001\tILL1\tPSI0001\n"
        "PSI0002\tPSI002_PSI002\tILL2\tPSI0002\n"
        "PSI0003\tPSI003_PSI003\tILL3\tPSI0003\n",
        encoding="utf-8",
    )


class RareScaffoldBridgeTest(unittest.TestCase):
    def make_inputs(self, root: Path, scaffold_phased: bool = True) -> dict[str, Path]:
        raw = root / "raw.vcf.gz"
        scaffold = root / "scaffold.vcf.gz"
        baseline = root / "baseline.vcf.gz"
        metadata = root / "metadata.tsv"
        raw_records = [
            (100, "A", "G", ["0/1", "0/0", "0/0"]),
            (200, "C", "T", ["0/1", "0/1", "0/0"]),
            (300, "G", "A", ["0/1", "0/1", "1/1"]),
            (400, "T", "C", ["1/1", "0/0", "0/0"]),
        ]
        separator = "|" if scaffold_phased else "/"
        scaffold_records = [
            (
                100,
                "A",
                "G",
                [f"0{separator}1", f"0{separator}0", f"0{separator}0", f"0{separator}0"],
            ),
            (
                200,
                "C",
                "T",
                [f"0{separator}1", f"0{separator}1", f"0{separator}0", f"0{separator}0"],
            ),
            (
                300,
                "G",
                "A",
                [f"0{separator}1", f"0{separator}1", f"1{separator}1", f"0{separator}0"],
            ),
            (
                400,
                "T",
                "C",
                [f"1{separator}1", f"0{separator}0", f"0{separator}0", f"0{separator}0"],
            ),
        ]
        baseline_records = [
            (100, "A", "G", ["0|0"]),
            (200, "C", "T", ["0|0"]),
            (300, "G", "A", ["0|0"]),
            (400, "T", "C", ["0|0"]),
        ]
        write_vcf(raw, ["ILL1", "ILL2", "ILL3"], raw_records)
        write_vcf(
            scaffold,
            ["PSI001_PSI001", "PSI002_PSI002", "PSI003_PSI003", "OTHER_OTHER"],
            scaffold_records,
        )
        write_vcf(baseline, ["Native-American_X"], baseline_records)
        write_metadata(metadata)

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
            "metadata": metadata,
            "preregistration": preregistration,
            "outdir": root / "out",
        }

    def test_direct_bridge_recomputes_minor_allele_and_preserves_privacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = self.make_inputs(Path(tmpdir))
            summary = AUDIT.run(type("Args", (), inputs)())
            self.assertEqual(summary["decision"], "GO_PC_RELATE_DONOR_AUDIT_ONLY")
            self.assertEqual(
                summary["gates"],
                {"B0": "PASS", "B1": "PASS", "B2": "PASS", "B3": "PASS", "B4": "PASS"},
            )

            rare = json.loads((inputs["outdir"] / "m27b_rare_support.json").read_text())
            self.assertEqual(rare["n_rare_sites"], 3)
            self.assertEqual(rare["n_rare_alt_major_sites"], 1)
            self.assertEqual(rare["n_rare_sites_with_one_carrier_individual"], 1)
            self.assertEqual(rare["n_rare_sites_with_at_least_two_carrier_individuals"], 2)
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
            self.assertNotIn("ILL1", emitted)

            identity = json.loads((inputs["outdir"] / "m27b_sample_identity.json").read_text())
            self.assertEqual(identity["n_raw_ids_present_in_scaffold"], 3)

    def test_unphased_scaffold_stops_before_pcrelate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = self.make_inputs(Path(tmpdir), scaffold_phased=False)
            summary = AUDIT.run(type("Args", (), inputs)())
            self.assertEqual(summary["decision"], "STOP_NO_DIRECT_RARE_PHASE_BRIDGE")
            self.assertEqual(summary["gates"]["B1"], "PASS")
            self.assertEqual(summary["gates"]["B3"], "FAIL")
            self.assertFalse(summary["pcrelate_executed"])

    def test_phase_is_not_evaluated_when_identity_aliases_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = self.make_inputs(Path(tmpdir))
            inputs["metadata"].write_text(
                "IID\tSample_ID(Aliases)\tIllumina_ID\toriginal_IID\n",
                encoding="utf-8",
            )
            summary = AUDIT.run(type("Args", (), inputs)())
            self.assertEqual(summary["decision"], "STOP_SAMPLE_IDENTITY")
            self.assertEqual(summary["gates"]["B1"], "FAIL")
            self.assertEqual(summary["gates"]["B3"], "NOT_EVALUABLE_UPSTREAM_IDENTITY")

    def test_nextflow_workflow_is_isolated_and_cloud_labeled(self) -> None:
        workflow = (ROOT / "workflows/m27b_rare_scaffold_bridge.nf").read_text(encoding="utf-8")
        module = (ROOT / "modules/27B_RARE_SCAFFOLD_BRIDGE.nf").read_text(encoding="utf-8")
        cloud = (ROOT / "conf/google_batch.config").read_text(encoding="utf-8")
        self.assertIn("AUDIT_RARE_SCAFFOLD_BRIDGE", workflow)
        self.assertIn("rare_scaffold_bridge_metadata", workflow)
        self.assertIn("--metadata ${metadata}", module)
        self.assertIn("--input ${metadata}", module)
        self.assertNotIn("PCRelate", module)
        self.assertNotIn("king", module.lower())
        self.assertNotIn("gnomix.py", module)
        self.assertNotIn("msprime", module)
        self.assertIn("resourceLabels = [team: 'frank']", cloud)
        self.assertIn("withName: 'AUDIT_RARE_SCAFFOLD_BRIDGE'", cloud)
        self.assertIn("disk = '20 GB'", cloud)


if __name__ == "__main__":
    unittest.main()
