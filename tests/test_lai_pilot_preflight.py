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
    "audit_lai_pilot_preflight", ROOT / "bin/audit_lai_pilot_preflight.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def write_vcf(path: Path, samples: list[str], records: list[tuple]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(samples) + "\n")
        for pos, ref, alt, af, genotypes in records:
            handle.write(
                f"chr22\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\tAF={af}\tGT\t"
                + "\t".join(genotypes)
                + "\n"
            )


class LaiPilotPreflightTest(unittest.TestCase):
    def make_inputs(self, root: Path) -> dict[str, Path]:
        baseline = root / "baseline.vcf.gz"
        external = root / "external.vcf.gz"
        write_vcf(
            baseline,
            ["African_A", "Native-American_HGDP00982"],
            [
                (100, "A", "G", 0.5, ["0|0", "0|1"]),
                (200, "C", "T", 0.5, ["0|1", "1|1"]),
                (300, "G", "A", 0.5, ["1|1", "0|0"]),
                (400, "T", "C", 0.5, ["0|1", "0|1"]),
            ],
        )
        write_vcf(
            external,
            ["A_A", "HG00982_HG00982", "PSI001_PSI001"],
            [
                (100, "A", "G", 0.005, ["0|0", "0|1", "0|0"]),
                (200, "C", "T", 0.2, ["0|1", "1|1", "0|0"]),
                (300, "G", "A", 0.4, ["1|1", "0|0", "0|1"]),
            ],
        )
        model = root / "model.pkl"
        model.write_bytes(b"frozen-model")
        config = root / "config.txt"
        config.write_text("A\t3\nC\t4\ncalibrate\tFalse\n", encoding="utf-8")
        genetic_map = root / "map.txt"
        genetic_map.write_text("22 100 0.1\n22 200 0.2\n22 400 0.4\n", encoding="utf-8")
        metadata = root / "metadata.tsv"
        metadata.write_text(
            "Population\tIID\tAncestry\tSource\tExclude\tMaximum_unrelated_dataset\tMaximum_unrelated_dataset_2nd\n"
            "Yoruba\tA\tAfrican\tTEST\tFALSE\tTRUE\tTRUE\n"
            "Tupiniquim\tPSI0001\tNative_American\tPSI\tFALSE\tTRUE\tTRUE\n"
            "French\tE1\tEuropean\tTEST\tFALSE\tTRUE\tTRUE\n",
            encoding="utf-8",
        )
        top95 = root / "top95.tsv"
        top95.write_text(
            "IID\tAfrican\tAmerican\tEuropean\nPSI001\t0.0\t0.99\t0.01\n",
            encoding="utf-8",
        )
        keep = root / "keep.txt"
        keep.write_text("PSI0001 PSI0001\n", encoding="utf-8")
        prereg = root / "prereg.json"
        prereg.write_text(
            (ROOT / "conf/m27_lai_pilot_preflight_preregistration.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return {
            "gnomix_reference_vcf": baseline,
            "external_panel_vcf": external,
            "gnomix_model": model,
            "gnomix_config": config,
            "genetic_map": genetic_map,
            "metadata": metadata,
            "top95_nam": top95,
            "nam_unrelated_keep": keep,
            "preregistration": prereg,
            "outdir": root / "out",
        }

    def test_fail_closed_below_marker_floor_and_fingerprints_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = self.make_inputs(Path(tmpdir))
            summary = AUDIT.run(type("Args", (), inputs)())
            self.assertEqual(summary["decision"], "STOP_BASELINE_NOT_EXECUTABLE_FOR_SCIENTIFIC_PILOT")
            self.assertEqual(summary["gates"]["G0"], "PASS")
            self.assertEqual(summary["gates"]["G2"], "FAIL")
            self.assertFalse(summary["simulation_performed"])
            g1 = json.loads((inputs["outdir"] / "g1_donor_identity_and_parentals.json").read_text())
            self.assertEqual(g1["configured_ambiguous_identity_fingerprints"][0]["status"], "MATCH")
            self.assertNotIn("HGDP00982", json.dumps(g1))

    def test_vcf_audit_detects_out_of_order_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unordered.vcf.gz"
            write_vcf(
                path,
                ["S_S"],
                [(200, "A", "G", 0.1, ["0|1"]), (100, "C", "T", 0.1, ["0|0"])],
            )
            observed = AUDIT.audit_vcf(path)
            self.assertFalse(observed.ordered)

    def test_nextflow_preflight_is_isolated_and_cloud_labeled(self) -> None:
        workflow = (ROOT / "workflows/m27_lai_pilot_preflight.nf").read_text(encoding="utf-8")
        module = (ROOT / "modules/27_LAI_PILOT_PREFLIGHT.nf").read_text(encoding="utf-8")
        cloud = (ROOT / "conf/google_batch.config").read_text(encoding="utf-8")
        self.assertIn("AUDIT_LAI_PILOT_PREFLIGHT", workflow)
        self.assertNotIn("gnomix.py", module)
        self.assertNotIn("haptools", module)
        self.assertIn("resourceLabels = [team: 'frank']", cloud)
        self.assertIn("withName: 'AUDIT_LAI_PILOT_PREFLIGHT'", cloud)


if __name__ == "__main__":
    unittest.main()
