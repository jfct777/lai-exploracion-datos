"""Known-answer tests for the M28C Gnomix training smoke."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m28c_gnomix_training_smoke", REPO / "bin" / "m28c_gnomix_training_smoke.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestAllocation(unittest.TestCase):
    def test_reserves_every_bin_and_is_deterministic(self):
        markers = []
        for bin_index, count in enumerate((2, 4, 6)):
            for within_bin in range(count):
                markers.append(
                    {
                        "chrom": "chr22",
                        "position": 1000 * bin_index + within_bin,
                        "cm": 1.01 + 0.2 * bin_index + 0.01 * within_bin,
                    }
                )
        markers.sort(key=lambda marker: marker["position"])
        contract = {
            "subset": {
                "markers": 8,
                "bin_width_cm": 0.2,
                "bin_origin_cm": 1.0,
                "bins": 3,
            }
        }
        selected_a, audit_a = MODULE.allocate_subset(markers, contract)
        selected_b, audit_b = MODULE.allocate_subset(markers, contract)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(audit_a, audit_b)
        self.assertEqual(len(selected_a), 8)
        self.assertEqual({marker["bin_index"] for marker in selected_a}, {0, 1, 2})
        self.assertTrue(all(item["selected"] >= 1 for item in audit_a))

    def test_rejects_missing_bin(self):
        markers = [
            {"chrom": "chr22", "position": 1, "cm": 1.01},
            {"chrom": "chr22", "position": 2, "cm": 1.41},
        ]
        contract = {
            "subset": {
                "markers": 2,
                "bin_width_cm": 0.2,
                "bin_origin_cm": 1.0,
                "bins": 3,
            }
        }
        with self.assertRaisesRegex(ValueError, "missing bins"):
            MODULE.allocate_subset(markers, contract)


class TestPredictionParsers(unittest.TestCase):
    def test_parses_msp_and_fb(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            msp = root / "test.msp"
            msp.write_text(
                "#Subpopulation order/codes: AFR=0\tEUR=1\tASIA=2\n"
                "#chm\tspos\tepos\tsgpos\tegpos\tn snps\tT0.0\tT0.1\n"
                "22\t100\t200\t1.0\t1.2\t2\t0\t1\n",
                encoding="utf-8",
            )
            fb = root / "test.fb"
            fb.write_text(
                "#reference_panel_population:\tAFR\tEUR\tASIA\n"
                "chromosome\tphysical position\tgenetic_position\tgenetic_marker_index"
                "\tT0:::hap1:::AFR\tT0:::hap1:::EUR\tT0:::hap1:::ASIA"
                "\tT0:::hap2:::AFR\tT0:::hap2:::EUR\tT0:::hap2:::ASIA\n"
                "22\t150\t1.1\t.\t0.8\t0.1\t0.1\t0.2\t0.3\t0.5\n",
                encoding="utf-8",
            )
            parsed_msp = MODULE.parse_msp(msp)
            parsed_fb = MODULE.parse_fb(fb)
            self.assertEqual(parsed_msp["populations"], ["AFR", "EUR", "ASIA"])
            self.assertEqual(parsed_msp["header"][6:], ["T0.0", "T0.1"])
            self.assertEqual(parsed_fb["populations"], ["AFR", "EUR", "ASIA"])
            self.assertEqual(len(parsed_fb["probability_header"]), 6)


class TestContract(unittest.TestCase):
    def test_canonical_contig_equivalence(self):
        self.assertEqual(MODULE.canonical_autosome("chr22"), "22")
        self.assertEqual(MODULE.canonical_autosome("22"), "22")

    def test_frozen_scope_and_dimensions(self):
        contract = MODULE.load_contract(
            REPO / "conf" / "m28c_gnomix_training_smoke_preregistration.json"
        )
        self.assertEqual(contract["version"], 2)
        self.assertEqual(len(contract["amendment_history"]), 2)
        self.assertEqual(contract["subset"]["markers"], 10000)
        self.assertEqual(contract["subset"]["bins"], 363)
        self.assertEqual(
            contract["gnomix_parameters"]["derived_expected"],
            {"C": 10000, "M": 27, "W": 370, "A": 3, "S": 75, "context_markers_each_side": 13},
        )
        self.assertIn("target_vcf", contract["execution"]["training_forbidden_inputs"])
        self.assertIn("truth", contract["execution"]["training_forbidden_inputs"])
        self.assertIn("synthetic admixed haplotypes", contract["execution"]["internal_validation_policy"])
        self.assertEqual(contract["software"]["scikit_learn"], "1.7.2")
        self.assertEqual(
            contract["software"]["container_image"], "dnabr-m28c-gnomix:e758f2c-r4"
        )

    def test_config_hash_is_frozen(self):
        contract = json.loads(
            (REPO / "conf" / "m28c_gnomix_training_smoke_preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            MODULE.sha256(REPO / "conf" / "m28c_gnomix_training_smoke.yaml"),
            contract["authenticated_inputs"]["gnomix_config_sha256"],
        )

    def test_full_b0_dimensions_and_residual_policy(self):
        contract = MODULE.load_contract(
            REPO / "conf" / "m28c_gnomix_full_b0_preregistration.json"
        )
        with tempfile.TemporaryDirectory() as name:
            genetic_map = Path(name) / "map.tsv"
            genetic_map.write_text("22\t1\t0.0\n22\t50791300\t74.109562\n", encoding="utf-8")
            observed = MODULE.calculate_dimensions(79791, genetic_map, contract)
        self.assertEqual(observed, contract["gnomix_parameters"]["derived_expected"])
        self.assertEqual(observed["terminal_window_markers"], 241)
        self.assertEqual(observed["modeled_markers"], 79791)
        self.assertEqual(MODULE.report_stem(contract), "m28c_gnomix_full_b0")
        self.assertEqual(
            MODULE.sha256(REPO / "conf" / "m28c_gnomix_full_b0.yaml"),
            contract["authenticated_inputs"]["gnomix_config_sha256"],
        )


class TestFullComparison(unittest.TestCase):
    def write_inference(self, root: Path, replicate: str) -> tuple[Path, Path]:
        inference = root / f"inference_{replicate}"
        results = inference / "results"
        results.mkdir(parents=True)
        (results / "query_results.msp").write_text(
            "#Subpopulation order/codes: AFR=0\tEUR=1\tASIA=2\n"
            "#chm\tspos\tepos\tsgpos\tegpos\tn snps\tT0.0\tT0.1\n"
            "22\t100\t200\t1.0\t1.2\t2\t0\t1\n",
            encoding="utf-8",
        )
        (results / "query_results.fb").write_text(
            "#reference_panel_population:\tAFR\tEUR\tASIA\n"
            "chromosome\tphysical position\tgenetic_position\tgenetic_marker_index"
            "\tT0:::hap1:::AFR\tT0:::hap1:::EUR\tT0:::hap1:::ASIA\n"
            "22\t150\t1.1\t.\t0.8\t0.1\t0.1\n",
            encoding="utf-8",
        )
        report = root / f"inference_{replicate}.json"
        contract_hash = MODULE.sha256(
            REPO / "conf" / "m28c_gnomix_full_b0_preregistration.json"
        )
        report.write_text(
            json.dumps(
                {
                    "replicate": replicate,
                    "decision": "GO_REPLICATE_COMPARISON_NO_TRUTH",
                    "contract_sha256": contract_hash,
                }
            ),
            encoding="utf-8",
        )
        return inference, report

    def write_training_report(self, root: Path, replicate: str, data_hash: str) -> Path:
        report = root / f"training_{replicate}.json"
        report.write_text(
            json.dumps(
                {
                    "replicate": replicate,
                    "decision": "GO_FROZEN_MODEL_INFERENCE_NO_TRUTH",
                    "contract_sha256": MODULE.sha256(
                        REPO / "conf" / "m28c_gnomix_full_b0_preregistration.json"
                    ),
                    "generated_data_sha256": {"generated_data/matrix.npy": data_hash},
                }
            ),
            encoding="utf-8",
        )
        return report

    def arguments(self, root: Path, hash_b: str = "same") -> SimpleNamespace:
        inference_a, report_a = self.write_inference(root, "A")
        inference_b, report_b = self.write_inference(root, "B")
        train_a = self.write_training_report(root, "A", "same")
        train_b = self.write_training_report(root, "B", hash_b)
        contract_hash = MODULE.sha256(
            REPO / "conf" / "m28c_gnomix_full_b0_preregistration.json"
        )

        def write_gate(replicate: str, decision: str, train: Path, infer: Path) -> Path:
            path = root / f"resource_gate_{replicate}.json"
            path.write_text(
                json.dumps(
                    {
                        "replicate": replicate,
                        "decision": decision,
                        "contract_sha256": contract_hash,
                        "train_report_sha256": MODULE.sha256(train),
                        "inference_report_sha256": MODULE.sha256(infer),
                        "gates": {"F0": True, "F6": True},
                    }
                ),
                encoding="utf-8",
            )
            return path

        return SimpleNamespace(
            preregistration=REPO / "conf" / "m28c_gnomix_full_b0_preregistration.json",
            inference_a=inference_a,
            report_a=report_a,
            inference_b=inference_b,
            report_b=report_b,
            train_report_a=train_a,
            train_report_b=train_b,
            resource_gate_a=write_gate(
                "A", "GO_LAUNCH_FULL_B0_REPLICATE_B", train_a, report_a
            ),
            resource_gate_b=write_gate(
                "B", "GO_COMPARE_FULL_B0_REPLICATES", train_b, report_b
            ),
            outdir=root / "comparison",
        )

    def test_full_comparison_requires_identical_generated_training_data(self):
        with tempfile.TemporaryDirectory() as name:
            report = MODULE.comparison(self.arguments(Path(name)))
        self.assertTrue(report["generated_training_data_hashes_exact"])
        self.assertTrue(report["gates"]["T6_REPRODUCIBILITY"])
        self.assertTrue(report["gates"]["T7_RESOURCES"])
        self.assertEqual(report["decision"], "PASS_FULL_B0_TECHNICAL_BENCHMARK")

    def test_full_comparison_rejects_different_generated_training_data(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, "reproducibility failed"):
                MODULE.comparison(self.arguments(Path(name), hash_b="different"))


if __name__ == "__main__":
    unittest.main()
