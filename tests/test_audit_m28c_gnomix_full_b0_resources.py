"""Known-answer tests for the protected full-B0 resource gate."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))
SPEC = importlib.util.spec_from_file_location(
    "audit_m28c_gnomix_full_b0_resources",
    REPO / "bin" / "audit_m28c_gnomix_full_b0_resources.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestFullResourceGate(unittest.TestCase):
    def write_fixture(self, root: Path, peak_rss: str = "13.6 GB") -> tuple[Path, Path, Path, Path]:
        dimensions = {
            "C": 79791,
            "M": 215,
            "W": 371,
            "A": 3,
            "S": 75,
            "context_markers_each_side": 107,
            "remainder_markers": 26,
            "terminal_window_markers": 241,
            "modeled_markers": 79791,
        }
        contract = root / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "stage": "M28C_GNOMIX_FULL_B0_RESOURCE_BENCHMARK",
                    "status": "PRE_FROZEN_BEFORE_FULL_B0",
                    "gnomix_parameters": {"derived_expected": dimensions},
                    "source_panel": {"target_samples": 30},
                }
            ),
            encoding="utf-8",
        )
        contract_hash = MODULE.sha256(contract)
        core = {key: dimensions[key] for key in ("C", "M", "W", "A", "S", "context_markers_each_side")}
        train = root / "train.json"
        train.write_text(
            json.dumps(
                {
                    "replicate": "A",
                    "decision": "GO_FROZEN_MODEL_INFERENCE_NO_TRUTH",
                    "contract_sha256": contract_hash,
                    "model_audit": {"derived_dimensions": core},
                    "target_input_present": False,
                    "truth_accessed": False,
                }
            ),
            encoding="utf-8",
        )
        infer = root / "infer.json"
        infer.write_text(
            json.dumps(
                {
                    "replicate": "A",
                    "decision": "GO_REPLICATE_COMPARISON_NO_TRUTH",
                    "contract_sha256": contract_hash,
                    "train_report_sha256": MODULE.sha256(train),
                    "prediction_audit": {
                        "target_samples": 30,
                        "windows": 371,
                        "msp_marker_count_sum": 79791,
                        "msp_terminal_window_markers": 241,
                        "population_order": ["AFR", "EUR", "ASIA"],
                    },
                    "truth_accessed": False,
                    "target_truth_accuracy_computed": False,
                }
            ),
            encoding="utf-8",
        )
        trace = root / "trace.tsv"
        trace.write_text(
            "name\tstatus\texit\tduration\tpeak_rss\n"
            f"TRAIN_M28C_GNOMIX_FULL_B0 (m28c_gnomix_full_b0_train_A)\tCOMPLETED\t0\t6m 18s\t{peak_rss}\n",
            encoding="utf-8",
        )
        return trace, train, infer, contract

    def test_a_pass_authorizes_b(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.write_fixture(Path(name))
            report = MODULE.audit(*inputs, replicate="A")
        self.assertEqual(report["decision"], "GO_LAUNCH_FULL_B0_REPLICATE_B")
        self.assertTrue(report["gates"]["F2_RESIDUAL_POLICY"])
        self.assertTrue(report["gates"]["F6_RESOURCES"])

    def test_a_memory_review_stops_before_b(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.write_fixture(Path(name), peak_rss="22 GiB")
            report = MODULE.audit(*inputs, replicate="A")
        self.assertEqual(report["decision"], "STOP_BEFORE_REPLICATE_B")
        self.assertFalse(report["gates"]["F6_RESOURCES"])


if __name__ == "__main__":
    unittest.main()
