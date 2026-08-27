import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "bin" / "summarize_m34_nam_replication.py"
SPEC = importlib.util.spec_from_file_location("m34_summary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestM34NamReplicationSummary(unittest.TestCase):
    def test_parse_roots_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate root"):
                MODULE.parse_roots([f"R0={directory}", f"R0={directory}"])

    def test_guardrail_value_uses_primary_boundary_radius(self):
        payload = {
            "boundary": {"0.2": {"false_transitions_per_cM": 0.025}},
            "macro_ancestry_dose_MAE": 0.01,
        }
        self.assertEqual(MODULE.guardrail_value(payload, "false_transitions_per_cM"), 0.025)
        self.assertEqual(MODULE.guardrail_value(payload, "macro_ancestry_dose_MAE"), 0.01)

    def test_plot_is_dependency_free_svg(self):
        result = {
            "root_audit": [{"rotation": root} for root in ("R0", "R1", "R2")],
            "model_decisions": [{
                "family": family,
                "RE_minus_RD_F1_0.2cM": {"per_root": {root: value for root in ("R0", "R1", "R2")}},
                "RE_minus_F0_F1_0.2cM": {"per_root": {root: -value for root in ("R0", "R1", "R2")}},
            } for family, value in (("bilstm", 0.001), ("unet_1d", 0.002))],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "effects.svg"
            MODULE.plot_summary(result, output)
            svg = output.read_text(encoding="utf-8")
        self.assertIn("<svg", svg)
        self.assertIn("mínimo preregistrado", svg)

    def test_frozen_real_replication_stops_both_models(self):
        real = Path("/tmp/m34-nam-128-post-20260827a.qgLqaQ")
        plan = Path("/tmp/m34-128-pre-20260827a/m34_128.replication.plan.json")
        if not real.is_dir() or not plan.is_file():
            self.skipTest("real M34 replication artifacts are not present")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = MODULE.summarize(Namespace(
                plan=plan,
                contract=Path(__file__).parents[1] / "conf" / "m34_adaptive_sweep_contract.json",
                root=[f"R0={real / 'r0'}", f"R1={real / 'r1'}", f"R2={real / 'r2'}"],
                output_json=output / "summary.json",
                output_tsv=output / "summary.tsv",
            ))
        self.assertEqual(result["status"], "STOP_FROZEN_DIPLOID_FINALISTS")
        self.assertEqual(len(result["rows"]), 6)
        self.assertTrue(all(not model["promotion_pass"] for model in result["model_decisions"]))
        self.assertTrue(all(audit["valid_sample_count"] == 32 for audit in result["root_audit"]))
        self.assertFalse(result["test_opened"])


if __name__ == "__main__":
    unittest.main()
