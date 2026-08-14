import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from reconcile_m27c_header_contract import reconcile  # noqa: E402


class TestReconcileM27CHeaderContract(unittest.TestCase):
    def setUp(self):
        self.full_summary = {
            "gates": {"C0": "FAIL", "C1": "PASS", "C2": "PASS", "C3": "PASS"},
            "primary_candidate_panel_ready_fraction": 0.843,
            "minimum_marker_fraction": 0.8,
            "robustness_classification": "PASS_THRESHOLD_SENSITIVE",
        }
        self.full_input = {
            "n_gvcf": 128,
            "gcs_input_manifest": {"sha256": "same"},
        }
        self.header = {
            "expected_samples": 128,
            "input_manifest_sha256": "same",
            "header_contract_pass": True,
        }

    def test_reconciles_only_c0_and_preserves_threshold_review(self):
        result = reconcile(self.full_summary, self.full_input, self.header)
        self.assertEqual(result["gates"]["C0"], "PASS")
        self.assertEqual(result["decision"], "REVIEW_THRESHOLD_SENSITIVITY")
        self.assertTrue(result["c1_c2_c3_reused_without_recomputation"])

    def test_rejects_different_input_manifest(self):
        self.header["input_manifest_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different input manifests"):
            reconcile(self.full_summary, self.full_input, self.header)


if __name__ == "__main__":
    unittest.main()
