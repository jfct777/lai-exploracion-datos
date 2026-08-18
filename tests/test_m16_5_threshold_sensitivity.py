#!/usr/bin/env python3
"""Contract and isolation tests for the separate M16.5 sensitivity run."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bin" / "evaluate_m16_5_threshold_sensitivity.py"
CONTRACT = REPO / "conf" / "m16_5_threshold_sensitivity_preregistration.json"
MODULE = REPO / "modules" / "16_5_THRESHOLD_SENSITIVITY.nf"
WORKFLOW = REPO / "workflows" / "m16_5_threshold_sensitivity.nf"


def load_module():
    spec = importlib.util.spec_from_file_location("m16_5_threshold_sensitivity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ThresholdSensitivityContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()
        cls.contract = cls.mod.load_contract(CONTRACT)
        cls.configs = cls.mod.build_configurations(cls.contract)

    def test_preregistered_grid_and_controls_are_exact(self):
        primary = [c for c in self.configs
                   if c["role"] in {"main_grid", "main_anchor"}]
        self.assertEqual(len(primary), 9)
        self.assertEqual(
            {(c["minimum_total_shared_bp"], c["minimum_longest_segment_bp"])
             for c in primary},
            {(edge, segment)
             for edge in (2_000_000, 3_000_000, 5_000_000)
             for segment in (1_000_000, 1_500_000, 2_000_000)},
        )
        roles = {c["role"] for c in self.configs}
        self.assertIn("identity_control", roles)
        self.assertIn("stress_control", roles)
        self.assertEqual(len(self.configs), 11)

    def test_neighbor_comparisons_cover_only_adjacent_grid_cells(self):
        pairs = self.mod.neighboring_grid_pairs(self.configs)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all("identity" not in left + right for left, right in pairs))
        self.assertTrue(all("stress" not in left + right for left, right in pairs))

    def test_identity_control_targets_the_five_megabase_anchor(self):
        identity = self.contract["configuration_design"]["identity_control"]
        self.assertEqual(identity["minimum_total_shared_bp"], 5_000_000)
        self.assertEqual(identity["minimum_longest_segment_bp"], 500_000)
        self.assertEqual(identity["must_match"], "edge5Mb_seg1Mb")
        self.assertEqual(self.contract["cohort"]["expected_pair_rows"], 54_522)
        self.assertEqual(
            self.contract["cohort"]["minimum_observed_longest_segment_bp"],
            1_000_000)

    def test_no_model_or_finestructure_selection_is_allowed(self):
        self.assertFalse(self.contract["scope"]["runs_nmf"])
        self.assertFalse(
            self.contract["scope"]["selects_configuration_by_finestructure"])
        self.assertFalse(
            self.contract["decision_rules"]["automatic_winner_selection"])

    def test_confounding_diagnostics_are_frozen_and_pcrelate_only(self):
        inputs = self.contract["inputs"]
        diagnostic = self.contract["fixed_parameters"]["ancestry_assignment_diagnostic"]
        self.assertEqual(len(inputs["autosomal_q_columns"]), 4)
        self.assertEqual(diagnostic["n_folds"], 5)
        self.assertEqual(diagnostic["evaluation"], "out_of_fold_roc_auc")
        self.assertEqual(
            self.contract["fixed_parameters"]["pcrelate"]["kinship_threshold"],
            0.0442)
        self.assertEqual(
            self.contract["fixed_parameters"]["pcrelate"]["expected_input_rows"],
            4_062_675)
        self.assertEqual(
            self.contract["fixed_parameters"]["pcrelate"][
                "expected_related_pairs_in_m14_observed_cohort"], 1_069)
        combined = "\n".join([
            SCRIPT.read_text(encoding="utf-8"),
            MODULE.read_text(encoding="utf-8"),
            WORKFLOW.read_text(encoding="utf-8"),
        ]).lower()
        self.assertIn("pcrelate", combined)
        self.assertNotIn("king", combined)

    def test_contract_rejects_duplicate_configuration_ids(self):
        altered = json.loads(CONTRACT.read_text(encoding="utf-8"))
        altered["configuration_design"]["stress_control"]["id"] = (
            altered["configuration_design"]["identity_control"]["id"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                self.mod.load_contract(path)

    def test_nextflow_path_is_separate_and_never_calls_all_mode(self):
        text = MODULE.read_text(encoding="utf-8") + WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("--mode all", text)
        self.assertNotIn("IBD_COMMUNITY_ENHANCED(", text)
        self.assertIn("RUN_M16_5_THRESHOLD_SENSITIVITY", text)
        self.assertIn("canonical M16.5 run directory is immutable", text)

    def test_config_does_not_modify_global_nextflow_configuration(self):
        dedicated = REPO / "conf" / "m16_5_threshold_sensitivity.config"
        self.assertTrue(dedicated.exists())
        self.assertIn("m16_5_sensitivity_results_dir = null",
                      dedicated.read_text(encoding="utf-8"))

    def test_google_batch_config_is_isolated_labeled_and_bounded(self):
        cloud = REPO / "conf" / "m16_5_threshold_sensitivity_google_batch.config"
        text = cloud.read_text(encoding="utf-8")
        self.assertIn("executor = 'google-batch'", text)
        self.assertIn("resourceLabels = [team: 'frank']", text)
        self.assertIn("maxForks = 2", text)
        self.assertIn("executor.queueSize = 2", text)
        self.assertIn("m16-5-threshold-sensitivity-20260818a", text)
        self.assertNotIn("m16-5-minor-20260806d", text)


if __name__ == "__main__":
    unittest.main()
