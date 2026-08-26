from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BoundaryWeightAmendmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent_path = ROOT / "conf" / "m33_pre4_preregistration.json"
        self.path = ROOT / "conf" / "m33_pre4a_boundary_weight_amendment.json"
        self.parent = json.loads(self.parent_path.read_text(encoding="utf-8"))
        self.value = json.loads(self.path.read_text(encoding="utf-8"))

    def test_delta_preserves_load_bearing_design(self) -> None:
        self.assertEqual(self.value["status"], "FROZEN_BEFORE_PRE4A_SCREEN")
        self.assertEqual(self.parent["root_registry"]["DEVELOPMENT"],
                         [386357765, 2024931463, 1324432253])
        self.assertEqual(self.parent["root_registry"]["EVAL_reserved_not_generated"],
                         [1341407242, 2049644864, 693524843, 1896826422, 166187460])
        self.assertEqual(self.value["screen"]["rotation"], "R0")
        self.assertTrue(self.value["screen"]["no_claim_from_screen"])

    def test_derived_beta_hits_requested_weight_share(self) -> None:
        counts = {"R0": 679, "R1": 645, "R2": 648}
        total = 9_574_800
        requested = (0.01, 0.05, 0.2)
        for rotation, transitions in counts.items():
            p = transitions / total
            betas = self.value["boundary_weight_parameterization"]["derived_beta_by_rotation"][rotation]
            self.assertEqual(betas[0], 0.0)
            for target, beta in zip(requested, betas[1:]):
                multiplier = 1.0 + beta
                observed = multiplier * p / ((1.0 - p) + multiplier * p)
                self.assertTrue(math.isclose(observed, target, rel_tol=0.0, abs_tol=1e-12))

    def test_screen_cardinality_and_stop_rule(self) -> None:
        screen = self.value["screen"]
        expected = (len(screen["families"]) * len(screen["radii_cM"]) *
                    len(screen["target_transition_weight_share_q"]))
        self.assertEqual(screen["candidate_count"], expected)
        self.assertEqual(screen["training_run_count"], 2 * expected)
        self.assertIn("RE_above_both_RD_and_F0", self.value["stop_rule"])


if __name__ == "__main__":
    unittest.main()
