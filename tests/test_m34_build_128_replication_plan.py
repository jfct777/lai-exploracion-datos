import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m34_build_128_replication_plan import build_plan  # noqa: E402


class ReplicationPlanTest(unittest.TestCase):
    def audit_payload(self) -> dict:
        return {
            "stage": "M34_RARE_LOCUS_DISTRIBUTION_AUDIT",
            "status": "PASS_DESCRIPTIVE_AUDIT_NO_MODEL_SELECTION",
            "selection": {
                "selected_loci": 660,
                "minor_alt_loci": 373,
                "minor_ref_loci": 287,
            },
            "ancestry": {
                ancestry: {"minimum_callability": 1.0}
                for ancestry in ("AFR", "EUR", "NAM")
            },
            "nam_enrichment": {
                "loci_nam_af_ge_0_05_and_afr_eur_below_0_01": 211,
                "of_these_in_ge_2_nam_units": 199,
                "of_these_in_ge_3_nam_units": 120,
            },
        }

    def test_plan_has_four_paired_tasks_per_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.json"
            audit.write_text(json.dumps(self.audit_payload()), encoding="utf-8")
            plan = build_plan(ROOT / "conf/m34_adaptive_sweep_contract.json", audit)
        self.assertEqual(plan["task_count"], 12)
        self.assertFalse(plan["test_opened"])
        self.assertEqual(plan["warmup_updates"], 400)
        self.assertEqual(plan["validation_every_updates"], 200)
        for root in ("R0", "R1", "R2"):
            tasks = [task for task in plan["tasks"] if task["rotation"] == root]
            self.assertEqual(len(tasks), 4)
            self.assertEqual({task["config_id"] for task in tasks},
                             {"bilstm_r1", "unet_r1"})
            self.assertEqual({task["arm"] for task in tasks}, {"RD", "RE"})
            self.assertEqual({task["maximum_updates"] for task in tasks}, {3200})
            self.assertEqual({task["sweep_stage"] for task in tasks}, {"replication_128"})
            self.assertEqual({task["radius_cM"] for task in tasks}, {0.2})

    def test_orientation_mismatch_stops_plan(self) -> None:
        payload = self.audit_payload()
        payload["selection"]["minor_ref_loci"] = 286
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.json"
            audit.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "orientation counts"):
                build_plan(ROOT / "conf/m34_adaptive_sweep_contract.json", audit)


if __name__ == "__main__":
    unittest.main()
