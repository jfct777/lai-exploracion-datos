from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_m16_target_viability", ROOT / "bin/audit_m16_target_viability.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def prereg(expected_n: int = 8) -> dict:
    data = json.loads(
        (ROOT / "conf/m26_m16_target_audit_preregistration.json").read_text(encoding="utf-8")
    )
    data["population"]["expected_train_validation_samples"] = expected_n
    return data


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_ids = [f"S{i}" for i in range(10)]
    folds = [0, 1, 2, 4, 0, 1, 2, 4, 3, 3]
    master = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "community_res_1": [1, 1, 1, 1, 2, 2, -1, -1, 1, -1],
            "Q_NAM": [0.1] * 10,
            "Q_EUR": [0.6] * 10,
            "Q_EAS": [0.0] * 10,
            "Q_AFR": [0.3] * 10,
            "cohort": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
            "region": ["R1"] * 10,
            "state": ["ST1"] * 10,
            "rare_density": [0.01] * 10,
            "rare_carrier_site_count": [10] * 10,
            "rare_gt_nonmissing_sites": [100] * 10,
            "rare_missing_sites": [0] * 10,
            "qc_red": [False] * 10,
        }
    )
    split = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "eligible": [True] * 10,
            "fold": folds,
            "split": ["TRAIN"] * 8 + ["TEST"] * 2,
            "split_group_key": [f"G{i}" for i in range(10)],
            "y": [1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        }
    )
    minor = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "community_res_1": [1, 1, 1, 1, -1, -1, -1, -1, 1, -1],
        }
    )
    return minor, master, split


class M16TargetViabilityTest(unittest.TestCase):
    def test_prepare_analysis_excludes_test_and_defines_disjoint_states(self) -> None:
        minor, master, split = synthetic_inputs()
        data, integrity = AUDIT.prepare_analysis(minor, master, split, prereg())
        self.assertEqual(len(data), 8)
        self.assertNotIn(3, set(data["fold"]))
        self.assertEqual(integrity["n_forbidden_test_rows_analyzed"], 0)
        self.assertEqual(
            data["target_state"].value_counts().to_dict(),
            {"minor_assigned": 4, "historical_lost": 2, "nonassigned": 2},
        )

    def test_effective_groups_detect_family_concentration(self) -> None:
        frame = pd.DataFrame({"split_group_key": ["A", "A", "A", "B"]})
        observed = AUDIT.effective_group_metrics(frame)
        self.assertAlmostEqual(observed["effective_groups"], 1.6)
        self.assertAlmostEqual(observed["max_group_share"], 0.75)

    def test_missing_community_support_closes_target(self) -> None:
        minor, master, split = synthetic_inputs()
        minor.loc[minor["sample_id"].eq("S4"), "community_res_1"] = 2
        data, _ = AUDIT.prepare_analysis(minor, master, split, prereg())
        fold_support, community_support = AUDIT.support_tables(data)
        continuous = AUDIT.continuous_effects(data)
        categorical = AUDIT.categorical_composition(data, suppress_n=1)
        verdict, reasons, _ = AUDIT.decide(
            data, fold_support, community_support, continuous, categorical, prereg()
        )
        self.assertEqual(verdict, "CLOSE_INTERNAL_TARGET")
        self.assertIn("COMMUNITY_ABSENT_FROM_ASSESSMENT_FOLD", reasons)

    def test_categorical_small_cells_are_suppressed(self) -> None:
        minor, master, split = synthetic_inputs()
        data, _ = AUDIT.prepare_analysis(minor, master, split, prereg())
        observed = AUDIT.categorical_composition(data, suppress_n=5)
        self.assertTrue(observed["suppressed"].any())
        self.assertFalse(observed.loc[observed["suppressed"], "category"].str.contains("ST1").any())

    def test_preregistration_rejects_test_fold_drift(self) -> None:
        data = prereg()
        data["population"]["forbidden_outer_fold"] = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "fold 3"):
                AUDIT.load_preregistration(path)

    def test_nextflow_contract_is_isolated_and_labeled(self) -> None:
        workflow = (ROOT / "workflows/m26_m16_target_viability.nf").read_text(encoding="utf-8")
        cloud = (ROOT / "conf/google_batch.config").read_text(encoding="utf-8")
        self.assertIn("AUDIT_M16_TARGET_VIABILITY", workflow)
        for forbidden in ("EVALUATE_TEST", "RARE_BENCH", "NMF", "IBD_COMMUNITY_ENHANCED"):
            self.assertNotIn(forbidden, workflow)
        self.assertIn("resourceLabels = [team: 'frank']", cloud)
        self.assertIn("withName: 'AUDIT_M16_TARGET_VIABILITY'", cloud)


if __name__ == "__main__":
    unittest.main()
