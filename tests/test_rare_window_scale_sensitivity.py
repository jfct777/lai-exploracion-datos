from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_module("build_rare_window_features", ROOT / "bin/build_rare_window_features.py")
load_module("evaluate_fold_rare_window_pca", ROOT / "bin/evaluate_fold_rare_window_pca.py")
SCALE = load_module(
    "evaluate_rare_window_scale_sensitivity",
    ROOT / "bin/evaluate_rare_window_scale_sensitivity.py",
)
REFERENCE_SCHEME = SCALE.REFERENCE_SCHEME
aggregate_columns = SCALE.aggregate_columns
build_schemes = SCALE.build_schemes
equal_site_window_map = SCALE.equal_site_window_map
fixed_window_map = SCALE.fixed_window_map
paired_group_bootstrap = SCALE.paired_group_bootstrap
validate_preregistration = SCALE.validate_preregistration


def synthetic_windows() -> pd.DataFrame:
    rows = []
    selected = [10, 20, 0, 0, 8, 12, 30, 5]
    cohort = [12, 24, 0, 0, 10, 15, 35, 6]
    for index in range(8):
        rows.append(
            {
                "outer_fold": 0,
                "chrom": "chr22",
                "window_id": f"chr22_w{index:04d}",
                "start_0based": index * 250_000,
                "end_0based": (index + 1) * 250_000,
                "window_bp": 250_000,
                "n_input_cohort_rare_sites": cohort[index],
                "n_fold_train_rare_sites": selected[index],
            }
        )
    return pd.DataFrame(rows)


class RareWindowScaleSensitivityTest(unittest.TestCase):
    def test_fixed_half_step_grid_is_separate_phase(self) -> None:
        windows = synthetic_windows()
        phase0 = fixed_window_map(windows, 500_000, 0, "core")
        phase1 = fixed_window_map(windows, 500_000, 250_000, "boundary_phase")
        self.assertEqual(
            [row["source_indices"] for row in phase0],
            [(0, 1), (2, 3), (4, 5), (6, 7)],
        )
        self.assertEqual(
            [row["source_indices"] for row in phase1],
            [(1, 2), (3, 4), (5, 6), (7,)],
        )
        self.assertEqual(phase1[-1]["is_partial"], 1)
        self.assertTrue(
            set(phase0[0]["source_indices"]).isdisjoint(phase0[1]["source_indices"])
        )

    def test_equal_site_map_never_crosses_structural_gap(self) -> None:
        records = equal_site_window_map(synthetic_windows())
        informative = [row for row in records if not row["is_structural_gap"]]
        gap = [row for row in records if row["is_structural_gap"]]
        self.assertEqual(len(gap), 1)
        self.assertEqual(gap[0]["source_indices"], (2, 3))
        for row in informative:
            indices = set(row["source_indices"])
            self.assertFalse(indices & {0, 1} and indices & {4, 5, 6, 7})
        self.assertEqual(sum(row["n_fold_train_rare_sites"] for row in informative), 85)

    def test_aggregation_sums_counts_instead_of_averaging_rates(self) -> None:
        matrix = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        windows = synthetic_windows().iloc[:4].copy()
        records = fixed_window_map(windows, 500_000, 0, "core")
        observed = aggregate_columns(matrix, records)
        np.testing.assert_array_equal(observed, np.asarray([[3, 7], [11, 15]], dtype=float))

    def test_build_schemes_has_preregistered_order(self) -> None:
        schemes = build_schemes(synthetic_windows())
        self.assertEqual(
            tuple(schemes),
            (
                REFERENCE_SCHEME,
                "physical_500_o0",
                "physical_500_o250",
                "physical_1000_o0",
                "physical_1000_o500",
                "equal_site_approx",
            ),
        )

    def test_paired_group_bootstrap_is_deterministic_and_keeps_all_schemes(self) -> None:
        rows = []
        schemes = (REFERENCE_SCHEME, "physical_500_o0")
        for fold in (0, 1):
            for group in (f"f{fold}a", f"f{fold}b"):
                for scheme_index, scheme in enumerate(schemes):
                    rows.append(
                        {
                            "outer_fold": fold,
                            "split_group_key": group,
                            "scheme": scheme,
                            "rank": 1,
                            "pca_sse": 8.0 - scheme_index,
                            "burden_sse": 10.0,
                            "pca_weighted_sse": 16.0 - 2 * scheme_index,
                            "burden_weighted_sse": 20.0,
                        }
                    )
        frame = pd.DataFrame(rows)
        first = paired_group_bootstrap(frame, schemes, (1,), 100, 17)
        second = paired_group_bootstrap(frame, schemes, (1,), 100, 17)
        pd.testing.assert_frame_equal(first, second)
        self.assertAlmostEqual(
            first.set_index("scheme").loc[REFERENCE_SCHEME, "observed_skill_vs_burden"],
            0.2,
        )
        self.assertGreater(
            first.set_index("scheme").loc["physical_500_o0", "delta_skill_vs_reference"],
            0,
        )

    def test_preregistration_rejects_scheme_drift(self) -> None:
        source = Path("conf/m25c_scale_sensitivity_preregistration.json")
        self.assertEqual(
            validate_preregistration(source)["stage"],
            "M25C_RARE_WINDOW_SCALE_SENSITIVITY",
        )
        data = json.loads(source.read_text(encoding="utf-8"))
        data["fixed_schemes"][1]["id"] = "changed"
        with tempfile.TemporaryDirectory() as tmpdir:
            changed = Path(tmpdir) / "changed.json"
            changed.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "scheme list"):
                validate_preregistration(changed)


if __name__ == "__main__":
    unittest.main()
