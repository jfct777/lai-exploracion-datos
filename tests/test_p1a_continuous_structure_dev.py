from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p1a_dev", ROOT / "bin/run_p1a_continuous_structure_dev.py"
)
DEV = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEV)


def contract() -> dict:
    return json.loads(
        (ROOT / "conf/p1a_continuous_structure_preregistration.json").read_text(
            encoding="utf-8"
        )
    )


def graph_fixture() -> tuple[pd.DataFrame, set[str], set[str], set[str]]:
    anchors = {f"t{x}" for x in range(14)}
    rows = []
    ordered = sorted(anchors)
    # Connected, non-regular TRAIN graph with enough nodes for k=d+1.
    for index, left in enumerate(ordered):
        right = ordered[(index + 1) % len(ordered)]
        rows.append((left, right, 2_000_000 + 10_000 * index))
        if index % 2 == 0:
            rows.append((left, ordered[(index + 3) % len(ordered)], 3_000_000 + index))
    rows.extend([("e0", "t1", 4_000_000), ("e1", "t2", 5_000_000)])
    edges = pd.DataFrame(rows, columns=["sample_a", "sample_b", "total_shared_bp"])
    edges["n_shared_variants_total"] = 2
    return edges, anchors, {"t0", "t1", "t2"}, {"e0", "e1"}


class P1AContinuousStructureDevTest(unittest.TestCase):
    def test_hash_contract_rejects_wrong_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.tsv"
            path.write_text("x\n", encoding="utf-8")
            cfg = {"inputs": {"expected_sha256": {"pairs": "0" * 64}}}
            with self.assertRaisesRegex(DEV.ContractError, "sha256"):
                DEV.verify_hashes({"pairs": path}, cfg)

    def test_nystrom_ignores_eval_eval_edges(self) -> None:
        edges, anchors, train_targets, eval_targets = graph_fixture()
        base, base_diag = DEV.spectral_fold_features(
            edges, anchors, train_targets, eval_targets, "binary", 2, contract()
        )
        changed = pd.concat(
            [
                edges,
                pd.DataFrame(
                    {
                        "sample_a": ["e0"],
                        "sample_b": ["e1"],
                        "total_shared_bp": [999_000_000],
                        "n_shared_variants_total": [999],
                    }
                ),
            ],
            ignore_index=True,
        )
        observed, observed_diag = DEV.spectral_fold_features(
            changed, anchors, train_targets, eval_targets, "binary", 2, contract()
        )
        pd.testing.assert_frame_equal(base, observed, check_exact=False, atol=1e-11, rtol=1e-11)
        self.assertEqual(base_diag["w_eval_eval_used"], 0)
        self.assertEqual(observed_diag["w_eval_eval_used"], 0)

    def test_edge_order_and_orientation_do_not_change_features(self) -> None:
        edges, anchors, train_targets, eval_targets = graph_fixture()
        expected, _ = DEV.spectral_fold_features(
            edges, anchors, train_targets, eval_targets, "binary", 2, contract()
        )
        reversed_edges = edges.rename(
            columns={"sample_a": "sample_b", "sample_b": "sample_a"}
        )[["sample_a", "sample_b", "total_shared_bp", "n_shared_variants_total"]]
        reversed_edges = reversed_edges.sample(frac=1.0, random_state=19).reset_index(drop=True)
        observed, _ = DEV.spectral_fold_features(
            reversed_edges, anchors, train_targets, eval_targets, "binary", 2, contract()
        )
        expected = expected.sort_values("sample_id").reset_index(drop=True)
        observed = observed.sort_values("sample_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, observed, check_exact=False, atol=1e-10, rtol=1e-10)

    def test_gcc_gate_rejects_too_many_unprojectable_targets(self) -> None:
        edges, anchors, _, eval_targets = graph_fixture()
        # Add six anchor isolates; three of four target TRAIN nodes are outside GCC.
        isolated = {f"iso{x}" for x in range(6)}
        with self.assertRaisesRegex(DEV.ContractError, "GCC/projectability"):
            DEV.spectral_fold_features(
                edges,
                anchors | isolated,
                {"t0", "iso0", "iso1", "iso2"},
                eval_targets,
                "binary",
                2,
                contract(),
            )

    def test_linear_spectral_block_is_rotation_invariant(self) -> None:
        rng = np.random.default_rng(7)
        n = 120
        frame = pd.DataFrame(
            {
                "region": np.repeat(["A", "B", "C"], n // 3),
                "Q_NAM": rng.uniform(0.01, 0.2, n),
                "Q_EUR": rng.uniform(0.55, 0.8, n),
                "Q_EAS": rng.uniform(0.001, 0.02, n),
                "Q_AFR": rng.uniform(0.05, 0.3, n),
                "burden": rng.uniform(0.001, 0.01, n),
                "log1p_missing": rng.uniform(0, 5, n),
                "log1p_degree_to_gcc": rng.uniform(0, 4, n),
                "log1p_weight_strength_to_gcc": rng.uniform(0, 4, n),
                "projectable": np.ones(n),
                "z1": rng.normal(size=n),
                "z2": rng.normal(size=n),
            }
        )
        q = np.linalg.qr(rng.normal(size=(2, 2)))[0]
        rotated = frame.copy()
        rotated[["z1", "z2"]] = frame[["z1", "z2"]].to_numpy() @ q
        train, evaluation = frame.iloc[:90], frame.iloc[90:]
        rotated_train, rotated_eval = rotated.iloc[:90], rotated.iloc[90:]
        classes = ["A", "B", "C"]
        expected = DEV.fit_predict(train, evaluation, "A", 2, 1.0, classes, contract())
        observed = DEV.fit_predict(
            rotated_train, rotated_eval, "A", 2, 1.0, classes, contract()
        )
        np.testing.assert_allclose(expected, observed, atol=2e-6, rtol=2e-6)

    def test_eigenvector_sign_is_canonical(self) -> None:
        values = np.asarray([[-0.2, 0.1], [0.8, -0.7], [0.1, 0.2]])
        observed = DEV.canonicalize_eigenvectors(values)
        self.assertGreater(observed[1, 0], 0)
        self.assertGreater(observed[1, 1], 0)

    def test_one_se_prefers_stronger_ridge_and_smaller_dimension(self) -> None:
        scores = {
            ("A", 0.01, 2): [1.03, 1.01, 1.02],
            ("A", 0.01, 4): [1.02, 1.00, 1.01],
            ("A", 1.0, 8): [0.96, 1.00, 1.04],
        }
        selected, threshold = DEV.select_one_standard_error(scores, "A")
        self.assertEqual(selected, ("A", 0.01, 2))
        self.assertGreaterEqual(threshold, 1.02)

    def test_nextflow_manifest_receives_complete_provenance(self) -> None:
        module = (ROOT / "modules/P1A_CONTINUOUS_STRUCTURE_DEV.nf").read_text(encoding="utf-8")
        workflow = (ROOT / "workflows/p1a_continuous_structure_dev.nf").read_text(
            encoding="utf-8"
        )
        self.assertIn("--provenance-b64 '${provenance_b64}'", module)
        self.assertIn("container_path   : params.p1a_container_image", workflow)
        self.assertIn("container_sha256 : params.p1a_container_digest", workflow)
        self.assertIn("p1a_region_state_counts.tsv", module)
        self.assertIn("--input ${runner_py} --input ${manifest_py}", module)

    def test_reserved_fold_and_primary_target_are_frozen(self) -> None:
        cfg = contract()
        self.assertFalse(cfg["scope"]["uses_reserved_fold_3"])
        self.assertEqual(cfg["population"]["reserved_fold"], 3)
        self.assertEqual(cfg["population"]["development_folds"], [0, 1, 2, 4])
        self.assertEqual(
            cfg["population"]["primary_regions"],
            ["NORTHEASTERN", "SOUTHEASTERN", "SOUTHERN"],
        )
        self.assertEqual(cfg["population"]["expected_primary_target_samples"], 212)
        self.assertEqual(
            cfg["population"]["expected_primary_region_counts"],
            {"NORTHEASTERN": 39, "SOUTHEASTERN": 137, "SOUTHERN": 36},
        )
        self.assertEqual(
            cfg["population"]["expected_primary_fold_counts"],
            {"0": 66, "1": 50, "2": 41, "4": 55},
        )
        self.assertEqual(
            cfg["population"]["expected_primary_fold_region_counts"]["2"],
            {"NORTHEASTERN": 5, "SOUTHEASTERN": 29, "SOUTHERN": 7},
        )

    def test_secondary_metrics_reward_perfect_predictions(self) -> None:
        classes = ["A", "B", "C"]
        y = np.asarray(classes, dtype=object)
        probabilities = np.eye(3)
        self.assertEqual(DEV.balanced_accuracy(y, probabilities, classes), 1.0)
        self.assertEqual(DEV.macro_brier(y, probabilities, classes), 0.0)

    def test_exact_graph_strength_matches_weight_mode(self) -> None:
        edges, anchors, train_targets, eval_targets = graph_fixture()
        binary, _ = DEV.spectral_fold_features(
            edges, anchors, train_targets, eval_targets, "binary", 2, contract()
        )
        np.testing.assert_allclose(
            binary["weight_strength_to_train_total"], binary["degree_to_train_total"]
        )
        np.testing.assert_allclose(
            binary["weight_strength_to_gcc"], binary["degree_to_gcc"]
        )
        weighted, weighted_diag = DEV.spectral_fold_features(
            edges, anchors, train_targets, eval_targets, "log_length", 2, contract()
        )
        _, binary_diag = DEV.spectral_fold_features(
            edges, anchors, train_targets, eval_targets, "binary", 2, contract()
        )
        self.assertEqual(binary_diag["n_anchor_train"], weighted_diag["n_anchor_train"])
        e0 = weighted.set_index("sample_id").loc["e0"]
        self.assertAlmostEqual(e0["degree_to_train_total"], 1.0)
        self.assertAlmostEqual(e0["weight_strength_to_train_total"], np.log1p(4.0))
        self.assertAlmostEqual(e0["bp_to_train_total"], 4_000_000.0)

    def test_b1_uses_gcc_matched_not_whole_train_connectivity(self) -> None:
        edges, anchors, train_targets, eval_targets = graph_fixture()
        anchors = anchors | {"s0", "s1"}
        edges = pd.concat(
            [
                edges,
                pd.DataFrame(
                    {
                        "sample_a": ["s0", "e0"],
                        "sample_b": ["s1", "s0"],
                        "total_shared_bp": [6_000_000, 7_000_000],
                        "n_shared_variants_total": [3, 4],
                    }
                ),
            ],
            ignore_index=True,
        )
        graph, _ = DEV.spectral_fold_features(
            edges, anchors, train_targets, eval_targets, "log_length", 2, contract()
        )
        e0 = graph.set_index("sample_id").loc["e0"]
        self.assertEqual(e0["degree_to_train_total"], 2.0)
        self.assertEqual(e0["degree_to_gcc"], 1.0)
        self.assertGreater(
            e0["weight_strength_to_train_total"], e0["weight_strength_to_gcc"]
        )

        ids = sorted(train_targets | eval_targets)
        base = pd.DataFrame(
            {
                "sample_id": ids,
                "region": ["A", "B", "C", "A", "B"],
                "Q_NAM": [0.1] * len(ids),
                "Q_EUR": [0.7] * len(ids),
                "Q_EAS": [0.01] * len(ids),
                "Q_AFR": [0.19] * len(ids),
                "burden": np.linspace(0.001, 0.005, len(ids)),
                "rare_missing_sites": np.arange(len(ids)),
            }
        )
        attached = DEV.attach_graph_features(base, graph)
        changed = graph.copy()
        changed["degree_to_train_total"] += 1000
        changed["weight_strength_to_train_total"] += 1000
        changed["bp_to_train_total"] += 1_000_000_000
        attached_changed = DEV.attach_graph_features(base, changed)
        train = attached[attached["sample_id"].isin(train_targets)]
        evaluation = attached[attached["sample_id"].isin(eval_targets)]
        train_changed = attached_changed[attached_changed["sample_id"].isin(train_targets)]
        evaluation_changed = attached_changed[
            attached_changed["sample_id"].isin(eval_targets)
        ]
        x_train, x_eval = DEV.build_design(train, evaluation, "B1", 0, contract())
        changed_train, changed_eval = DEV.build_design(
            train_changed, evaluation_changed, "B1", 0, contract()
        )
        np.testing.assert_allclose(x_train, changed_train)
        np.testing.assert_allclose(x_eval, changed_eval)

    def test_preflight_missing_hash_fails_closed(self) -> None:
        cfg = contract()
        observed = {"pairs": "abc", "metadata": "def"}
        frozen = cfg["preflight_contract"]
        preflight = {
            "schema_version": frozen["schema_version"],
            "decision": frozen["decision"],
            "scope": frozen["scope"],
            "counts": frozen["expected_counts"],
            "gates": {key: True for key in frozen["required_true_gates"]},
            "input_sha256": {"pairs": "abc"},
        }
        with self.assertRaisesRegex(DEV.ContractError, "lacks frozen hash for metadata"):
            DEV.validate_preflight(preflight, observed, cfg)

    def test_projectability_denominators_and_coverage_gates(self) -> None:
        rows = []
        classes = ["NORTHEASTERN", "SOUTHEASTERN", "SOUTHERN"]
        for fold in DEV.DEV_FOLDS:
            for region in classes:
                rows.extend(
                    [
                        {"fold": fold, "region": region, "projectable": 1.0},
                        {"fold": fold, "region": region, "projectable": 0.0},
                    ]
                )
        counts = DEV.projectability_count_table(pd.DataFrame(rows), classes)
        for _, frame in counts.groupby(["fold", "region"]):
            self.assertEqual(
                frame["n"].sum(), frame["region_fold_denominator"].iloc[0]
            )
        self.assertTrue(all(DEV.projectability_coverage_gates(counts, classes).values()))
        failed = counts.copy()
        failed.loc[
            failed["fold"].eq("2")
            & failed["region"].eq("SOUTHERN")
            & failed["projectable"].eq(1),
            "n",
        ] = 0
        gates = DEV.projectability_coverage_gates(failed, classes)
        self.assertFalse(gates["all_regions_projectable_in_each_outer_fold"])
        self.assertFalse(gates["southern_projectable_in_each_outer_fold"])

    def test_non_target_rows_never_become_labels(self) -> None:
        samples = pd.DataFrame(
            {
                "sample_id": ["rht_high", "non_rht", "rht_low"],
                "region": ["SOUTHERN", "SOUTHERN", "SOUTHERN"],
                "is_target": [True, False, False],
            }
        )
        observed = DEV.select_target_rows(samples)
        self.assertEqual(observed["sample_id"].tolist(), ["rht_high"])

    def test_reserved_fold_edges_do_not_change_features_or_predictions(self) -> None:
        edges, anchors, _, eval_targets = graph_fixture()
        target_train = set(anchors)
        base_graph, base_diag = DEV.spectral_fold_features(
            edges, anchors, target_train, eval_targets, "binary", 2, contract(), {"r3"}
        )
        changed_edges = pd.concat(
            [
                edges,
                pd.DataFrame(
                    {
                        "sample_a": ["r3", "r3"],
                        "sample_b": ["t1", "e0"],
                        "total_shared_bp": [9_000_000, 8_000_000],
                        "n_shared_variants_total": [9, 8],
                    }
                ),
            ],
            ignore_index=True,
        )
        changed_graph, changed_diag = DEV.spectral_fold_features(
            changed_edges,
            anchors,
            target_train,
            eval_targets,
            "binary",
            2,
            contract(),
            {"r3"},
        )
        pd.testing.assert_frame_equal(base_graph, changed_graph)
        self.assertEqual(base_diag["reserved_fold_endpoints_used"], 0)
        self.assertEqual(changed_diag["reserved_fold_endpoints_used"], 0)

        rng = np.random.default_rng(31)
        ids = sorted(target_train | eval_targets)
        frame = pd.DataFrame(
            {
                "sample_id": ids,
                "region": ["A", "B", "C"] * 4 + ["A", "B", "C", "A"],
                "Q_NAM": rng.uniform(0.01, 0.2, len(ids)),
                "Q_EUR": rng.uniform(0.55, 0.8, len(ids)),
                "Q_EAS": rng.uniform(0.001, 0.02, len(ids)),
                "Q_AFR": rng.uniform(0.05, 0.3, len(ids)),
                "burden": rng.uniform(0.001, 0.01, len(ids)),
                "rare_missing_sites": rng.uniform(0, 10, len(ids)),
            }
        )
        train_base = frame[frame["sample_id"].isin(target_train)]
        eval_base = frame[frame["sample_id"].isin(eval_targets)]
        train_a = DEV.attach_graph_features(train_base, base_graph)
        eval_a = DEV.attach_graph_features(eval_base, base_graph)
        train_b = DEV.attach_graph_features(train_base, changed_graph)
        eval_b = DEV.attach_graph_features(eval_base, changed_graph)
        expected = DEV.fit_predict(train_a, eval_a, "B1", 0, 0.1, ["A", "B", "C"], contract())
        observed = DEV.fit_predict(train_b, eval_b, "B1", 0, 0.1, ["A", "B", "C"], contract())
        np.testing.assert_allclose(expected, observed, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
