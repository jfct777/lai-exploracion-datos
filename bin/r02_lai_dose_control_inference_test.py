#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "dose_inference", HERE / "r02_lai_dose_control_inference.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class DoseInferenceTests(unittest.TestCase):
    def data(self):
        return {
            "query_valid": np.ones((1, 2), dtype=bool),
            "truth_observed": np.ones((1, 2), dtype=bool),
            "truth_state": np.array([[0, 4]], dtype=np.int64),
        }

    def test_probability_metrics_match_hand_calculation(self):
        prediction = np.array([[[1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0]]], dtype=np.float64)
        result = M.probability_statistics(prediction, self.data())
        self.assertEqual(result["aggregation"],
                         "QUERY_POOLED_SUMS_OVER_ONE_FIXED_SCORE_MASK")
        self.assertEqual(result["observed_queries"], 2)
        self.assertEqual(result["mean_log_loss"], 0)
        self.assertEqual(result["mean_brier"], 0)
        self.assertEqual(result["state_error_rate"], 0)
        self.assertEqual(result["dosage_mae_AFR_EUR_NAM"], [0, 0, 0])

    def test_probability_change_uses_only_explicit_score_mask(self):
        original = np.full((1, 2, 6), 1 / 6, dtype=np.float64)
        changed = original.copy()
        changed[0, 0] = [1, 0, 0, 0, 0, 0]
        changed[0, 1] = [0, 1, 0, 0, 0, 0]
        result = M.prediction_change(
            changed, original, self.data(), np.array([[True, False]]))
        self.assertEqual(result["observed_queries"], 1)
        self.assertEqual(result["argmax_state_changes"], 0)
        self.assertAlmostEqual(result["mean_abs_probability_change"], 5 / 18)

    def test_saved_original_requires_exact_by_default_and_reports_tolerance(self):
        original = np.array([0.2, 0.8], dtype=np.float32)
        exact = M.compare_saved_original(original.copy(), original, 0)
        self.assertTrue(exact["exact"])
        changed = original.copy(); changed[0] += np.float32(5e-7)
        with self.assertRaisesRegex(ValueError, "differs"):
            M.compare_saved_original(changed, original, 0)
        accepted = M.compare_saved_original(changed, original, 1e-6)
        self.assertFalse(accepted["exact"])
        self.assertLessEqual(accepted["max_abs_delta_probability"], 1e-6)

    def test_state_mutation_during_any_scenario_is_rejected(self):
        model = SimpleNamespace(state=0)
        def digest(value): return str(value.state)
        def mutate(value, _view):
            value.state += 1
            return np.zeros((1, 1, 6))
        with self.assertRaisesRegex(ValueError, "RNG changed"):
            M.predict_preserving_state(model, {"ORIGINAL": {}}, mutate, digest)

    def test_score_support_requires_exact_mask_and_unique_coordinates(self):
        data = self.data()
        data.update({"query_core": np.array([[True, True]]),
                     "query_chrom": np.array([[1, 1]]),
                     "query_pos_bp": np.array([[10, 20]])})
        support = M.score_support_identity(data, data["truth_observed"])
        self.assertEqual(support["owned_coordinate_count"], 2)
        self.assertEqual(support["observed_query_count"], 2)
        changed = dict(data); changed["query_pos_bp"] = np.array([[10, 10]])
        with self.assertRaisesRegex(ValueError, "repeated"):
            M.score_support_identity(changed, changed["truth_observed"])
        with self.assertRaisesRegex(ValueError, "exactly"):
            M.score_support_identity(data, np.array([[True, False]]))

    def test_control_package_rejects_tampered_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            names = {
                "bin/r02_lai_control_views.py", "bin/r02_lai_dose_controls.py",
                "bin/r02_lai_dose_control_training.py", "bin/r02_lai_person_microbatch.py",
                "bin/r02_lai_fit_monitor.py", "bin/r02_lai_dose_control_inference.py",
            }
            for name in names:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name + "\n")
                files[name] = M.sha256_file(path)
            manifest = root / "MANIFEST.json"
            manifest.write_text(json.dumps({"schema": M.PACKAGE_SCHEMA, "files": files}))
            digest = M.sha256_file(manifest)
            M.validate_control_package(root, manifest, digest)
            (root / sorted(names)[0]).write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "member differs"):
                M.validate_control_package(root, manifest, digest)

    def test_idempotent_receipt_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "region.json"
            value = {"schema": M.SCHEMA, "status": M.STATUS,
                     "request_sha256": "a" * 64}
            self.assertTrue(M.write_idempotent(path, value, "a" * 64))
            before = path.read_bytes()
            self.assertFalse(M.write_idempotent(path, {}, "a" * 64))
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "another request"):
                M.write_idempotent(path, {}, "b" * 64)

    def test_existing_result_reauthenticates_current_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / "fh"; package.mkdir()
            sources = {}
            for name in ("fh/MANIFEST.json", "control.json", "fj.py", "route.json",
                         "ex.py", "predictions.json", "products.json"):
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name + "\n"); sources[name] = M.sha256_file(path)
            args = SimpleNamespace(
                package=str(package), package_manifest_sha256=sources["fh/MANIFEST.json"],
                control_package_manifest=str(root / "control.json"),
                control_package_manifest_sha256=sources["control.json"],
                fj_diagnostic=str(root / "fj.py"), fj_diagnostic_sha256=sources["fj.py"],
                none_route=str(root / "route.json"), none_route_sha256=sources["route.json"],
                ex_helper=str(root / "ex.py"), ex_helper_sha256=sources["ex.py"],
                prediction_index=str(root / "predictions.json"),
                prediction_index_sha256=sources["predictions.json"],
                product_plan=str(root / "products.json"),
                product_plan_sha256=sources["products.json"],
                case_id="case", region_id="region-00")
            resolved = root / "resolved.json"; resolved.write_text("{}\n")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"case_id": "case", "region_id": "region-00"}))
            entry = {"path": "/entry", "sha256": "1" * 64,
                     "receipt_path": str(receipt), "receipt_sha256": M.sha256_file(receipt)}
            contract = root / "contract.json"
            contract.write_text(json.dumps({"batches": {"FIT": [entry]}}))
            route_value = {"contract": {"path": str(contract), "sha256": M.sha256_file(contract)},
                           "resolved": {"path": str(resolved), "sha256": M.sha256_file(resolved)}}
            route = root / "route.json"; route.write_text(json.dumps(route_value))
            args.none_route_sha256 = M.sha256_file(route)
            checkpoint_root = root / "checkpoint"; checkpoint_root.mkdir()
            payload = checkpoint_root / "payload.npz"; payload.write_bytes(b"payload")
            checkpoint_manifest = checkpoint_root / "checkpoint.json"
            checkpoint_manifest.write_text(json.dumps({"next_step": 96,
                "payload": payload.name, "payload_sha256": M.sha256_file(payload)}))
            current = checkpoint_root / "CURRENT"
            current.write_text(json.dumps({"next_step": 96,
                "manifest": checkpoint_manifest.name}))
            args.checkpoint_root = str(checkpoint_root)
            saved_prediction = root / "prediction.npy"; saved_prediction.write_bytes(b"prediction")
            prediction_index = root / "predictions.json"
            prediction_index.write_text(json.dumps([{"input": {"sha256": entry["sha256"]},
                "path": str(saved_prediction), "sha256": M.sha256_file(saved_prediction)}]))
            args.prediction_index = str(prediction_index)
            args.prediction_index_sha256 = M.sha256_file(prediction_index)
            metric = {"aggregation": "QUERY_POOLED_SUMS_OVER_ONE_FIXED_SCORE_MASK",
                      "observed_queries": 2, "log_loss_sum": 1.0, "brier_sum": 1.0,
                      "state_error_sum": 1, "dosage_abs_sum_AFR_EUR_NAM": [1, 1, 1]}
            value = {
                "schema": M.SCHEMA, "status": M.STATUS, "request_sha256": "a" * 64,
                "selection": {"case_id": "case", "region_id": "region-00", "role": "FIT",
                    "people": 128, "people_per_microbatch": 8,
                    "model_condition": "NONE", "checkpoint_step": 96},
                "authentication": {"FH_package_manifest_sha256": args.package_manifest_sha256,
                    "FL_control_package_manifest_sha256": args.control_package_manifest_sha256,
                    "execution_image": M.FROZEN_IMAGE,
                    "NONE_route_sha256": args.none_route_sha256,
                    "FI_prediction_index_sha256": args.prediction_index_sha256,
                    "FK_product_plan_sha256": args.product_plan_sha256,
                    "NONE_contract_sha256": route_value["contract"]["sha256"],
                    "resolved_sha256": route_value["resolved"]["sha256"],
                    "FIT_entry_sha256": entry["sha256"],
                    "FIT_entry_receipt_sha256": entry["receipt_sha256"],
                    "checkpoint": {"pointer_sha256": M.sha256_file(current),
                        "manifest_sha256": M.sha256_file(checkpoint_manifest),
                        "payload_sha256": M.sha256_file(payload)},
                    "FI_prediction_sha256": M.sha256_file(saved_prediction),
                    "original_reproduction": {"exact": True}},
                "score_support": {},
                "scenarios": {name: {"metrics": metric} for name in
                              ("FLARE", "ORIGINAL", *M.CONDITIONS)},
                "state": {"unchanged": True, "optimizer_constructed": False,
                          "backward_called": False},
                "access": {"fit_entries_reconstructed": 1,
                           "select_entries_reconstructed": 0, "evaluate_opened": False},
                "resources": {},
                "stored_outputs": "AGGREGATE_SUMMARIES_AND_AUTHENTICATED_IDENTITIES_ONLY",
                "interpretation_limit": "x", "dependence_limit": "x"}
            output = root / "output.json"; output.write_bytes(M.canonical(value))
            M.validate_existing_output(output, args, "a" * 64)
            route.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "source identity"):
                M.validate_existing_output(output, args, "a" * 64)

    def test_prepared_plan_has_exact_fit_coverage_and_bounded_resources(self):
        root = HERE.parent
        path = root / ".claude/runs/r02-dl-training-path-20261102fm/lai-controls/EXECUTION_PLAN.json"
        plan = json.loads(path.read_text())
        tasks = plan["tasks"]
        self.assertEqual(len(tasks), 48)
        self.assertEqual(len({row["id"] for row in tasks}), 48)
        self.assertEqual({row["region_id"] for row in tasks},
                         {f"region-{value:02d}" for value in range(16)})
        self.assertEqual(len({row["case_id"] for row in tasks}), 3)
        self.assertFalse(plan["coverage"]["select_opened"])
        self.assertFalse(plan["coverage"]["evaluate_opened"])
        self.assertEqual(plan["resources"]["per_task"]["cpus"], 1)
        self.assertEqual(plan["resources"]["per_task"]["memory_bytes"], 4 * 1024 ** 3)
        self.assertEqual(plan["resources"]["per_task"]["max_wall_seconds"], 900)
        self.assertEqual(plan["resources"]["max_parallel_tasks"], 3)
        self.assertFalse(plan["launch_authorized"])

    def test_packaged_executable_is_exact_source(self):
        root = HERE.parent
        packaged = root / ".claude/runs/r02-dl-training-path-20261102fm/lai-controls/package/bin/r02_lai_dose_control_inference.py"
        self.assertEqual(M.sha256_file(packaged),
                         M.sha256_file(HERE / "r02_lai_dose_control_inference.py"))


if __name__ == "__main__":
    unittest.main()
