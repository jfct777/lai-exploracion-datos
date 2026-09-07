"""Throughput plumbing on synthetic tensors and metadata; no genomic data reads."""
import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m39_profile_ordered_throughput as P
from test_m39_ordered_models import fixture, model_for
from test_m39_throughput_sampling import MetadataStore


class OrderedThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.profile = ROOT / "conf" / "m39_ordered_throughput_profile.json"
        self.cfg = json.loads(self.profile.read_text())
        self.store = MetadataStore()
        self.sampling = P.build_sampling_manifest(self.store, self.cfg["seed"])

    def changed_profile(self, cfg):
        with patch.object(Path, "read_text", return_value=json.dumps(cfg)):
            return P.load_profile(Path("artificial.json"))

    def records(self, batch_size=2, policy="grouped"):
        rows = P.steps_for_case(self.sampling, batch_size, policy)
        return [self.record(row, index + 1, False) for index, row in enumerate(rows)]

    @staticmethod
    def record(row, index, warmup):
        elapsed = 2.0 if row["stratum"] is None else float(row["stratum"] + 1)
        return {**row, "warmup": warmup, "step": index, "end_to_end_seconds": elapsed,
                "peak_rss_kib": 1000, "true_site_tokens": sum(row["anchor_lengths"]),
                "padded_site_tokens": len(row["pairs"]) * max(row["anchor_lengths"]),
                **{key: elapsed / 12 for key in P.PHASE_FIELDS}}

    def test_six_small_real_cases_and_source_dependency_inventory(self):
        cfg = P.load_profile(self.profile)
        expected = {(family, policy, batch) for family in ("cnn", "attention")
                    for policy, batch in (("grouped", 1), ("grouped", 2), ("random", 2))}
        self.assertEqual({(c["family"], c["policy"], c["batch_size"]) for c in cfg["cases"]}, expected)
        self.assertTrue(all(c["arm"] == "real" for c in cfg["cases"]))
        self.assertEqual(len(P.SOURCE_FILES), len(set(P.SOURCE_FILES)))
        self.assertEqual(set(P.SOURCE_FILES), set(P.HISTORICAL_SOURCE_FILES) |
                         {"m39_profile_ordered_throughput.py", "m39_throughput_sampling.py"})
        self.assertTrue(all((ROOT / "bin" / path).is_file() for path in P.SOURCE_FILES))

    def test_scope_hash_roster_and_unknown_fields_rejected(self):
        changes = ({"schema_version": "old"}, {"scope": "training"}, {"device": "cuda"},
                   {"fold": False}, {"fold": 1}, {"store_case": "people_16/radius_1cm"},
                   {"folds_sha256": "F" * 64}, {"parent_receipt_sha256": "f" * 63},
                   {"store_manifest_sha256": "../file"}, {"truth_path": "unused"})
        for change in changes:
            cfg = copy.deepcopy(self.cfg)
            cfg.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.changed_profile(cfg)
        del cfg["scope"]
        with self.assertRaises(ValueError):
            self.changed_profile(cfg)

    def test_sampling_resource_and_optimizer_limits_are_closed(self):
        changes = ({"warmup_steps": 0}, {"warmup_steps": True}, {"sample_strata": 3},
                   {"anchors_per_stratum": 5}, {"queries_per_anchor": 1}, {"core_sites": 128},
                   {"seed": -1}, {"seed": 2**32}, {"torch_threads": 3}, {"torch_threads": 1.0},
                   {"max_input_bytes": 67108865}, {"max_rss_kib": 6710887},
                   {"max_seconds": 901}, {"max_seconds": 0}, {"weight_decay": 0.01},
                   {"learning_rate": 0.0001})
        for change in changes:
            cfg = copy.deepcopy(self.cfg)
            cfg.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.changed_profile(cfg)

    def test_recipe_case_drift_and_duplicates_rejected(self):
        for change in ({"width": 64}, {"depth": 4}, {"kernels": [3, 15]},
                       {"width": 32.0}, {"dilations": [True, 2]}, {"dropout": 0.1}):
            cfg = copy.deepcopy(self.cfg)
            cfg["recipe"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.changed_profile(cfg)
        for change in ({"family": "global"}, {"policy": "random"}, {"batch_size": True},
                       {"batch_size": 3}, {"arm": "common"}, {"id": "free-id"}, {"size": "small"}):
            cfg = copy.deepcopy(self.cfg)
            cfg["cases"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.changed_profile(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["cases"][1] = cfg["cases"][0]
        with self.assertRaises(ValueError):
            self.changed_profile(cfg)

    def test_summary_excludes_warmup_and_uses_one_exact_weighted_estimator(self):
        for batch in (1, 2):
            rows = self.records(batch)
            warmup = self.record(self.sampling["warmup"], 0, True)
            warmup["end_to_end_seconds"] = 10000.0
            with patch.object(P, "estimate_grouped_epoch", wraps=P.estimate_grouped_epoch) as estimate:
                result = P.summarize_steps([warmup, *rows], policy="grouped", batch_size=batch,
                                           sampling=self.sampling)
                estimate.assert_called_once()
            self.assertEqual(result["measured_steps"], 32 // batch)
            self.assertEqual(result["measured_query_anchor_pairs"], 32)
            value = result["grouped_fullpass_estimate"]
            self.assertEqual(value["seconds"], 165 * 48 / batch * (1 + 2 + 3 + 4))
            self.assertEqual(value["seconds"], value["estimated_grouped_pass_seconds"])
            self.assertIsNone(value["confidence_interval"])
            self.assertFalse(value["convergence_time_estimated"])
            self.assertEqual(value["population_query_anchor_pairs"], 31680)
            self.assertTrue(all(x["sampled_anchors"] == 4 for x in value["strata"]))

    def test_random_replay_never_receives_population_estimate(self):
        rows = self.records(2, "random")
        with patch.object(P, "estimate_grouped_epoch", side_effect=AssertionError("must not estimate")):
            result = P.summarize_steps(rows, policy="random", batch_size=2, sampling=self.sampling)
        self.assertEqual(result["measured_query_anchor_pairs"], 32)
        self.assertIsNone(result["grouped_fullpass_estimate"])
        self.assertIn("not_population", result["random_scope"])

    def test_missing_duplicate_or_wrong_stratum_fails_grouped_summary(self):
        for mode in ("empty", "missing", "duplicate", "stratum", "nonfinite"):
            rows = self.records()
            if mode == "empty":
                rows = []
            elif mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = copy.deepcopy(rows[0])
            elif mode == "stratum":
                rows[0]["stratum"] = (rows[0]["stratum"] + 1) % 4
            else:
                rows[0]["end_to_end_seconds"] = float("nan")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                P.summarize_steps(rows, policy="grouped", batch_size=2, sampling=self.sampling)

    def test_time_and_rss_guards_fail_closed(self):
        usage = SimpleNamespace(ru_maxrss=100)
        with patch.object(P.time, "monotonic", return_value=100), \
                patch.object(P.resource, "getrusage", return_value=usage):
            self.assertEqual(P.check_limits(99, self.cfg), 100)
            with self.assertRaisesRegex(ValueError, "wall time"):
                P.check_limits(-1000, self.cfg)
            usage.ru_maxrss = self.cfg["max_rss_kib"] + 1
            with self.assertRaisesRegex(ValueError, "RSS"):
                P.check_limits(99, self.cfg)

    def test_actual_synthetic_steps_load_fresh_and_update_both_model_families(self):
        for family in ("cnn", "attention"):
            model = model_for(family, dtype=torch.float32, depth=1, dilations=(1,), core_sites=5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
            batch = fixture(count=1, candidates=2, length=9, dtype=torch.float32)
            store = SimpleNamespace(window_bounds=lambda anchor: (0, 9))
            row = {"stratum": 0, "pairs": [[0, 0]], "anchor_lengths": [9]}
            case = {"batch_size": 1, "arm": "real"}
            before = P.tensor_digest(batch)
            with patch.object(P, "pack_batch", side_effect=lambda *args, **kw:
                              {key: value.clone() for key, value in batch.items()}) as pack, \
                    patch.object(P, "estimate_batch_bytes", return_value={"bounded": True}):
                for step in range(2):
                    record = P.measure_step(store, model, optimizer, row, case, self.cfg,
                                            step_index=step, warmup=step == 0, case_started=time.monotonic())
                    self.assertGreater(record["optimizer_state_bytes"], 0)
                    self.assertTrue(all(value > 0 for value in record["gradient_norm_by_component"].values()))
                    self.assertTrue(all(value > 0 for value in record["updated_tensors_by_component"].values()))
                    self.assertTrue(all(record[key] >= 0 for key in P.PHASE_FIELDS))
                    self.assertGreaterEqual(record["end_to_end_seconds"], sum(record[k] for k in P.PHASE_FIELDS))
                    self.assertEqual(record["true_site_tokens"], 9)
                    self.assertEqual(record["padded_site_tokens"], 9)
                    self.assertEqual(record["tensor_sha256"], before)
                self.assertEqual(pack.call_count, 2)
            self.assertEqual(P.tensor_digest(batch), before)

    def test_window_truncation_and_input_mutation_are_rejected(self):
        for mode in ("truncation", "mutation"):
            model = model_for("cnn", dtype=torch.float32, depth=1, dilations=(1,))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
            batch = fixture(count=1, candidates=2, length=9, dtype=torch.float32)
            if mode == "truncation":
                batch["site_mask"][0, -1] = False
            with patch.object(P, "pack_batch", return_value=batch), \
                    patch.object(P, "estimate_batch_bytes", return_value={}), \
                    patch.object(P, "tensor_digest", side_effect=["a" * 64, "b" * 64]):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "retained|mutated"):
                    P.measure_step(SimpleNamespace(window_bounds=lambda j: (0, 9)), model, optimizer,
                                   {"pairs": [[0, 0]], "anchor_lengths": [9], "stratum": 0},
                                   {"batch_size": 1, "arm": "real"}, self.cfg,
                                   step_index=0, warmup=True, case_started=time.monotonic())

    def mocked_run(self, output, *, fail_at=None):
        calls = []
        def measured(store, model, optimizer, row, case, cfg, **kwargs):
            if len(calls) == fail_at:
                raise TimeoutError("artificial timeout")
            record = self.record(row, kwargs["step_index"], kwargs["warmup"])
            calls.append(record)
            return record
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(P, "authenticate", return_value=(self.store, {})))
            stack.enter_context(patch.object(P, "measure_step", side_effect=measured))
            stack.enter_context(patch.object(P, "recheck_inputs"))
            stack.enter_context(patch.object(P.torch, "set_num_interop_threads"))
            stack.enter_context(patch.object(P.torch, "set_num_threads"))
            stack.enter_context(patch.object(P, "check_limits", return_value=1000))
            return P.run_case(Path("unused-store"), Path("unused-parent"), Path("unused-folds"),
                              self.profile, "cnn-small-grouped-b2-real", output, "a" * 40)

    def test_completed_mock_run_writes_manifest_steps_and_only_final_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile"
            result = self.mocked_run(output)
            self.assertEqual(result["decision"], P.DECISION)
            self.assertEqual(result["summary"]["measured_steps"], 16)
            self.assertEqual(result["summary"]["measured_query_anchor_pairs"], 32)
            self.assertEqual(len(list((output / "steps").glob("step-*.json"))), 17)
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ["profile.json", "sampling-manifest.json", "steps"])
            self.assertEqual(P.sha256_file(output / "sampling-manifest.json"),
                             result["sampling_manifest_sha256"])
            self.assertEqual(json.loads((output / "steps/step-0000.json").read_text()), result["steps"][0])
            self.assertTrue(result["scope"]["partial_step_receipts_are_not_PASS"])
            self.assertFalse(result["scope"]["truth_opened"])
            self.assertFalse(result["scope"]["weights_saved"])
            self.assertFalse(result["scope"]["accuracy_evaluated"])

    def test_timeout_keeps_completed_exclusive_steps_without_partial_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile"
            with self.assertRaises(TimeoutError):
                self.mocked_run(output, fail_at=2)
            self.assertTrue((output / "sampling-manifest.json").is_file())
            self.assertEqual(len(list((output / "steps").glob("step-*.json"))), 2)
            self.assertFalse((output / "profile.json").exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.mocked_run(output)

    def test_invalid_commit_case_and_existing_output_fail_before_authentication(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(P, "authenticate") as auth:
            for commit, case, output in (("bad", "cnn-small-grouped-b2-real", Path(directory) / "new"),
                                         ("a" * 40, "unknown", Path(directory) / "new"),
                                         ("a" * 40, "cnn-small-grouped-b2-real", Path(directory))):
                with self.assertRaises(ValueError):
                    P.run_case(Path("none"), Path("none"), Path("none"), self.profile, case, output, commit)
            auth.assert_not_called()


if __name__ == "__main__":
    unittest.main()
