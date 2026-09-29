"""Real Optuna + SQLite tests on invented scalar losses; no model or genomic I/O."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import optuna

SOURCE = Path(__file__).resolve().parents[1] / "bin" / "genomic_adaptive_search.py"
SPEC = importlib.util.spec_from_file_location("genomic_adaptive_search", SOURCE)
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def fixture_plan(full=False):
    sha = search.digest
    folds = [{"id": f"dev{k}", "fit_groups_sha256": sha(["fit", k]),
              "development_groups_sha256": sha(["development", k]),
              "query_sha256": sha(["queries", k]), "truth_sha256": sha(["truth", k]), "n_evaluated": 4}
             for k in range(2)]
    return {
        "schema": search.SCHEMA, "campaign_id": "synthetic-test", "scope": "DEVELOPMENT",
        "max_recipes": 24 if full else 4, "startup_trials": 12 if full else 2,
        "sampler_seed": 109, "training_seeds": [17, 29], "pruning": "none",
        "search_space": {"radius_cm": [.05, .2, 1.] if full else [.2],
                         "width": [32, 64] if full else [32],
                         "learning_rate": {"low": .0001, "high": .001, "log": True}},
        "provenance": {"dataset_sha256": sha("dataset"), "split_manifest_sha256": sha("split"),
                       "feature_contract_sha256": sha("features"), "trainer_code_sha256": sha("trainer"),
                       "container_digest": "sha256:" + sha("image"),
                       "development_only_audit_sha256": sha("role audit")},
        "objectives": {name: {"families": ["interaction_set", "context_attention"],
                              "primary_metric": "macro_mse" if name == "structure" else "log_loss",
                              "secondary_metrics": [] if name == "structure" else ["mae_AFR", "mae_EUR", "mae_NAM"],
                              "fixed_config": {"optimizer": "AdamW", "batch_size": 8, "depth": 2},
                              "checkpoint_step": 16, "folds": copy.deepcopy(folds), "evaluation_unit": "family_macro",
                              "arm_input_sha256": {arm: sha([name, arm]) for arm in search.ARMS},
                              "range_rationale": "Synthetic test bounds only; not a biological or training recommendation."}
                       for name in ("structure", "lai")},
    }


def receipt(recipe, none=2., rare=1.):
    out = {key: copy.deepcopy(recipe[key]) for key in
           ("schema", "scope", "objective", "trial_number", "recipe_sha256", "plan_sha256", "provenance")}
    names = [recipe["primary_metric"], *recipe["secondary_metrics"]]
    out["records"] = [{**copy.deepcopy(task), "status": "COMPLETE", "scope": "DEVELOPMENT",
                       "metrics": {name: none if task["arm"] == "NONE" else rare for name in names},
                       "checkpoint_sha256": search.digest([task, "checkpoint"]),
                       "prediction_sha256": search.digest([task, "prediction"]),
                       "n_evaluated": 4, "evaluation_unit": "family_macro"}
                      for task in recipe["tasks"]]
    return out


class AdaptiveSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = fixture_plan()
        self.controller = search.SearchController(self.root / "state", self.plan)

    def submit(self, controller, proposed, result=None):
        result = receipt(proposed) if result is None else result
        path = self.root / f"receipt-{proposed['objective']}-{proposed['trial_number']}.json"
        path.write_text(json.dumps(result, allow_nan=False))
        _, sha = search.load_json(path)
        return controller.tell(proposed["objective"], path, sha), path, sha

    def test_paired_recipe_resume_and_idempotent_tell(self):
        proposed = self.controller.ask("structure")
        require_keys = {"family", "radius_cm", "width", "learning_rate", "optimizer", "batch_size", "depth"}
        self.assertEqual(set(proposed["config"]), require_keys)
        self.assertEqual(len(proposed["tasks"]), 8)
        self.assertEqual({t["seed"] for t in proposed["tasks"]}, {17, 29})
        self.assertEqual({t["checkpoint_step"] for t in proposed["tasks"]}, {16})
        resumed = search.SearchController(self.root / "state", self.plan)
        self.assertEqual(resumed.ask("structure"), proposed)
        self.assertEqual(resumed.status("structure")["issued"], 1)
        summary, path, sha = self.submit(resumed, proposed)
        self.assertEqual(summary["selection_loss"], 1.5)
        self.assertEqual(summary["delta_rare_minus_none"]["macro_mse"], -1.)
        self.assertEqual(resumed.tell("structure", path, sha), summary)
        self.assertEqual(resumed.status("structure")["trials"][0]["summary"], summary)
        modified = receipt(proposed, rare=.5)
        with self.assertRaisesRegex(search.ContractError, "overwritten"):
            self.submit(resumed, proposed, modified)

    def test_both_absolute_losses_not_only_favorable_delta(self):
        proposed = self.controller.ask("structure")
        misleading_delta = self.controller.validate_receipt(receipt(proposed, none=100., rare=50.), proposed)
        good_absolute = self.controller.validate_receipt(receipt(proposed, none=1., rare=2.), proposed)
        self.assertGreater(misleading_delta["selection_loss"], good_absolute["selection_loss"])
        self.assertEqual(good_absolute["delta_rare_minus_none"]["macro_mse"], 1.)

    def test_structure_only_plan_can_advance_without_lai_contract(self):
        plan = fixture_plan()
        del plan["objectives"]["lai"]
        controller = search.SearchController(self.root / "structure-only", plan)
        proposed = controller.ask("structure")
        summary, _, _ = self.submit(controller, proposed)
        self.assertEqual(summary["selection_loss"], 1.5)
        with self.assertRaisesRegex(search.ContractError, "unknown objective"):
            controller.ask("lai")
        with self.assertRaisesRegex(search.ContractError, "another plan"):
            search.SearchController(self.root / "structure-only", fixture_plan()).ask("lai")
        with self.assertRaisesRegex(search.ContractError, "frozen plan mismatch"):
            search.SearchController(self.root / "structure-only", fixture_plan()).ask("structure")
        for objectives in ({}, {"unknown": plan["objectives"]["structure"]}):
            bad = copy.deepcopy(plan)
            bad["objectives"] = objectives
            with self.assertRaisesRegex(search.ContractError, "nonempty subset"):
                search.validate_plan(bad)

    def test_rejects_non_development_unpaired_and_incomplete_metrics(self):
        proposed = self.controller.ask("lai")
        variants = []
        for role in ("TEST", "SCORE", "VALID", "TRAIN", "REFIT"):
            bad = receipt(proposed); bad["scope"] = role; variants.append(bad)
            bad = receipt(proposed); bad["records"][0]["scope"] = role; variants.append(bad)
        for field, value in (("seed", 999), ("fold", "unknown"), ("checkpoint_step", 8),
                             ("checkpoint_rule", "independent_best"), ("config_sha256", search.digest("other")),
                             ("input_sha256", search.digest("other")), ("n_evaluated", 3),
                             ("evaluation_unit", "raw_pairs"), ("status", "PRUNED")):
            bad = receipt(proposed); bad["records"][0][field] = value; variants.append(bad)
        bad = receipt(proposed); bad["records"].pop(); variants.append(bad)
        bad = receipt(proposed); bad["records"][0] = copy.deepcopy(bad["records"][1]); variants.append(bad)
        bad = receipt(proposed); bad["records"][0]["metrics"].pop("mae_NAM"); variants.append(bad)
        bad = receipt(proposed); bad["provenance"]["trainer_code_sha256"] = search.digest("other"); variants.append(bad)
        bad = receipt(proposed); bad["records"][0]["fold_contract"]["query_sha256"] = search.digest("other"); variants.append(bad)
        bad = receipt(proposed); bad["records"][0]["metrics"]["log_loss"] = float("nan"); variants.append(bad)
        for index, bad in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(search.ContractError):
                self.controller.validate_receipt(bad, proposed)
        self.assertEqual(self.controller.status("lai")["trials"][0]["state"], "RUNNING")

    def test_receipt_hash_and_duplicate_json_keys_fail_closed(self):
        proposed = self.controller.ask("structure")
        path = self.root / "bad.json"
        path.write_text(json.dumps(receipt(proposed)))
        with self.assertRaisesRegex(search.ContractError, "SHA256"):
            self.controller.tell("structure", path, search.digest("wrong"))
        path.write_text('{"scope":"DEVELOPMENT","scope":"TEST"}')
        with self.assertRaisesRegex(search.ContractError, "duplicate"):
            search.load_json(path)

    def test_plan_drift_and_hard_budget_are_rejected(self):
        self.controller.ask("structure")
        mutations = []
        for key, value in (("campaign_id", "renamed"), ("training_seeds", [7]), ("sampler_seed", 900)):
            bad = copy.deepcopy(self.plan); bad[key] = value; mutations.append(bad)
        bad = copy.deepcopy(self.plan); bad["objectives"]["lai"]["checkpoint_step"] = 99; mutations.append(bad)
        for bad in mutations:
            other = search.SearchController(self.root / "state", bad)
            with self.assertRaises(search.ContractError):
                other.ask("structure")
            with self.assertRaises(search.ContractError):
                other.ask("lai")
        bad = copy.deepcopy(self.plan); bad["max_recipes"] = 25
        with self.assertRaisesRegex(search.ContractError, "24"):
            search.SearchController(self.root / "bad", bad)
        bad = copy.deepcopy(self.plan); bad["pruning"] = "median"
        with self.assertRaisesRegex(search.ContractError, "pruning"):
            search.validate_plan(bad)

    def test_failed_startup_counts_and_does_not_prune_a_family(self):
        proposed = self.controller.ask("structure")
        self.controller.fail("structure", proposed["trial_number"], "synthetic operational failure")
        second = self.controller.ask("structure")
        self.assertNotEqual(proposed["config"]["family"], second["config"]["family"])
        self.submit(self.controller, second)
        with self.assertRaisesRegex(search.ContractError, "startup incomplete"):
            self.controller.ask("structure")
        status = self.controller.status("structure")
        self.assertEqual(status["issued"], 2)
        self.assertEqual(status["trials"][0]["state"], "FAIL")
        self.assertEqual(status["pruning"], "none")

    def test_journal_recovers_crash_between_validation_and_tell(self):
        proposed = self.controller.ask("structure")
        with patch.object(optuna.study.Study, "tell", side_effect=RuntimeError("simulated process loss")):
            with self.assertRaises(RuntimeError):
                self.submit(self.controller, proposed)
        with self.assertRaisesRegex(search.ContractError, "journalled"):
            self.controller.fail("structure", 0, "cannot discard validated metric receipt")
        resumed = search.SearchController(self.root / "state", self.plan)
        self.assertEqual(resumed.ask("structure"), proposed)
        self.submit(resumed, proposed)
        self.assertEqual(resumed.status("structure")["trials"][0]["state"], "COMPLETE")

    def test_real_tpe_budget_24_startup_coverage_and_separate_objectives(self):
        plan = fixture_plan(full=True)
        controller = search.SearchController(self.root / "full", plan)
        startup = set()
        calls = []
        original = optuna.samplers.TPESampler._sample

        def observe_real_tpe(sampler, *args, **kwargs):
            calls.append(True)
            return original(sampler, *args, **kwargs)

        with patch.object(optuna.samplers.TPESampler, "_sample", observe_real_tpe):
            for number in range(24):
                # Reconstruct controller each time: storage, not Python memory, is authoritative.
                controller = search.SearchController(self.root / "full", plan)
                proposed = controller.ask("structure")
                self.assertEqual(proposed["trial_number"], number)
                self.assertEqual(proposed["phase"], "STARTUP" if number < 12 else "TPE")
                if number < 12:
                    self.assertFalse(calls)
                    startup.add(tuple(proposed["config"][k] for k in ("family", "radius_cm", "width")))
                loss = 1 + abs(proposed["config"]["learning_rate"] - .0003) * 100
                self.submit(controller, proposed, receipt(proposed, none=loss, rare=loss + .1))
        self.assertEqual(len(startup), 12)
        self.assertTrue(calls)  # Optuna's actual Parzen sampler ran, not just labelled random trials.
        with self.assertRaisesRegex(search.ContractError, "budget"):
            controller.ask("structure")
        self.assertEqual(controller.ask("lai")["trial_number"], 0)
        selection = controller.freeze("structure")
        self.assertEqual(selection["refit"], "SEPARATE_PLAN_AND_AUTHORIZATION_REQUIRED")
        self.assertFalse(selection["training_launched"])
        self.assertEqual(controller.freeze("structure"), selection)
        with self.assertRaisesRegex(search.ContractError, "frozen"):
            controller.ask("structure")

    def test_seed_schedule_restart_reproducibility_and_metric_feedback(self):
        plan = fixture_plan()
        configs = []
        for branch, reverse in (("a", False), ("b", False), ("c", True)):
            controller = search.SearchController(self.root / branch, plan)
            for number in range(2):
                proposal = controller.ask("structure")
                loss = float(1 + (1 - number if reverse else number) * 20)
                self.submit(controller, proposal, receipt(proposal, none=loss, rare=loss))
                controller = search.SearchController(self.root / branch, plan)
            configs.append(controller.ask("structure")["config"])
        self.assertEqual(configs[0], configs[1])
        self.assertNotEqual(configs[0], configs[2])

    def test_real_cli_emits_recipe_and_accepts_receipt(self):
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(self.plan))
        command = [sys.executable, str(SOURCE), "--plan", str(plan_path), "--state-dir",
                   str(self.root / "cli"), "--objective", "lai"]
        proposed = json.loads(subprocess.check_output([*command, "ask"], text=True))
        result_path = self.root / "cli-receipt.json"
        result_path.write_text(json.dumps(receipt(proposed)))
        _, sha = search.load_json(result_path)
        summary = json.loads(subprocess.check_output([*command, "tell", "--receipt", str(result_path),
                                                      "--receipt-sha256", sha], text=True))
        self.assertEqual(summary["selection_loss"], 1.5)
        status = json.loads(subprocess.check_output([*command, "status"], text=True))
        self.assertEqual(status["trials"][0]["state"], "COMPLETE")

    def test_concurrent_local_ask_does_not_spend_two_slots(self):
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(self.plan))
        command = [sys.executable, "-B", str(SOURCE), "--plan", str(plan_path), "--state-dir",
                   str(self.root / "concurrent"), "--objective", "structure", "ask"]
        children = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for _ in range(2)]
        proposals = []
        try:
            for child in children:
                stdout, stderr = child.communicate(timeout=30)
                self.assertEqual(child.returncode, 0, stderr)
                proposals.append(json.loads(stdout))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        self.assertEqual(proposals[0], proposals[1])
        controller = search.SearchController(self.root / "concurrent", self.plan)
        self.assertEqual(controller.status("structure")["issued"], 1)

    def test_example_requires_real_contracts_before_any_study(self):
        example, _ = search.load_json(SOURCE.parents[1] / "conf" / "genomic_adaptive_search.example.json")
        self.assertEqual(example["max_recipes"], 24)
        self.assertEqual(example["startup_trials"], 12)
        for objective in ("structure", "lai"):
            self.assertEqual([fold["id"] for fold in example["objectives"][objective]["folds"]],
                             ["dev0", "dev1", "dev2"])
        lai = example["objectives"]["lai"]
        self.assertEqual(lai["primary_metric"], "one_minus_boundary_f1_0p2cm")
        self.assertEqual(lai["secondary_metrics"], ["log_loss", "brier", "mae_AFR", "mae_EUR", "mae_NAM"])
        with self.assertRaises(search.ContractError):
            search.SearchController(self.root / "example", example)
        self.assertFalse((self.root / "example" / "search.sqlite3").exists())

    def test_boundary_loss_receipt_with_three_folds(self):
        plan = fixture_plan()
        lai = plan["objectives"]["lai"]
        lai["primary_metric"] = "one_minus_boundary_f1_0p2cm"
        lai["secondary_metrics"] = ["log_loss", "brier", "mae_AFR", "mae_EUR", "mae_NAM"]
        lai["folds"].append({**copy.deepcopy(lai["folds"][0]), "id": "dev2"})
        controller = search.SearchController(self.root / "boundary", plan)
        proposed = controller.ask("lai")
        self.assertEqual(len(proposed["tasks"]), 3 * 2 * 2)
        summary, _, _ = self.submit(controller, proposed, receipt(proposed, none=.6, rare=.4))
        self.assertAlmostEqual(summary["selection_loss"], .5)
        self.assertAlmostEqual(summary["delta_rare_minus_none"]["one_minus_boundary_f1_0p2cm"], -.2)

    def test_boundary_loss_is_bounded_without_capping_other_losses(self):
        metric = "one_minus_boundary_f1_0p2cm"
        for primary in (True, False):
            with self.subTest(primary=primary):
                plan = fixture_plan()
                spec = plan["objectives"]["lai"]
                if primary:
                    spec["primary_metric"] = metric
                    spec["secondary_metrics"].append("log_loss")
                else:
                    spec["secondary_metrics"].append(metric)
                controller = search.SearchController(self.root / f"bounded-{primary}", plan)
                proposed = controller.ask("lai")
                for value in (-.01, 1.000001):
                    invalid = receipt(proposed, none=5., rare=3.)
                    for row in invalid["records"]:
                        row["metrics"][metric] = .5
                    invalid["records"][0]["metrics"][metric] = value
                    with self.subTest(value=value), self.assertRaises(search.ContractError):
                        controller.validate_receipt(invalid, proposed)
                valid = receipt(proposed, none=5., rare=3.)
                for row in valid["records"]:
                    row["metrics"][metric] = 0. if row["arm"] == "NONE" else 1.
                result = controller.validate_receipt(valid, proposed)
                self.assertEqual(result["selection_loss"], .5 if primary else 4.)


if __name__ == "__main__":
    unittest.main()
