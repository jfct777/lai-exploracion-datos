"""Versioned conditional search tests; synthetic losses, no training or genomic I/O."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import optuna

LEGACY_SPEC = importlib.util.spec_from_file_location(
    "adaptive_search_test_helpers", Path(__file__).with_name("test_genomic_adaptive_search.py"))
legacy = importlib.util.module_from_spec(LEGACY_SPEC)
LEGACY_SPEC.loader.exec_module(legacy)
search = legacy.search


def fixture_plan(full=False):
    plan = legacy.fixture_plan(full)
    plan["schema"] = search.CONTEXTUAL_SCHEMA
    plan["campaign_id"] = "synthetic-contextual-v2"
    space = plan["search_space"]
    space["common_radius_cm"] = space.pop("radius_cm")
    space.update({"rare_radius_cm": [.05, .2, 1.], "depth": [1, 2, 3],
                  "dropout": [0., .1, .2], "heads": [2, 4, 8]})
    for objective, spec in plan["objectives"].items():
        spec["families"] = list(search.CONTEXTUAL_FAMILIES[objective])
        del spec["fixed_config"]["depth"]
    return plan


class ContextualSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = fixture_plan()
        self.controller = search.SearchController(self.root / "state", self.plan)

    submit = legacy.AdaptiveSearchTests.submit

    def test_conditional_config_and_optuna_domains_for_all_four_families(self):
        for objective in ("structure", "lai"):
            for family in search.CONTEXTUAL_FAMILIES[objective]:
                recipe = self.controller.ask(objective)
                config = recipe["config"]
                self.assertEqual(recipe["schema"], search.CONTEXTUAL_SCHEMA)
                self.assertEqual(config["family"], family)
                self.assertEqual(set(config), set(self.plan["objectives"][objective]["fixed_config"])
                                 | search.CONTEXTUAL_FIELDS | {"family"})
                self.assertNotIn("radius_cm", config)
                params = self.controller.status(objective)["trials"][-1]["params"]
                if family == "structure_deep_sets":
                    self.assertIsNone(config["rare_radius_cm"])
                    self.assertNotIn("rare_radius_cm", params)
                else:
                    self.assertIn(config["rare_radius_cm"], self.plan["search_space"]["rare_radius_cm"])
                    self.assertIn("rare_radius_cm", params)
                if family.endswith("_local_attention"):
                    self.assertEqual(config["width"] % config["heads"], 0)
                    self.assertIn("heads", params)
                else:
                    self.assertIsNone(config["heads"])
                    self.assertNotIn("heads", params)
                self.assertIn(config["depth"], [1, 2, 3])
                self.assertIn(config["dropout"], [0., .1, .2])
                self.assertEqual({task["config_sha256"] for task in recipe["tasks"]}, {search.digest(config)})
                self.assertEqual({task["arm"] for task in recipe["tasks"]}, {"NONE", "RARE"})
                self.assertEqual(self.controller.ask(objective), recipe)
                self.submit(self.controller, recipe)

    def test_common_and_rare_radii_are_not_implicitly_tied(self):
        plan = fixture_plan()
        plan["search_space"]["common_radius_cm"] = [.05]
        plan["search_space"]["rare_radius_cm"] = [1.]
        controller = search.SearchController(self.root / "separate-radii", plan)
        for _ in range(2):
            recipe = controller.ask("lai")
            self.assertEqual(recipe["config"]["common_radius_cm"], .05)
            self.assertEqual(recipe["config"]["rare_radius_cm"], 1.)
            self.submit(controller, recipe)

    def test_conditional_domain_rejected_before_allocating_any_state(self):
        invalid_spaces = [
            ("heads", [3]), ("heads", [0]), ("heads", [True]),
            ("width", [33]), ("width", [32, 32]), ("depth", [0]), ("depth", [1.5]),
            ("dropout", [-.1]), ("dropout", [1.]), ("dropout", [float("nan")]),
            ("dropout", [True]), ("rare_radius_cm", [0.]), ("rare_radius_cm", []),
            ("common_radius_cm", [float("inf")]), ("heads", [[2]]),
        ]
        for index, (name, choices) in enumerate(invalid_spaces):
            bad = fixture_plan()
            bad["search_space"][name] = choices
            destination = self.root / f"bad-domain-{index}"
            with self.subTest(name=name, choices=choices), self.assertRaises(search.ContractError):
                search.SearchController(destination, bad)
            self.assertFalse(destination.exists())
        for field in search.CONTEXTUAL_FIELDS | {"radius_cm"}:
            bad = fixture_plan()
            bad["objectives"]["structure"]["fixed_config"][field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(search.ContractError, "override"):
                search.validate_plan(bad)
        bad = fixture_plan()
        bad["objectives"]["structure"]["families"] = list(search.CONTEXTUAL_FAMILIES["lai"])
        with self.assertRaisesRegex(search.ContractError, "canonical"):
            search.validate_plan(bad)

    def test_v1_v2_schemas_and_existing_state_cannot_be_reinterpreted(self):
        for original in (legacy.fixture_plan(), fixture_plan()):
            bad = copy.deepcopy(original)
            bad["schema"] = search.CONTEXTUAL_SCHEMA if original["schema"] == search.SCHEMA else search.SCHEMA
            with self.assertRaisesRegex(search.ContractError, "search_space"):
                search.validate_plan(bad)
        old = search.SearchController(self.root / "legacy", legacy.fixture_plan())
        original = old.ask("structure")
        self.assertEqual(original["schema"], search.SCHEMA)
        self.assertIn("radius_cm", original["config"])
        self.assertNotIn("common_radius_cm", original["config"])
        new = search.SearchController(self.root / "legacy", fixture_plan())
        with self.assertRaisesRegex(search.ContractError, "frozen plan mismatch"):
            new.ask("structure")
        with self.assertRaisesRegex(search.ContractError, "another plan"):
            new.ask("lai")
        self.assertEqual(old.ask("structure"), original)
        wrong_schema = legacy.receipt(original)
        wrong_schema["schema"] = search.CONTEXTUAL_SCHEMA
        with self.assertRaisesRegex(search.ContractError, "schema"):
            old.validate_receipt(wrong_schema, original)

    def test_rejects_old_controller_digest_without_implicit_migration(self):
        self.controller.ask("structure")
        study = optuna.load_study(study_name="structure", storage=self.controller.storage)
        study.set_user_attr("controller_sha256", search.digest("previous controller"))
        with self.assertRaisesRegex(search.ContractError, "controller code drift"):
            self.controller.ask("structure")

    def test_v2_receipts_remain_paired_and_development_only(self):
        legacy.AdaptiveSearchTests.test_rejects_non_development_unpaired_and_incomplete_metrics(self)
        recipe = self.controller.ask("structure")
        bad = legacy.receipt(recipe)
        bad["schema"] = search.SCHEMA
        with self.assertRaisesRegex(search.ContractError, "schema"):
            self.controller.validate_receipt(bad, recipe)
        for field in ("common_radius_cm", "rare_radius_cm", "depth", "heads", "dropout"):
            config = copy.deepcopy(recipe["config"])
            config[field] = "changed"
            bad = legacy.receipt(recipe)
            bad["records"][0]["config_sha256"] = search.digest(config)
            with self.subTest(field=field), self.assertRaisesRegex(search.ContractError, "unpaired"):
                self.controller.validate_receipt(bad, recipe)
        summary, path, sha = self.submit(self.controller, recipe)
        self.assertEqual(self.controller.tell("structure", path, sha), summary)

    def test_v2_journal_recovers_after_tell_crash(self):
        legacy.AdaptiveSearchTests.test_journal_recovers_crash_between_validation_and_tell(self)

    def test_v2_failed_startup_counts_without_replacement(self):
        legacy.AdaptiveSearchTests.test_failed_startup_counts_and_does_not_prune_a_family(self)

    def test_real_conditional_tpe_24_budget_startup_restart_and_freeze(self):
        plan = fixture_plan(full=True)
        startup = set()
        observed = {"depth": set(), "dropout": set(), "heads": set(), "rare_radius_cm": set()}
        tpe_calls = []
        original = optuna.samplers.TPESampler._sample

        def observe(sampler, *args, **kwargs):
            tpe_calls.append(True)
            return original(sampler, *args, **kwargs)

        with patch.object(optuna.samplers.TPESampler, "_sample", observe):
            for number in range(24):
                controller = search.SearchController(self.root / "full-v2", plan)
                recipe = controller.ask("structure")
                self.assertEqual(recipe["trial_number"], number)
                self.assertEqual(controller.ask("structure"), recipe)
                config = recipe["config"]
                if number < 12:
                    self.assertEqual(recipe["phase"], "STARTUP")
                    self.assertFalse(tpe_calls)
                    startup.add(tuple(config[k] for k in ("family", "common_radius_cm", "width")))
                    for name in observed:
                        if config[name] is not None:
                            observed[name].add(config[name])
                else:
                    self.assertEqual(recipe["phase"], "TPE")
                self.submit(controller, recipe, legacy.receipt(recipe, none=1. + config["dropout"], rare=1.1))
        self.assertEqual(startup, set(itertools.product(plan["objectives"]["structure"]["families"],
                                                       plan["search_space"]["common_radius_cm"],
                                                       plan["search_space"]["width"])))
        # Extra parameters already vary during startup; there is no later-stage switch.
        for name, values in observed.items():
            self.assertGreater(len(values), 1, name)
        self.assertTrue(tpe_calls)
        with self.assertRaisesRegex(search.ContractError, "budget"):
            controller.ask("structure")
        selection = controller.freeze("structure")
        self.assertEqual(selection["schema"], search.CONTEXTUAL_SCHEMA)
        self.assertFalse(selection["training_launched"])
        self.assertEqual(selection["scope"], "DEVELOPMENT_SELECTION_NOT_CONFIRMATORY")
        self.assertEqual(controller.freeze("structure"), selection)
        self.assertEqual(controller.ask("lai")["trial_number"], 0)

    def test_v2_restart_schedule_is_reproducible(self):
        sequences = []
        for name in ("repeat-a", "repeat-b"):
            sequence = []
            for number in range(4):
                controller = search.SearchController(self.root / name, self.plan)
                recipe = controller.ask("lai")
                sequence.append(recipe["config"])
                self.submit(controller, recipe, legacy.receipt(recipe, none=2. + number, rare=1. + number))
            sequences.append(sequence)
        self.assertEqual(*sequences)

    def test_failed_adaptive_trial_spends_the_last_slot(self):
        for _ in range(3):
            recipe = self.controller.ask("lai")
            self.submit(self.controller, recipe)
        final = self.controller.ask("lai")
        self.controller.fail("lai", final["trial_number"], "synthetic operational failure")
        with self.assertRaisesRegex(search.ContractError, "budget"):
            search.SearchController(self.root / "state", self.plan).ask("lai")
        self.assertEqual(self.controller.status("lai")["issued"], 4)
        self.assertFalse(self.controller.freeze("lai")["training_launched"])

    def test_v2_cli_and_nonrunnable_example(self):
        plan_path = self.root / "contextual.json"
        plan_path.write_text(json.dumps(self.plan))
        command = [sys.executable, str(legacy.SOURCE), "--plan", str(plan_path), "--state-dir",
                   str(self.root / "cli-v2"), "--objective", "lai"]
        proposed = json.loads(subprocess.check_output([*command, "ask"], text=True))
        self.assertEqual(proposed["schema"], search.CONTEXTUAL_SCHEMA)
        self.assertEqual(proposed["config"]["family"], "lai_cnn")
        result_path = self.root / "cli-v2-receipt.json"
        result_path.write_text(json.dumps(legacy.receipt(proposed)))
        _, sha = search.load_json(result_path)
        result = json.loads(subprocess.check_output([*command, "tell", "--receipt", str(result_path),
                                                     "--receipt-sha256", sha], text=True))
        self.assertEqual(result["selection_loss"], 1.5)
        example, _ = search.load_json(legacy.SOURCE.parents[1] / "conf" / "genomic_adaptive_search.contextual.example.json")
        self.assertEqual(example["schema"], search.CONTEXTUAL_SCHEMA)
        self.assertEqual(set(example["search_space"]), search.CONTEXTUAL_FIELDS)
        self.assertEqual(example["max_recipes"], 24)
        self.assertEqual(example["startup_trials"], 12)
        with self.assertRaises(search.ContractError):
            search.SearchController(self.root / "not-runnable", example)
        self.assertFalse((self.root / "not-runnable").exists())


if __name__ == "__main__":
    unittest.main()
