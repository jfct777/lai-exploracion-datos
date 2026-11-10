"""Small integration checks; no project genotypes or training are loaded."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch

import r02_lai_donor_experiment as experiment
from r02_lai_catalogue_policy import AVAILABLE_SCOPE
from r02_lai_donor_head_test import model, batch


ROOT = Path(__file__).resolve().parents[1]
GL = ROOT / ".claude/runs/r02-lai-integrated-inputs-20261108gl"
FROZEN = ROOT / ".claude/runs/r02-lai-parallel-20261108gg/training-v4/artifact/extracted/bin/r02_lai_donor_context.py"
FROZEN_SHA = "eef15415dabb13aadc00c6b7d61e8bbdc9edb83b6b57ec445f2774849b4082fe"


class DonorIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((GL / "training/prepared/lr1e3.v2.json").read_text())
        self.policy = dict(scope=AVAILABLE_SCOPE, input_spec_sha256="a"*64,
                           material_axis_sha256="b"*64, source_rare_total=9,
                           material_loci_total=7, source_rare_intersect_material=3)

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            return experiment.load_config(path)

    def new_config(self):
        self.config.pop("require_full_catalogue_coverage")
        self.config.update(schema=experiment.AVAILABLE_AXIS_SCHEMA,
                           catalogue_policy=self.policy)
        return self.config

    def test_legacy_guard_not_disabled_by_false(self):
        self.assertEqual(self.load(self.config), self.config)
        self.config["require_full_catalogue_coverage"] = False
        with self.assertRaisesRegex(ValueError, "legacy training requires"):
            self.load(self.config)

    def test_explicit_version_requires_policy_and_rejects_legacy_boolean(self):
        config = self.new_config()
        self.assertEqual(self.load(config), config)
        with self.assertRaises(ValueError):
            self.load({**config, "require_full_catalogue_coverage": False})
        config.pop("catalogue_policy")
        with self.assertRaises(ValueError):
            self.load(config)

    def test_head_options_are_propagated_and_both_required(self):
        config = self.new_config()
        for mode in ("factorized_linear", "mlp"):
            config["model"].update(final_head_mode=mode, final_head_hidden_width=6)
            built = experiment._model(self.load(config))
            self.assertEqual(built.configuration["final_head"]["mode"], mode)
        del config["model"]["final_head_hidden_width"]
        with self.assertRaisesRegex(ValueError, "specified together"):
            self.load(config)

    def test_guard_uses_authenticated_classified_axis_not_source_rows(self):
        reader = MagicMock()
        reader.__enter__.return_value.catalogue_coverage = dict(
            source_rare_total=9, material_loci_total=7, source_rare_intersect_material=3,
            source_rare_missing_from_material=6, complete_catalogue_covered=False,
            coverage_scope="LOCUS_AXIS_MEMBERSHIP_ONLY",
            classified_material_axis_sha256="b"*64)
        with patch.object(experiment, "DonorInputReader", return_value=reader):
            result = experiment.require_training_catalogue_coverage(
                object(), policy=self.policy, input_spec_sha256="a"*64)
            self.assertTrue(result["locus_scope_check_passed"])
            reader.__enter__.return_value.catalogue_coverage.pop(
                "classified_material_axis_sha256")
            with self.assertRaisesRegex(ValueError, "material axis differs"):
                experiment.require_training_catalogue_coverage(
                    object(), policy=self.policy, input_spec_sha256="a"*64)

    def test_failed_guard_creates_no_output_model_or_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            spec.write_text("{}")
            with patch.object(experiment, "load_config", return_value=self.new_config()), \
                 patch.object(experiment, "sources_from_spec", return_value=object()), \
                 patch.object(experiment, "require_training_catalogue_coverage",
                              side_effect=ValueError("scope rejected")), \
                 patch.object(experiment, "_model") as create_model, \
                 patch.object(experiment, "build_adam") as create_optimizer:
                with self.assertRaisesRegex(ValueError, "scope rejected"):
                    experiment.run(spec, root/"config.json", "NO_SOURCE_RARE", root/"output")
                self.assertFalse((root/"output").exists())
                create_model.assert_not_called()
                create_optimizer.assert_not_called()

    def test_default_matches_actual_frozen_module(self):
        self.assertEqual(hashlib.sha256(FROZEN.read_bytes()).hexdigest(), FROZEN_SHA)
        spec = importlib.util.spec_from_file_location("gp_frozen_donor", FROZEN)
        frozen = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = frozen
        spec.loader.exec_module(frozen)
        with patch.dict(model.__globals__, AutonomousDonorContextLAI=frozen.AutonomousDonorContextLAI):
            torch.manual_seed(731)
            old = model()
            old_rng = torch.get_rng_state().clone()
        torch.manual_seed(731)
        new = model()
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertEqual(old.configuration, new.configuration)
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)
        old.eval(); new.eval()
        values = batch().__dict__
        old_output = old(frozen.DonorContextBatch(**values))
        new_output = new(batch())
        for key, value in old_output.__dict__.items():
            torch.testing.assert_close(value, new_output.__dict__[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
