import importlib.util
import copy
import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODULE_PATH = Path(os.environ.get("M32_CONTRACT_SCRIPT_PATH", ROOT / "bin" / "m32_locus_contract.py"))
SPEC = importlib.util.spec_from_file_location("m32_locus_contract", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class M32ContractTest(unittest.TestCase):
    def test_frozen_contract_is_smoke_only_and_order_preserving(self):
        contract = MODULE.load_contract(ROOT / "conf" / "m32_locus_sequence_smoke_preregistration.json")
        self.assertTrue(contract["representation"]["preserve_each_locus"])
        self.assertTrue(contract["representation"]["preserve_genetic_order"])
        self.assertTrue(contract["representation"]["full_flare_grid"])
        self.assertFalse(contract["future_metrics"]["scientific_run_authorized"])
        self.assertEqual(contract["future_split_contract"]["independent_unit"], "complete_diploid_individual")
        self.assertEqual(contract["future_split_contract"]["root17_root18"], "consumed_known_answer_only")

    def test_contract_rejects_aggregation_and_truth_in_producer(self):
        source = json.loads((ROOT / "conf" / "m32_locus_sequence_smoke_preregistration.json").read_text())
        source["representation"]["preserve_each_locus"] = False
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "individual loci"):
                MODULE.load_contract(path)
        contract = json.loads((ROOT / "conf" / "m32_locus_sequence_smoke_preregistration.json").read_text())
        self.assertEqual(contract["future_split_contract"]["truth_access"]["tensor_producer"], "never")

    def test_contract_rejects_load_bearing_mutations(self):
        source = json.loads((ROOT / "conf" / "m32_locus_sequence_smoke_preregistration.json").read_text())
        mutations = []
        bad = copy.deepcopy(source)
        bad["controls"]["rare_channel_disabled_ablation"]["same_parameter_count"] = False
        mutations.append(bad)
        bad = copy.deepcopy(source)
        bad["controls"]["matched_common_locus_control"]["same_tensor_slots"] = False
        mutations.append(bad)
        bad = copy.deepcopy(source)
        bad["future_split_contract"]["truth_access"]["tensor_producer"] = "TRAIN"
        mutations.append(bad)
        bad = copy.deepcopy(source)
        bad["future_split_contract"]["role_disjunction_required"].remove("IBD_component")
        mutations.append(bad)
        bad = copy.deepcopy(source)
        bad["future_model_screen"]["maximum_cnn_configurations"] = 80
        mutations.append(bad)
        bad = copy.deepcopy(source)
        bad["future_metrics"]["minimum_relevant_delta_F1"] = 0.0
        mutations.append(bad)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for index, contract in enumerate(mutations):
                path = Path(tmp) / f"bad-{index}.json"
                path.write_text(json.dumps(contract))
                with self.assertRaises(ValueError):
                    MODULE.load_contract(path)

    def test_git_commit_is_exact(self):
        self.assertEqual(MODULE.validate_git_commit("a" * 40), "a" * 40)
        with self.assertRaisesRegex(ValueError, "git_commit"):
            MODULE.validate_git_commit("unknown")


if __name__ == "__main__":
    unittest.main()
