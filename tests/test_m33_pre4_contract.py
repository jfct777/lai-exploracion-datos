import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m33_pre4_contract", ROOT / "bin/m33_pre4_contract.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)
CONTRACT = ROOT / "conf/m33_pre4_preregistration.json"


def payload():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


class M33ContractTests(unittest.TestCase):
    def test_contract_passes_and_all_execution_is_closed(self):
        value = MOD.load_contract(CONTRACT)
        self.assertEqual(value["status"], MOD.STATUS)
        self.assertEqual(MOD.sha256_file(CONTRACT), MOD.EXACT_CONTRACT_SHA256)
        self.assertFalse(value["execution_authorization"]["prospective_asset_generation"])
        self.assertFalse(value["execution_authorization"]["no_gradient_forward"])
        self.assertFalse(value["execution_authorization"]["training"])
        self.assertEqual(value["asset_manifest_gate"]["status"], "BLOCKED_PENDING_MANIFESTS")

    def test_roots_rotations_are_deterministic_unique_and_not_consumed(self):
        value = MOD.load_contract(CONTRACT)
        roots = value["root_registry"]
        found = []
        for role, count in MOD.ROLES.items():
            base = "EVAL" if role.startswith("EVAL_") else role
            expected = [MOD.derived_seed(base, index) for index in range(count)]
            self.assertEqual(roots[role], expected)
            found.extend(expected)
        self.assertEqual(len(found), len(set(found)))
        self.assertTrue(set(found).isdisjoint(roots["consumed_technical_only"]))
        self.assertEqual({row["score_only_root"] for row in roots["development_rotations"]},
                         set(roots["DEVELOPMENT"]))

    def test_every_top_level_scientific_block_is_byte_immutable(self):
        original = payload()
        for key in original:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(original)
                value = changed[key]
                if isinstance(value, dict):
                    value["unexpected_mutation"] = True
                elif isinstance(value, list):
                    value.append("unexpected_mutation")
                else:
                    changed[key] = f"{value}_unexpected_mutation"
                path = Path(directory) / "contract.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "immutable contract byte hash drift"):
                    MOD.load_contract(path)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.json"
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    MOD.strict_json(path)

    def test_forged_source_auth_is_rejected(self):
        staged = {relative: ROOT / relative for relative in MOD.REQUIRED_SOURCES}
        commit = "a" * 40
        forged = {
            "stage": "M33_PRE4_SOURCE_AUTH",
            "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
            "git_commit": commit,
            "source_sha256": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory incomplete"):
                MOD.validate_source_auth(path, commit, staged)

    def test_wrong_staged_hash_is_rejected(self):
        commit = "b" * 40
        hashes = {relative: MOD.sha256_file(ROOT / relative) for relative in MOD.REQUIRED_SOURCES}
        auth = {"stage": "M33_PRE4_SOURCE_AUTH", "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
                "git_commit": commit, "source_sha256": hashes}
        staged = {relative: ROOT / relative for relative in MOD.REQUIRED_SOURCES}
        with tempfile.TemporaryDirectory() as directory:
            auth_path = Path(directory) / "auth.json"
            auth_path.write_text(json.dumps(auth), encoding="utf-8")
            changed = Path(directory) / "changed.py"
            changed.write_text("changed", encoding="utf-8")
            staged["bin/m33_pre4_contract.py"] = changed
            with self.assertRaisesRegex(ValueError, "changed after authentication"):
                MOD.validate_source_auth(auth_path, commit, staged)

    def test_full_hash_auth_with_fake_commit_is_rejected_against_git(self):
        if shutil.which("git") is None:
            self.skipTest("Git binding is exercised by the host-only contract process")
        commit = "a" * 40
        hashes = {relative: MOD.sha256_file(ROOT / relative) for relative in MOD.REQUIRED_SOURCES}
        auth = {"stage": "M33_PRE4_SOURCE_AUTH", "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
                "git_commit": commit, "source_sha256": hashes}
        staged = {relative: ROOT / relative for relative in MOD.REQUIRED_SOURCES}
        with tempfile.TemporaryDirectory() as directory:
            auth_path = Path(directory) / "auth.json"
            auth_path.write_text(json.dumps(auth), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Git HEAD differs"):
                MOD.validate_source_auth(auth_path, commit, staged, ROOT)

    def test_controls_and_primary_estimand_are_explicit(self):
        value = MOD.load_contract(CONTRACT)
        controls = value["controls"]
        self.assertIn("conditional_on_rare_site_geometry", controls["rare_disabled"]["estimand"])
        self.assertEqual(controls["target_same_locus_sham"]["replicates"], 3)
        self.assertIn("per_person_dosage", controls["target_same_locus_sham"]["does_not_preserve"])
        self.assertEqual(controls["diagnostic_rule"],
                         "real_must_exceed_max_of_three_shams_or_controls_no_permutation_p_value")
        self.assertEqual(value["metrics"]["primary"],
                         "paired_boundary_F1_at_0.2_cM_RE_minus_RD_per_simulation_root")
        self.assertEqual(value["metrics"]["primary_inference_unit"], "simulation_root")

    def test_models_and_budget_are_fully_frozen_but_not_authorized(self):
        value = MOD.load_contract(CONTRACT)["model_screen"]
        self.assertEqual(value["families"], ["local_linear", "small_residual_cnn_1d"])
        self.assertEqual(value["candidate_grid"]["total_candidates"], 16)
        self.assertEqual(value["candidate_grid"]["training_seeds"], [1103, 2207, 3301])
        self.assertEqual(value["training"]["max_updates"], 2000)
        self.assertIn("RD_only_selects", value["training"]["checkpoint_rule"])
        self.assertIn("average_their_probability_tensors", value["selection"]["seed_aggregation"])
        self.assertTrue(value["training_blocked_until_T0_T1_POST"])

    def test_runtime_anchors_are_exact(self):
        self.assertEqual(MOD.EXPECTED_NEXTFLOW_VERSION, "26.04.6")
        self.assertEqual(MOD.EXPECTED_CONTAINER_DIGEST,
                         "sha256:2c30d018028636ac1b7a4890641e04b3e15be8c79d991dfade35b90db0e17bd1")


if __name__ == "__main__":
    unittest.main()
