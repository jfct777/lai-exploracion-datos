import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


def load(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SELECTOR = load("prepare_m29_root_b0", "bin/prepare_m29_root_b0.py")
MATERIALIZER = load("materialize_m29_root_b0", "bin/materialize_m29_root_b0.py")
INGEST = load("ingest_m29_root_b0", "bin/ingest_m29_root_b0.py")


class M29RootB0ProductionTest(unittest.TestCase):
    def test_contract_has_two_distinct_authenticated_roots(self):
        contract = json.loads((ROOT / "conf/m29_root_b0_production_preregistration.json").read_text())
        self.assertEqual([row["root_seed"] for row in contract["roots"]], [20260817, 20260818])
        self.assertEqual([row["m28b_mode"] for row in contract["roots"]], ["development", "validation"])
        for row in contract["roots"]:
            self.assertEqual(set(row["sha256"]), {"tree", "pools", "preflight_report", "preflight_manifest", "mosaic_events"})
        self.assertNotEqual(contract["roots"][0]["sha256"]["tree"], contract["roots"][1]["sha256"]["tree"])

    def test_selector_writes_b0_even_without_geometry_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            m28b = tmp / "m28b.json"
            m28b.write_text('{"stage":"stub"}')
            contract = tmp / "contract.json"
            contract.write_text(json.dumps({"stage": "M29_ROOT_B0_PRODUCTION", "status": "PRE_FROZEN_BEFORE_B0_SELECTION", "roots": [{"root_seed": 20260817, "m28b_mode": "development", "sha256": {"tree": "x", "pools": "x", "preflight_report": "x", "preflight_manifest": "x"}}], "expected": {"b0_markers": 2}, "shared_inputs": {"m28b_v5_contract_sha256": "contract-hash", "m28_contract_sha256": "x", "reproducibility_receipt_sha256": "x", "genetic_map_sha256": "x", "baseline_template": {"sha256": "x"}}}))
            args = SimpleNamespace(production_contract=contract, root_seed=20260817, m28b_contract=m28b, preregistration=m28b, outdir=tmp / "out")
            prepared = {"b0": [object(), object()], "common": [1, 2, 3], "hashes": {"tree_sequence": "x", "pool_manifest": "x", "development_preflight_report": "x", "development_preflight_manifest": "x", "m28_preregistration": "x", "m28_v2_reproducibility": "x", "genetic_map": "x", "baseline_template": "x"}}
            def fake_write(path, *_args, **_kwargs):
                path.write_bytes(b"B0")
            with patch.object(SELECTOR, "sha256", side_effect=lambda path: "contract-hash" if path == m28b else "output-hash"), patch.object(SELECTOR, "prepare_markers", return_value=prepared) as prepare, patch.object(SELECTOR, "write_marker_manifest", side_effect=fake_write):
                report = SELECTOR.run(args)
            prepare.assert_called_once()
            self.assertEqual(report["counts"]["B0"], 2)
            self.assertFalse(report["BR_BS_geometry_evaluated"])
            self.assertFalse(report["truth_read"])

    def test_selector_rejects_root_hash_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            m28b = tmp / "m28b.json"
            m28b.write_text('{}')
            contract = tmp / "contract.json"
            contract.write_text(json.dumps({"stage": "M29_ROOT_B0_PRODUCTION", "status": "PRE_FROZEN_BEFORE_B0_SELECTION", "roots": [{"root_seed": 20260817, "m28b_mode": "development", "sha256": {"tree": "wrong", "pools": "x", "preflight_report": "x", "preflight_manifest": "x"}}], "expected": {"b0_markers": 1}, "shared_inputs": {"m28b_v5_contract_sha256": "contract-hash", "m28_contract_sha256": "x", "reproducibility_receipt_sha256": "x", "genetic_map_sha256": "x", "baseline_template": {"sha256": "x"}}}))
            args = SimpleNamespace(production_contract=contract, root_seed=20260817, m28b_contract=m28b, preregistration=m28b, outdir=tmp / "out")
            prepared = {"b0": [object()], "common": [], "hashes": {"tree_sequence": "x", "pool_manifest": "x", "development_preflight_report": "x", "development_preflight_manifest": "x", "m28_preregistration": "x", "m28_v2_reproducibility": "x", "genetic_map": "x", "baseline_template": "x"}}
            with patch.object(SELECTOR, "sha256", return_value="contract-hash"), patch.object(SELECTOR, "prepare_markers", return_value=prepared):
                with self.assertRaisesRegex(ValueError, "root inputs disagree"):
                    SELECTOR.run(args)

    def test_materializer_authenticates_root_and_reuses_existing_logic(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            files = {}
            for name in ("tree", "pools", "mosaic", "b0"):
                files[name] = tmp / name
                files[name].write_text(name)
            contract = tmp / "contract.json"
            hashes = {"tree": "tree-h", "pools": "pool-h", "mosaic_events": "mosaic-h"}
            contract.write_text(json.dumps({"stage": "M29_ROOT_B0_PRODUCTION", "status": "PRE_FROZEN_BEFORE_B0_SELECTION", "roots": [{"root_seed": 20260817, "sha256": hashes}]}))
            selection = tmp / "selection.json"
            selection.write_text(json.dumps({"stage": "M29_ROOT_B0_SELECTION", "root_seed": 20260817, "decision": "GO_ROOT_B0_MATERIALIZATION", "production_contract_sha256": "meta-h", "output_sha256": {"b0": "b0-h"}}))
            out = tmp / "out"
            out.mkdir()
            args = SimpleNamespace(production_contract=contract, root_seed=20260817, selection_report=selection, b0_markers=files["b0"], tree_sequence=files["tree"], pool_manifest=files["pools"], mosaic_events=files["mosaic"], outdir=out)
            mapping = {files["tree"]: "tree-h", files["pools"]: "pool-h", files["mosaic"]: "mosaic-h", files["b0"]: "b0-h"}
            with patch.object(MATERIALIZER, "sha256", side_effect=lambda path: mapping.get(path, "meta-h")), patch.object(MATERIALIZER, "materialize", return_value={"decision": "GO_EXTERNAL_GNOMIX_INGEST_VALIDATION", "root_seed": 20260817}) as reused:
                report = MATERIALIZER.run(args)
            reused.assert_called_once()
            self.assertTrue(report["dev_root_only"])

    def test_materializer_rejects_selection_from_another_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            contract = tmp / "contract.json"
            contract.write_text(json.dumps({"stage": "M29_ROOT_B0_PRODUCTION", "status": "PRE_FROZEN_BEFORE_B0_SELECTION", "roots": [{"root_seed": 20260817, "sha256": {}}]}))
            selection = tmp / "selection.json"
            selection.write_text(json.dumps({"stage": "M29_ROOT_B0_SELECTION", "root_seed": 20260817, "decision": "GO_ROOT_B0_MATERIALIZATION", "production_contract_sha256": "wrong"}))
            args = SimpleNamespace(production_contract=contract, root_seed=20260817, selection_report=selection)
            with patch.object(MATERIALIZER, "sha256", return_value="right"):
                with self.assertRaisesRegex(ValueError, "another production contract"):
                    MATERIALIZER.run(args)

    def test_ingest_rejects_materialization_from_another_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            contract = tmp / "contract.json"
            contract.write_text(json.dumps({"stage": "M29_ROOT_B0_PRODUCTION", "status": "PRE_FROZEN_BEFORE_B0_SELECTION", "roots": [{"root_seed": 20260817}], "software": {"gnomix_commit": "x"}}))
            materialization = tmp / "materialization.json"
            materialization.write_text(json.dumps({"m29_stage": "M29_ROOT_B0_MATERIALIZATION", "root_seed": 20260818, "m29_production_contract_sha256": "hash"}))
            args = SimpleNamespace(production_contract=contract, root_seed=20260817, materialization_report=materialization)
            with patch.object(INGEST, "sha256", return_value="hash"):
                with self.assertRaisesRegex(ValueError, "another stage or root"):
                    INGEST.run(args)

    def test_workflow_contains_no_training_or_truth_input(self):
        workflow = (ROOT / "workflows/m29_root_b0_production.nf").read_text().lower()
        module = (ROOT / "modules/29_ROOT_B0_PRODUCTION.nf").read_text().lower()
        self.assertNotIn("train_m28", workflow + module)
        self.assertNotIn("lai_truth", workflow + module)
        self.assertIn("prepare_m29_root_b0.py", workflow)
        self.assertIn("materialize_m28c_b0_inputs.py", workflow)
        self.assertIn("audit_m28c_gnomix_ingest.py", workflow)
        self.assertIn("write_m29_root_b0_provenance", workflow + module)
        self.assertIn("dnabr_git_commit", workflow)
        self.assertIn("run_provenance", module)

    def test_nextflow_config_is_isolated_and_fail_closed(self):
        config = (ROOT / "conf/m29_root_b0_production.config").read_text()
        self.assertIn("maxRetries = 0", config)
        self.assertIn("executor = 'local'", config)
        self.assertNotIn("google", config.lower())
        run_config = (ROOT / "conf" / "m29_root_b0_production_20260818a.config").read_text()
        self.assertIn("--network none --user 1017:1020", run_config)


if __name__ == "__main__":
    unittest.main()
