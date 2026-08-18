import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load("build_m29_root_gnomix_contract", "bin/build_m29_root_gnomix_contract.py")
BINDER = load("bind_m29_b0_inputs", "bin/bind_m29_b0_inputs.py")
RSS = load("run_with_rss_guard", "bin/run_with_rss_guard.py")


class M29RootGnomixB0Test(unittest.TestCase):
    def test_preregistration_authenticates_unchanged_m28c_inputs_and_resources(self):
        pre = json.loads((ROOT / "conf/m29_root_gnomix_b0_preregistration.json").read_text())
        expected = pre["authenticated_templates"]
        self.assertEqual(BUILDER.sha256(ROOT / "conf/m29_root_b0_production_preregistration.json"), expected["production_contract_sha256"])
        self.assertEqual(BUILDER.sha256(ROOT / "conf/m28c_gnomix_full_b0_preregistration.json"), expected["full_b0_contract_sha256"])
        self.assertEqual(BUILDER.sha256(ROOT / "conf/m28c_gnomix_full_b0.yaml"), expected["gnomix_config_sha256"])
        self.assertEqual(BUILDER.sha256(ROOT / "bin/m28c_gnomix_training_smoke.py"), expected["runner_sha256"])
        self.assertEqual(pre["resources"]["memory_per_root"], "8 GB")
        self.assertEqual(pre["resources"]["fail_closed_peak_rss_gib"], 6.4)
        self.assertEqual(pre["resources"]["max_parallel_roots"], 2)

    def test_builder_binds_one_root_and_frozen_resource_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            template = tmp / "template.json"
            template.write_text(json.dumps({
                "stage": "M28C_GNOMIX_FULL_B0_RESOURCE_BENCHMARK",
                "status": "PRE_FROZEN_AMENDED_BEFORE_SUCCESSFUL_FULL_B0",
                "scope": "old",
                "source_panel": {"first_position": 1, "last_position": 2},
                "execution": {}, "numerical_guard": {"probability_tolerance": 1e-12, "negative_mass_tolerance": 1e-12},
                "resources": {},
            }))
            paths = {}
            for name in ("pre", "producer", "selection", "selection_manifest", "materialization", "materialization_manifest", "ingest", "ingest_manifest", "reference", "reference_tbi", "target", "target_tbi", "sample_map", "b0", "map", "config", "runner"):
                paths[name] = tmp / name
                paths[name].write_text(name)
            paths["pre"].write_text(json.dumps({"stage": "M29_ROOT_GNOMIX_B0", "status": "PRE_FROZEN_BEFORE_ROOT_TRAINING", "roots": {"root17": 20260817}, "resources": {"memory_per_root": "8 GB", "fail_closed_peak_rss_gib": 6.4, "max_parallel_roots": 2}, "authenticated_templates": {"production_contract_sha256": "producer-hash", "full_b0_contract_sha256": "template-hash", "gnomix_config_sha256": "config-hash", "runner_sha256": "runner-hash"}}))
            paths["producer"].write_text(json.dumps({"stage": "M29_ROOT_B0_PRODUCTION", "status": "PRE_FROZEN_BEFORE_B0_SELECTION", "roots": [{"label": "root17", "root_seed": 20260817}]}))
            paths["selection"].write_text(json.dumps({"root_seed": 20260817, "decision": "GO_ROOT_B0_MATERIALIZATION", "production_contract_sha256": "producer-hash", "output_sha256": {"b0": "b0-hash"}}))
            paths["materialization"].write_text(json.dumps({"root_seed": 20260817, "m29_stage": "M29_ROOT_B0_MATERIALIZATION", "decision": "GO_EXTERNAL_GNOMIX_INGEST_VALIDATION", "m29_production_contract_sha256": "producer-hash", "m29_selection_report_sha256": "hash-selection"}))
            paths["ingest"].write_text(json.dumps({"root_seed": 20260817, "m29_stage": "M29_ROOT_B0_GNOMIX_INGEST", "decision": "GO_M29_ROOT_B0_READY_FOR_TRAINING", "m29_production_contract_sha256": "producer-hash", "output_sha256": {"reference": "reference-hash", "reference_tbi": "reference_tbi-hash", "target": "target-hash", "target_tbi": "target_tbi-hash"}}))
            out = tmp / "out.json"
            args = SimpleNamespace(
                root_label="root17", root_seed=20260817, preregistration=paths["pre"], production_contract=paths["producer"],
                template_contract=template, selection_report=paths["selection"], selection_manifest=paths["selection_manifest"],
                materialization_report=paths["materialization"], materialization_manifest=paths["materialization_manifest"],
                ingest_report=paths["ingest"], ingest_manifest=paths["ingest_manifest"], reference_vcf=paths["reference"],
                reference_tbi=paths["reference_tbi"], target_vcf=paths["target"], target_tbi=paths["target_tbi"],
                sample_map=paths["sample_map"], b0_markers=paths["b0"], genetic_map=paths["map"], gnomix_config=paths["config"],
                runner=paths["runner"], gnomix_root=tmp, out=out,
            )
            hash_map = {paths["producer"]: "producer-hash", template: "template-hash", paths["config"]: "config-hash", paths["runner"]: "runner-hash", paths["b0"]: "b0-hash", paths["reference"]: "reference-hash", paths["reference_tbi"]: "reference_tbi-hash", paths["target"]: "target-hash", paths["target_tbi"]: "target_tbi-hash"}
            with patch.object(BUILDER, "sha256", side_effect=lambda path: hash_map.get(path, f"hash-{path.name}")), patch.object(BUILDER, "manifest_authenticates"), patch.object(BUILDER, "marker_positions", return_value=list(range(1, 79792))), patch.object(BUILDER, "audit_breakpoint_probability_map", return_value={"negative_probability_count": 7}):
                contract = BUILDER.run(args)
            self.assertEqual(contract["m29_binding"]["root_seed"], 20260817)
            self.assertEqual(contract["numerical_guard"]["expected_negative_probability_count"], 7)
            self.assertEqual(contract["resources"]["memory_per_training"], "8 GB")
            self.assertEqual(contract["resources"]["peak_rss_stop_gib"], 6.4)

    def _binding_fixture(self, tmp: Path) -> list[str]:
        fb, msp = tmp / "query_results.fb", tmp / "query_results.msp"
        fb.write_text("fb")
        msp.write_text("msp")
        runtime = tmp / "runtime.json"
        runtime.write_text(json.dumps({"resources": {"memory_per_training": "8 GB", "peak_rss_stop_gib": 6.4}, "m29_binding": {"stage": "M29_ROOT_GNOMIX_B0", "root_label": "root17", "root_seed": 20260817}}))
        report = tmp / "inference.json"
        report.write_text(json.dumps({"decision": "GO_REPLICATE_COMPARISON_NO_TRUTH", "replicate": "root17", "contract_sha256": BINDER.sha256(runtime), "output_sha256": {fb.name: BINDER.sha256(fb), msp.name: BINDER.sha256(msp)}}))
        rss = tmp / "training_rss_gate.json"
        rss.write_text(json.dumps({"stage": "M29_PROCESS_TREE_RSS_GATE", "decision": "PASS_RSS_GATE", "threshold_exceeded": False, "max_rss_gib": 6.4, "peak_rss_gib": 3.5}))
        manifest = tmp / "manifest.json"
        manifest.write_text(json.dumps({"stage": "M29_ROOT_GNOMIX_INFER_root17", "inputs": {rss.name: BINDER.sha256(rss)}, "sha256": {fb.name: BINDER.sha256(fb), msp.name: BINDER.sha256(msp)}}))
        provenance = tmp / "run_provenance.json"
        provenance.write_text(json.dumps({"scientific_scope": "M29 root-specific B0 training and inference; no truth or effect estimation"}))
        return ["bind_m29_b0_inputs.py", "--root-seed", "20260817", "--fb", str(fb), "--msp", str(msp), "--inference-report", str(report), "--inference-manifest", str(manifest), "--runtime-contract", str(runtime), "--training-rss-gate", str(rss), "--run-provenance", str(provenance), "--out", str(tmp / "binding.json")]

    def test_binding_is_exact_and_fails_on_cross_root_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            argv = self._binding_fixture(tmp)
            with patch.object(sys, "argv", argv):
                BINDER.main()
            result = json.loads((tmp / "binding.json").read_text())
            self.assertEqual(result["stage"], "M29_AUTHENTICATED_B0_BINDING")
            report_path = Path(argv[argv.index("--inference-report") + 1])
            report = json.loads(report_path.read_text())
            report["replicate"] = "root18"
            report_path.write_text(json.dumps(report))
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "another contract/root"):
                    BINDER.main()

    def test_rss_tree_accounting_and_command_failure_are_fail_closed(self):
        table = {10: (1, 100), 11: (10, 200), 12: (11, 300), 20: (1, 999)}
        self.assertEqual(RSS.process_tree_rss_kib(10, table), 600)
        with tempfile.TemporaryDirectory() as temporary:
            report = RSS.run([sys.executable, "-c", "raise SystemExit(3)"], Path(temporary) / "rss.json", 1.0, 0.01)
            self.assertEqual(report["decision"], "STOP_COMMAND_FAILED")
            self.assertEqual(report["command_returncode"], 3)
        with tempfile.TemporaryDirectory() as temporary:
            report = RSS.run([sys.executable, "-c", "import time; x=bytearray(4*1024*1024); time.sleep(0.2)"], Path(temporary) / "rss.json", 0.001, 0.01)
            self.assertEqual(report["decision"], "STOP_RSS_LIMIT_EXCEEDED")
            self.assertTrue(report["threshold_exceeded"])

    def test_workflow_is_parallel_bounded_and_training_has_no_target_or_truth(self):
        workflow = (ROOT / "workflows/m29_root_gnomix_b0.nf").read_text()
        module = (ROOT / "modules/29_ROOT_GNOMIX_B0.nf").read_text()
        config = (ROOT / "conf/m29_root_gnomix_b0.config").read_text()
        training = module.split("process TRAIN_M29_ROOT_GNOMIX_B0", 1)[1].split("process INFER_M29_ROOT_GNOMIX_B0", 1)[0]
        training_inputs = training.split("input:", 1)[1].split("output:", 1)[0]
        self.assertNotIn("target_vcf", training_inputs)
        self.assertNotIn("truth", training_inputs.lower())
        self.assertNotIn("mosaic", workflow.lower() + module.lower())
        self.assertIn(".join(VALIDATE_M29_ROOT_GNOMIX_B0.out.targets_ready, by: 0)", workflow)
        self.assertIn("maxForks params.m29_gnomix_max_parallel_roots", module)
        self.assertIn('m29_gnomix_max_parallel_roots = 2', config)
        self.assertIn('m29_gnomix_memory = "8 GB"', config)
        self.assertIn('m29_gnomix_peak_rss_stop_gib = 6.4', config)
        self.assertIn("--network none --user 1017:1020", config)
        self.assertIn("maxRetries = 0", config)

    def test_same_locus_root_a_is_seed17_b_without_changing_scientific_arrays(self):
        contract = json.loads((ROOT / "conf/m29_same_locus_dev_preregistration.json").read_text())
        root = contract["roots"]["root_a"]
        self.assertEqual(root["source_attempt"], "seed17-b")
        self.assertEqual(root["sha256"]["tree"], "566946ec93cbc28c7f36d27878e12d091f1c87ebdd6fb6423b940f4f778391f7")
        self.assertEqual(root["sha256"]["catalog"], "3162022c6882aa184094f96f6e8d49b6917b5fbe5a45ecfabf6cc4a8265b366c")
        self.assertEqual(root["sha256"]["haplotypes"], "015f7c420730537cb2d9173971e4767d4d561d11ed22ce2b46149c9eb7832122")
        self.assertEqual(root["sha256"]["truth"], "bf059f94ba40b033a75bb2d4ea782ce2d8f6cb86098a77e4812ee019f07da303")


if __name__ == "__main__":
    unittest.main()
