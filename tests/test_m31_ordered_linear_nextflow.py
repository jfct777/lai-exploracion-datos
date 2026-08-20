import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "conf" / "m31_ordered_linear.config"
MODULE = ROOT / "modules" / "31_ORDERED_LINEAR.nf"
WORKFLOW = ROOT / "workflows" / "m31_ordered_linear.nf"


class M31OrderedLinearNextflowContractTest(unittest.TestCase):
    def test_workflow_wires_every_frozen_input_for_both_roots(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        for root in ("root17", "root18"):
            for item in ("sites", "target", "tree", "pools", "truth", "flare_vcf", "flare_audit"):
                self.assertIn(f"m31_ordered_linear_{root}_{item}", text)
        self.assertIn("m31_ordered_linear_genetic_map", text)
        self.assertIn("m31_ordered_linear_preregistration", text)
        self.assertIn("20260817", text)
        self.assertIn("20260818", text)

    def test_workflow_freezes_code_and_container_provenance(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        module = MODULE.read_text(encoding="utf-8")
        self.assertIn("repositoryHead", workflow)
        self.assertIn("[0-9a-f]{40}", workflow)
        self.assertIn("MessageDigest.getInstance('SHA-256')", workflow)
        self.assertIn("[0-9a-f]{64}", workflow)
        self.assertIn("@sha256:", workflow)
        self.assertIn("git_commit: repositoryHead", workflow)
        self.assertIn("code_sha256: codeSha256", workflow)
        self.assertIn("preregistration_sha256: contractSha256", workflow)
        self.assertIn("base64 -d", module)

    def test_module_is_fail_closed_bounded_and_never_overwrites(self):
        text = MODULE.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("overwrite: false", text)
        self.assertIn("params.m31_ordered_linear_cpus", text)
        self.assertIn("params.m31_ordered_linear_memory", text)
        self.assertIn("params.m31_ordered_linear_time", text)
        self.assertIn("m31_ordered_linear.selftest.json", text)
        self.assertIn("m31_ordered_linear.input_sha256.tsv", text)
        self.assertIn("m31_ordered_linear.provenance.json", text)
        self.assertIn("--contract", text)
        self.assertIn("--selftest", text)
        self.assertIn("--genetic-map", text)
        self.assertIn("--root17-flare-audit", text)
        self.assertIn("--root18-flare-audit", text)
        self.assertNotIn("sbatch", text)

    def test_config_defaults_to_single_local_task_with_fixed_resources(self):
        text = CONFIG.read_text(encoding="utf-8")
        self.assertIn("executor = 'local'", text)
        self.assertIn("maxRetries = 0", text)
        self.assertIn("maxForks = 1", text)
        self.assertIn("m31_ordered_linear_cpus = 8", text)
        self.assertIn("m31_ordered_linear_memory = '16 GB'", text)
        self.assertIn("m31_ordered_linear_time = '2h'", text)
        self.assertIn("m31_ordered_linear_container_image = null", text)
        self.assertIn("m31_ordered_linear_container_digest = null", text)
        self.assertNotIn("google-batch", text)
        self.assertNotIn("resourceLabels", text)

    def test_results_are_run_scoped_and_existing_directories_fail_closed(self):
        config = CONFIG.read_text(encoding="utf-8")
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("DNABR_RUN_ID", config)
        self.assertIn("results/runs/", config)
        self.assertIn("resultsDir.exists()", workflow)
        self.assertIn("will not be reused or overwritten", workflow)

    def test_no_validation_or_test_partition_can_be_wired(self):
        combined = "\n".join(
            path.read_text(encoding="utf-8").lower()
            for path in (CONFIG, MODULE, WORKFLOW)
        )
        self.assertNotIn("source_valid", combined)
        self.assertNotIn("source_test", combined)
        self.assertNotIn("validation", combined)
        self.assertNotIn("test_partition", combined)


if __name__ == "__main__":
    unittest.main()
