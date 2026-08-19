import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class M31NextflowContractTest(unittest.TestCase):
    def test_workflow_uses_only_preflight_inputs_for_two_authenticated_roots(self):
        text = (ROOT / "workflows" / "m31_ordered_rare_preflight.nf").read_text()
        for token in ("root17", "20260817", "root18", "20260818", "tree", "pools", "catalog", "haplotypes"):
            self.assertIn(token, text)
        for prohibited in ("truth", "gnomix", "flare", "KING", "king"):
            self.assertNotIn(prohibited, text)
        self.assertIn("repositoryHead", text)
        self.assertIn("[0-9a-f]{40}", text)

    def test_module_is_nextflow_first_bounded_and_exports_manifest(self):
        text = (ROOT / "modules" / "31_ORDERED_RARE_PREFLIGHT.nf").read_text()
        self.assertIn("set -euo pipefail", text)
        self.assertIn("m31_ordered_rare.manifest.json", text)
        self.assertIn("params.m31_preflight_memory", text)
        self.assertIn("overwrite: false", text)
        self.assertIn("--git-commit", text)
        self.assertNotIn("sbatch", text)

    def test_dedicated_config_is_local_smoke_and_has_no_cloud_labels(self):
        text = (ROOT / "conf" / "m31_ordered_rare_preflight.config").read_text()
        self.assertIn("executor = 'local'", text)
        self.assertIn("maxRetries = 0", text)
        self.assertIn("maxForks = 1", text)
        self.assertNotIn("google-batch", text)
        self.assertNotIn("resourceLabels", text)


if __name__ == "__main__":
    unittest.main()
