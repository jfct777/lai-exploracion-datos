import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class M32NextflowContractTest(unittest.TestCase):
    def test_module_is_bounded_and_non_overwriting(self):
        text = (ROOT / "modules" / "32_LOCUS_SEQUENCE_SMOKE.nf").read_text()
        self.assertIn("overwrite: false", text)
        self.assertIn("set -euo pipefail", text)
        self.assertIn("params.m32_smoke_cpus", text)
        self.assertIn("params.m32_smoke_memory", text)
        self.assertIn("--repository-root", text)
        self.assertIn("conf/m32_locus_sequence_smoke.config", text)
        self.assertNotIn("sbatch", text)

    def test_workflow_has_no_scientific_or_private_inputs(self):
        text = (ROOT / "workflows" / "m32_locus_sequence_smoke.nf").read_text().lower()
        for prohibited in ("truth", "root17", "root18", "gnomix", "king", "gs://"):
            self.assertNotIn(prohibited, text)
        self.assertIn("m32_smoke_git_commit", text)
        self.assertIn("m32_smoke_run_id", text)

    def test_config_is_local_one_cpu_two_gb_and_no_cloud(self):
        text = (ROOT / "conf" / "m32_locus_sequence_smoke.config").read_text()
        self.assertIn("m32_smoke_cpus = 1", text)
        self.assertIn("m32_smoke_memory = '2 GB'", text)
        self.assertIn("executor = 'local'", text)
        self.assertIn("maxForks = 1", text)
        self.assertNotIn("google-batch", text)
        self.assertNotIn("resourceLabels", text)


if __name__ == "__main__":
    unittest.main()
