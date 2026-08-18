import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class M29NextflowContractTest(unittest.TestCase):
    def test_workflow_requires_root_specific_b0_bindings(self):
        text = (ROOT / "workflows" / "m29_same_locus_dev.nf").read_text()
        for token in ("m29_root_a_fb", "m29_root_a_msp", "m29_root_a_binding", "m29_root_b_fb", "m29_root_b_msp", "m29_root_b_binding"):
            self.assertIn(token, text)
        self.assertIn("historical M28C predictions are not valid substitutes", text)

    def test_module_stops_on_failure_and_exports_a_manifest(self):
        text = (ROOT / "modules" / "29_SAME_LOCUS_DEV.nf").read_text()
        self.assertIn("set -euo pipefail", text)
        self.assertIn("m29_dev.manifest.json", text)
        self.assertEqual(text.count("stageAs: 'root_a/*'"), 10)
        self.assertEqual(text.count("stageAs: 'root_b/*'"), 10)
        self.assertNotIn("M28E", text)

    def test_global_nextflow_config_is_not_part_of_m29(self):
        text = (ROOT / "conf" / "m29_same_locus_dev.config").read_text()
        self.assertIn("maxRetries = 0", text)
        self.assertIn("executor = 'local'", text)


if __name__ == "__main__":
    unittest.main()
