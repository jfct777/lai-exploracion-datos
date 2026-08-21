import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "modules/33_ASSET_EXECUTION_CONTRACT.nf").read_text(encoding="utf-8")
WORKFLOW = (ROOT / "workflows/m33_asset_execution_contract.nf").read_text(encoding="utf-8")
CONFIG = (ROOT / "conf/m33_asset_execution_contract.config").read_text(encoding="utf-8")


class M33AssetExecutionNextflowTests(unittest.TestCase):
    def test_workflow_has_only_source_auth_and_fixture_contract_processes(self):
        self.assertEqual(len(re.findall(r"^process ", MODULE, flags=re.MULTILINE)), 2)
        self.assertIn("M33_AUTHENTICATE_ASSET_EXECUTION_SOURCES.out.auth", WORKFLOW)
        self.assertIn("unittest discover", MODULE)
        self.assertIn("overwrite: false", MODULE)

    def test_no_scientific_asset_input_or_cloud_command(self):
        combined = "\n".join((MODULE, WORKFLOW, CONFIG)).lower()
        for forbidden in (
            "path truth", "path target_vcf", "path ref_vcf", "path tree_sequence",
            "path genetic_map", "path mosaic_events", "path donor_to_target_provenance",
            "gsutil", "gcloud", "docker", "kubectl", "--allow-real-assets",
            "forward", "training", "nextflow run main.nf",
        ):
            self.assertNotIn(forbidden, combined)

    def test_local_fail_closed_resources(self):
        self.assertIn("executor = 'local'", CONFIG)
        self.assertIn("maxRetries = 0", CONFIG)
        self.assertIn("maxForks = 1", CONFIG)
        self.assertNotIn("google-batch", combined := "\n".join((MODULE, WORKFLOW, CONFIG)).lower())


if __name__ == "__main__":
    unittest.main()
