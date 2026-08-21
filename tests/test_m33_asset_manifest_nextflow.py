import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "modules/33_ASSET_MANIFEST_CONTRACT.nf").read_text(encoding="utf-8")
WORKFLOW = (ROOT / "workflows/m33_asset_manifest_contract.nf").read_text(encoding="utf-8")
CONFIG = (ROOT / "conf/m33_asset_manifest_contract.config").read_text(encoding="utf-8")


class M33AssetManifestNextflowTests(unittest.TestCase):
    def test_workflow_is_contract_only(self):
        combined = "\n".join((MODULE, WORKFLOW, CONFIG)).lower()
        for forbidden in ("m28_simulation_preflight.py", "train", "predict", "--truth", "gsutil", "gcloud"):
            self.assertNotIn(forbidden, combined)
        self.assertEqual(len(re.findall(r"^process ", MODULE, flags=re.MULTILINE)), 2)

    def test_source_auth_precedes_contract_validation(self):
        self.assertIn("M33_AUTHENTICATE_ASSET_CONTRACT_SOURCES.out.auth", WORKFLOW)
        self.assertIn("overwrite: false", MODULE)
        self.assertIn("maxRetries = 0", CONFIG)
        self.assertIn("maxForks = 1", CONFIG)

    def test_no_scientific_assets_enter_any_process(self):
        for forbidden_input in ("path truth", "path targets", "path sites", "path flare", "path tree_sequence"):
            self.assertNotIn(forbidden_input, MODULE.lower())


if __name__ == "__main__":
    unittest.main()
