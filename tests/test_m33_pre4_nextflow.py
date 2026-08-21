import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class M33NextflowContractTests(unittest.TestCase):
    def test_workflow_is_contract_only(self):
        workflow = (ROOT / "workflows/m33_pre4_contract.nf").read_text(encoding="utf-8")
        self.assertIn("M33_VALIDATE_PRE4_CONTRACT", workflow)
        for forbidden in ("truth", "train", "optimizer", "checkpoint"):
            self.assertNotIn(forbidden, workflow.lower())

    def test_module_has_no_truth_or_training_interface(self):
        module = (ROOT / "modules/33_PRE4_CONTRACT.nf").read_text(encoding="utf-8")
        self.assertIn("overwrite: false", module)
        self.assertIn("--network none", (ROOT / "conf/m33_pre4_contract.config").read_text(encoding="utf-8"))
        for forbidden in ("path truth", "--truth", "train_model", "optimizer"):
            self.assertNotIn(forbidden, module.lower())

    def test_validator_rechecks_every_staged_source_and_publishes_auth(self):
        module = (ROOT / "modules/33_PRE4_CONTRACT.nf").read_text(encoding="utf-8")
        for path in ("bin/m33_pre4_source_auth.py", "bin/m33_pre4_contract.py",
                     "conf/m33_pre4_preregistration.json", "conf/m33_pre4_contract.config",
                     "modules/33_PRE4_CONTRACT.nf", "workflows/m33_pre4_contract.nf",
                     "tests/test_m33_pre4_contract.py", "tests/test_m33_pre4_nextflow.py"):
            self.assertIn(f"--staged-source {path}=", module)
        self.assertGreaterEqual(module.count("publishDir"), 2)

    def test_runtime_sources_are_complete(self):
        auth = (ROOT / "bin/m33_pre4_source_auth.py").read_text(encoding="utf-8")
        for path in ("bin/m33_pre4_contract.py", "conf/m33_pre4_preregistration.json",
                     "modules/33_PRE4_CONTRACT.nf", "workflows/m33_pre4_contract.nf"):
            self.assertIn(path, auth)


if __name__ == "__main__":
    unittest.main()
