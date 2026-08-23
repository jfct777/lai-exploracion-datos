#!/usr/bin/env python3

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FactorizedLazyNextflowTests(unittest.TestCase):
    def test_workflow_is_contract_and_synthetic_only(self):
        module = (ROOT / "modules" / "33_M0_FACTORIZED_LAZY_CONTRACT.nf").read_text(encoding="utf-8")
        workflow = (ROOT / "workflows" / "m33_m0_factorized_lazy_contract.nf").read_text(encoding="utf-8")
        config = (ROOT / "conf" / "m33_m0_factorized_lazy_contract.config").read_text(encoding="utf-8")
        combined = "\n".join((module, workflow, config)).lower()
        self.assertIn("m33_validate_factorized_lazy_contract", combined)
        self.assertIn("unittest discover", module)
        self.assertIn("--network none", config)
        self.assertIn("overwrite: false", module)
        for forbidden in ("gsutil", "gcloud", "path truth", "train_model", "optimizer", "backward"):
            self.assertNotIn(forbidden, combined)

    def test_resources_are_small_and_fail_closed(self):
        config = (ROOT / "conf" / "m33_m0_factorized_lazy_contract.config").read_text(encoding="utf-8")
        self.assertIn("executor = 'local'", config)
        self.assertIn("maxRetries = 0", config)
        self.assertIn("maxForks = 1", config)


if __name__ == "__main__":
    unittest.main()
