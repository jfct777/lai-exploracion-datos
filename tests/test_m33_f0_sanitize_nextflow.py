#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NextflowContractTests(unittest.TestCase):
    def test_module_is_cache_off_and_probability_boundary_is_explicit(self) -> None:
        module = (ROOT / "modules" / "33_SANITIZE_FLARE_F0.nf").read_text()
        self.assertIn("cache false", module)
        self.assertIn("--flare-anc", module)
        self.assertIn("--target-rare-diploid", module)
        self.assertNotIn("truth", module.lower())

    def test_workflow_uses_manifest_and_exact_source_hash(self) -> None:
        workflow = (ROOT / "workflows" / "m33_f0_sanitize.nf").read_text()
        self.assertIn("roots_manifest", workflow)
        self.assertIn("[0-9a-f]{40}", workflow)
        self.assertIn("source_auth", workflow)
        self.assertIn("checkIfExists:true", workflow)

    def test_config_pins_container_by_digest_and_has_bounded_resources(self) -> None:
        config = (ROOT / "conf" / "m33_f0_sanitize.config").read_text()
        self.assertIn("@sha256:", config)
        self.assertIn("m33_f0_sanitize_cpus = 2", config)
        self.assertIn("m33_f0_sanitize_memory = '8 GB'", config)
        self.assertIn("overwrite: false", config)


if __name__ == "__main__":
    unittest.main()
