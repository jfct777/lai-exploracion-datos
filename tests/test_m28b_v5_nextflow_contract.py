"""Static integration checks for the isolated M28B-v5 Nextflow entrypoint."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class TestM28BV5NextflowContract(unittest.TestCase):
    def setUp(self):
        self.module = (
            REPO / "modules" / "28B_V5_INDIVIDUAL_SAFE_MATCHING_AUDIT.nf"
        ).read_text(encoding="utf-8")
        self.workflow = (
            REPO / "workflows" / "m28b_v5_individual_safe_matching_audit.nf"
        ).read_text(encoding="utf-8")
        self.config = (
            REPO / "conf" / "m28b_v5_individual_safe_matching_audit.config"
        ).read_text(encoding="utf-8")

    def test_v5_has_dedicated_process_and_result_names(self):
        self.assertIn("RUN_M28B_V5_DEVELOPMENT", self.module)
        self.assertIn("RUN_M28B_V5_VALIDATION", self.module)
        self.assertIn("m28b_v5_dev.public.json", self.module)
        self.assertIn("m28b_v5_validation.public.json", self.module)
        self.assertNotIn("m28b_v4_results_dir", self.module + self.workflow + self.config)

    def test_both_phases_authenticate_corrected_preflight(self):
        self.assertGreaterEqual(self.module.count("--preflight-report"), 2)
        self.assertGreaterEqual(self.module.count("--preflight-manifest"), 2)
        self.assertGreaterEqual(self.module.count("--preflight-reproducibility"), 2)
        self.assertIn("m28_lai_simulation_preflight_preregistration.v2.json", self.config)

    def test_scope_excludes_lai_target_and_truth_inputs(self):
        process_inputs = {
            line.strip().removeprefix("path ")
            for line in self.module.splitlines()
            if line.strip().startswith("path ")
        }
        self.assertNotIn("target", process_inputs)
        self.assertNotIn("truth", process_inputs)
        self.assertNotIn("mosaic_events", process_inputs)

    def test_resource_defaults_are_small_and_retry_is_disabled(self):
        self.assertIn('m28b_v5_cpus = 1', self.config)
        self.assertIn('m28b_v5_memory = "4 GB"', self.config)
        self.assertIn('maxRetries = 0', self.config)


if __name__ == "__main__":
    unittest.main()
