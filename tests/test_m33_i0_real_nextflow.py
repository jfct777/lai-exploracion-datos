#!/usr/bin/env python3

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "modules" / "33_I0_REAL.nf").read_text(encoding="utf-8")
WORKFLOW = (ROOT / "workflows" / "m33_i0_real.nf").read_text(encoding="utf-8")
CONFIG = (ROOT / "conf" / "m33_i0_real.config").read_text(encoding="utf-8")
AUTH = json.loads((ROOT / "conf" / "m33_i0_real_authorization.json").read_text())


class M33I0RealNextflowTests(unittest.TestCase):
    def test_dag_is_stage_index_aggregate_only(self):
        self.assertEqual(len(re.findall(r"^process ", MODULE, flags=re.MULTILINE)), 3)
        for process in ("M33_I0_STAGE_REAL_SOURCE", "M33_I0_DERIVE_REAL_INDEX", "M33_I0_AGGREGATE_REAL"):
            self.assertIn(process, WORKFLOW)
        self.assertEqual(MODULE.count("cache false"), 3)
        self.assertIn("maxForks = 2", CONFIG)

    def test_exact_roots_and_generations_are_frozen(self):
        self.assertEqual(set(AUTH["roots"]), {"root17", "root18"})
        self.assertEqual(AUTH["roots"]["root17"]["generation"], "1787175566795248")
        self.assertEqual(AUTH["roots"]["root18"]["generation"], "1787175916753131")
        self.assertEqual(AUTH["expected_record_count"], 79791)
        self.assertIn("Channel.of('root17', 'root18')", WORKFLOW)

    def test_index_container_is_pinned_offline_and_stage_is_host_only(self):
        self.assertEqual(MODULE.count("container params.m33_i0_real_tabix_image"), 2)
        self.assertEqual(MODULE.count("--network none"), 2)
        self.assertRegex(CONFIG, r"m33-tabix@sha256:[0-9a-f]{64}")
        stage_block = MODULE.split("process M33_I0_DERIVE_REAL_INDEX", 1)[0]
        self.assertNotIn("container ", stage_block)
        self.assertIn("stageInMode 'copy'", stage_block)
        self.assertIn("--helper-script", stage_block)
        self.assertIn("--repo-root", stage_block)
        index_block = MODULE.split("process M33_I0_DERIVE_REAL_INDEX", 1)[1].split(
            "process M33_I0_AGGREGATE_REAL", 1
        )[0]
        self.assertIn("--helper-script", index_block)
        self.assertNotIn("--repo-root", index_block)
        self.assertIn("stageInMode 'copy'", index_block)
        aggregate_block = MODULE.split("process M33_I0_AGGREGATE_REAL", 1)[1]
        self.assertIn("--sources", aggregate_block)
        self.assertIn("--indexes", aggregate_block)
        self.assertIn("--helper-script", aggregate_block)
        self.assertNotIn("--repo-root", aggregate_block)

    def test_no_laboratory_sink_or_downstream_stage(self):
        combined = "\n".join((MODULE, WORKFLOW, CONFIG)).lower()
        for forbidden in ("google-batch", "safe_bridge true", "materialize true", "training true", "path truth"):
            self.assertNotIn(forbidden, combined)
        self.assertNotIn("gs://projects-usp", combined)
        self.assertIn("m33_i0_real_ready?.tostring() != 'false'", combined)
        self.assertFalse(AUTH["local_output_policy"]["publish_gcs_during_workflow"])

    def test_only_derived_outputs_are_published_locally(self):
        self.assertEqual(MODULE.count("publishDir"), 2)
        self.assertNotRegex(MODULE, r"publishDir[^\n]*source_vcf")
        self.assertIn("filename.endsWith('.vcf.gz') ? null", MODULE)
        self.assertIn("I0_REAL_PASS_NON_CONSUMABLE", MODULE)
        self.assertNotIn("path 'READY'", MODULE)
        self.assertIn("local_results.canonicalPath.startsWith('/tmp/')", WORKFLOW)
        self.assertIn("local_results.exists()", WORKFLOW)

    def test_resume_is_rejected_and_retries_are_zero(self):
        self.assertIn("workflow.resume", WORKFLOW)
        self.assertIn("maxRetries = 0", CONFIG)
        self.assertIn("errorStrategy = 'terminate'", CONFIG)


if __name__ == "__main__":
    unittest.main()
