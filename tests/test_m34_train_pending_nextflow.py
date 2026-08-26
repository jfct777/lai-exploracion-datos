#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "m34_train_pending.nf"
STANDARD_MODULE = ROOT / "modules" / "34_NAM_TRAIN_FACTORIZED.nf"
TRANSFORMER_MODULE = ROOT / "modules" / "34_NAM_TRAIN_TRANSFORMER_FACTORIZED.nf"
SCORE_MODULE = ROOT / "modules" / "34_NAM_SCORE.nf"


class PendingWorkflowTests(unittest.TestCase):
    def test_pending_workflow_is_exact_append_safe_and_scores_all_predictions(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pending.task_count < 1", workflow)
        self.assertIn("M34_LOCAL_EXPANSION_PLAN", workflow)
        self.assertIn("pending.input_sha256.factorized_manifest", workflow)
        self.assertIn("MessageDigest.getInstance('SHA-256')", workflow)
        self.assertIn("pending.completed_count + pending.pending_count", workflow)
        self.assertIn("family != 'transformer_small'", workflow)
        self.assertIn("family == 'transformer_small'", workflow)
        self.assertIn("completedPredictions", workflow)
        self.assertIn(".mix(standardPredictions)", workflow)
        self.assertIn(".mix(transformerPredictions)", workflow)
        self.assertIn("M34_NAM_SCORE_VALID(allPredictions", workflow)
        self.assertIn("if (runResults.exists())", workflow)
        self.assertIn("def runResults = file(", workflow)
        self.assertIn("checkIfExists: false", workflow)
        self.assertNotIn("def runResults = new File(", workflow)

    def test_complete_task_identity_namespaces_models_and_metrics(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        standard = STANDARD_MODULE.read_text(encoding="utf-8")
        transformer = TRANSFORMER_MODULE.read_text(encoding="utf-8")
        score = SCORE_MODULE.read_text(encoding="utf-8")
        self.assertIn("String m34RadiusToken", workflow)
        self.assertIn("task.sweep_stage", workflow)
        self.assertIn('"seed${task.seed}"', workflow)
        self.assertIn('"u${task.maximum_updates}"', workflow)
        self.assertIn("radiusCm, radiusToken, taskToken", workflow)
        for module in (standard, transformer, score):
            self.assertIn("val(radiusCm), val(radiusToken)", module)
            self.assertIn("${taskToken}", module)
        self.assertIn("--task task.json", score)

    def test_transformer_module_writes_batching_receipt_and_keeps_logical_limits(self):
        module = TRANSFORMER_MODULE.read_text(encoding="utf-8")
        self.assertIn("test '${family}' = transformer_small", module)
        self.assertIn("m34_train_transformer_factorized.py", module)
        self.assertIn("transformer_batching.receipt.json", module)
        self.assertIn("--maximum-rows-per-batch", module)
        self.assertIn("--maximum-tokens-per-batch", module)
        self.assertIn("--validation-every", module)


if __name__ == "__main__":
    unittest.main()
