from __future__ import annotations

import subprocess
import shutil
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class M36CoraSetNextflowTests(unittest.TestCase):
    def test_workflow_is_isolated_with_explicit_plan_smoke_train_modes(self) -> None:
        workflow = (ROOT / "workflows/m36_cora_set.nf").read_text(encoding="utf-8")
        module = (ROOT / "modules/36_CORA_SET.nf").read_text(encoding="utf-8")
        combined = workflow + module
        self.assertIn("M36_CORA_SET_PLAN", combined)
        self.assertIn("M36_CORA_SET_TRAIN", combined)
        self.assertIn("M36_CORA_MATERIALIZE", combined)
        self.assertIn("M36_CORA_MATERIALIZE_TRAIN", combined)
        self.assertIn("--smoke-only", module)
        self.assertIn("m36_cora_materialization_receipt", workflow)
        self.assertIn("--synthetic-smoke", module)
        self.assertIn("mode == 'smoke'", module)
        self.assertIn("m36_cora_train_publication_receipt", module)
        self.assertIn("m36_cora_train_cpus", module)
        self.assertIn("m36_cora_synthetic_smoke=true is permitted only", workflow)
        self.assertIn("m36_cora_zero_negative_ratios", module)
        self.assertIn("m36_cora_pytorch_image", module)
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos/runs/${params.m36_cora_run_id}", module)
        self.assertIn("pcrelate_component", (ROOT / "bin/m36_cora_set.py").read_text(encoding="utf-8"))
        for forbidden in ("M14", "M16", "chromopainter truth"):
            self.assertNotIn(forbidden.lower(), combined.lower())

    @unittest.skipUnless(shutil.which("nextflow"), "Nextflow is exercised in the workflow runtime")
    def test_nextflow_configuration_parses(self) -> None:
        result = subprocess.run(
            ["nextflow", "-C", str(ROOT / "conf/m36_cora_set_smoke.config"), "config", "-flat"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_google_batch_scope_is_m36_and_team_labeled(self) -> None:
        config = (ROOT / "conf/m36_cora_google_batch.config").read_text(encoding="utf-8")
        self.assertIn("executor = 'google-batch'", config)
        self.assertIn("resourceLabels = [team: 'frank', lane: 'm36-cora']", config)
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos", config)
        self.assertNotIn("main.nf", config)


if __name__ == "__main__":
    unittest.main()
