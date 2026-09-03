from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class M38BLooWorkflowTest(unittest.TestCase):
    def test_workflow_is_ref_train_only_and_personal_bucket_only(self) -> None:
        workflow = (ROOT / "workflows/m38b_freeze_loo_subset.nf").read_text()
        module = (ROOT / "modules/38B_LOO_SUBSET.nf").read_text()
        config = (ROOT / "conf/m38b_freeze_loo_subset.config").read_text()
        combined = workflow + module + config
        self.assertIn("m38b_build_loo_subset.py", combined)
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos", config)
        self.assertNotIn("VALID", combined)
        self.assertNotIn("TEST", combined)
        self.assertNotIn("KING", combined.upper())
        self.assertIn("team: 'frank'", config)
        self.assertIn("overwrite: false", module)

    def test_batch_runtime_can_read_and_write_the_gcs_workdir(self) -> None:
        config = (ROOT / "conf/m38b_freeze_loo_subset.config").read_text()
        self.assertIn("m38b_loo_container_user = '0:0'", config)
        self.assertIn(
            'containerOptions = { "--network none --user '
            '${params.m38b_loo_container_user}" }',
            config,
        )


if __name__ == "__main__":
    unittest.main()
