import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))

import reconcile_m31_pre2_stop as RECONCILE


class AuthorizedSourceReconstructionTest(unittest.TestCase):
    def _manifest(self):
        staged = {
            name: str(index) * 64
            for index, name in enumerate(
                sorted(set(RECONCILE.EXECUTION_SOURCE_PATHS) - {"m31_pre2_pipeline.py"}),
                start=1,
            )
        }
        return {
            "context": {
                "source_sha256": staged,
                "verified_code_sha256": {"orchestrator": "a" * 64},
            }
        }

    def test_joins_five_staged_sources_and_separate_orchestrator(self):
        observed = RECONCILE.authorized_sources_from_manifest(self._manifest())
        self.assertEqual(set(observed), set(RECONCILE.EXECUTION_SOURCE_PATHS))
        self.assertEqual(observed["m31_pre2_pipeline.py"], "a" * 64)

    def test_rejects_missing_extra_or_duplicated_source(self):
        missing = self._manifest()
        missing["context"]["source_sha256"].pop("m31_pre2_receipt.py")
        with self.assertRaisesRegex(RECONCILE.ReconciliationError, "five-source"):
            RECONCILE.authorized_sources_from_manifest(missing)
        extra = self._manifest()
        extra["context"]["source_sha256"]["unexpected.py"] = "b" * 64
        with self.assertRaisesRegex(RECONCILE.ReconciliationError, "five-source"):
            RECONCILE.authorized_sources_from_manifest(extra)
        duplicate = self._manifest()
        duplicate["context"]["source_sha256"]["m31_pre2_pipeline.py"] = "c" * 64
        with self.assertRaisesRegex(RECONCILE.ReconciliationError, "five-source"):
            RECONCILE.authorized_sources_from_manifest(duplicate)


class StopOnlyBoundaryTest(unittest.TestCase):
    def test_accepts_only_the_frozen_stop(self):
        RECONCILE.require_stop_only({"decision": {"status": "STOP_PRE2_BEFORE_ROOT18"}})
        for status in ("OPEN_ROOT18", "UNKNOWN", None):
            with self.subTest(status=status), self.assertRaisesRegex(
                RECONCILE.ReconciliationError, "refuses"
            ):
                RECONCILE.require_stop_only({"decision": {"status": status}})

    def test_cli_has_no_root18_truth_or_open_token_argument(self):
        actions = {action.dest for action in RECONCILE.build_parser()._actions}
        self.assertNotIn("root18_truth", actions)
        self.assertNotIn("root18_truth_source", actions)
        self.assertNotIn("open_token", actions)


if __name__ == "__main__":
    unittest.main()
