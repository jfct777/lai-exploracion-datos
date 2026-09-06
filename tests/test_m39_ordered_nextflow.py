"""Regression checks for the bounded ordered-context Nextflow interface."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OrderedWorkflowTests(unittest.TestCase):
    def test_fold_path_is_single_item_not_iterable_path_components(self):
        source = (ROOT / "workflows/m39_ordered_context_profile.nf").read_text()
        self.assertIn("inputs.add(folds)", source)
        self.assertNotIn("inputs += folds", source)
        self.assertIn("inputs.size() != bridge.inputs.size() + 1", source)
        self.assertIn("java.nio.file.Files.isRegularFile(it)", source)

    def test_profile_process_has_no_training_and_bounded_local_resources(self):
        source = (ROOT / "modules/39_ORDERED_CONTEXT_PROFILE.nf").read_text()
        self.assertIn("cpus 2", source)
        self.assertIn("memory '6 GB'", source)
        self.assertIn("time '1h'", source)
        self.assertIn("maxForks 1", source)
        self.assertIn("m39_profile_ordered.py", source)
        self.assertNotIn("m39_anchor_screen.py", source)


if __name__ == "__main__":
    unittest.main()
