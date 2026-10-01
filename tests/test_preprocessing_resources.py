"""New operational configuration is explicit; legacy runs are not reinterpreted."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('resource_pipeline', ROOT/'bin/r02_autosome_pipeline.py')
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class PreprocessResourceTests(unittest.TestCase):
    def test_legacy_has_no_added_parameters(self):
        self.assertEqual(pipeline.preprocessing_resources({}, 10), {})

    def test_explicit_budget_for_new_runs(self):
        config = dict(preprocess_checkpointed_m01=True, container_cpus=6,
                      free_disk_min_gib=12)
        result = pipeline.preprocessing_resources(config, 1000)
        self.assertEqual(result['preprocess_required_free_bytes'], 3000+32*1024**3)
        self.assertEqual(result['annotation_cpus'], 6)
        self.assertEqual(config['container_cpus'], 6)

    def test_invalid_budget_is_not_silently_coerced(self):
        valid = dict(preprocess_checkpointed_m01=True, container_cpus=6)
        for change in ({'preprocess_checkpointed_m01': 'true'}, {'container_cpus': 0},
                       {'container_cpus': 1},
                       {'container_cpus': True}, {'preprocess_scratch_input_multiplier': 0},
                       {'preprocess_scratch_reserve_gib': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                pipeline.preprocessing_resources({**valid, **change}, 1000)
        with self.assertRaises(ValueError):
            pipeline.preprocessing_resources(valid, 0)


if __name__ == '__main__':
    unittest.main()
