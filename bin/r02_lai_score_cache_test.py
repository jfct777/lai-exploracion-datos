import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from r02_lai_score_cache import LaiScoreCache


class ScoreCacheTests(unittest.TestCase):
    def test_resume_and_mutated_model_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            model = torch.nn.Linear(2, 3)
            entry = {'path': 'authenticated-source', 'sha256': 'a'*64}
            value = np.array([[[.2, .3, .5], [np.nan]*3]], dtype=np.float32)
            cache = LaiScoreCache(temp, model, {'scope': 'FIT'}, {'seed': 1}, 'FIT', 'NONE')
            cache.save(0, entry, value)
            resumed = LaiScoreCache(temp, model, {'scope': 'FIT'}, {'seed': 1}, 'FIT', 'NONE')
            np.testing.assert_array_equal(resumed.load(0, entry), value)
            with torch.no_grad(): model.weight.add_(1)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                LaiScoreCache(temp, model, {'scope': 'FIT'}, {'seed': 1}, 'FIT', 'NONE')

    def test_corruption_and_partial_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            model = torch.nn.Linear(1, 1); entry = {'path': 'source'}
            cache = LaiScoreCache(temp, model, {}, {}, 'FIT', 'NONE')
            value = np.ones((2, 4, 6), np.float32)
            np.save(Path(temp)/'predictions-00000.npy', value)
            cache.save(0, entry, value)  # Complete lost receipt without replacing result.
            self.assertIsNone(cache.load(1, entry))
            with self.assertRaisesRegex(ValueError, 'source or bytes changed'):
                cache.load(0, {'path': 'different'})
            (Path(temp)/'predictions-00000.npy').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'source or bytes changed'):
                cache.load(0, entry)

    def test_conflicting_prediction_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            model = torch.nn.Linear(1, 1)
            cache = LaiScoreCache(temp, model, {}, {}, 'FIT', 'NONE')
            old = np.zeros((1, 2, 6), np.float32)
            new = np.ones_like(old)
            np.save(Path(temp)/'predictions-00000.npy', old)
            with self.assertRaisesRegex(ValueError, 'both preserved'):
                cache.save(0, {'path': 'same-source'}, new)
            np.testing.assert_array_equal(np.load(Path(temp)/'predictions-00000.npy'), old)
            conflicts = list(Path(temp).glob('.conflict-*.npy'))
            self.assertEqual(len(conflicts), 1)
            np.testing.assert_array_equal(np.load(conflicts[0]), new)


if __name__ == '__main__': unittest.main()
