import unittest
import numpy as np
from r02_lai_error_strata import paired_strata, truth_boundary_distance


class StrataTests(unittest.TestCase):
    def test_missing_never_bridges_or_creates_boundary(self):
        y = np.array([0, 1, -1, 2, 2]); m = y >= 0
        d = truth_boundary_distance(y, m, np.arange(5.))
        np.testing.assert_equal(d, [1., 0., np.inf, np.inf, np.inf])

    def test_near_far_and_corrections_damage_partition(self):
        y = np.array([0, 0, 1, 1]); f = np.eye(6)[[0, 1, 1, 1]]
        n = np.eye(6)[[1, 0, 1, 1]]; m = np.ones(4, bool)
        rows = paired_strata(f, n, y, m, np.arange(4.) / 10,
                            np.array([[0,0,0],[0,0,1],[1,1,1],[1,2,1]]),
                            cuts_cm=[.1,.2,.5], floor=1e-12)
        self.assertEqual(sum(r['corrected'] for r in rows), 1)
        self.assertEqual(sum(r['introduced'] for r in rows), 1)
        self.assertEqual(sum(r['count'] for r in rows), 4)
        self.assertEqual(sum(r['NEURAL']['true_zero_count'] for r in rows), 1)

    def test_reject_invalid_burden(self):
        with self.assertRaises(ValueError):
            paired_strata(np.eye(6)[:2], np.eye(6)[:2], np.array([0,1]),
                          np.ones(2,bool), np.arange(2.), np.array([[2,2,1],[0,0,0]]),
                          cuts_cm=[.1], floor=1e-12)

    def test_perfect_neural_no_spurious_damage(self):
        y = np.array([0,1,2]); p = np.eye(6)[y]
        rows = paired_strata(p,p,y,np.ones(3,bool),np.arange(3.),
                            np.zeros((3,3)),cuts_cm=[.1],floor=1e-12)
        self.assertEqual(sum(r['both_correct'] for r in rows),3)
        self.assertEqual(sum(r['introduced'] for r in rows),0)

if __name__ == '__main__': unittest.main()
