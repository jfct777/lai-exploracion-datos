import unittest
import numpy as np
from r02_lai_fit_query_plan import allocate_blocks, make_fit_plan


class QueryPlanTests(unittest.TestCase):
    def test_exact_expectation_and_circular_boundaries(self):
        truth = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 1, 3, 3, 5]])
        loss = np.arange(truth.size, dtype=float).reshape(truth.shape) ** 2
        counts = np.bincount(truth.ravel(), minlength=6)
        quota = allocate_blocks(counts, 8)
        for width in (1, 3, 6):
            expectation = 0.0
            total_mass = 0.0
            for person, start in np.ndindex(truth.shape):
                state = truth[person, start]
                q = (start + np.arange(width)) % truth.shape[1]
                coefficient = counts[state] / (truth.size * quota[state] * width)
                expected_draws = quota[state] / counts[state]
                expectation += expected_draws * coefficient * loss[person, q].sum()
                total_mass += expected_draws * coefficient * width
            self.assertAlmostEqual(total_mass, 1.)
            self.assertAlmostEqual(expectation, loss.mean(), places=12)

    def test_replay_and_state_coverage(self):
        truth = np.tile(np.arange(6), (2, 1)); observed = np.ones_like(truth, bool)
        args = dict(role="FIT", seed=9, updates=3, blocks_per_update=8, width=3)
        a = make_fit_plan(truth, observed, **args); b = make_fit_plan(truth, observed, **args)
        np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
        for update in range(3):
            rows = a[0][a[0][:, 0] == update]
            self.assertEqual(set(truth[rows[:, 2], rows[:, 3]]), set(range(6)))
        self.assertFalse(a[2]["training_admitted"])
        for update in range(3):
            self.assertAlmostEqual(float(a[1][a[0][:, 0] == update].sum()), 1.)

    def test_repeated_cells_preserve_occurrences(self):
        truth = np.arange(6).reshape(1, 6)
        rows, weights, _ = make_fit_plan(truth, np.ones_like(truth, bool),
            role="FIT", seed=1, updates=1, blocks_per_update=6, width=6)
        self.assertEqual(len(rows), 36)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertEqual(len(set(map(tuple, rows[:, 2:4]))), 6)

    def test_roles_missingness_budget_and_absent_states(self):
        truth = np.arange(6).reshape(1, 6); mask = np.ones_like(truth, bool)
        args = dict(seed=1, updates=1, blocks_per_update=6, width=3)
        with self.assertRaises(ValueError): make_fit_plan(truth, mask, role="SELECT", **args)
        mask[0, 0] = False
        with self.assertRaises(ValueError): make_fit_plan(truth, mask, role="FIT", **args)
        with self.assertRaises(ValueError): allocate_blocks(np.ones(6, int), 5)
        np.testing.assert_array_equal(allocate_blocks(np.array([3, 1, 0, 0, 0, 0]), 4),
                                      [3, 1, 0, 0, 0, 0])

    def test_state_and_boundary_subtotals_have_correct_expectation(self):
        truth = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 1, 3, 3, 5]])
        loss = np.arange(12, dtype=float).reshape(truth.shape) ** 2
        counts = np.bincount(truth.ravel(), minlength=6)
        quota = allocate_blocks(counts, 8)
        masks = [truth == s for s in range(6)]
        masks += [np.array([[0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0]], bool)]
        for mask in masks:
            expected = 0.
            for person, start in np.ndindex(truth.shape):
                s = truth[person, start]; q = (start + np.arange(3)) % 6
                coefficient = counts[s] / (truth.size * quota[s] * 3)
                expected += quota[s] / counts[s] * coefficient * np.sum(
                    loss[person, q] * mask[person, q])
            self.assertAlmostEqual(expected, np.mean(loss * mask), places=12)
            self.assertAlmostEqual(expected / mask.mean(), loss[mask].mean(), places=12)

    def test_actual_random_draws_and_heterogeneous_duplicate_weights(self):
        truth = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 1, 3, 3, 5]])
        loss = np.arange(12, dtype=float).reshape(truth.shape) ** 2
        estimates=[]; heterogeneous_duplicate_seen = False
        for seed in range(400):
            rows, weights, _ = make_fit_plan(truth, np.ones_like(truth, bool),
                role="FIT", seed=seed, updates=1, blocks_per_update=8, width=3)
            aggregate = np.zeros_like(loss)
            np.add.at(aggregate, (rows[:, 2], rows[:, 3]), weights)
            value = float(np.sum(weights * loss[rows[:, 2], rows[:, 3]]))
            self.assertAlmostEqual(value, float(np.sum(aggregate * loss)), places=12)
            estimates.append(value)
            for person, query in set(map(tuple, rows[:, 2:4])):
                same = weights[(rows[:, 2] == person) & (rows[:, 3] == query)]
                heterogeneous_duplicate_seen |= len(np.unique(same)) > 1
        self.assertTrue(heterogeneous_duplicate_seen)
        # Fixed reproducible seed range; tolerance checks the actual sampler,
        # not independent genomic cells or biological uncertainty.
        se = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        self.assertLess(abs(np.mean(estimates) - loss.mean()), 6 * se + 1e-12)


if __name__ == "__main__": unittest.main()
