"""Static regression tests for dynamic M28 process directives."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class TestM28NextflowContract(unittest.TestCase):
    def test_seed_dependent_directives_are_dynamic_closures(self):
        module = (REPO / "modules" / "28_LAI_SIMULATION_PREFLIGHT.nf").read_text()
        self.assertIn('tag { "m28_seed_${root_seed}" }', module)
        self.assertIn(
            'publishDir { "${params.m28_results_dir}/seed-${root_seed}" }',
            module,
        )


if __name__ == "__main__":
    unittest.main()
