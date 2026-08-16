#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))
from plot_m27f_valid_transfer import build_primary_nam_matrix  # noqa: E402


class TestPlotM27FValidTransfer(unittest.TestCase):
    def test_matrix_is_ordered_and_privacy_safe(self):
        rows = []
        for position, states in [(30, ["ABSENT", "ABSENT", "ABSENT"]), (10, ["ABSENT", "PRESENT", "ABSENT"]), (20, ["ABSENT", "ABSENT", "ABSENT"])]:
            for index, state in enumerate(states, start=1):
                rows.append({
                    "pos": str(position),
                    "primary_for_local_transfer": "True",
                    "ancestry": "Native_American",
                    "block_token": f"native_american_block_{index:02d}",
                    "state": state,
                })
        patterns, blocks, matrix = build_primary_nam_matrix(rows)
        self.assertEqual(patterns, ["Patrón 1", "Patrón 2", "Patrón 3"])
        self.assertEqual(blocks, ["Bloque 1", "Bloque 2", "Bloque 3"])
        self.assertEqual(matrix, [[0, 1, 0], [0, 0, 0], [0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
