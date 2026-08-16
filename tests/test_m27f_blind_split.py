#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))
import audit_m27f_blind_split as m27f


class TestM27FBlindSplit(unittest.TestCase):
    def test_assignment_is_deterministic_and_balances_block_counts(self):
        blocks = [(f"{index:064x}", size) for index, size in enumerate([40, 30, 20, 10, 8, 6, 4, 2])]
        first = m27f.deterministic_assignment(blocks)
        second = m27f.deterministic_assignment(list(reversed(blocks)))
        self.assertEqual(first, second)
        counts = {role: list(first.values()).count(role) for role in m27f.ROLES}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_assignment_never_uses_randomness(self):
        source = (REPO / "bin/audit_m27f_blind_split.py").read_text()
        self.assertNotIn("import random", source)
        self.assertNotIn("genotype", m27f.deterministic_assignment.__doc__.lower())

    def test_nextflow_wrapper_is_narrow_and_auditable(self):
        workflow = (REPO / "workflows/m27f_blind_role_split.nf").read_text()
        module = (REPO / "modules/27F_BLIND_ROLE_SPLIT.nf").read_text()
        self.assertIn("AUDIT_M27F_BLIND_ROLE_SPLIT", workflow)
        self.assertIn("WRITE_M27F_SPLIT_RUN_PROVENANCE", workflow)
        self.assertIn("workflow.commandLine", workflow)
        self.assertIn('"genotypes_parsed":false', module)
        self.assertNotIn("SOURCE_TEST", module)
        self.assertNotIn("KING", workflow + module)


if __name__ == "__main__":
    unittest.main()
