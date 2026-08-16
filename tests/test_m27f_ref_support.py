#!/usr/bin/env python3
"""Contracts for the M27F REF-only projection and support gate."""

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))
import audit_m27f_ref_support as support  # noqa: E402


class TestM27FRefSupport(unittest.TestCase):
    def test_frozen_orientation_carrier_logic(self):
        self.assertTrue(support.usable_carrier(1, True, True, True))
        self.assertFalse(support.usable_carrier(1, False, True, True))
        self.assertFalse(support.usable_carrier(0, True, True, True))
        self.assertTrue(support.usable_carrier(0, False, True, False))
        self.assertFalse(support.usable_carrier(2, True, True, False))
        self.assertFalse(support.usable_carrier(0, True, False, False))

    def test_catalog_digest_is_order_invariant_and_orientation_sensitive(self):
        first = {("22", 20, "A", "G"): True, ("22", 10, "C", "T"): False}
        second = dict(reversed(list(first.items())))
        self.assertEqual(support.catalog_digest(first), support.catalog_digest(second))
        changed = dict(first)
        changed[("22", 20, "A", "G")] = False
        self.assertNotEqual(support.catalog_digest(first), support.catalog_digest(changed))

    def test_preregistration_seals_validation_and_test(self):
        prereg = json.loads(
            (REPO / "conf/m27f_ref_support_preregistration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(prereg["version"], 1)
        self.assertFalse(prereg["support_contract"]["select_using_validation"])
        self.assertEqual(prereg["projection_contract"]["variant_filters"], [])
        self.assertEqual(prereg["projection_contract"]["genotype_filters"], [])

    def test_projection_is_positive_allowlist_without_variant_filters(self):
        source = (REPO / "bin/project_m27f_ref_panel.py").read_text(encoding="utf-8")
        self.assertIn('"--samples-file"', source)
        self.assertIn('"--no-update"', source)
        self.assertNotIn('"--include"', source)
        self.assertNotIn('"--exclude"', source)
        self.assertNotIn("KING", source)

    def test_nextflow_keeps_discovery_work_only(self):
        module = (REPO / "modules/27F_REF_SUPPORT_AUDIT.nf").read_text(encoding="utf-8")
        workflow = (REPO / "workflows/m27f_ref_support_audit.nf").read_text(encoding="utf-8")
        self.assertIn("pattern: 'm27f_ref*'", module)
        self.assertEqual(module.count("container params.m27f_ref_container_image"), 3)
        self.assertIn("chmod 600 m27f_ref_site_support.private.tsv.gz", module)
        self.assertIn("source_valid_opened: false", workflow)
        self.assertIn("source_test_opened: false", workflow)
        self.assertNotIn("KING", module + workflow)


if __name__ == "__main__":
    unittest.main()
