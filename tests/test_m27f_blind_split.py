#!/usr/bin/env python3
"""Focused contracts for the M27F metadata/IBD-only split."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))
import audit_m27f_blind_split as m27f  # noqa: E402


FRACTIONS = {"REF_TRAIN": 0.5, "SOURCE_VALID": 0.25, "SOURCE_TEST": 0.25}


class TestM27FBlindSplit(unittest.TestCase):
    def test_hamilton_quotas_follow_preregistered_fraction_and_ties(self):
        self.assertEqual(
            m27f.hamilton_quotas(19, FRACTIONS),
            {"REF_TRAIN": 9, "SOURCE_VALID": 5, "SOURCE_TEST": 5},
        )
        self.assertEqual(
            m27f.hamilton_quotas(31, FRACTIONS),
            {"REF_TRAIN": 15, "SOURCE_VALID": 8, "SOURCE_TEST": 8},
        )
        self.assertEqual(
            m27f.hamilton_quotas(52, FRACTIONS),
            {"REF_TRAIN": 26, "SOURCE_VALID": 13, "SOURCE_TEST": 13},
        )

    def test_assignment_is_order_invariant_and_meets_exact_quotas(self):
        units = [
            (hashlib.sha256(str(index).encode()).hexdigest(), size, populations)
            for index, (size, populations) in enumerate(
                [(40, 2), (30, 1), (20, 1), (10, 1), (8, 1), (6, 1), (4, 1), (2, 1)]
            )
        ]
        first, receipt = m27f.deterministic_assignment(units, FRACTIONS)
        second, _ = m27f.deterministic_assignment(list(reversed(units)), FRACTIONS)
        self.assertEqual(first, second)
        self.assertEqual(receipt["quotas"], {"REF_TRAIN": 4, "SOURCE_VALID": 2, "SOURCE_TEST": 2})
        self.assertEqual(
            {role: list(first.values()).count(role) for role in m27f.ROLES},
            receipt["quotas"],
        )

    def test_duplicate_atomic_unit_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "not unique"):
            m27f.deterministic_assignment(
                [("same", 2, 1), ("same", 3, 1)], FRACTIONS
            )

    def test_upstream_manifest_authenticates_reused_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "ibd_chr_1.ibd"
            payload.write_text("frozen\n", encoding="utf-8")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = root / "m27e.manifest.json"
            body = {
                "stage": "M27E_IBD_RARE_TRANSFER_FEASIBILITY",
                "git_commit": "abc",
                "inputs": {payload.name: digest},
            }
            manifest.write_text(json.dumps(body), encoding="utf-8")
            prereg = {
                "upstream_contract": {
                    "m27e_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "m27e_manifest_stage": body["stage"],
                    "m27e_generator_commit": body["git_commit"],
                }
            }
            receipt = m27f.validate_upstream_manifest(manifest, prereg, [payload])
            self.assertEqual(receipt["n_authenticated_inputs"], 1)
            payload.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not authenticated"):
                m27f.validate_upstream_manifest(manifest, prereg, [payload])

    def test_splitter_has_no_vcf_or_random_interface(self):
        source = (REPO / "bin/audit_m27f_blind_split.py").read_text(encoding="utf-8")
        workflow = (REPO / "workflows/m27f_blind_role_split.nf").read_text(encoding="utf-8")
        module = (REPO / "modules/27F_BLIND_ROLE_SPLIT.nf").read_text(encoding="utf-8")
        config = (REPO / "nextflow.config").read_text(encoding="utf-8")
        combined = workflow + module
        self.assertNotIn("import random", source)
        self.assertNotIn("panel_vcf", combined)
        self.assertNotIn("discovery_vcf", combined)
        self.assertNotIn("m27f_split_panel_vcf", config)
        self.assertNotIn("KING", combined)
        self.assertIn('"vcf_inputs_declared":false', module)

    def test_nextflow_wrapper_is_narrow_and_auditable(self):
        workflow = (REPO / "workflows/m27f_blind_role_split.nf").read_text(encoding="utf-8")
        module = (REPO / "modules/27F_BLIND_ROLE_SPLIT.nf").read_text(encoding="utf-8")
        self.assertIn("AUDIT_M27F_BLIND_ROLE_SPLIT", workflow)
        self.assertIn("WRITE_M27F_SPLIT_RUN_PROVENANCE", workflow)
        self.assertIn("workflow.commandLine", workflow)
        self.assertIn("--upstream-m27e-manifest", module)
        self.assertIn("chmod 600 m27f_split.private.tsv", module)
        self.assertIn('"source_test_opened":false', module)


if __name__ == "__main__":
    unittest.main()
