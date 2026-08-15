"""The provenance record must not be able to contradict the gate that let a run start.

`m27d-pass0-20260815a` was launched with `--donor_kinship_full_run_authorized true` and
published `full_run_authorized: false`.  Nothing was wrong with the run: the launch gate
asked whether the phase was `pass0` or `audit`, the provenance record asked whether it
was `audit`, and only the second statement reached the artifact.  Two copies of one rule
drifted, and the trail of a correct run read as if it had never been authorized.

These tests pin the fix at the level of the mechanism, not the symptom: the rule has one
authority, three separate facts get three separate names, and a base record that still
carries the ambiguous field is refused instead of silently overwritten.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))

import m27d_run_provenance as provenance  # noqa: E402


CONTRACT = json.loads(
    (REPO / "conf" / "m27d_donor_kinship_preregistration.json").read_text(encoding="utf-8")
)

BASE = {
    "git_commit": "0" * 40,
    "run_id": "m27d-test",
    "phase": "pass0",
}


def encode(record: dict) -> str:
    return base64.b64encode(json.dumps(record).encode("utf-8")).decode("ascii")


class TestAuthorizationRecord(unittest.TestCase):
    def test_requested_and_effective_are_separate_facts(self):
        record = provenance.authorization_record("pass0", True, CONTRACT)
        self.assertTrue(record["full_run_authorized_requested"])
        self.assertTrue(record["full_run_authorization_required_by_phase"])
        self.assertTrue(record["full_run_authorized_effective"])
        self.assertEqual(record["phase_executed"], "pass0")

    def test_pass0_is_a_gated_phase_which_is_the_case_that_regressed(self):
        """The historical record reported false for exactly this combination."""
        record = provenance.authorization_record("pass0", True, CONTRACT)
        self.assertNotEqual(record["full_run_authorized_effective"], False)

    def test_ungated_phase_records_the_request_without_spending_it(self):
        record = provenance.authorization_record("strata", True, CONTRACT)
        self.assertTrue(record["full_run_authorized_requested"])
        self.assertFalse(record["full_run_authorization_required_by_phase"])
        self.assertFalse(record["full_run_authorized_effective"])
        self.assertIn("was not spent", record["full_run_authorization_note"])

    def test_gated_phase_without_a_request_cannot_report_an_effective_yes(self):
        record = provenance.authorization_record("audit", False, CONTRACT)
        self.assertFalse(record["full_run_authorized_effective"])

    def test_every_gated_phase_round_trips(self):
        for phase in CONTRACT["authorization"]["phases_requiring_explicit_authorization"]:
            record = provenance.authorization_record(phase, True, CONTRACT)
            self.assertTrue(record["full_run_authorized_effective"], msg=phase)

    def test_unknown_phase_is_refused(self):
        with self.assertRaises(SystemExit):
            provenance.authorization_record("audir", True, CONTRACT)

    def test_gated_phase_outside_the_declared_phases_is_refused(self):
        broken = json.loads(json.dumps(CONTRACT))
        broken["authorization"]["phases_requiring_explicit_authorization"].append("ghost")
        with self.assertRaises(SystemExit):
            provenance.authorization_record("pass0", True, broken)

    def test_missing_authorization_block_fails_closed(self):
        broken = json.loads(json.dumps(CONTRACT))
        del broken["authorization"]
        with self.assertRaises(SystemExit):
            provenance.authorization_record("pass0", True, broken)

    def test_empty_string_is_not_read_as_false(self):
        """Groovy renders an unset param as an empty string, not as a boolean."""
        with self.assertRaises(SystemExit):
            provenance.parse_boolean("")

    def test_boolean_words_are_accepted_in_both_directions(self):
        self.assertTrue(provenance.parse_boolean("true"))
        self.assertFalse(provenance.parse_boolean("false"))


class TestProvenanceWriter(unittest.TestCase):
    def write(self, base: dict, phase: str, requested: str):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run_provenance.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "bin" / "m27d_run_provenance.py"),
                    "--base-b64", encode(base),
                    "--phase", phase,
                    "--authorization-requested", requested,
                    "--preregistration",
                    str(REPO / "conf" / "m27d_donor_kinship_preregistration.json"),
                    "--out", str(out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
            return result, payload

    def test_authorized_pass0_publishes_an_effective_authorization(self):
        result, payload = self.write(BASE, "pass0", "true")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(payload["full_run_authorized_requested"])
        self.assertTrue(payload["full_run_authorized_effective"])
        self.assertEqual(payload["git_commit"], BASE["git_commit"])

    def test_the_ambiguous_field_is_never_emitted(self):
        _, payload = self.write(BASE, "pass0", "true")
        self.assertNotIn("full_run_authorized", payload)

    def test_a_base_record_carrying_the_old_field_is_refused(self):
        """A workflow that kept its own copy of the rule must fail loudly."""
        result, _ = self.write(dict(BASE, full_run_authorized=False), "pass0", "true")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ambiguous", result.stderr)

    def test_a_base_record_presetting_an_authorization_key_is_refused(self):
        result, _ = self.write(
            dict(BASE, full_run_authorized_effective=True), "pass0", "true"
        )
        self.assertNotEqual(result.returncode, 0)


class TestSingleAuthorityForTheRule(unittest.TestCase):
    WORKFLOW = (REPO / "workflows" / "m27d_donor_kinship_audit.nf").read_text(encoding="utf-8")
    MODULE = (REPO / "modules" / "27D_DONOR_KINSHIP_AUDIT.nf").read_text(encoding="utf-8")

    def test_the_workflow_holds_no_literal_list_of_phases(self):
        """The drift was two literal phase lists; the contract is now the only one.

        Single-phase routing such as `if( phase == 'pass0' )` stays legal: it decides
        where the DAG stops, not whether an authorization was spent.  What is banned is
        a *list* of phases, because that is the shape the gate and the record each kept
        their own copy of.
        """
        phases = "|".join(CONTRACT["authorization"]["phases"])
        literal_list = re.compile(rf"\[\s*'({phases})'\s*,")
        found = literal_list.findall(self.WORKFLOW)
        self.assertEqual(found, [], msg=f"phase list restated in Groovy: {found}")

    def test_the_workflow_reads_the_policy_from_the_preregistration(self):
        self.assertIn("phases_requiring_explicit_authorization", self.WORKFLOW)
        self.assertIn("authorizationPolicy(contract)", self.WORKFLOW)

    def test_the_workflow_never_computes_an_authorization_field_itself(self):
        for field in (
            "full_run_authorized_requested",
            "full_run_authorized_effective",
            "full_run_authorization_required_by_phase",
        ):
            self.assertNotIn(field, self.WORKFLOW, msg=f"{field} is computed in Groovy")

    def test_no_stage_manifest_hardcodes_a_run_level_authorization(self):
        """It is a run-level fact; a per-stage copy is a second place to drift."""
        self.assertNotIn('"full_run_authorized"', self.MODULE)

    def test_the_provenance_process_invokes_the_single_authority(self):
        self.assertIn("bin/m27d_run_provenance.py", self.WORKFLOW)
        self.assertIn("python3 ${run_provenance_py}", self.MODULE)
        self.assertIn("--authorization-requested", self.MODULE)

    def test_the_process_no_longer_writes_the_record_by_decoding_a_literal(self):
        """Decoding a literal is what let the workflow decide the answer alone."""
        self.assertNotIn("base64 -d > run_provenance.json", self.MODULE)


if __name__ == "__main__":
    unittest.main()
