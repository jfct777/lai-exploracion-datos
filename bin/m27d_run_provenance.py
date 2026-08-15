#!/usr/bin/env python3
"""Write the M27D run-level provenance record.

The authorization rule lives here, and only here, on the Python side.  It used to be
restated inside the workflow: the launch gate asked whether the phase was ``pass0`` or
``audit``, while the provenance record asked whether it was ``audit``.  Both statements
looked correct in isolation, so the drift only surfaced in the published artifact of a
real run, which reported ``full_run_authorized=false`` for a run launched with the flag
set to true.

Three distinct facts are reported under three distinct names:

``full_run_authorized_requested``
    what the operator passed on the command line, recorded verbatim;
``full_run_authorization_required_by_phase``
    whether this phase consults that request at all;
``full_run_authorized_effective``
    whether the request was actually spent to let the phase start.

A reader of a single boolean cannot tell which of the three it is looking at, and that
ambiguity is what made a correct run look like an unauthorized one.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


BOOLEAN_WORDS = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "1": True,
    "0": False,
}


def parse_boolean(value: str) -> bool:
    """Refuse anything that is not unambiguously a boolean.

    Groovy renders ``null`` as the empty string, and treating an empty string as false
    would silently record 'not authorized' for a parameter that was never wired up.
    """
    key = str(value).strip().lower()
    if key not in BOOLEAN_WORDS:
        raise SystemExit(f"Expected a boolean, got {value!r}")
    return BOOLEAN_WORDS[key]


def authorization_policy(contract: dict) -> tuple[list[str], list[str]]:
    block = contract.get("authorization")
    if not isinstance(block, dict):
        raise SystemExit("Preregistration does not declare an authorization block")
    phases = block.get("phases")
    gated = block.get("phases_requiring_explicit_authorization")
    if not isinstance(phases, list) or not phases:
        raise SystemExit("Preregistration does not declare authorization.phases")
    if not isinstance(gated, list):
        raise SystemExit(
            "Preregistration does not declare authorization.phases_requiring_explicit_authorization"
        )
    unknown = sorted(set(gated) - set(phases))
    if unknown:
        raise SystemExit(f"Gated phases are not declared phases: {', '.join(unknown)}")
    return [str(phase) for phase in phases], [str(phase) for phase in gated]


def authorization_record(phase: str, requested: bool, contract: dict) -> dict[str, object]:
    phases, gated = authorization_policy(contract)
    if phase not in phases:
        raise SystemExit(f"Unknown M27D phase {phase!r}; declared: {', '.join(phases)}")
    required = phase in gated
    note = (
        f"Phase {phase} refuses to start unless --donor_kinship_full_run_authorized is "
        "true, so the effective value equals the requested one."
        if required
        else f"Phase {phase} never reads --donor_kinship_full_run_authorized; the "
        "requested value is recorded but was not spent."
    )
    return {
        "phase_executed": phase,
        "full_run_authorized_requested": requested,
        "full_run_authorization_required_by_phase": required,
        "full_run_authorized_effective": bool(requested and required),
        "full_run_authorization_note": note,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    base = json.loads(base64.b64decode(args.base_b64).decode("utf-8"))
    if not isinstance(base, dict):
        raise SystemExit("Base provenance must decode to an object")
    contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if contract.get("pcrelate", {}).get("king_allowed"):
        raise SystemExit("M27D forbids KING")

    record = authorization_record(args.phase, parse_boolean(args.authorization_requested), contract)
    # The ambiguous name is refused rather than overwritten: a base record still carrying
    # it would mean the workflow kept its own copy of the rule after this script took it
    # over, which is exactly the situation that produced the discrepancy.
    if "full_run_authorized" in base:
        raise SystemExit(
            "Base provenance still carries the ambiguous 'full_run_authorized' field"
        )
    collisions = sorted(set(base) & set(record))
    if collisions:
        raise SystemExit(f"Base provenance already sets: {', '.join(collisions)}")

    provenance = dict(base)
    provenance.update(record)
    args.out.write_text(json.dumps(provenance, indent=4) + "\n", encoding="utf-8")
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-b64", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--authorization-requested", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
