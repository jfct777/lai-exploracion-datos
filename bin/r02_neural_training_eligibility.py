#!/usr/bin/env python3
"""Freeze role-separation evidence before either human neural trainer can run.

The receipt contains identities and hashes, never EVALUATE responses.  Structure
and LAI share this gate, while retaining their different biological units.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


SCHEMA = "r02_neural_training_eligibility_request_v1"
RECEIPT_SCHEMA = "r02_neural_training_eligibility_receipt_v1"
ROLES = ("FIT", "SELECT", "EVALUATE")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, expected=None):
    if expected is not None:
        require(sha256(path) == expected, "JSON SHA-256 differs: " + str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def descriptor(value, name):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"},
            "invalid descriptor: " + name)
    path = Path(value["path"])
    require(path.is_absolute() and path.is_file() and sha256(path) == value["sha256"],
            "pinned source differs: " + name)
    return path


def identity_sha256(rows, roles=("FIT", "SELECT")):
    digest = hashlib.sha256()
    selected = sorted((str(row["sample_id"]), str(row["component"]), str(row["role"]))
                      for row in rows if row["role"] in roles)
    for fields in selected:
        digest.update(("\t".join(fields) + "\n").encode())
    return digest.hexdigest()


def structure_branch(spec):
    require(set(spec) == {"kind", "persons", "expected_role_counts", "responses_opened"}
            and spec["kind"] == "STRUCTURE_PERSON_COMPONENT_ROSTER",
            "invalid structure eligibility specification")
    require(spec["responses_opened"] == {"FIT": True, "SELECT": True, "EVALUATE": False},
            "structure response roles differ or EVALUATE was opened")
    path = descriptor(spec["persons"], "structure persons")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["sample_id", "component", "role"],
                "structure roster columns differ")
        rows = list(reader)
    require(rows and all(row["sample_id"] and row["component"] and row["role"] in ROLES
                         for row in rows), "empty structure identity/component/role")
    require(len({row["sample_id"] for row in rows}) == len(rows),
            "a structure person crosses roles or is duplicated")
    counts = Counter(row["role"] for row in rows)
    expected = spec["expected_role_counts"]
    require(set(expected) == set(ROLES) and all(type(expected[role]) is int and expected[role] > 0
                                                for role in ROLES),
            "explicit positive role counts required")
    require(dict(counts) == expected, "structure role counts differ")
    component_roles = defaultdict(set)
    for row in rows:
        component_roles[row["component"]].add(row["role"])
    crossed = sorted(component for component, roles in component_roles.items() if len(roles) > 1)
    require(not crossed, "structure family/IBD component crosses roles")
    return {
        "status": "ELIGIBLE_FOR_FIT_SELECT_TRAINING_EVALUATE_RESPONSES_CLOSED",
        "eligibility_gate_passed": True,
        "role_counts": {role: counts[role] for role in ROLES},
        "person_cross_role_collisions": 0,
        "family_ibd_component_cross_role_collisions": 0,
        "donor_separation": "NOT_APPLICABLE_STRUCTURE_USES_PERSON_AND_FAMILY_IBD_COMPONENTS",
        "evaluate_metadata_present": True,
        "evaluate_responses_opened": False,
        "development_identity_sha256": identity_sha256(rows),
        "all_role_identity_sha256": identity_sha256(rows, ROLES),
        "source": spec["persons"],
        "blockers": [],
    }


def lai_branch(spec):
    require(set(spec) == {"kind", "separation_request", "responses_opened"}
            and spec["kind"] == "LAI_PERSON_DONOR_MOSAIC_SEPARATION_AUDIT",
            "invalid LAI eligibility specification")
    require(spec["responses_opened"] == {"FIT": True, "SELECT": True, "EVALUATE": False},
            "LAI response roles differ or EVALUATE was opened")
    path = descriptor(spec["separation_request"], "LAI separation request")
    from r02_lai_separation import audit
    result = audit(path, spec["separation_request"]["sha256"])
    blockers = set(result["blockers"])
    collisions = result["pooled_collision_counts"]
    if collisions.get("donor") or collisions.get("family") or collisions.get("atomic_unit"):
        blockers.add("DONOR_OR_FAMILY_IBD_INDEPENDENCE_NOT_VERIFIED")
    if not result["evaluate_assignments"]:
        blockers.add("INDEPENDENT_EVALUATE_METADATA_ABSENT")
    allowed = (not blockers and result["training_allowed"] and not result["evaluate_opened"])
    return {
        "status": ("ELIGIBLE_FOR_FIT_SELECT_TRAINING_EVALUATE_RESPONSES_CLOSED" if allowed
                   else "BLOCKED_BEFORE_NEURAL_TRAINING"),
        "eligibility_gate_passed": allowed,
        "pooled_collision_counts": collisions,
        "protocols": {name: value["collision_counts"]
                      for name, value in result["protocols"].items()},
        "donor_separation": "VERIFIED" if allowed else "NOT_VERIFIED",
        "evaluate_metadata_present": bool(result["evaluate_assignments"]),
        "evaluate_responses_opened": result["evaluate_opened"],
        "source": spec["separation_request"],
        "separation_audit_request_sha256": result["request_sha256"],
        "blockers": sorted(blockers),
    }


def build(request_path, expected_sha256):
    request = read_json(request_path, expected_sha256)
    require(set(request) == {"schema", "evaluate_opened", "branches"}
            and request["schema"] == SCHEMA and request["evaluate_opened"] is False,
            "invalid joint neural eligibility request or EVALUATE opened")
    require(set(request["branches"]) == {"structure", "lai"},
            "both structure and LAI branches are required")
    branches = {
        "structure": structure_branch(request["branches"]["structure"]),
        "lai": lai_branch(request["branches"]["lai"]),
    }
    return {
        "schema": RECEIPT_SCHEMA,
        "status": ("ALL_BRANCH_SEPARATION_GATES_PASS" if all(value["eligibility_gate_passed"] for value in branches.values())
                   else "ONE_OR_MORE_BRANCHES_BLOCKED"),
        "request_sha256": expected_sha256,
        "verifier_sha256": sha256(__file__),
        "evaluate_opened": False,
        "scope": "IDENTITY_FAMILY_IBD_DONOR_MOSAIC_AND_EVALUATE_ROLE_SEPARATION_ONLY",
        "does_not_certify": ["complete scientific inputs", "channel comparability",
                             "hyperparameter search", "resource readiness", "biological validity"],
        "branches": branches,
    }


def require_contract_eligibility(contract):
    spec = contract.get("eligibility_manifest")
    require(isinstance(spec, dict) and set(spec) == {"path", "sha256"},
            "development training requires a pinned joint eligibility manifest")
    path = descriptor(spec, "joint neural eligibility manifest")
    receipt = read_json(path, spec["sha256"])
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("evaluate_opened") is False,
            "joint eligibility receipt differs or opened EVALUATE")
    objective = contract.get("objective")
    require(objective in receipt.get("branches", {}), "objective missing from eligibility receipt")
    branch = receipt["branches"][objective]
    require(branch.get("eligibility_gate_passed") is True,
            objective + " training blocked by eligibility receipt: " + "; ".join(branch.get("blockers", [])))
    if objective == "structure":
        rows = [{"sample_id": person, "component": value["component"], "role": value["role"]}
                for person, value in contract["persons"].items()]
        require(identity_sha256(rows) == branch.get("development_identity_sha256"),
                "structure FIT/SELECT identities differ from eligibility receipt")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.request, args.request_sha256)
    output = Path(args.output)
    require(not output.exists() and output.parent.is_dir(), "fresh eligibility output required")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "structure_eligibility_gate_passed": result["branches"]["structure"]["eligibility_gate_passed"],
                      "lai_eligibility_gate_passed": result["branches"]["lai"]["eligibility_gate_passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
