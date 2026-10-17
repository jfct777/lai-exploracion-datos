#!/usr/bin/env python3
"""Audit existing LAI lineage; reciprocal protocols are dependent, never pooled.

This is an admission check, not an estimator of biological independence. Historical
population/IBD labels and sensitivity signals retain their original meanings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from r02_lai_target_dependencies import read_json, require, sha256

PROTOCOLS = {
    "FOLD_A_TO_FOLD_B": {"FOLD_A": "FIT", "FOLD_B": "SELECT"},
    "FOLD_B_TO_FOLD_A": {"FOLD_A": "SELECT", "FOLD_B": "FIT"},
}
DEPENDENCIES = (
    "full_scenario_universe", "independent_separation", "common_event_channels",
    "flare_baseline", "quantity_density_identity_controls",
    "ancestry_probability_boundary_metrics", "justified_search",
    "checkpoint_recovery", "measured_resources",
)
IDENTITIES = ("person", "donor", "family", "atomic_unit", "population", "root", "truth", "mosaic")


def authenticated(spec):
    require(isinstance(spec, dict) and {"path", "sha256"} <= spec.keys(), "missing pinned evidence")
    path = Path(spec["path"])
    require(path.is_file() and sha256(path) == spec["sha256"], "evidence differs: " + str(path))
    return path


def rows(spec):
    with authenticated(spec).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def identities(spec):
    values = authenticated(spec).read_text().splitlines()
    require(values and len(set(values)) == len(values) and all(values), "invalid identity list")
    require(spec.get("count", len(values)) == len(values), "identity count differs")
    return set(values)


def cross_roles(units):
    """Return exact collisions across roles, including renamed mosaics."""
    by_key = {key: defaultdict(lambda: defaultdict(set)) for key in IDENTITIES}
    for unit in units:
        for key in IDENTITIES:
            for identity in unit["identities"][key]:
                by_key[key][identity][unit["role"]].add(unit["case_id"])
    return {key: [{"identity": identity, "roles": {r: sorted(c) for r, c in role_map.items()}}
                  for identity, role_map in sorted(values.items()) if len(role_map) > 1]
            for key, values in by_key.items()}


def audit(request_path, expected_sha=None):
    if expected_sha:
        require(sha256(request_path) == expected_sha, "audit request differs")
    request = read_json(request_path)
    require(request.get("schema") == "r02_lai_separation_request_v1", "unsupported separation request")
    require(request.get("policy") == "SEPARATE_DEPENDENT_RECIPROCAL_PROTOCOLS", "protocol policy differs")
    require(request.get("evaluate_opened") is False, "EVALUATE must remain closed")
    require(request.get("cross_protocol_selection_allowed") is False, "cross-protocol selection forbidden")
    sources = request["sources"]
    inventory = read_json(authenticated(sources["units"]))
    design = read_json(authenticated(sources["design_manifest"]))
    roster_rows = rows(sources["roles"])
    original_rows = rows(sources["original_roles"])
    require(len({r["sample_id"] for r in roster_rows}) == len(roster_rows), "duplicate donor roster identity")
    require(len({r["sample_id"] for r in original_rows}) == len(original_rows), "duplicate original identity")
    roster = {r["sample_id"]: r for r in roster_rows}
    original = {r["sample_id"]: r for r in original_rows}
    require(design["outputs_sha256"]["roles.private.tsv"] == sources["roles"]["sha256"], "unbound donor roster")
    require(design["inputs"]["original_roles"]["sha256"] == sources["original_roles"]["sha256"], "unbound original units")
    require(design["outputs_sha256"]["kinship_checks.json"] == sources["kinship_checks"]["sha256"], "unbound kinship sensitivity")
    authenticated(sources["kinship_checks"])
    blocked, units, assignments = [], [], []
    seen = set()
    for item in inventory["units"]:
        case, direction, fold = item["case_id"], item["direction"], item["fold"]
        require(direction in PROTOCOLS and fold in PROTOCOLS[direction], "unknown reciprocal protocol")
        require(item["role"] == PROTOCOLS[direction][fold], "role differs from frozen protocol")
        require((case, direction) not in seen, "duplicate query unit")
        seen.add((case, direction))
        people, donors = identities(item["sample_ids"]), identities(item["donor_ids"])
        for donor in donors:
            require(donor in roster and donor in original, "unmapped donor identity")
            require(roster[donor]["role"] == original[donor]["role"] == "SOURCE_VALID", "donor role changed")
            require(roster[donor]["split"] == fold, "generating donor crosses source fold")
        # Authenticate small primary receipts; array hashes are inventoried separately.
        for name in ("source_manifest", "material_manifest", "truth", "validated_split", "overlay_receipt", "request"):
            authenticated(item["evidence"][name])
        source = read_json(item["evidence"]["source_manifest"]["path"])
        require(item["source_manifest_sha256"] == item["evidence"]["source_manifest"]["sha256"], "source root differs")
        require(item["truth_sha256"] == item["evidence"]["truth"]["sha256"] == source["outputs_sha256"]["truth.private.tsv.gz"], "truth lineage differs")
        split = read_json(item["evidence"]["validated_split"]["path"])
        match = [r for r in split["assignments"] if r["case_id"] == case]
        require(len(match) == 1 and match[0]["role"] == item["role"] and match[0]["component_id"] == item["component_id"], "component lineage differs")
        if "mosaic_hashes" not in item:
            blocked.append("MOSAIC_IDENTITY_NOT_VERIFIED")
        mosaic_rows = rows(item["mosaic_hashes"]) if "mosaic_hashes" in item else []
        if mosaic_rows:
            require(len(mosaic_rows) == len(people) and {r["sample_id"] for r in mosaic_rows} == people,
                    "mosaic and person axes differ")
        mosaics = {r["canonical_mosaic_sha256"] for r in mosaic_rows}
        unit = {"case_id": case, "protocol_id": direction, "role": item["role"], "identities": {
            "person": people, "donor": donors,
            "family": {roster[d]["family_component"] for d in donors},
            "atomic_unit": {original[d]["atomic_unit_id"] for d in donors},
            "population": {original[d]["canonical_population"] for d in donors},
            "root": {item["source_manifest_sha256"]},
            "truth": {item["canonical_truth_sha256"]}, "mosaic": mosaics,
        }}
        require(all(unit["identities"][key] and "" not in unit["identities"][key] for key in IDENTITIES if key != "mosaic"), "unknown separation identity")
        units.append(unit)
        assignments.append({"case_id": case, "protocol_id": direction, "role": item["role"],
                            "component_id": item["component_id"], "overlay_sha256": item["overlay_sha256"],
                            "source_manifest_sha256": item["source_manifest_sha256"]})
    require(len(units) == inventory["unit_count"] == request["expected_query_units"], "incomplete query universe")
    protocols = {}
    for protocol in PROTOCOLS:
        subset = [u for u in units if u["protocol_id"] == protocol]
        require({u["role"] for u in subset} == {"FIT", "SELECT"}, "both development roles required")
        collisions = cross_roles(subset)
        if any(collisions.values()): blocked.append("CROSS_ROLE_COLLISION:" + protocol)
        protocols[protocol] = {"collision_counts": {k: len(v) for k, v in collisions.items()},
                               "collisions": collisions, "query_units": len(subset)}
    pooled = cross_roles(units)
    blocked += ["INDEPENDENT_EVALUATE_COMPONENT_ABSENT", "FULL_SCIENTIFIC_DEPENDENCIES_NOT_VERIFIED"]
    summary = design["summary"]
    if summary["cross_reference_donor_flagged_pairs_at_lowest_phi"]:
        blocked.append("REFERENCE_DONOR_KINSHIP_SENSITIVITY_UNRESOLVED")
    if not summary["independence_certified"]:
        blocked.append("BIOLOGICAL_INDEPENDENCE_NOT_CERTIFIED")
    kinship_dependencies = None
    if 'pair_review_manifest' in sources:
        from r02_lai_kinship_dependencies import assess
        review = read_json(authenticated(sources['pair_review_manifest']))
        require(design['inputs']['pair_review']['sha256'] == sources['pair_review_manifest']['sha256'],
                'pair review not bound to original design')
        require(review['outputs_sha256']['pair_review.private.tsv'] == sources['pair_review']['sha256'],
                'pair table not bound to review')
        kinship_dependencies = assess(roster_rows, original_rows, rows(sources['pair_review']),
                                     read_json(authenticated(sources['kinship_checks'])))
        if 'ibd_manifest' in sources:
            from r02_lai_kinship_dependencies import link_existing_ibd
            ibd = read_json(authenticated(sources['ibd_manifest']))
            for key, filename in [('ibd_pairs', 'pairs.private.tsv'), ('ibd_summary', 'summary.json')]:
                require(ibd['outputs_sha256'][filename] == sources[key]['sha256'], 'unbound IBD evidence')
            kinship_dependencies['existing_ibd'] = link_existing_ibd(
                kinship_dependencies, rows(sources['ibd_pairs']), read_json(authenticated(sources['ibd_summary'])))
            blocked.append('IBD_DETECTABILITY_NOT_CALIBRATED')
    else:
        blocked.append('PAIR_LEVEL_KINSHIP_DEPENDENCIES_NOT_TRACED')
    return {"schema": "r02_lai_separation_audit_v1", "status": "AUDITED_DEVELOPMENT_PROTOCOLS_PRODUCTION_BLOCKED",
            "request_sha256": sha256(request_path), "assignments": assignments, "protocols": protocols,
            "pooled_collision_counts": {k: len(v) for k, v in pooled.items()},
            "pooling_allowed": False, "cross_protocol_selection_allowed": False,
            "independent_replicates": False, "evaluate_opened": False, "evaluate_assignments": [],
            "training_allowed": False, "blockers": sorted(set(blocked)),
            "scientific_dependencies": {key: "NOT_VERIFIED_FOR_PRODUCTION" for key in DEPENDENCIES},
            "reference_donor_flagged_pairs": summary["cross_reference_donor_flagged_pairs_at_lowest_phi"],
            "kinship_dependency_sensitivity": kinship_dependencies,
            "allowed_use": "PINNED_INPUT_REUSE_WITHIN_ONE_DEPENDENT_DEVELOPMENT_PROTOCOL",
            "limits": ["Nominal population and component disjointness is not biological independence.",
                       "Content identity denotes supplied simulated mosaics, not validated parental phase.",
                       "A fixed external reference panel does not resolve its kinship sensitivity."]}


def require_single_protocol(requests):
    """Used before Nextflow stages any neural input, including cached tasks."""
    directions = {r.get("protocol_id") for r in requests}
    require(len(directions) == 1 and next(iter(directions), None) in PROTOCOLS, "one explicit LAI protocol required; pooling is forbidden")
    direction = next(iter(directions))
    require(all(r.get("direction") == direction for r in requests), "protocol and direction differ")
    return direction


def require_training_separation(contract):
    """Fail closed for real LAI; synthetic code tests do not assert admission.

    The current audit cannot certify a three-role independent population. A future
    admission needs its own reviewed evidence/schema, not a Boolean flip here.
    """
    spec = contract.get("lai_separation_request")
    require(spec is not None, "LAI production requires authenticated joint separation and scientific admission")
    result = audit(authenticated(spec), spec["sha256"])
    require(result["training_allowed"], "LAI production blocked: " + "; ".join(result["blockers"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.request, args.request_sha256)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    if output.exists():
        require(output.read_text() == payload, "existing audit differs; preserve and use a new revision")
    else:
        with output.open("x") as stream: stream.write(payload)
    print(json.dumps({"status": result["status"], "training_allowed": result["training_allowed"]}))


if __name__ == "__main__":
    main()
