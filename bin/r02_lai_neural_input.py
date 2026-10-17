#!/usr/bin/env python3
"""Admit and assemble LAI neural batches without using truth to choose inputs.

The metadata audit is intentionally usable without NumPy.  Assembly requires
NumPy and only joins an authenticated genomic channel batch with an already
materialized truth-blind query overlay.  FLARE probabilities remain a model
input/comparator; recorded simulation ancestry remains a target only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re


REQUEST_SCHEMA = "r02_lai_neural_input_request_v1"
BATCH_SCHEMA = "r02_lai_neural_input_batch_v1"
DIRECTIONS = {
    "FOLD_A_TO_FOLD_B": {"FIT": "FOLD_A", "SELECT": "FOLD_B"},
    "FOLD_B_TO_FOLD_A": {"FIT": "FOLD_B", "SELECT": "FOLD_A"},
}
REQUIRED_CONTROLS = {
    "NONE": "ALL_ADMITTED_CONTEXT",
    "NO_RARE": "COMMON_AND_FLARE_ONLY",
    "COUNT_MATCHED": "RARE_COUNT_FIXED_IDENTITY_REPLACED",
    "DENSITY_MATCHED": "RARE_DENSITY_FIXED_IDENTITY_REPLACED",
    "IDENTITY_SHAM": "RARE_LOCUS_AND_COUNT_FIXED_PERSON_RELATION_BROKEN",
}
STAGED_INPUTS = None


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def descriptor(spec, name, required=True):
    if spec is None and not required:
        return None
    require(isinstance(spec, dict) and set(spec) == {"path", "sha256"}, f"invalid {name} descriptor")
    require(re.fullmatch(r"[0-9a-f]{64}", spec["sha256"] or ""), f"invalid {name} SHA-256")
    path = Path(spec["path"])
    if STAGED_INPUTS is not None:
        matches = [candidate for candidate in STAGED_INPUTS.iterdir()
                   if candidate.is_file() and sha(candidate) == spec["sha256"]]
        require(len(matches) == 1, f"staged {name} is absent or ambiguous")
        path = matches[0]
    require(path.is_file() and sha(path) == spec["sha256"], f"{name} differs")
    return path


def tsv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_case(case_id):
    match = re.fullmatch(r"[^.]+\.(PULSE_T(?:5|10|16)_(?:ANCHOR|SENSITIVITY_[12]))\.chr([0-9]+)\.(FOLD_[AB])\.r([1-9][0-9]*)", case_id)
    require(match, "case identity is not an admitted scenario/chromosome/fold/realization")
    scenario, chromosome, donor_fold, replicate = match.groups()
    require(1 <= int(chromosome) <= 22, "non-autosomal LAI case")
    return scenario, chromosome, donor_fold, int(replicate)


def validate_controls(value):
    require(isinstance(value, list), "control definitions must be a list")
    controls = {row.get("name"): row for row in value if isinstance(row, dict)}
    require(set(controls) == set(REQUIRED_CONTROLS), "exact registered control family required")
    for name, operation in REQUIRED_CONTROLS.items():
        row = controls[name]
        require(row.get("operation") == operation, f"control semantics differ: {name}")
        require(row.get("truth_used") is False and row.get("preserves_people") is True,
                f"control may not use truth or change people: {name}")
    return controls


def audit_request(request_path, expected_sha=None):
    request_path = Path(request_path)
    if expected_sha is not None:
        require(sha(request_path) == expected_sha, "request SHA-256 differs")
    request = read_json(request_path)
    require(request.get("schema") == REQUEST_SCHEMA and request.get("operation") == "ASSEMBLE_ONLY_NO_TRAINING",
            "explicit assembly-only request required")
    direction = request.get("direction")
    require(direction in DIRECTIONS, "unknown reciprocal donor direction")
    scenario, chromosome, donor_fold, replicate = parse_case(request.get("case_id", ""))
    expected_role = "FIT" if donor_fold == DIRECTIONS[direction]["FIT"] else "SELECT"
    require(request.get("expected_role") == expected_role, "donor fold and FIT/SELECT role differ")

    inputs = request.get("inputs", {})
    required = {"material_manifest", "genotype_request", "truthblind_request", "overlay", "overlay_receipt"}
    require(required <= set(inputs), "incomplete genotype/truth/query input inventory")
    paths = {name: descriptor(inputs[name], name) for name in required}
    material = read_json(paths["material_manifest"])
    genotype = read_json(paths["genotype_request"])
    truthblind = read_json(paths["truthblind_request"])
    overlay_receipt = read_json(paths["overlay_receipt"])
    require(material.get("schema") == "r02_lai_genotype_material_v1" and material.get("case_id") == request["case_id"],
            "genotype material identity differs")
    require(str(material.get("chromosome")) == chromosome and material.get("donor_split") == donor_fold,
            "genotype chromosome/donor fold differs")
    require(genotype.get("schema") == "r02_lai_genotype_material_request_v1" and
            genotype.get("inputs", {}).get("manifest", {}).get("sha256") == material.get("source_manifest_sha256"),
            "genotype request does not bind the source simulation")
    require(truthblind.get("schema") == "r02_lai_truthblind_context_request_v2" and
            truthblind.get("case_id") == request["case_id"] and truthblind.get("direction") == direction,
            "truth-blind request identity differs")
    require(truthblind.get("material_manifest_sha256") == inputs["material_manifest"]["sha256"],
            "truth-blind request does not bind genotype material")
    require(genotype["inputs"]["truth"]["sha256"] == truthblind.get("truth_path_sha256"),
            "genotype and query targets do not originate from the same recorded truth")
    require(overlay_receipt.get("schema") == "r02_lai_truthblind_context_receipt_v2" and
            overlay_receipt.get("request_sha256") == inputs["truthblind_request"]["sha256"] and
            overlay_receipt.get("output_sha256") == inputs["overlay"]["sha256"],
            "overlay receipt does not authenticate request/output")
    require(overlay_receipt.get("truth_read_for_query_geometry") is False and
            overlay_receipt.get("n_people") == material.get("n_people") and
            overlay_receipt.get("truth_observed_count", 0) > 0,
            "overlay truth-blindness, people, or observed target support differs")
    require(overlay_receipt.get("roles") == [expected_role], "serialized person role differs from donor direction")

    controls = validate_controls(request.get("controls"))
    base_path = descriptor(inputs.get("channel_batch"), "channel_batch", required=False)
    base_receipt_path = descriptor(inputs.get("channel_receipt"), "channel_receipt", required=False)
    require((base_path is None) == (base_receipt_path is None), "channel batch and receipt must be supplied together")
    blockers = []
    channel = None
    if base_path is None:
        blockers.append("AUTHENTICATED_COMMON_RARE_REFERENCE_AND_FLARE_CHANNEL_BATCH_MISSING")
    else:
        channel = read_json(base_receipt_path)
        require(channel.get("schema") == "r02_lai_neural_channel_receipt_v1" and
                channel.get("batch_sha256") == inputs["channel_batch"]["sha256"],
                "channel receipt does not authenticate batch")
        require(channel.get("case_id") == request["case_id"] and channel.get("direction") == direction and
                channel.get("expected_role") == expected_role, "channel case/direction/role differs")
        require(channel.get("source_genotype_material_sha256") == inputs["material_manifest"]["sha256"] and
                channel.get("source_overlay_sha256") == inputs["overlay"]["sha256"] and
                re.fullmatch(r"[0-9a-f]{64}", channel.get("person_axis_sha256", "")),
                "channel lineage or person axis differs")
        require(channel.get("missing_policy") == "OBSERVED_MASK_SEPARATE_FROM_ZERO_DOSAGE" and
                channel.get("flare_semantics") == "COMPARATOR_INPUT_NOT_TRUTH", "channel missingness/FLARE semantics differ")
        validate_controls(channel.get("controls"))
        require(channel.get("truth_read_for_site_selection") is False, "truth selected genomic channels")

    limits = request.get("resources", {})
    require(set(limits) == {"cpus", "memory_bytes", "timeout_seconds", "max_input_bytes", "max_output_bytes"}
            and all(type(value) is int and value > 0 for value in limits.values()), "explicit positive resource bounds required")
    resolved_inputs = list(paths.values()) + [p for p in (base_path, base_receipt_path) if p is not None]
    total = sum(path.stat().st_size for path in resolved_inputs)
    require(total <= limits["max_input_bytes"], "input byte bound exceeded")
    return {
        "schema": "r02_lai_neural_input_admission_v1",
        "status": "READY_FOR_BOUNDED_ASSEMBLY" if not blockers else "BLOCKED_BEFORE_ASSEMBLY",
        "case_id": request["case_id"], "scenario_id": scenario, "chromosome": chromosome,
        "donor_fold": donor_fold, "replicate": replicate, "direction": direction,
        "expected_role": expected_role, "fit_donor_fold": DIRECTIONS[direction]["FIT"],
        "select_donor_fold": DIRECTIONS[direction]["SELECT"], "n_people": material["n_people"],
        "n_genotype_loci": material["n_loci"], "n_queries": overlay_receipt["n_queries"],
        "genotype_missing_cells": material["gt_missing_cells"],
        "truth_missing_cells_at_genotype_loci": material["truth_coverage"]["missing_cells"],
        "truth_observed_query_cells": overlay_receipt["truth_observed_count"],
        "query_valid_cells": overlay_receipt["query_valid_count"], "blockers": blockers,
        "controls": sorted(controls), "truth_used_for_site_selection": False,
        "flare_is_truth": False, "request_sha256": sha(request_path), "input_bytes": total,
    }


def assemble(request_path, expected_sha, output_dir):
    from r02_lai_separation import require_single_protocol
    require_single_protocol([read_json(request_path)])
    admission = audit_request(request_path, expected_sha)
    require(admission["status"] == "READY_FOR_BOUNDED_ASSEMBLY", "; ".join(admission["blockers"]))
    import numpy as np
    request = read_json(request_path); inputs = request["inputs"]
    channel_path = descriptor(inputs["channel_batch"], "channel_batch")
    overlay_path = descriptor(inputs["overlay"], "overlay")
    with np.load(channel_path, allow_pickle=False) as source:
        base = {key: source[key] for key in source.files}
    with np.load(overlay_path, allow_pickle=False) as source:
        overlay = {key: source[key] for key in source.files}
    required_base = {"person_id", "role", "component_id", "common_features", "common_cm", "common_chrom",
                     "common_pos_bp", "common_valid", "common_core", "event_metadata", "event_dosage", "event_cm",
                     "event_chrom", "event_pos_bp", "event_valid", "event_core", "baseline_probabilities"}
    required_overlay = {"person_id", "role", "component_id", "query_features", "query_cm", "query_chrom",
                        "query_pos_bp", "query_valid", "query_score_weight_cm", "truth_state", "truth_observed"}
    require(required_base <= set(base) and required_overlay <= set(overlay), "neural batch arrays incomplete")
    for key in ("person_id", "role", "component_id"):
        require(np.array_equal(base[key].astype(str), overlay[key].astype(str)), f"{key} axis differs")
    n, q = overlay["query_valid"].shape
    p = base["baseline_probabilities"]
    require(p.shape == (n, q, 6) and np.isfinite(p).all() and (p >= 0).all(), "FLARE comparator shape/domain differs")
    require(np.allclose(p.sum(-1), 1.0, atol=5e-5), "FLARE six-state probability mass differs")
    require(not (overlay["truth_observed"] & ~overlay["query_valid"]).any(), "truth extends query support")
    require(np.isin(overlay["truth_state"][overlay["truth_observed"]], np.arange(6)).all(), "invalid truth state")
    collisions = (set(base) & set(overlay)) - {"person_id", "role", "component_id"}
    require(not collisions, "channel batch predefines query/truth arrays: " + ",".join(sorted(collisions)))
    base.update({key: value for key, value in overlay.items() if key not in {"person_id", "role", "component_id"}})
    output_dir = Path(output_dir); require(not output_dir.exists() and output_dir.parent.is_dir(), "fresh output directory required")
    output_dir.mkdir(mode=0o700); output = output_dir / "batch.npz"
    np.savez_compressed(output, **base)
    receipt = {**admission, "schema": BATCH_SCHEMA, "status": "ASSEMBLED_NOT_TRAINED_OR_EVALUATED",
               "batch_sha256": sha(output), "arrays": sorted(base), "training_authorized": False,
               "model_selection_authorized": False, "evaluate_opened": False}
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    global STAGED_INPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True); parser.add_argument("--request-sha256")
    parser.add_argument("--output-dir"); parser.add_argument("--audit-output"); parser.add_argument("--staged-inputs")
    args = parser.parse_args(); os.umask(0o077)
    if args.staged_inputs:
        STAGED_INPUTS = Path(args.staged_inputs)
        require(STAGED_INPUTS.is_dir(), "staged input directory absent")
    if args.output_dir:
        require(args.request_sha256, "assembly requires pinned request SHA-256")
        value = assemble(args.request, args.request_sha256, args.output_dir)
    else:
        value = audit_request(args.request, args.request_sha256)
        if args.audit_output:
            Path(args.audit_output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
