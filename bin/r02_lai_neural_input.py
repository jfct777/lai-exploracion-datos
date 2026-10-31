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
REFERENCE_HASHES = {"event_catalogue_sha256", "ref_train_receipt_sha256",
                    "ref_support_by_provenance_sha256", "reference_vcf_sha256",
                    "reference_roles_sha256"}
STAGED_INPUTS = None
REGIONAL_TRUTH_REQUEST_SCHEMA = "r02_lai_regional_truth_join_request_v1"
REGIONAL_TRUTH_RECEIPT_SCHEMA = "r02_lai_regional_truth_join_receipt_v1"
REGIONAL_TRUTH_STATUS = "REGIONAL_TRUTH_AND_ONE_OF_FOUR_CONTROLS_PREPARED_NO_TRAINING"
REGIONAL_TRUTH_REQUEST_SCHEMA_V2 = "r02_lai_regional_truth_join_request_v2"
REGIONAL_TRUTH_RECEIPT_SCHEMA_V2 = "r02_lai_regional_truth_join_receipt_v2"
REGIONAL_TRUTH_STATUS_V2 = "REGIONAL_TRUTH_CONTROL_AND_QUERY_BURDEN_PREPARED_NO_TRAINING"
REGIONAL_FACTORIZED_LINK_RECEIPT_SCHEMA = "r02_lai_regional_truth_factorized_link_receipt_v1"
REGIONAL_FACTORIZED_LINK_STATUS = "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION"
REGIONAL_FALSE_FIELDS = ("training_authorized", "selection_authorized", "evaluate_opened",
                         "controls_materialized")


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
        require(channel.get("controls_materialized") is False, "channel claims materialized LAI controls")
        reference_identity = channel.get("reference_identity")
        require(channel.get("event_reference_feature_schema") == "r02_lai_event_reference_features_v1"
                and channel.get("event_reference_features") == 15
                and channel.get("event_reference_feature_formula") ==
                    "PER_ANCESTRY(AC_0,AN_0,AF_0,KNOWN,FREQUENCY_DEFINED);AF_0=AC/AN_IFF_KNOWN_AND_AN_GT_0"
                and channel.get("event_reference_ancestry_order") ==
                    ["African", "European", "Native_American"]
                and isinstance(reference_identity, dict)
                and set(reference_identity) == REFERENCE_HASHES
                and all(re.fullmatch(r"[0-9a-f]{64}", value or "")
                        for value in reference_identity.values()), "channel LAI reference identity differs")
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
                     "event_chrom", "event_pos_bp", "event_valid", "event_core", "event_padding",
                     "event_reference_features", "baseline_probabilities"}
    required_overlay = {"person_id", "role", "component_id", "query_features", "query_cm", "query_chrom",
                        "query_pos_bp", "query_valid", "query_score_weight_cm", "truth_state", "truth_observed"}
    require(required_base <= set(base) and required_overlay <= set(overlay), "neural batch arrays incomplete")
    for key in ("person_id", "role", "component_id"):
        require(np.array_equal(base[key].astype(str), overlay[key].astype(str)), f"{key} axis differs")
    n, q = overlay["query_valid"].shape
    p = base["baseline_probabilities"]
    require(p.shape == (n, q, 6) and np.isfinite(p).all() and (p >= 0).all(), "FLARE comparator shape/domain differs")
    require(np.allclose(p.sum(-1), 1.0, atol=5e-5), "FLARE six-state probability mass differs")
    require(base["event_reference_features"].shape == (*base["event_valid"].shape, 15)
            and np.isfinite(base["event_reference_features"]).all(),
            "LAI EVENT reference feature tensor differs")
    from r02_lai_reference_features import reference_axes_identity, reference_features
    rebuilt_reference_features = reference_features(base, np)
    require(np.array_equal(base["event_reference_features"], rebuilt_reference_features),
            "LAI EVENT reference features do not match the explicit REF/ALT join")
    axes_identity = reference_axes_identity(base, np)
    channel_receipt = read_json(descriptor(inputs["channel_receipt"], "channel_receipt"))
    require(channel_receipt.get("person_axis_sha256") == axes_identity["person_axis_sha256"],
            "channel receipt and assembled person axis differ")
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
               "model_selection_authorized": False, "evaluate_opened": False,
               "controls_materialized": False,
               "reference_identity": channel_receipt["reference_identity"],
               "reference_join_key": ["chromosome", "position_bp", "REF", "ALT"],
               "reference_orientation_fields": ["event_counted_allele", "event_is_alt"],
               **axes_identity}
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _regional_truth_request(path, expected_sha):
    require(sha(path) == expected_sha, "regional truth request SHA-256 differs")
    value = read_json(path)
    required = {"schema", "operation", "task_id", "case_id", "direction", "role", "chromosome",
                "condition", "regional_request", "regional_receipt", "regional_batch",
                "overlay", "overlay_receipt", "control_request", "control_receipt", "control_batch",
                *REGIONAL_FALSE_FIELDS}
    v2 = value.get("schema") == REGIONAL_TRUTH_REQUEST_SCHEMA_V2
    if v2:
        required |= {"event_burden_prefix", "event_burden_prefix_receipt", "rare_radius_cm",
                     "lai_burden_scope", "lai_burden_boundary", "lai_burden_count_names"}
    require(set(value) == required and value.get("schema") in
            {REGIONAL_TRUTH_REQUEST_SCHEMA, REGIONAL_TRUTH_REQUEST_SCHEMA_V2}
            and value.get("operation") == "PREPARE", "regional truth PREPARE request differs")
    require(value.get("condition") in
            {"NONE", "NO_RARE", "COUNT_MATCHED", "DENSITY_MATCHED", "IDENTITY_SHAM"},
            "condition is not one of the five defined views")
    require(all(value.get(k) is False for k in REGIONAL_FALSE_FIELDS), "forbidden operation is open")
    paths = {k: descriptor(value[k], k) for k in ("regional_request", "regional_receipt", "regional_batch",
             "overlay", "overlay_receipt", "control_request", "control_receipt", "control_batch")}
    if v2:
        paths.update({k: descriptor(value[k], k) for k in
                      ("event_burden_prefix", "event_burden_prefix_receipt")})
    return value, paths


def _take_query(array, people, query_indices, np):
    value = np.asarray(array)
    require(value.ndim >= 2 and value.shape[0] > max(people, default=-1)
            and value.shape[1] > max(query_indices, default=-1), "overlay query axes are out of range")
    return value[np.asarray(people)[:, None], np.asarray(query_indices)[None, :]]


def authenticate_regional_radius(request, regional_receipt):
    """Bind the requested burden radius to both stored regional geometries."""
    radius = request.get("rare_radius_cm")
    recipe = regional_receipt.get("layout_recipe", {})
    resolved = regional_receipt.get("resolved_model_config", {})
    require(type(radius) in (int, float) and math.isfinite(radius) and radius > 0
            and regional_receipt.get("scale_cm") == radius
            and recipe.get("common_radius_cm") == radius
            and recipe.get("rare_radius_cm") == radius
            and resolved.get("common_radius_cm") == radius
            and resolved.get("rare_radius_cm") == radius,
            "request, regional layout, and resolved model radii differ")
    return float(radius)


def join_regional_truth(request_path, expected_sha, output_dir, factorized_link=False,
                        *, _reconstruct_only=False):
    """Join authenticated overlay targets to revised regional predictors for PREPARE only."""
    import numpy as np
    import r02_lai_regional_channel_bridge as regional_bridge
    import r02_lai_control_views as control_views
    request, paths = _regional_truth_request(request_path, expected_sha)
    regional_request_sha = request["regional_request"]["sha256"]
    regional_receipt_sha = request["regional_receipt"]["sha256"]
    regional = regional_bridge.load(paths["regional_request"], regional_request_sha,
                                    paths["regional_receipt"], regional_receipt_sha,
                                    paths["regional_batch"], action="PREPARE")
    regional_request = read_json(paths["regional_request"])
    regional_receipt = read_json(paths["regional_receipt"])
    if request["schema"] == REGIONAL_TRUTH_REQUEST_SCHEMA_V2:
        authenticate_regional_radius(request, regional_receipt)
    control_receipt_value = read_json(paths["control_receipt"])
    if control_receipt_value.get("schema") == "r02_lai_five_control_receipt_v2":
        import r02_lai_density_control as density_control
        require(control_receipt_value.get("request_sha256") ==
                request["control_request"]["sha256"], "density control request identity differs")
        controls, control_receipt = density_control.load_materialized(
            paths["control_receipt"], request["control_receipt"]["sha256"],
            paths["control_batch"], action="PREPARE")
        require(request["condition"] == "DENSITY_MATCHED"
                and control_receipt.get("radius_cm") == request.get("rare_radius_cm"),
                "density control radius differs")
    else:
        controls, control_receipt = control_views.load(
            paths["control_request"], request["control_request"]["sha256"],
            paths["control_receipt"], request["control_receipt"]["sha256"],
            paths["control_batch"], action="PREPARE")
        require(request["condition"] != "DENSITY_MATCHED",
                "DENSITY_MATCHED requires the radius-specific five-control product")
    for key in ("task_id", "case_id", "direction", "role"):
        require(request[key] == regional_receipt.get(key) == control_receipt.get(key), key + " differs")
    require(str(request["chromosome"]) == str(regional_receipt.get("chromosome")) ==
            str(control_receipt.get("chromosome")), "chromosome differs")
    overlay_receipt = read_json(paths["overlay_receipt"])
    require(overlay_receipt.get("schema") == "r02_lai_truthblind_context_receipt_v2"
            and overlay_receipt.get("output_sha256") == request["overlay"]["sha256"]
            and overlay_receipt.get("truth_read_for_query_geometry") is False,
            "truth-blind overlay receipt differs")
    with np.load(paths["overlay"], allow_pickle=False) as z:
        overlay = {k: z[k] for k in z.files}
    expected_overlay = {"person_id", "role", "component_id", "query_features", "query_cm",
                        "query_chrom", "query_pos_bp", "query_valid", "query_score_weight_cm",
                        "truth_state", "truth_observed"}
    require(expected_overlay <= set(overlay), "overlay arrays incomplete")
    require(not any("flare" in k.lower() or "baseline" in k.lower() for k in overlay),
            "overlay must not replace revised regional FLARE probabilities")
    overlay_ids = overlay["person_id"].astype(str).tolist()
    require(len(set(overlay_ids)) == len(overlay_ids), "overlay person IDs are not unique")
    lookup = {v: i for i, v in enumerate(overlay_ids)}
    regional_ids = regional["person_id"].astype(str).tolist()
    require(all(v in lookup for v in regional_ids), "regional person absent from overlay")
    people = [lookup[v] for v in regional_ids]
    for name in ("role", "component_id"):
        require(np.array_equal(regional[name].astype(str), overlay[name][people].astype(str)), name + " differs")
    qidx = np.asarray(regional["query_global_index"])
    require(qidx.ndim == 1 and np.issubdtype(qidx.dtype, np.integer) and len(np.unique(qidx)) == len(qidx)
            and np.all(qidx >= 0), "query_global_index differs")
    for name in ("query_features", "query_cm", "query_chrom", "query_pos_bp", "query_valid"):
        selected = _take_query(overlay[name], people, qidx.tolist(), np)
        require(np.array_equal(regional[name], selected, equal_nan=True), name + " differs")
    truth = {name: _take_query(overlay[name], people, qidx.tolist(), np)
             for name in ("query_score_weight_cm", "truth_state", "truth_observed")}
    require(not (truth["truth_observed"] & ~regional["query_valid"]).any(), "truth extends query support")
    require(np.isin(truth["truth_state"][truth["truth_observed"]], np.arange(6)).all(), "truth state differs")
    control_ids = controls["person_id"].astype(str).tolist()
    require(len(set(control_ids)) == len(control_ids), "control person IDs are not unique")
    control_lookup = {v: i for i, v in enumerate(control_ids)}
    require(all(v in control_lookup for v in regional_ids), "regional person absent from controls")
    cp = [control_lookup[v] for v in regional_ids]
    eg = np.asarray(regional["event_global_index"])
    require(eg.ndim == 1 and np.issubdtype(eg.dtype, np.integer) and len(np.unique(eg)) == len(eg)
            and np.all((eg >= 0) & (eg < control_receipt["event_loci"])), "event_global_index differs")
    condition = request["condition"]
    dose = controls[condition + "_event_dosage"][np.asarray(cp)[:, None], eg[None, :]]
    valid = controls[condition + "_event_valid"][np.asarray(cp)[:, None], eg[None, :]]
    require(dose.shape == regional["event_dosage"].shape == valid.shape, "regional EVENT cut differs")
    result = dict(regional); result["event_dosage"] = dose; result["event_valid"] = valid
    if condition == "NO_RARE": result["event_core"] = np.zeros_like(regional["event_core"], dtype=bool)
    burden_receipt = None
    if request["schema"] == REGIONAL_TRUTH_REQUEST_SCHEMA_V2:
        import r02_lai_query_burden as query_burden
        require(request["lai_burden_scope"] == query_burden.SCOPE
                and request["lai_burden_boundary"] == query_burden.BOUNDARY
                and request["lai_burden_count_names"] == list(query_burden.COUNT_NAMES),
                "query burden specification differs")
        burden_arrays, burden_receipt = query_burden.load_prefix(
            paths["event_burden_prefix"], paths["event_burden_prefix_receipt"], condition)
        require(burden_receipt["event_input_sha256"] == request["control_batch"]["sha256"]
                and burden_receipt["control_receipt_sha256"] == request["control_receipt"]["sha256"]
                and burden_receipt["event_axis_receipt_sha256"] ==
                    regional_receipt["factorized_receipt_sha256"]
                and burden_receipt["reference_identity"] == control_receipt["reference_identity"]
                and burden_receipt["radius_selected"] is False,
                "query burden source identity differs")
        for key in ("task_id", "case_id", "direction", "role"):
            require(burden_receipt[key] == request[key], "query burden identity differs: " + key)
        require(str(burden_receipt["chromosome"]) == str(request["chromosome"]),
                "query burden chromosome differs")
        burden_ids = burden_arrays["person_id"].astype(str).tolist()
        require(len(set(burden_ids)) == len(burden_ids), "query burden person IDs are not unique")
        burden_lookup = {value: index for index, value in enumerate(burden_ids)}
        require(all(value in burden_lookup for value in regional_ids),
                "regional person absent from query burden")
        bpeople = np.asarray([burden_lookup[value] for value in regional_ids])
        selected_prefix = dict(burden_arrays)
        selected_prefix["prefix_counts"] = burden_arrays["prefix_counts"][bpeople]
        inverse = np.empty(len(burden_arrays["event_order"]), dtype=np.int64)
        inverse[burden_arrays["event_order"]] = np.arange(len(inverse))
        sorted_event = inverse[eg]
        require(np.array_equal(burden_arrays["event_chrom"][sorted_event],
                               regional["event_chrom"][0])
                and np.array_equal(burden_arrays["event_cm"][sorted_event],
                                   regional["event_cm"][0]),
                "regional EVENT axis differs from burden prefix")
        result["query_burden"] = query_burden.project(
            selected_prefix, regional["query_cm"], regional["query_chrom"],
            request["rare_radius_cm"], np)
        require(condition != "NO_RARE" or not result["query_burden"].any(),
                "NO_RARE query burden leaks EVENT information")
    result.update(truth)
    # revisedBASE is carried byte-for-byte from the authenticated regional product.
    require(np.array_equal(result["baseline_probabilities"], regional["baseline_probabilities"]),
            "revised regional FLARE probabilities changed")
    require(not _reconstruct_only or (factorized_link and output_dir is None),
            "in-memory reconstruction requires a factorized link and no output directory")
    if not _reconstruct_only:
        output_dir = Path(output_dir)
        require(not output_dir.exists() and output_dir.parent.is_dir(), "fresh output directory required")
        output_dir.mkdir(mode=0o700)
        batch = output_dir / "batch.npz"
        if not factorized_link:
            np.savez_compressed(batch, **result)
    current = request["schema"] == REGIONAL_TRUTH_REQUEST_SCHEMA_V2
    receipt = {"schema": (REGIONAL_FACTORIZED_LINK_RECEIPT_SCHEMA if factorized_link else
                          REGIONAL_TRUTH_RECEIPT_SCHEMA_V2 if current else REGIONAL_TRUTH_RECEIPT_SCHEMA),
               "request_sha256": expected_sha,
               "batch_sha256": None if factorized_link else sha(batch), "arrays": sorted(result),
               "task_id": request["task_id"], "case_id": request["case_id"],
               "direction": request["direction"], "role": request["role"],
               "chromosome": str(request["chromosome"]), "condition": condition,
               "regional_receipt_sha256": regional_receipt_sha,
               "control_receipt_sha256": request["control_receipt"]["sha256"],
               "overlay_receipt_sha256": request["overlay_receipt"]["sha256"],
               "person_indices": regional_request["person_indices"],
               "region_id": regional_request["region_id"], "query_global_index_sha256": regional_receipt["query_global_index_sha256"],
               "baseline_probabilities_source": "REVISED_BASE_GEN10_FROM_REGIONAL_SOURCE",
               "baseline_probabilities_sha256": regional_receipt["arrays_sha256"]["baseline_probabilities"],
               "condition_operation": control_receipt["condition_operations"][condition],
               "lai_burden_scope": request.get("lai_burden_scope"),
               "lai_burden_boundary": request.get("lai_burden_boundary"),
               "lai_burden_count_names": request.get("lai_burden_count_names"),
               "rare_radius_cm": request.get("rare_radius_cm"),
               "event_burden_prefix_receipt_sha256":
                   (request["event_burden_prefix_receipt"]["sha256"] if current else None),
               "fixed_revised_base": control_receipt["fixed_revised_base"],
               "no_rare_scope": ("NO_INDIVIDUAL_EVENT_CHANNEL; COMMON_AND_FIXED_REVISED_BASE_RETAINED"
                                 if condition == "NO_RARE" else None),
               "no_rare_interpretation_limit":
                   ("Does not claim absence of signals correlated with EVENT alleles in COMMON or fixed revisedBASE."
                    if condition == "NO_RARE" else None),
               "event_fields_retained_for_identity_but_masked_by_event_valid":
                   (["EVENT_COORDINATES", "EVENT_12_METADATA", "EVENT_15_REFERENCE_FEATURES"]
                    if condition == "NO_RARE" else []),
               "truth_joined": True, "training_authorized": False, "selection_authorized": False,
               "evaluate_opened": False, "controls_materialized": False,
               "density_matched_status": ("MATERIALIZED_FOR_THIS_RADIUS_PREPARE_ONLY"
                                          if condition == "DENSITY_MATCHED" else
                                          "SEPARATE_RADIUS_SPECIFIC_PRODUCT"),
               "factorized_link": factorized_link,
               "regional_batch_reused_without_copy": factorized_link,
               "reconstructed_arrays_sha256": ({name: regional_bridge.array_sha256(value, np)
                                                  for name, value in result.items()}
                                                 if factorized_link else None),
               "status": (REGIONAL_FACTORIZED_LINK_STATUS if factorized_link else
                          REGIONAL_TRUTH_STATUS_V2 if current else REGIONAL_TRUTH_STATUS),
               "scientific_limit": "PREPARE join for one of five defined conditions; no training, selection, evaluation, stability, or biological result."}
    if _reconstruct_only:
        return result, receipt
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_factorized_link(request_path, request_sha256, receipt_path, receipt_sha256,
                         *, reconstruction_limits, action="PREPARE"):
    """Rebuild a stored regional condition in memory and authenticate every array.

    The source files, role and geometry are checked by the same operation that
    produced the receipt. No temporary combined NPZ is created. This interface
    retains PREPARE admission; a future scientific training entry point must
    explicitly admit its complete design rather than overriding this restriction.
    """
    import zipfile
    require(action == "PREPARE", "factorized PREPARE link cannot authorize TRAIN")
    require(isinstance(reconstruction_limits, dict)
            and set(reconstruction_limits) == {"max_factorized_source_bytes",
                "max_reconstructed_join_bytes", "max_source_plus_reconstructed_bytes"}
            and all(type(v) is int and v > 0 for v in reconstruction_limits.values()),
            "explicit source, joined-array, and combined reconstruction limits are required")
    require(sha(receipt_path) == receipt_sha256, "factorized receipt SHA-256 differs")
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == REGIONAL_FACTORIZED_LINK_RECEIPT_SCHEMA
            and receipt.get("status") == REGIONAL_FACTORIZED_LINK_STATUS
            and receipt.get("request_sha256") == request_sha256
            and receipt.get("factorized_link") is True
            and receipt.get("regional_batch_reused_without_copy") is True
            and receipt.get("batch_sha256") is None,
            "factorized receipt identity differs")
    _, paths = _regional_truth_request(request_path, request_sha256)
    total = 0
    for name in ("regional_batch", "control_batch", "overlay", "event_burden_prefix"):
        with zipfile.ZipFile(paths[name]) as archive:
            total += sum(item.file_size for item in archive.infolist())
    require(total <= reconstruction_limits["max_factorized_source_bytes"],
            "factorized sources exceed reconstruction memory budget")
    data, rebuilt = join_regional_truth(request_path, request_sha256, None,
                                       factorized_link=True, _reconstruct_only=True)
    require(rebuilt == receipt, "reconstructed factorized arrays or provenance differ")
    reconstructed = sum(value.nbytes for value in data.values())
    require(reconstructed <= reconstruction_limits["max_reconstructed_join_bytes"],
            "reconstructed joined arrays exceed measured technical limit")
    require(total + reconstructed <= reconstruction_limits["max_source_plus_reconstructed_bytes"],
            "factorized sources plus reconstructed arrays exceed reconstruction memory budget")
    return data


def main():
    global STAGED_INPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True); parser.add_argument("--request-sha256")
    parser.add_argument("--output-dir"); parser.add_argument("--audit-output"); parser.add_argument("--staged-inputs")
    parser.add_argument("--regional-truth-join", action="store_true")
    parser.add_argument("--factorized-regional-link", action="store_true")
    args = parser.parse_args(); os.umask(0o077)
    if args.staged_inputs:
        STAGED_INPUTS = Path(args.staged_inputs)
        require(STAGED_INPUTS.is_dir(), "staged input directory absent")
    require(not (args.regional_truth_join and args.factorized_regional_link),
            "choose materialized join or factorized link")
    if args.regional_truth_join or args.factorized_regional_link:
        require(args.output_dir and args.request_sha256, "regional truth join requires output and pinned request")
        value = join_regional_truth(args.request, args.request_sha256, args.output_dir,
                                    factorized_link=args.factorized_regional_link)
    elif args.output_dir:
        require(args.request_sha256, "assembly requires pinned request SHA-256")
        value = assemble(args.request, args.request_sha256, args.output_dir)
    else:
        value = audit_request(args.request, args.request_sha256)
        if args.audit_output:
            Path(args.audit_output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
