#!/usr/bin/env python3
"""Parametric synthetic DENSITY_MATCHED operation; human materialization stays blocked."""
from __future__ import annotations

import hashlib
import json
import math
import argparse
import os
from pathlib import Path

ALGORITHM_VERSION = "DENSITY_CM_BIN_SHA256_PCG64_V1"
REQUEST_SCHEMA = "r02_lai_density_synthetic_request_v1"
MATERIAL_REQUEST_SCHEMA = "r02_lai_density_material_request_v1"
MATERIAL_RECEIPT_SCHEMA = "r02_lai_five_control_receipt_v2"
MATERIAL_STATUS = "FIVE_CONDITIONS_MATERIALIZED_FOR_ONE_RADIUS_PREPARE_ONLY"
EVENT_AXIS_FIELDS = (
    "event_source_index", "event_position_bp", "event_ref", "event_alt",
    "event_counted_allele", "event_is_alt",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_nonhuman_request(request):
    """Reject scientific/human use while width and geometry remain unadmitted."""
    required = {"schema", "operation", "width_cm", "offset_cm", "seed", "algorithm_version",
                "human_material", "training_authorized", "width_scientifically_admitted"}
    require(isinstance(request, dict) and set(request) == required, "density request fields differ")
    require(request["schema"] == REQUEST_SCHEMA and request["operation"] == "SYNTHETIC_TRANSFORM_ONLY",
            "only synthetic density transformation is implemented")
    require(request["human_material"] is False, "human DENSITY_MATCHED materialization is blocked")
    require(request["training_authorized"] is False, "DENSITY_MATCHED does not authorize TRAIN")
    require(request["width_scientifically_admitted"] is False,
            "this module does not admit or select a scientific width")
    require(request["algorithm_version"] == ALGORITHM_VERSION, "density algorithm version differs")
    _validate_parameters(request["width_cm"], request["offset_cm"], request["seed"])
    return request


def _validate_parameters(width_cm, offset_cm, seed):
    require(type(width_cm) in (int, float) and math.isfinite(width_cm) and width_cm > 0,
            "width_cm must be finite and positive")
    require(type(offset_cm) in (int, float) and math.isfinite(offset_cm)
            and 0 <= offset_cm < width_cm, "offset_cm must satisfy 0 <= offset < width")
    require(type(seed) is int and 0 <= seed < 2**63, "seed differs")


def density_bin_ids(chromosome, position_cm, width_cm, offset_cm, np):
    """Return chromosome-local nonoverlapping half-open bin IDs.

    The first and last bins are truncated at the observed chromosome range.
    An offset greater than zero creates a first bin ``[min_cm,min_cm+offset)``.
    """
    _validate_parameters(width_cm, offset_cm, 0)
    chrom = np.asarray(chromosome).astype(str); cm = np.asarray(position_cm, dtype=np.float64)
    require(chrom.ndim == cm.ndim == 1 and len(chrom) == len(cm) and len(cm) > 0,
            "chromosome/cM axes differ")
    require(np.isfinite(cm).all(), "EVENT cM must be finite")
    result = np.empty(len(cm), dtype=np.int64)
    next_id = 0
    # Preserve first-seen chromosome order; the identifier never permits cross-chromosome exchange.
    for value in dict.fromkeys(chrom.tolist()):
        loci = np.flatnonzero(chrom == value); start = float(cm[loci].min()); origin = start + offset_cm
        local = np.where(cm[loci] < origin, -1,
                         np.floor((cm[loci] - origin) / width_cm).astype(np.int64))
        unique = sorted(set(local.tolist()))
        mapping = {key: next_id + rank for rank, key in enumerate(unique)}
        result[loci] = np.asarray([mapping[x] for x in local], dtype=np.int64)
        next_id += len(unique)
    return result


def _rng(seed, algorithm_version, case_id, direction, role, chromosome, person, width, offset, bin_id, np):
    pieces = (algorithm_version, str(seed), case_id, direction, role, str(chromosome), str(person),
              float(width).hex(), float(offset).hex(), str(bin_id))
    derived = int.from_bytes(hashlib.sha256("\0".join(pieces).encode()).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(derived))


def transform_density(event_dosage, event_valid, event_chromosome, event_cm, *, width_cm,
                      offset_cm, seed, case_id, direction, role,
                      algorithm_version=ALGORITHM_VERSION, np=None):
    """Permute observed diploid doses within person/chromosome/cM bin.

    Locus masks stay fixed. This preserves each person's observed opportunities,
    dose histogram, carried-locus count and allele-copy sum in every bin.
    """
    if np is None:
        import numpy as np
    _validate_parameters(width_cm, offset_cm, seed)
    require(algorithm_version == ALGORITHM_VERSION, "density algorithm version differs")
    require(all(isinstance(x, str) and x for x in (case_id, direction, role)), "density identity differs")
    dose = np.asarray(event_dosage); valid = np.asarray(event_valid)
    chrom = np.asarray(event_chromosome).astype(str); cm = np.asarray(event_cm, dtype=np.float64)
    require(dose.ndim == 2 and valid.shape == dose.shape and valid.dtype == np.bool_,
            "EVENT dosage/mask axes differ")
    require(chrom.shape == cm.shape == (dose.shape[1],), "EVENT locus axes differ")
    require(np.isin(dose[valid], [0, 1, 2]).all(), "observed EVENT dosage is not diploid")
    bins = density_bin_ids(chrom, cm, width_cm, offset_cm, np)
    # The locus catalogue is shared by all people. Group it once, avoiding a
    # full chromosome scan for every person and every interval.
    grouped = {}
    for locus, bin_id in enumerate(bins.tolist()):
        grouped.setdefault(bin_id, []).append(locus)
    grouped = {key: np.asarray(value, dtype=np.int64)
               for key, value in sorted(grouped.items())}
    output = dose.copy(); rows = []
    totals = {"bins": 0, "bins_with_observed": 0, "degenerate_bins": 0,
              "constant_dosage_bins": 0, "observed_cells": 0,
              "permutable_cells": 0, "moved_indices": 0, "values_changed": 0}
    for person in range(dose.shape[0]):
        for bin_id, all_loci in grouped.items():
            loci = all_loci[valid[person, all_loci]]
            totals["bins"] += 1; totals["observed_cells"] += len(loci)
            if len(loci): totals["bins_with_observed"] += 1
            degenerate = len(loci) < 2
            constant = len(loci) > 0 and len(np.unique(dose[person, loci])) == 1
            if degenerate: totals["degenerate_bins"] += 1
            if constant: totals["constant_dosage_bins"] += 1
            moved = changed = 0
            if len(loci) > 1:
                chromosome = chrom[all_loci[0]]
                perm = _rng(seed, algorithm_version, case_id, direction, role, chromosome,
                            person, width_cm, offset_cm, bin_id, np).permutation(len(loci))
                source = loci[perm]; before = dose[person, loci]
                output[person, loci] = dose[person, source]
                moved = int(np.count_nonzero(source != loci)); changed = int(np.count_nonzero(before != output[person, loci]))
                totals["permutable_cells"] += len(loci); totals["moved_indices"] += moved
                totals["values_changed"] += changed
            rows.append({"person_index": person, "chromosome": str(chrom[all_loci[0]]),
                         "bin_id": int(bin_id), "loci": int(len(all_loci)),
                         "observed": int(len(loci)), "degenerate": bool(degenerate),
                         "constant_dosage": bool(constant), "moved_indices": moved,
                         "values_changed": changed})
    metrics = {"algorithm_version": algorithm_version, "seed": seed,
               "width_cm": float(width_cm), "offset_cm": float(offset_cm),
               "case_id": case_id, "direction": direction, "role": role,
               "totals": totals, "by_person_chromosome_bin": rows,
               "claim_limit": "Preserves density only in these nonoverlapping cM bins; not in every overlapping network window.",
               "human_materialization_authorized": False, "training_authorized": False}
    return output, valid.copy(), bins, metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _descriptor(value, name):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"},
            name + " descriptor differs")
    path = Path(value["path"])
    require(path.is_file() and sha256(path) == value["sha256"], name + " differs")
    return path


def authenticate_event_axis(control_arrays, factorized_arrays, np):
    """Require the control EVENT loci to equal the factorized EVENT loci in order.

    A matching locus count or chromosome is insufficient: two catalogues can have
    the same dimensions while referring to different variants or counted alleles.
    """
    missing_control = [name for name in EVENT_AXIS_FIELDS if name not in control_arrays]
    missing_factorized = [name for name in EVENT_AXIS_FIELDS if name not in factorized_arrays]
    require(not missing_control, "control EVENT identity fields absent: " + ",".join(missing_control))
    require(not missing_factorized,
            "factorized EVENT identity fields absent: " + ",".join(missing_factorized))
    for name in EVENT_AXIS_FIELDS:
        left = np.asarray(control_arrays[name])
        right = np.asarray(factorized_arrays[name])
        if left.dtype.kind in "USO" or right.dtype.kind in "USO":
            left, right = left.astype(str), right.astype(str)
        require(np.array_equal(left, right), "control/factorized EVENT axis differs: " + name)
    return {name: np.asarray(control_arrays[name]) for name in EVENT_AXIS_FIELDS}


def materialize(request_path, expected_sha256, output_dir):
    """Add one radius-specific DENSITY_MATCHED view to an existing four-view product."""
    import numpy as np
    import r02_lai_control_views as controls

    require(sha256(request_path) == expected_sha256, "density material request hash differs")
    request = json.loads(Path(request_path).read_text())
    required = {"schema", "operation", "task_id", "case_id", "direction", "role",
                "chromosome", "radius_cm", "width_cm", "offset_cm", "seed",
                "algorithm_version", "four_control_receipt",
                "four_control_batch", "event_axis", "event_axis_receipt",
                "training_authorized", "selection_authorized", "evaluate_opened",
                "truth_loaded"}
    require(set(request) == required and request.get("schema") == MATERIAL_REQUEST_SCHEMA
            and request.get("operation") == "PREPARE", "density material request differs")
    require(all(request.get(key) is False for key in
                ("training_authorized", "selection_authorized", "evaluate_opened", "truth_loaded")),
            "density request opened a forbidden operation")
    radius = request["radius_cm"]
    require(type(radius) in (int, float) and math.isfinite(radius) and radius > 0
            and request["width_cm"] == 2 * radius and request["offset_cm"] == 0,
            "density interval must be the full EVENT window width 2r with zero offset")
    require(request["algorithm_version"] == ALGORITHM_VERSION, "density algorithm version differs")
    paths = {name: _descriptor(request[name], name) for name in
             ("four_control_receipt", "four_control_batch",
              "event_axis", "event_axis_receipt")}
    source = json.loads(paths["four_control_receipt"].read_text())
    require(source.get("schema") == "r02_lai_four_control_receipt_v1"
            and source.get("status") == "FOUR_DEFINED_CONDITIONS_MATERIALIZED_DENSITY_BLOCKED"
            and source.get("batch_sha256") == request["four_control_batch"]["sha256"],
            "four-control source receipt differs")
    with np.load(paths["four_control_batch"], allow_pickle=False) as batch_source:
        require(sorted(batch_source.files) == source.get("arrays"),
                "four-control source inventory differs")
        arrays = {key: batch_source[key] for key in batch_source.files}
    require(all(controls.array_sha256(value, np) == source["arrays_sha256"].get(key)
                for key, value in arrays.items()), "four-control source array differs")
    axis_receipt = json.loads(paths["event_axis_receipt"].read_text())
    require(axis_receipt.get("schema") == "r02_lai_factorized_predictors_v1"
            and axis_receipt.get("outputs_sha256", {}).get("indices.npz") ==
                request["event_axis"]["sha256"], "density EVENT axis receipt differs")
    for key in ("task_id", "case_id", "direction", "role"):
        require(request[key] == source.get(key) == axis_receipt.get(key), key + " differs")
    require(str(request["chromosome"]) == str(source.get("chromosome")) ==
            str(axis_receipt.get("chromosome")), "chromosome differs")
    with np.load(paths["event_axis"], allow_pickle=False) as axis_source:
        axis = {name: axis_source[name] for name in axis_source.files}
    authenticate_event_axis(arrays, axis, np)
    event_cm = axis["event_position_cm"].astype(np.float64)
    event_chrom = axis["event_chromosome"].astype(str)
    axis_people = axis["person_id"].astype(str)
    require(np.array_equal(arrays["person_id"].astype(str), axis_people)
            and len(event_cm) == source["event_loci"]
            and np.array_equal(arrays["event_chromosome"].astype(str), event_chrom),
            "density control and EVENT axes differ")
    density_dose, density_valid, bins, metrics = transform_density(
        arrays["NONE_event_dosage"], arrays["NONE_event_valid"], event_chrom, event_cm,
        width_cm=request["width_cm"], offset_cm=request["offset_cm"], seed=request["seed"],
        case_id=request["case_id"], direction=request["direction"], role=request["role"],
        algorithm_version=request["algorithm_version"], np=np)
    output = dict(arrays)
    output["DENSITY_MATCHED_event_dosage"] = density_dose
    output["DENSITY_MATCHED_event_valid"] = density_valid
    output["density_bin_id"] = bins
    out = Path(output_dir)
    require(not out.exists() and out.parent.is_dir(), "fresh density output directory required")
    out.mkdir(mode=0o700)
    batch = out / "control-views.npz"
    np.savez_compressed(batch, **output)
    array_hashes = {key: controls.array_sha256(value, np) for key, value in output.items()}
    operations = dict(source["condition_operations"])
    operations["DENSITY_MATCHED"] = (
        "EVENT_DOSAGE_PERMUTED_WITHIN_PERSON_CHROMOSOME_NONOVERLAPPING_CM_INTERVAL_"
        "MASK_FIXED_WIDTH_EQUALS_TWO_TIMES_RARE_RADIUS")
    receipt = {
        "schema": MATERIAL_RECEIPT_SCHEMA, "status": MATERIAL_STATUS,
        "request_sha256": expected_sha256, "batch_sha256": sha256(batch),
        "arrays": sorted(output), "arrays_sha256": array_hashes,
        "task_id": request["task_id"], "case_id": request["case_id"],
        "direction": request["direction"], "role": request["role"],
        "chromosome": str(request["chromosome"]), "people": source["people"],
        "event_loci": source["event_loci"], "reference_identity": source["reference_identity"],
        "factorized_receipt_sha256": request["event_axis_receipt"]["sha256"],
        "event_annotation_schema": source["event_annotation_schema"],
        "fixed_revised_base": source["fixed_revised_base"],
        "conditions_materialized": ["NONE", "NO_RARE", "COUNT_MATCHED",
                                    "DENSITY_MATCHED", "IDENTITY_SHAM"],
        "condition_operations": operations, "condition_display_aliases": {"NONE": "FULL"},
        "radius_cm": float(radius), "width_cm": float(request["width_cm"]),
        "offset_cm": 0.0, "seed": request["seed"],
        "algorithm_version": request["algorithm_version"], "density_metrics": metrics,
        "density_claim_limit": metrics["claim_limit"],
        "source_four_control_receipt_sha256": request["four_control_receipt"]["sha256"],
        "training_authorized": False, "selection_authorized": False,
        "evaluate_opened": False, "truth_loaded": False, "controls_materialized": True,
        "scientific_limit": "One radius-specific PREPARE view; no training, selection, evaluation, null-model claim, or biological result.",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_materialized(receipt_path, expected_receipt_sha256, batch_path, action="PREPARE"):
    import numpy as np
    import r02_lai_control_views as controls
    require(action == "PREPARE", "density product only permits PREPARE")
    require(sha256(receipt_path) == expected_receipt_sha256, "density receipt hash differs")
    receipt = json.loads(Path(receipt_path).read_text())
    require(receipt.get("schema") == MATERIAL_RECEIPT_SCHEMA
            and receipt.get("status") == MATERIAL_STATUS
            and receipt.get("batch_sha256") == sha256(batch_path), "density receipt differs")
    require(all(receipt.get(key) is False for key in
                ("training_authorized", "selection_authorized", "evaluate_opened", "truth_loaded"))
            and receipt.get("controls_materialized") is True, "density product opened forbidden operation")
    with np.load(batch_path, allow_pickle=False) as source:
        require(sorted(source.files) == receipt.get("arrays"), "density array inventory differs")
        arrays = {key: source[key] for key in source.files}
    require(all(controls.array_sha256(value, np) == receipt["arrays_sha256"].get(key)
                for key, value in arrays.items()), "density array hash differs")
    return arrays, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); os.umask(0o077)
    print(json.dumps(materialize(args.request, args.request_sha256, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
