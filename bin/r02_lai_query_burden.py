#!/usr/bin/env python3
"""Cache LAI EVENT prefix sums and project query-local counts after radius resolution.

Input dosage/masks must already represent one complete chromosome condition.
This module does not permute genotypes, choose a radius, read truth, or train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PREFIX_SCHEMA = "r02_lai_event_burden_prefix_v1"
RECEIPT_SCHEMA = "r02_lai_event_burden_prefix_receipt_v1"
SCOPE = "QUERY_LOCAL_RARE_RADIUS_V1"
BOUNDARY = "SAME_CHROMOSOME_ABS_CM_LE_RADIUS"
# Preserve r02_context_train.event_burden_counts v1 execution order. A
# historical metadata list says carrier/observed/copies; it does not describe
# the arrays already computed by the trainer.
COUNT_NAMES = ("carrier_sites", "minor_copies", "observed_sites")
CONDITIONS = ("NONE", "NO_RARE", "COUNT_MATCHED", "DENSITY_MATCHED", "IDENTITY_SHAM")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value, np):
    value = np.asarray(value); digest = hashlib.sha256()
    digest.update(value.dtype.str.encode() + b"\0")
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode() + b"\0")
    digest.update(value.tobytes(order="C")); return digest.hexdigest()


def _shared_axis(value, width, name, np, finite=True):
    value = np.asarray(value)
    if value.ndim == 1:
        require(value.shape == (width,), name + " shape differs"); row = value
    else:
        require(value.ndim == 2 and value.shape[1] == width and value.shape[0] > 0,
                name + " shape differs")
        require(np.array_equal(value, np.broadcast_to(value[:1], value.shape), equal_nan=True),
                name + " differs between people")
        row = value[0]
    if finite: require(np.isfinite(row).all(), name + " contains nonfinite values")
    return row


def build_prefix(event_dosage, event_valid, event_cm, event_chrom, condition, np):
    """Return sorted axes and cumulative carrier/copy/observed counts."""
    dose, valid = np.asarray(event_dosage), np.asarray(event_valid)
    require(condition in CONDITIONS, "unknown LAI condition")
    require(dose.ndim == 2 and dose.shape == valid.shape and valid.dtype == np.bool_,
            "EVENT dosage/mask shape or dtype differs")
    require(np.isin(dose[valid], (0, 1, 2)).all(), "observed EVENT dosage is not diploid")
    people, events = dose.shape
    cm = _shared_axis(event_cm, events, "event_cm", np)
    chrom = _shared_axis(event_chrom, events, "event_chrom", np)
    require(np.isin(chrom, np.arange(1, 23)).all(), "non-autosomal EVENT chromosome")
    order = np.lexsort((np.arange(events), cm, chrom))
    cm, chrom = cm[order].astype(np.float64), chrom[order].astype(np.int16)
    if condition == "NO_RARE":
        fields = [np.zeros((people, events), dtype=np.int64)] * 3
    else:
        measured = np.where(valid, dose, 0).astype(np.int64, copy=False)[:, order]
        observed = valid.astype(np.int64)[:, order]
        fields = [(measured > 0).astype(np.int64), measured, observed]
    prefix = np.stack([np.concatenate((np.zeros((people, 1), dtype=np.int64),
                       np.cumsum(value, axis=1, dtype=np.int64)), axis=1) for value in fields], axis=2)
    return {"event_order": order.astype(np.int64), "event_cm": cm,
            "event_chrom": chrom, "prefix_counts": prefix}


def project(prefix, query_cm, query_chrom, radius_cm, np):
    """Return int64 [person, query, carrier/copy/observed] counts."""
    counts = np.asarray(prefix["prefix_counts"])
    cm, chrom = np.asarray(prefix["event_cm"]), np.asarray(prefix["event_chrom"])
    require(counts.ndim == 3 and counts.shape == (counts.shape[0], len(cm) + 1, 3)
            and np.issubdtype(counts.dtype, np.integer), "EVENT prefix shape/dtype differs")
    require(type(radius_cm) in (int, float) and math.isfinite(radius_cm) and radius_cm > 0,
            "rare_radius_cm must be finite and positive")
    qvalue = np.asarray(query_cm); people = counts.shape[0]
    require(qvalue.ndim == 2 and qvalue.shape[0] == people and qvalue.shape[1] > 0,
            "query_cm shape differs")
    queries = qvalue.shape[1]
    qcm = _shared_axis(qvalue, queries, "query_cm", np, finite=False)
    qchrom = _shared_axis(query_chrom, queries, "query_chrom", np)
    result = np.zeros((people, queries, 3), dtype=np.int64)
    for chromosome in sorted(set(int(x) for x in chrom.tolist())):
        first, stop = np.searchsorted(chrom, [chromosome, chromosome + 1])
        query_ids = np.flatnonzero((qchrom == chromosome) & np.isfinite(qcm))
        if not len(query_ids): continue
        positions = cm[first:stop]
        left = first + np.searchsorted(positions, qcm[query_ids] - radius_cm, side="left")
        right = first + np.searchsorted(positions, qcm[query_ids] + radius_cm, side="right")
        result[:, query_ids] = counts[:, right] - counts[:, left]
    return result


def calculate(event_dosage, event_valid, event_cm, event_chrom,
              query_cm, query_chrom, radius_cm, condition, np):
    return project(build_prefix(event_dosage, event_valid, event_cm, event_chrom,
                                condition, np), query_cm, query_chrom, radius_cm, np)


def prepare_prefix(event_path, control_receipt_path, event_axis_path,
                   event_axis_receipt_path, condition, output_dir):
    import numpy as np
    event_path, control_receipt_path = Path(event_path), Path(control_receipt_path)
    event_axis_path, event_axis_receipt_path = Path(event_axis_path), Path(event_axis_receipt_path)
    control_receipt = json.loads(control_receipt_path.read_text())
    axis_receipt = json.loads(event_axis_receipt_path.read_text())
    four = (control_receipt.get("schema") == "r02_lai_four_control_receipt_v1"
            and control_receipt.get("status") == "FOUR_DEFINED_CONDITIONS_MATERIALIZED_DENSITY_BLOCKED"
            and condition in {"NONE", "NO_RARE", "COUNT_MATCHED", "IDENTITY_SHAM"})
    density = (control_receipt.get("schema") == "r02_lai_five_control_receipt_v2"
               and control_receipt.get("status") ==
                   "FIVE_CONDITIONS_MATERIALIZED_FOR_ONE_RADIUS_PREPARE_ONLY"
               and condition == "DENSITY_MATCHED")
    require((four or density) and control_receipt.get("batch_sha256") == sha256(event_path)
            and condition in control_receipt.get("conditions_materialized", []),
            "control condition receipt differs")
    require(axis_receipt.get("schema") == "r02_lai_factorized_predictors_v1"
            and axis_receipt.get("status") == "PREDICTOR_INDEX_COMPLETE_NOT_TRAINING"
            and axis_receipt.get("outputs_sha256", {}).get("indices.npz") == sha256(event_axis_path),
            "factorized EVENT-axis receipt differs")
    reference = control_receipt.get("reference_identity")
    require(control_receipt.get("factorized_receipt_sha256") == sha256(event_axis_receipt_path)
            and control_receipt.get("event_annotation_schema") ==
                axis_receipt.get("event_annotation_schema") == "r02_lai_event_annotations_v1"
            and isinstance(reference, dict) and set(reference) == {
                "event_catalogue_sha256", "ref_train_receipt_sha256",
                "ref_support_by_provenance_sha256", "reference_vcf_sha256",
                "reference_roles_sha256"}
            and all(isinstance(value, str) and len(value) == 64 for value in reference.values()),
            "fixed source-minor/reference identity differs")
    for name in ("task_id", "case_id", "direction", "role", "chromosome", "reference_identity"):
        require(control_receipt.get(name) == axis_receipt.get(name), name + " differs between inputs")
    with np.load(event_path, allow_pickle=False) as source:
        control_arrays = {name: source[name] for name in source.files}
    people = control_arrays["person_id"]
    dose = control_arrays[condition + "_event_dosage"]
    valid = control_arrays[condition + "_event_valid"]
    global_index = control_arrays["event_global_index"]
    control_chrom = control_arrays["event_chromosome"]
    with np.load(event_axis_path, allow_pickle=False) as source:
        factorized_arrays = {name: source[name] for name in source.files}
        axis_people = factorized_arrays["person_id"]
        # Factorized indices preserve tabular coordinates as text. Apply the
        # same explicit conversion as r02_lai_regional_inputs.py; invalid text
        # must raise instead of becoming missing or selecting another locus.
        event_cm = factorized_arrays["event_position_cm"].astype(np.float64)
        event_chrom = factorized_arrays["event_chromosome"].astype(np.int64)
    from r02_lai_density_control import authenticate_event_axis
    authenticate_event_axis(control_arrays, factorized_arrays, np)
    require(np.array_equal(global_index, np.arange(len(event_cm)))
            and np.array_equal(people.astype(str), axis_people.astype(str))
            and np.array_equal(control_chrom.astype(str), event_chrom.astype(str))
            and len(people) == control_receipt["people"] == axis_receipt["people"]
            and len(event_cm) == control_receipt["event_loci"] == axis_receipt["loci"]["event"],
            "control and factorized EVENT axes differ")
    arrays = {"person_id": people,
              **build_prefix(dose, valid, event_cm, event_chrom, condition, np)}
    out = Path(output_dir); require(not out.exists() and out.parent.is_dir(), "fresh output directory required")
    out.mkdir(mode=0o700); batch = out / "event-burden-prefix.npz"
    np.savez_compressed(batch, **arrays)
    receipt = {"schema": RECEIPT_SCHEMA, "status": "EVENT_BURDEN_PREFIX_PREPARED_NO_TRAINING",
        "prefix_schema": PREFIX_SCHEMA, "count_names": list(COUNT_NAMES), "condition": condition,
        "batch_sha256": sha256(batch), "arrays_sha256": {k: array_sha256(v, np) for k, v in arrays.items()},
        "event_input_sha256": sha256(event_path), "event_axis_sha256": sha256(event_axis_path),
        "control_receipt_sha256": sha256(control_receipt_path),
        "event_axis_receipt_sha256": sha256(event_axis_receipt_path),
        "task_id": control_receipt["task_id"], "case_id": control_receipt["case_id"],
        "direction": control_receipt["direction"], "role": control_receipt["role"],
        "chromosome": str(control_receipt["chromosome"]),
        "reference_identity": control_receipt["reference_identity"],
        "rare_allele": "DNABR2723_FIXED_SOURCE_MINOR",
        "rare_phase": "DIPLOID_UNPHASED_NO_HAPLOTYPE_ASSIGNMENT",
        "people": len(people), "events": dose.shape[1], "radius_selected": False,
        "truth_loaded": False, "training_authorized": False, "selection_authorized": False,
        "evaluate_opened": False}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_prefix(batch_path, receipt_path, condition):
    import numpy as np
    receipt = json.loads(Path(receipt_path).read_text())
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("prefix_schema") == PREFIX_SCHEMA
            and receipt.get("count_names") == list(COUNT_NAMES) and receipt.get("condition") == condition
            and receipt.get("radius_selected") is False
            and receipt.get("rare_allele") == "DNABR2723_FIXED_SOURCE_MINOR"
            and receipt.get("rare_phase") == "DIPLOID_UNPHASED_NO_HAPLOTYPE_ASSIGNMENT"
            and receipt.get("batch_sha256") == sha256(batch_path), "EVENT burden prefix receipt differs")
    require(all(receipt.get(k) is False for k in
                ("truth_loaded", "training_authorized", "selection_authorized", "evaluate_opened")),
            "EVENT burden prefix receipt opens a forbidden operation")
    with np.load(batch_path, allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    require(set(arrays) == {"person_id", "event_order", "event_cm", "event_chrom", "prefix_counts"}
            and arrays["prefix_counts"].shape ==
                (len(arrays["person_id"]), len(arrays["event_cm"]) + 1, 3),
            "EVENT burden prefix arrays differ")
    order = arrays["event_order"]
    same_chromosome = arrays["event_chrom"][1:] == arrays["event_chrom"][:-1]
    require(np.array_equal(np.sort(order), np.arange(len(order)))
            and np.all(arrays["event_chrom"][1:] >= arrays["event_chrom"][:-1])
            and np.all(arrays["event_cm"][1:][same_chromosome] >=
                       arrays["event_cm"][:-1][same_chromosome])
            and not arrays["prefix_counts"][:, 0].any()
            and (np.diff(arrays["prefix_counts"], axis=1) >= 0).all(),
            "EVENT burden prefix ordering/monotonicity differs")
    require(all(array_sha256(arrays[k], np) == receipt["arrays_sha256"].get(k) for k in arrays),
            "EVENT burden prefix array hash differs")
    return arrays, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-prefix")
    build.add_argument("--event-conditions", required=True); build.add_argument("--control-receipt", required=True)
    build.add_argument("--event-axis", required=True); build.add_argument("--event-axis-receipt", required=True)
    build.add_argument("--condition", choices=CONDITIONS, required=True); build.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_prefix(args.event_conditions, args.control_receipt,
                                    args.event_axis, args.event_axis_receipt,
                                    args.condition, args.output_dir), sort_keys=True))


if __name__ == "__main__": main()
