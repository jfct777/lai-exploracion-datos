#!/usr/bin/env python3
"""Build PREPARE-only LAI COMMON/EVENT arrays from exact reviewed axes.

This producer reads target genotypes and the revised EVENT catalogue.  It
authenticates the catalogue against the DM reference-linkage receipt, then
requires every emitted locus to exist with the same chromosome, position,
REF and ALT in the target-genotype axis.  It never reads truth or creates a
training/control authorization.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


DECISION_SCHEMA = "r02_lai_explicit_channels_prepare_v2"
RECEIPT_SCHEMA = "r02_lai_predictor_channels_receipt_v2"
REFERENCE_RECEIPT_SCHEMA = "r02_lai_reference_linkage_receipt_v1"
BASES = "ACGT"
COMMON_FEATURES = ("alt_dose_fraction", "fit_alt_af", "ref_A", "ref_C", "ref_G", "ref_T",
                   "alt_A", "alt_C", "alt_G", "alt_T", "bp_fraction", "map_fraction")
EVENT_METADATA = ("source_minor_af", "counted_is_alt", "ref_A", "ref_C", "ref_G", "ref_T",
                  "alt_A", "alt_C", "alt_G", "alt_T", "bp_fraction", "map_fraction")
REFERENCE_HASHES = ("event_catalogue_sha256", "ref_train_receipt_sha256",
                    "ref_support_by_provenance_sha256", "reference_vcf_sha256",
                    "reference_roles_sha256")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path, expected_sha=None):
    path = Path(path)
    if expected_sha is not None:
        require(sha256(path) == expected_sha, "JSON SHA-256 differs: " + str(path))
    return json.loads(path.read_text())


def resolve_descriptor(value, name, staged=None):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"},
            "invalid descriptor: " + name)
    path = Path(value["path"])
    if staged is not None:
        matches = [candidate for candidate in Path(staged).iterdir()
                   if candidate.is_file() and sha256(candidate) == value["sha256"]]
        require(len(matches) == 1, "staged input absent or ambiguous: " + name)
        path = matches[0]
    require(path.is_file() and sha256(path) == value["sha256"], name + " differs")
    return path


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else Path(path).open()


def read_catalogue(path):
    with open_text(path) as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    require(rows, "revised EVENT catalogue is empty")
    return fields, rows


def exact_reference_case(receipt, task_id, role, event_catalogue_sha):
    require(receipt.get("schema") == REFERENCE_RECEIPT_SCHEMA
            and receipt.get("status") == "COUNTS_AND_ORIGINAL_UNITS_MATCH"
            and receipt.get("training_authorized") is False
            and receipt.get("controls_materialized") is False
            and receipt.get("truth_loaded") is False
            and receipt.get("evaluate_opened") is False,
            "reference linkage receipt is not PREPARE-only")
    matches = [row for row in receipt.get("cases", []) if row.get("task_id") == task_id]
    require(len(matches) == 1, "reference linkage case absent or duplicated")
    row = matches[0]
    require(row.get("role") == role and row.get("event_catalogue_sha256") == event_catalogue_sha,
            "reference linkage case role/catalogue differs")
    require(row.get("reference_join_key") == ["chromosome", "position_bp", "REF", "ALT"]
            and row.get("reference_duplicate_keys") == 0
            and row.get("reference_target_person_overlap") == 0
            and row.get("problems") == [], "reference linkage checks differ")
    require(all(isinstance(row.get(name), str) and len(row[name]) == 64 for name in REFERENCE_HASHES),
            "reference linkage identity is incomplete")
    return row


def validate_decision(decision):
    required = {"schema", "status", "task_id", "case_id", "direction", "role", "chromosome",
                "chromosome_length_bp", "map_start_cm", "map_end_cm", "regional_batch_sha256",
                "component_by_person", "common", "event_catalogue", "reference_linkage_receipt",
                "training_allowed", "controls_materialized", "evaluate_opened"}
    require(set(decision) == required and decision["schema"] == DECISION_SCHEMA
            and decision["status"] == "PREPARE_ADMITTED"
            and decision["training_allowed"] is False
            and decision["controls_materialized"] is False
            and decision["evaluate_opened"] is False, "explicit PREPARE decision differs")
    require(decision["role"] in {"FIT", "SELECT"} and decision["direction"] in
            {"FOLD_A_TO_FOLD_B", "FOLD_B_TO_FOLD_A"}, "role/direction differs")
    require(type(decision["chromosome"]) is int and 1 <= decision["chromosome"] <= 22,
            "invalid chromosome")
    require(type(decision["chromosome_length_bp"]) is int and decision["chromosome_length_bp"] > 0
            and decision["map_end_cm"] > decision["map_start_cm"], "invalid coordinate scale")
    require(isinstance(decision["component_by_person"], dict) and decision["component_by_person"],
            "person/component map absent")
    require(isinstance(decision["common"], list) and decision["common"], "COMMON catalogue absent")


def _catalogue_rows(rows, label, chromosome):
    keys = []
    for row in rows:
        require(str(row.get("chromosome", chromosome)) == str(chromosome), label + " chromosome differs")
        key = (chromosome, int(row["position_bp"]), row["ref"], row["alt"])
        require(row["ref"] in BASES and row["alt"] in BASES and row["ref"] != row["alt"],
                label + " contains a non-biallelic SNV")
        require(row.get("core") in (True, False, "True", "False", "true", "false"),
                label + " core flag differs")
        keys.append(key)
    require(len(keys) == len(set(keys)), "duplicate " + label + " locus")
    return keys


def _as_bool(value):
    return value if type(value) is bool else value.lower() == "true"


def assemble(regional_path, decision_path, decision_sha, output, staged=None):
    import numpy as np
    from r02_lai_event_annotations import build_index_arrays, counted_is_alt

    decision = load_json(decision_path, decision_sha)
    validate_decision(decision)
    regional_path = Path(regional_path)
    require(sha256(regional_path) == decision["regional_batch_sha256"], "regional genotype batch differs")
    event_path = resolve_descriptor(decision["event_catalogue"], "revised EVENT catalogue", staged)
    reference_path = resolve_descriptor(decision["reference_linkage_receipt"], "DM reference receipt", staged)
    fields, event_rows = read_catalogue(event_path)
    link = exact_reference_case(load_json(reference_path), decision["task_id"], decision["role"],
                                decision["event_catalogue"]["sha256"])
    require(link["event_loci"] == len(event_rows), "reference receipt and EVENT row count differ")
    annotation = build_index_arrays(fields, event_rows, np)

    allowed = {"person_id", "target_position_bp", "target_ref", "target_alt", "target_dosage",
               "target_observed", "case_id", "direction", "role", "query_features", "query_cm",
               "query_chrom", "query_pos_bp", "query_valid", "baseline_probabilities"}
    with np.load(regional_path, allow_pickle=False) as loaded:
        require(allowed <= set(loaded.files), "regional genotype/query arrays incomplete")
        source = {name: loaded[name] for name in allowed}
    person_axis = source["person_id"]
    require(person_axis.ndim == 1 and len(person_axis) > 0,
            "regional person_id must be a nonempty 1D axis")
    person = person_axis.astype(str)
    require(len(person) == len(set(person)), "regional person_id contains duplicates")
    require((str(source["case_id"].item()), str(source["direction"].item()), str(source["role"].item())) ==
            (decision["case_id"], decision["direction"], decision["role"]),
            "regional case/direction/role differs")
    require(len(person) == link["target_people"] and set(person) == set(decision["component_by_person"]),
            "regional person axis differs from decision/reference count")
    position = source["target_position_bp"]
    ref_axis, alt_axis = source["target_ref"], source["target_alt"]
    require(position.ndim == ref_axis.ndim == alt_axis.ndim == 1
            and len(position) == len(ref_axis) == len(alt_axis) > 0,
            "target position/REF/ALT must be nonempty 1D axes of equal length")
    require(np.issubdtype(position.dtype, np.integer)
            and np.all((position >= 1) & (position <= decision["chromosome_length_bp"]))
            and np.all(position[1:] > position[:-1]),
            "target position_bp must be valid strictly increasing integers")
    ref, alt = ref_axis.astype(str), alt_axis.astype(str)
    dose, observed = source["target_dosage"], source["target_observed"]
    require(observed.dtype == np.dtype(bool), "target_observed must have boolean dtype")
    require(dose.shape == observed.shape == (len(person), len(position)), "target genotype shape differs")
    target_keys = [(decision["chromosome"], int(p), r, a) for p, r, a in zip(position, ref, alt)]
    require(len(target_keys) == len(set(target_keys)), "duplicate target genotype locus")
    lookup = {key: index for index, key in enumerate(target_keys)}

    common_rows = decision["common"]
    common_keys = _catalogue_rows(common_rows, "COMMON", decision["chromosome"])
    event_keys = _catalogue_rows(event_rows, "EVENT", decision["chromosome"])
    require(not set(common_keys).intersection(event_keys), "COMMON and EVENT catalogues overlap")
    missing = [key for key in common_keys + event_keys if key not in lookup]
    if missing:
        raise ValueError("reviewed catalogue locus absent from target genotype axis: " + repr(missing[0]))

    n = len(person)
    arrays = {"person_id": person,
              "component_id": np.asarray([decision["component_by_person"][p] for p in person]),
              "role": np.full(n, decision["role"])}
    for name in ("query_features", "query_cm", "query_chrom", "query_pos_bp", "query_valid",
                 "baseline_probabilities"):
        arrays[name] = source[name]
    require(arrays["query_valid"].dtype == np.dtype(bool)
            and arrays["query_valid"].shape[0] == n
            and arrays["baseline_probabilities"].shape == (*arrays["query_valid"].shape, 6),
            "query/FLARE axes differ")

    for rows, keys, prefix in ((common_rows, common_keys, "common"), (event_rows, event_keys, "event")):
        indices = np.asarray([lookup[key] for key in keys], dtype=np.int64)
        valid = observed[:, indices]
        alt_dose = dose[:, indices].astype(np.float32)
        require(np.isin(alt_dose[valid], (0, 1, 2)).all(), "invalid " + prefix + " dosage")
        bp = np.asarray([key[1] for key in keys], dtype=np.int64)
        cm = np.asarray([float(row["position_cm"]) for row in rows], dtype=np.float64)
        fixed = np.column_stack(([[int(row["ref"] == base) for base in BASES] for row in rows],
                                 [[int(row["alt"] == base) for base in BASES] for row in rows]))
        geometry = np.column_stack((bp / decision["chromosome_length_bp"],
            (cm - decision["map_start_cm"]) / (decision["map_end_cm"] - decision["map_start_cm"])))
        arrays[prefix + "_cm"] = np.broadcast_to(cm, (n, len(rows))).copy()
        arrays[prefix + "_chrom"] = np.full((n, len(rows)), decision["chromosome"], dtype=np.int64)
        arrays[prefix + "_pos_bp"] = np.broadcast_to(bp, (n, len(rows))).copy()
        arrays[prefix + "_valid"] = valid
        arrays[prefix + "_core"] = np.broadcast_to(
            np.asarray([_as_bool(row["core"]) for row in rows], dtype=bool), valid.shape).copy()
        arrays[prefix + "_padding"] = np.zeros(valid.shape, dtype=bool)
        if prefix == "common":
            af = np.asarray([int(row["fit_alt_ac"]) / int(row["fit_an"]) for row in rows])
            metadata = np.column_stack((af, fixed, geometry)).astype(np.float32)
            arrays["common_features"] = np.concatenate((
                np.where(valid, alt_dose, 0)[..., None] / 2,
                np.broadcast_to(metadata, (n, *metadata.shape))), axis=2)
        else:
            is_alt = np.asarray([counted_is_alt(row["ref"], row["alt"], row["counted_allele"])
                                 for row in rows], dtype=bool)
            counted = np.where(is_alt[None, :], alt_dose, 2 - alt_dose)
            arrays["event_dosage"] = np.where(valid, counted, 0).astype(np.float32)
            maf = np.asarray([int(row["source_minor_ac"]) / int(row["source_an"]) for row in rows])
            metadata = np.column_stack((maf, is_alt.astype(float), fixed, geometry)).astype(np.float32)
            arrays["event_metadata"] = np.broadcast_to(metadata, (n, *metadata.shape)).copy()
            arrays.update(annotation)
            arrays["event_ref"] = np.asarray([row["ref"] for row in rows])
            arrays["event_alt"] = np.asarray([row["alt"] for row in rows])
            arrays["event_is_alt"] = is_alt
            arrays["event_global_index"] = indices

    require(arrays["common_features"].shape[2] == len(COMMON_FEATURES)
            and arrays["event_metadata"].shape[2] == len(EVENT_METADATA), "feature width differs")
    output = Path(output)
    require(not output.exists() and not output.with_suffix(output.suffix + ".json").exists(),
            "output exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    reference_identity = {name: link[name] for name in REFERENCE_HASHES}
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "PREPARED_EXACT_REVIEWED_AXES_NO_TRAINING",
        "training_allowed": False, "controls_materialized": False, "truth_loaded": False,
        "selection_performed": False, "evaluate_opened": False,
        "task_id": decision["task_id"], "case_id": decision["case_id"],
        "direction": decision["direction"], "role": decision["role"],
        "arrays_sha256": sha256(output), "regional_batch_sha256": sha256(regional_path),
        "decision_sha256": decision_sha, "reference_linkage_receipt_sha256": sha256(reference_path),
        "reference_identity": reference_identity,
        "reference_join_key": ["chromosome", "position_bp", "REF", "ALT"],
        "reference_duplicate_keys": 0, "reference_target_person_overlap": 0,
        "n_people": n, "n_common": len(common_rows), "n_event": len(event_rows),
        "common_features": list(COMMON_FEATURES), "event_metadata_features": list(EVENT_METADATA),
        "limit": "DM authenticates reference counts/units only; this producer separately checks target genotype axes."
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regional-batch", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--staged-inputs")
    args = parser.parse_args()
    print(json.dumps(assemble(args.regional_batch, args.decision, args.decision_sha256, args.output,
                              args.staged_inputs),
                     sort_keys=True))


if __name__ == "__main__":
    main()
