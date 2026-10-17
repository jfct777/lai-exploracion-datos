#!/usr/bin/env python3
"""Authenticate K03 FIT/SELECT pair responses and expose neural target axes.

This is a representation adapter.  It does not choose a target, block,
exclusion territory, missingness threshold, weighting rule, model, or training
parameter.  EVALUATE arrays are forbidden.  K03 pair weights and observed
marker denominators are retained even though the current neural reader uses
only ``pairs`` and ``target``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np


HEX = re.compile(r"[0-9a-f]{64}")
ROLES = ("FIT", "SELECT", "EVALUATE")
SOURCE_FIELDS = {"points"} | {
    role + "_" + name
    for role in ("FIT", "SELECT")
    for name in ("i", "j", "y", "n", "w", "extra")
}
OUTPUT_FIELDS = {
    "person_id", "component_id", "role", "pairs", "target",
    "pair_weight", "joint_marker_count",
}
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"nonfinite JSON value {value}")))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")


def pin(path, digest, limit, label):
    path = Path(path).resolve()
    require(HEX.fullmatch(digest or "") and path.is_file(), f"{label} descriptor invalid")
    require(path.stat().st_size <= limit and sha(path) == digest,
            f"{label} SHA256 or byte bound differs")
    return path


def table(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == ["sample_id", "component", "role"],
                "persons.tsv exact fields/order differ")
        rows = list(reader)
    require(rows and all(set(row) == set(reader.fieldnames) for row in rows),
            "persons.tsv is empty or malformed")
    ids = [row["sample_id"] for row in rows]
    require(all(ids) and len(ids) == len(set(ids)), "person IDs must be nonempty and unique")
    require(all(row["component"] and row["component"].strip() == row["component"]
                and row["role"] in ROLES for row in rows),
            "person component/role differs")
    component_roles = {}
    for row in rows:
        component_roles.setdefault(row["component"], set()).add(row["role"])
    require(all(len(value) == 1 for value in component_roles.values()),
            "component crosses frozen roles")
    return rows


def deterministic_npz(path, arrays):
    """Write byte-reproducible, uncompressed arrays with no pickle payload."""
    path = Path(path)
    require(not path.exists() and set(arrays) == OUTPUT_FIELDS, "new exact neural target output required")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED,
                             strict_timestamps=True) as archive:
            for name in sorted(arrays):
                value = np.ascontiguousarray(arrays[name])
                require(value.dtype.kind != "O", "object arrays are forbidden")
                stream = io.BytesIO()
                np.lib.format.write_array(stream, value, allow_pickle=False)
                member = zipfile.ZipInfo(name + ".npy", FIXED_ZIP_TIME)
                member.compress_type = zipfile.ZIP_STORED
                member.external_attr = 0o100400 << 16
                archive.writestr(member, stream.getvalue())
        os.replace(temporary, path)
        os.chmod(path, 0o400)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_identity(records, names=("i", "j", "y", "n", "w"), roles=("FIT", "SELECT")):
    digest = hashlib.sha256()
    for role in roles:
        for name in names:
            value = np.asarray(records[role][name])
            digest.update(f"{role}:{name}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def expected_component_weights(i, j, components):
    labels = [tuple(sorted((components[int(a)], components[int(b)]))) for a, b in zip(i, j)]
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    weights = np.asarray([1 / counts[label] for label in labels], dtype=np.float64)
    return weights / weights.mean()


def role_payload(persons, role, record, audit):
    global_rows = np.asarray([index for index, row in enumerate(persons) if row["role"] == role],
                             dtype=np.int64)
    require(len(global_rows) > 1, f"{role} person axis is empty")
    i, j, target, denominator, weight = (
        np.asarray(record[name]) for name in ("i", "j", "y", "n", "w")
    )
    require(i.dtype.kind in "iu" and j.dtype.kind in "iu" and denominator.dtype.kind in "iu"
            and i.ndim == j.ndim == target.ndim == denominator.ndim == weight.ndim == 1
            and i.shape == j.shape == target.shape == denominator.shape == weight.shape
            and len(i) > 0, f"{role} target dimensions differ")
    require(np.all(i < j) and np.all(np.diff(i.astype(np.int64) * len(persons) + j) > 0),
            f"{role} global pairs are not canonical ordered unique")
    left, right = np.searchsorted(global_rows, i), np.searchsorted(global_rows, j)
    require(np.all(left < len(global_rows)) and np.all(right < len(global_rows))
            and np.array_equal(global_rows[left], i) and np.array_equal(global_rows[right], j),
            f"{role} pair crosses its frozen role")
    local_pairs = np.column_stack((left, right)).astype(np.int64)
    components = np.asarray([persons[index]["component"] for index in global_rows], dtype=np.str_)
    require(np.all(local_pairs[:, 0] < local_pairs[:, 1])
            and np.all(components[local_pairs[:, 0]] != components[local_pairs[:, 1]]),
            f"{role} target contains a within-component pair")
    all_components = [row["component"] for row in persons]
    expected_weight = expected_component_weights(i, j, all_components)
    require(np.isfinite(target).all() and np.isfinite(weight).all() and np.all(denominator > 0)
            and np.all(weight > 0) and np.allclose(weight, expected_weight, rtol=1e-13, atol=0),
            f"{role} target/denominator/original pair weight differs")

    counts = {}
    for component in components:
        counts[str(component)] = counts.get(str(component), 0) + 1
    total = len(global_rows) * (len(global_rows) - 1) // 2
    within = sum(value * (value - 1) // 2 for value in counts.values())
    insufficient = total - within - len(local_pairs)
    excluded = dict(audit.get("excluded_pairs", {}))
    require(audit.get("pairs") == len(local_pairs)
            and excluded.get("within_component", 0) == within
            and excluded.get("insufficient_joint_markers", 0) == insufficient
            and audit.get("min_joint_markers") == int(denominator.min())
            and audit.get("max_joint_markers") == int(denominator.max())
            and math.isclose(audit.get("weight_sum", float("nan")), float(weight.sum()),
                             rel_tol=1e-13, abs_tol=0),
            f"{role} support/exclusion audit differs from arrays")

    ids = np.asarray([persons[index]["sample_id"] for index in global_rows], dtype=np.str_)
    arrays = dict(person_id=ids, component_id=components,
                  role=np.asarray([role] * len(ids), dtype=np.str_), pairs=local_pairs,
                  target=target.astype(np.float64), pair_weight=weight.astype(np.float64),
                  joint_marker_count=denominator.astype(np.int64))
    support = dict(
        role_people=len(ids), emitted_between_component_pairs=len(local_pairs),
        total_possible_pairs=total, excluded_within_component_pairs=within,
        excluded_insufficient_joint_marker_pairs=insufficient,
        minimum_joint_marker_count=int(denominator.min()),
        maximum_joint_marker_count=int(denominator.max()),
        pair_weight_sum=float(weight.sum()), target_missing_values=0,
        target_missing_policy="OMITTED_IF_BELOW_AUTHENTICATED_MIN_PAIR_MARKERS",
    )
    return arrays, support


def adapt(request_path, request_sha256, output, *, source_directory=None,
          target_contract_path=None):
    os.umask(0o077)
    request_path = pin(request_path, request_sha256, 2**20, "adapter request")
    request = load_json(request_path)
    require(set(request) == {"schema", "status", "source", "target_contract", "expected", "resources"}
            and request["schema"] == "r02_k03_neural_target_adapter_request_v1"
            and request["status"] == "PREPARE_TARGETS_NOT_TRAINING_ADMISSION",
            "exact target adapter request/status required")
    limits = request["resources"]
    require(set(limits) == {"max_metadata_bytes", "max_input_bytes", "max_array_bytes",
                            "max_output_bytes", "min_free_bytes"}
            and all(type(value) is int and value > 0 for value in limits.values()),
            "explicit positive adapter byte/disk limits required")
    source = request["source"]
    require(set(source) == {"directory", "manifest_sha256"}
            and Path(source["directory"]).is_absolute(), "exact absolute source descriptor required")
    root = Path(source_directory if source_directory is not None else source["directory"]).resolve()
    manifest_path = pin(root / "manifest.json", source["manifest_sha256"],
                        limits["max_metadata_bytes"], "K03 manifest")
    manifest = load_json(manifest_path)
    expected = request["expected"]
    require(set(expected) == {"evidence_scope", "scientific_spec_sha256", "block_id",
                              "exclusion_scope", "person_counts"}
            and expected["person_counts"] == {"FIT": 1460, "SELECT": 476, "EVALUATE": 683},
            "explicit frozen human roster counts required")
    require(manifest.get("schema") == "r02_k03_fit_select_arrays_v1"
            and manifest.get("stage") == "FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS"
            and manifest.get("evaluate_targets_computed") is False
            and manifest.get("alpha_selection_performed") is False
            and all(manifest.get(key) == expected[key] for key in
                    ("evidence_scope", "scientific_spec_sha256", "block_id", "exclusion_scope"))
            and set(manifest.get("outputs_sha256", {})) == {"persons.tsv", "arrays.private.npz"},
            "K03 source scope, geometry, or FIT/SELECT-only status differs")
    persons_path = pin(root / "persons.tsv", manifest["outputs_sha256"]["persons.tsv"],
                       limits["max_metadata_bytes"], "persons.tsv")
    arrays_path = pin(root / "arrays.private.npz", manifest["outputs_sha256"]["arrays.private.npz"],
                      limits["max_input_bytes"], "K03 arrays")
    require(persons_path.stat().st_size + arrays_path.stat().st_size <= limits["max_input_bytes"],
            "aggregate adapter input byte bound exceeded")
    with zipfile.ZipFile(arrays_path) as archive:
        require(len(archive.infolist()) == len(SOURCE_FIELDS)
                and {item.filename for item in archive.infolist()} ==
                {name + ".npy" for name in SOURCE_FIELDS}
                and sum(item.file_size for item in archive.infolist()) <= limits["max_array_bytes"],
                "exact bounded K03 FIT/SELECT array inventory required; EVALUATE forbidden")
    persons = table(persons_path)
    observed_counts = {role: sum(row["role"] == role for row in persons) for role in ROLES}
    require(observed_counts == expected["person_counts"] and len(persons) == 2619,
            "frozen DNABR human roster counts differ")
    with np.load(arrays_path, allow_pickle=False) as loaded:
        # ``points`` and paired-control ``extra`` are authenticated by the source
        # archive/hash but are not materialized: this adapter copies targets,
        # weights and denominators only and must stay O(number of target pairs).
        records = {role: {name: loaded[role + "_" + name]
                          for name in ("i", "j", "y", "n", "w")}
                   for role in ("FIT", "SELECT")}
    require(source_identity(records) == manifest.get("target_identity")
            and source_identity(records, roles=("SELECT",)) == manifest.get("select_target_identity"),
            "K03 authenticated target identity differs")
    audit = manifest.get("target_audit", {})
    require(audit.get("evaluate_targets_computed") is False
            and set(audit.get("roles", {})) == {"FIT", "SELECT"}
            and type(audit.get("min_pair_markers")) is int and audit["min_pair_markers"] > 0,
            "K03 target support audit is incomplete")

    contract = request["target_contract"]
    require(set(contract) == {"path", "sha256"} and Path(contract["path"]).is_absolute(),
            "exact absolute target-definition contract descriptor required")
    contract_path = pin(target_contract_path if target_contract_path is not None else contract["path"],
                        contract["sha256"], limits["max_metadata_bytes"],
                        "target-definition contract")
    out = Path(output).resolve()
    require(not out.exists() and out.parent.is_dir()
            and shutil.disk_usage(out.parent).free >= limits["min_free_bytes"] + limits["max_output_bytes"],
            "new output and explicit disk reserve required")
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage-", dir=out.parent))
    try:
        receipts = {}
        for role in ("FIT", "SELECT"):
            payload, support = role_payload(persons, role, records[role], audit["roles"][role])
            target_path = stage / f"{role}.target.private.npz"
            deterministic_npz(target_path, payload)
            receipt = dict(
                schema="r02_k03_neural_pair_target_v1",
                status="TARGET_REPRESENTATION_COMPLETE_NOT_TRAINING_ADMISSION",
                role=role, arrays_sha256=sha(target_path),
                target_contract_sha256=contract["sha256"],
                source_manifest_sha256=source["manifest_sha256"],
                source_arrays_sha256=manifest["outputs_sha256"]["arrays.private.npz"],
                source_target_identity=manifest["target_identity"],
                evidence_scope=expected["evidence_scope"],
                scientific_spec_sha256=expected["scientific_spec_sha256"],
                block_id=expected["block_id"], exclusion_scope=expected["exclusion_scope"],
                n_people=support["role_people"], n_pairs=support["emitted_between_component_pairs"],
                support=support, min_pair_markers=audit["min_pair_markers"],
                fields=sorted(OUTPUT_FIELDS), evaluate_targets_computed=False,
                training_allowed=False, biological_validation=False,
                neural_consumer_semantics={
                    "used_by_current_loss": ["pairs", "target", "pair_weight"],
                    "authenticated_but_not_used_by_current_loss":
                        ["joint_marker_count"],
                },
            )
            receipt_path = stage / f"{role}.target.receipt.json"
            write_json(receipt_path, receipt)
            receipts[role] = dict(
                path=target_path.name, sha256=receipt["arrays_sha256"],
                receipt_path=receipt_path.name, receipt_sha256=sha(receipt_path),
            )
        result = dict(
            schema="r02_k03_neural_target_adapter_result_v1",
            status="FIT_SELECT_TARGETS_COMPLETE_NOT_TRAINING_ADMISSION",
            request_sha256=request_sha256, source_manifest_sha256=source["manifest_sha256"],
            target_contract_sha256=contract["sha256"], targets=receipts,
            evaluate_targets_computed=False, training_allowed=False, biological_validation=False,
        )
        write_json(stage / "manifest.json", result)
        require(sum(path.stat().st_size for path in stage.iterdir()) <= limits["max_output_bytes"],
                "adapter output byte bound exceeded")
        require(sha(request_path) == request_sha256 and sha(manifest_path) == source["manifest_sha256"]
                and sha(persons_path) == manifest["outputs_sha256"]["persons.tsv"]
                and sha(arrays_path) == manifest["outputs_sha256"]["arrays.private.npz"]
                and sha(contract_path) == contract["sha256"], "adapter source changed during conversion")
        os.replace(stage, out)
        return load_json(out / "manifest.json")
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-directory")
    parser.add_argument("--target-contract")
    args = parser.parse_args()
    print(json.dumps(adapt(args.request, args.request_sha256, args.output,
                           source_directory=args.source_directory,
                           target_contract_path=args.target_contract), sort_keys=True))


if __name__ == "__main__":
    main()
