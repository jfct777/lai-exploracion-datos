#!/usr/bin/env python3
"""Build a truth-blind LAI query overlay and, optionally, a consumer batch.

The query grid and all model-facing arrays are finalized before the private
truth intervals are read.  Truth is projected only into ``truth_state`` and
``truth_observed``; the latter is consumed by the LAI loss, never as a feature
or query-validity mask.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pinned(spec, key, staged_dir=None):
    value = ((Path(staged_dir) / Path(spec[key]).name) if staged_dir is not None else Path(spec[key]))
    require(sha(value) == spec[key + "_sha256"], f"{key} changed")
    return value


def read_tsv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_grid(path):
    rows = read_tsv(path)
    require(rows, "empty query grid")
    required = {"chromosome", "position_bp", "query_cm", "score_weight_cm", "query_valid"}
    require(required <= set(rows[0]), "query grid columns differ")
    chrom = np.asarray([int(row["chromosome"].removeprefix("chr")) for row in rows], dtype=np.int16)
    pos = np.asarray([int(row["position_bp"]) for row in rows], dtype=np.int64)
    cm = np.asarray([float(row["query_cm"]) for row in rows], dtype=np.float64)
    weight = np.asarray([float(row["score_weight_cm"]) for row in rows], dtype=np.float64)
    valid = np.asarray([row["query_valid"].lower() == "true" for row in rows], dtype=bool)
    require(valid.all(), "producer grid contains technically invalid rows")
    require((np.diff(pos) > 0).all() and (np.diff(cm) >= 0).all(), "query grid not ordered")
    require(len(set(chrom.tolist())) == 1, "one chromosome required per overlay")
    return chrom, pos, cm, weight, valid


def load_people(persons_path, roles_path):
    people_rows, role_rows = read_tsv(persons_path), read_tsv(roles_path)
    people = [row["sample_key_sha256"] for row in people_rows]
    roles = {row["sample_key_sha256"]: (row["role"], row["component_id"]) for row in role_rows}
    require(len(people) == len(set(people)) and set(people) == set(roles),
            "persons and reciprocal roles must identify the same unique people")
    return (np.asarray(people), np.asarray([roles[p][0] for p in people]),
            np.asarray([roles[p][1] for p in people]))


def project_truth(path, people, chromosome, positions):
    """Project unordered diploid ancestry states without changing query arrays."""
    by_person = {person: {1: [], 2: []} for person in people.tolist()}
    ancestry = {"AFR": 0, "African": 0, "EUR": 1, "European": 1,
                "NAM": 2, "Native_American": 2}
    for row in read_tsv(path):
        if int(row["chromosome"].removeprefix("chr")) != chromosome:
            continue
        key = hashlib.sha256(row["sample_id"].encode()).hexdigest()
        require(key in by_person, "truth contains a person outside the authenticated material")
        hap = int(row["haplotype"])
        require(hap in (1, 2) and row["ancestry"] in ancestry, "unsupported truth state")
        by_person[key][hap].append((int(row["start_bp_inclusive"]),
                                    int(row["end_bp_inclusive"]), ancestry[row["ancestry"]]))
    state = np.full((len(people), len(positions)), -1, dtype=np.int8)
    observed = np.zeros(state.shape, dtype=bool)
    for person_index, person in enumerate(people.tolist()):
        hap_values = []
        for hap in (1, 2):
            intervals = sorted(by_person[person][hap])
            cursor, values = 0, np.full(len(positions), -1, dtype=np.int8)
            for query_index, position in enumerate(positions):
                while cursor < len(intervals) and intervals[cursor][1] < position:
                    cursor += 1
                if cursor < len(intervals) and intervals[cursor][0] <= position <= intervals[cursor][1]:
                    values[query_index] = intervals[cursor][2]
            hap_values.append(values)
        ok = (hap_values[0] >= 0) & (hap_values[1] >= 0)
        lo, hi = np.minimum(hap_values[0], hap_values[1]), np.maximum(hap_values[0], hap_values[1])
        # AA, AE, AN, EE, EN, NN
        state[person_index, ok] = np.asarray([[0, 1, 2], [-1, 3, 4], [-1, -1, 5]], dtype=np.int8)[lo[ok], hi[ok]]
        observed[person_index] = ok
    return state, observed


def predictor_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(k for k in arrays if k not in {"truth_state", "truth_observed"}):
        value = np.ascontiguousarray(arrays[key])
        h.update(key.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        h.update(value.tobytes())
    return h.hexdigest()


def apply_person_permutation(arrays, order):
    """Apply an admitted control permutation to every person-indexed array."""
    n = len(arrays["person_id"])
    order = np.asarray(order, dtype=np.int64)
    require(order.shape == (n,) and np.array_equal(np.sort(order), np.arange(n)),
            "person permutation must be a bijection")
    return {key: (value[order] if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == n else value)
            for key, value in arrays.items()}


def merge_base(base_path, overlay):
    with np.load(base_path, allow_pickle=False) as source:
        base = {key: source[key] for key in source.files}
    reserved = set(overlay) - {"person_id", "role", "component_id"}
    require(not (reserved & set(base)), "base batch already defines query or truth arrays")
    require("person_id" in base and np.array_equal(base["person_id"].astype(str), overlay["person_id"].astype(str)),
            "base batch person order differs")
    base.update({key: value for key, value in overlay.items() if key != "person_id"})
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--staged-inputs", type=Path,
                        help="resolve pinned input basenames in this Nextflow task directory")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    require(request.get("schema") == "r02_lai_truthblind_context_request_v2", "request schema differs")
    grid_path = pinned(request, "query_grid_path", args.staged_inputs)
    grid_receipt = pinned(request, "query_grid_receipt_path", args.staged_inputs)
    grid_meta = json.loads(grid_receipt.read_text())
    require(grid_meta.get("schema") == "r02_lai_truth_blind_query_grid_v1"
            and grid_meta.get("output_sha256") == sha(grid_path)
            and grid_meta.get("truth_read") is False, "query grid receipt does not prove truth blindness")
    people_path = pinned(request, "persons_path", args.staged_inputs)
    roles_path = pinned(request, "roles_path", args.staged_inputs)
    truth_path = pinned(request, "truth_path", args.staged_inputs)
    chrom, pos, cm, weight, valid = load_grid(grid_path)
    people, roles, components = load_people(people_path, roles_path)
    n = len(people)
    # Model-facing arrays are complete before truth is read below.
    overlay = {
        "person_id": people,
        "role": roles,
        "component_id": components,
        "query_features": np.broadcast_to(weight[None, :, None].astype(np.float32), (n, len(pos), 1)).copy(),
        "query_cm": np.broadcast_to(cm, (n, len(pos))).copy(),
        "query_chrom": np.broadcast_to(chrom, (n, len(pos))).copy(),
        "query_pos_bp": np.broadcast_to(pos, (n, len(pos))).copy(),
        "query_valid": np.broadcast_to(valid, (n, len(pos))).copy(),
        "query_score_weight_cm": np.broadcast_to(weight, (n, len(pos))).copy(),
    }
    before_truth = predictor_digest(overlay)
    state, observed = project_truth(truth_path, people, int(chrom[0]), pos)
    overlay.update(truth_state=state, truth_observed=observed)
    require(predictor_digest(overlay) == before_truth, "truth changed a model-facing array")
    require(observed.any(1).all() and not (observed & ~overlay["query_valid"]).any(),
            "truth support cannot define or extend technical query support")
    if "person_permutation" in request:
        overlay = apply_person_permutation(overlay, request["person_permutation"])
    output = (merge_base(pinned(request, "base_batch_path", args.staged_inputs), overlay)
              if request.get("base_batch_path") else overlay)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / ("batch.npz" if request.get("base_batch_path") else "query-overlay.npz")
    np.savez_compressed(output_path, **output)
    target_hash = hashlib.sha256(state.tobytes() + observed.tobytes()).hexdigest()
    receipt = {
        "schema": "r02_lai_truthblind_context_receipt_v2",
        "status": "CONSUMER_BATCH_PREPARED_NOT_TRAINING" if request.get("base_batch_path") else "QUERY_OVERLAY_PREPARED_NOT_TRAINING",
        "request_sha256": sha(args.request),
        "output": output_path.name,
        "output_sha256": sha(output_path),
        "n_people": n,
        "n_queries": len(pos),
        "query_valid_count": int(overlay["query_valid"].sum()),
        "truth_observed_count": int(overlay["truth_observed"].sum()),
        "predictor_arrays_sha256": predictor_digest(overlay),
        "target_arrays_sha256": target_hash,
        "truth_read_for_query_geometry": False,
        "truth_observed_used_for": ["LOSS", "SCORING"],
        "query_valid_used_for": ["MODEL_QUERY_SUPPORT"],
        "roles": sorted(set(roles.tolist())),
        "allowed_for": ["CONTEXT_BATCH_QUERY_OVERLAY", "SOFTWARE_VALIDATION"],
        "forbidden_for": ["SCIENTIFIC_TRAINING", "MODEL_SELECTION", "BIOLOGICAL_CONCLUSION"],
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
