#!/usr/bin/env python3
"""Load one complete authenticated FIT LAI case from the saved FT sources.

This module performs no inference or fitting.  It joins the same sixteen owned
query fragments used by FT and returns the original FLARE probabilities, truth,
observation mask, genetic positions and query-local rare-allele counts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


NEEDED = (
    "baseline_probabilities", "query_cm", "query_core", "query_global_index",
    "query_burden", "truth_state", "truth_observed", "query_valid",
    "person_id", "role",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated(descriptor: dict, label: str = "input") -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError(label + " descriptor differs")
    path = Path(descriptor["path"])
    if sha256(path) != descriptor["sha256"]:
        raise ValueError(label + " checksum differs: " + str(path))
    return path


def authenticated_json(descriptor: dict, label: str) -> dict:
    return json.loads(authenticated(descriptor, label).read_text())


def _array_hash(value) -> str:
    return hashlib.sha256(
        value.dtype.str.encode()
        + b"\0"
        + json.dumps(value.shape, separators=(",", ":")).encode()
        + b"\0"
        + value.tobytes()
    ).hexdigest()


def _load_cached_region(descriptor: dict, saved: dict, request: dict, np) -> dict:
    manifest_path = authenticated(descriptor, "duration cache manifest")
    manifest = json.loads(manifest_path.read_text())
    identity = manifest["identity"]
    if (manifest.get("status") != "COMPLETE"
            or identity.get("contract_sha256") != request["contract_sha256"]):
        raise ValueError("unverified or wrong cache contract")
    key = identity["wrapper_receipt_sha256"]
    if saved["input"].get("role") != "FIT":
        raise ValueError("non-FIT prediction input")
    wrapper = authenticated_json(
        {"path": saved["input"]["receipt_path"], "sha256": key},
        "prediction input wrapper",
    )
    authenticated(
        {"path": saved["input"]["path"], "sha256": saved["input"]["sha256"]},
        "prediction input request",
    )
    if (wrapper.get("request_sha256") != saved["input"]["sha256"]
            or wrapper.get("role") != "FIT"
            or wrapper.get("case_id") != request["case"]
            or wrapper.get("condition") != "NONE"
            or wrapper.get("region_id") != identity.get("region_id")):
        raise ValueError("prediction input receipt differs from cache")
    table = {value["name"]: value for value in manifest["arrays"]}
    if not set(NEEDED) <= set(table):
        raise ValueError("cache arrays are incomplete")
    data = {}
    for name in NEEDED:
        record = table[name]
        path = authenticated(
            {"path": manifest_path.parent / record["file"],
             "sha256": record["file_sha256"]},
            "cached array " + name,
        )
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != record["shape"] or value.dtype.str != record["dtype"]:
            raise ValueError("cached array shape or dtype differs: " + name)
        content = _array_hash(value)
        if (content != wrapper["reconstructed_arrays_sha256"].get(name)
                or content != record["array_sha256"]):
            raise ValueError("cached person/query/content identity differs: " + name)
        data[name] = value
    return data


def load_saved_fit_case(request: dict, np, *, started: float | None = None,
                        max_seconds: float | None = None,
                        additional_prediction_indices: dict | None = None,
                        include_fi96: bool = False) -> dict:
    """Return a complete case after reproducing FT's input authentication."""
    if request.get("schema") != "r02_saved_rare_boundary_diagnostic_v1":
        raise ValueError("FT request schema differs")
    begun = time.monotonic() if started is None else started
    limit = request["max_seconds"] if max_seconds is None else max_seconds
    for descriptor in request["code"]:
        authenticated(descriptor, "FT code")
    receipt = authenticated_json(request["training_receipt"], "FI training receipt")
    expected_receipt = {
        "status": "FIT_CHECKPOINT", "score_role": "FIT", "scope": "DEVELOPMENT",
        "arm": "NONE", "steps": 96,
        "contract_sha256": request["contract_sha256"],
        "prediction_sha256": request["prediction_index"]["sha256"],
    }
    for key, value in expected_receipt.items():
        if receipt.get(key) != value:
            raise ValueError("saved FI training receipt differs: " + key)
    contract = authenticated_json(request["contract"], "FT contract")
    if (request["contract"]["sha256"] != request["contract_sha256"]
            or contract.get("count_scope") !=
                "query-local DNABR2723 fixed-source-minor event counts within 1 cM"
            or contract.get("rare_allele") != "DNABR2723_FIXED_SOURCE_MINOR"):
        raise ValueError("fixed rare exposure meaning differs")
    index = authenticated_json(request["prediction_index"], "FI prediction index")
    chosen = [row for row in index if row["input"].get("lai_case_id") == request["case"]]
    by_receipt = {row["input"]["receipt_sha256"]: row for row in chosen}
    if len(chosen) != len(by_receipt) or len(chosen) != 16 or len(request["regions"]) != 16:
        raise ValueError("complete unique sixteen-region case required")

    prediction_sources = {"FI96": by_receipt} if include_fi96 else {}
    for name, descriptor in (additional_prediction_indices or {}).items():
        other = authenticated_json(descriptor, name + " prediction index")
        rows = [row for row in other if row["input"].get("lai_case_id") == request["case"]]
        mapped = {row["input"]["receipt_sha256"]: row for row in rows}
        if len(rows) != 16 or len(mapped) != 16 or set(mapped) != set(by_receipt):
            raise ValueError(name + " prediction inputs differ from FT case")
        for key in mapped:
            if mapped[key]["input"] != by_receipt[key]["input"]:
                raise ValueError(name + " prediction wrapper differs from FT")
        prediction_sources[name] = mapped

    arrays: dict[str, list] = {}
    predictions: dict[str, list] = {name: [] for name in prediction_sources}
    people = None
    seen = set()
    for descriptor in request["regions"]:
        if time.monotonic() - begun > limit:
            raise TimeoutError("saved FIT case loading budget exceeded")
        if request.get("reader", "AUTHENTICATED_CACHE") == "SAVED_QUERY_SOURCES":
            from r02_lai_saved_query_inputs import load_saved_queries
            key = descriptor["receipt_sha256"]
            if key not in by_receipt:
                raise ValueError("saved-query receipt is absent from prediction index")
            saved = by_receipt[key]
            data = load_saved_queries(
                saved["input"],
                {"case": request["case"], "people": request["people"],
                 "rare_allele": contract["rare_allele"]},
                np,
            )
        else:
            manifest = authenticated_json(descriptor, "duration cache manifest")
            key = manifest["identity"]["wrapper_receipt_sha256"]
            if key not in by_receipt:
                raise ValueError("cache receipt is absent from prediction index")
            saved = by_receipt[key]
            data = _load_cached_region(descriptor, saved, request, np)
        if key in seen:
            raise ValueError("duplicate saved query input")
        seen.add(key)
        if not np.all(data["role"] == "FIT"):
            raise ValueError("non-FIT rows")
        if people is None:
            people = data["person_id"].copy()
        if not np.array_equal(people, data["person_id"]):
            raise ValueError("person order differs")
        core_by_person = data["query_core"]
        if not np.array_equal(core_by_person, np.broadcast_to(core_by_person[:1], core_by_person.shape)):
            raise ValueError("person core differs")
        if not np.all(data["query_valid"][core_by_person]):
            raise ValueError("invalid owned query")
        core = core_by_person[0]
        for name, mapping in prediction_sources.items():
            prediction_path = authenticated(
                {field: mapping[key][field] for field in ("path", "sha256")},
                name + " saved prediction")
            prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
            if prediction.shape != data["baseline_probabilities"].shape:
                raise ValueError(name + " prediction axes differ")
            predictions[name].append(prediction[:, core].copy())
        for name in NEEDED:
            if name in {"person_id", "role", "query_core", "query_valid"}:
                continue
            value = data[name]
            arrays.setdefault(name, []).append(
                (value[core] if value.ndim == 1 else value[:, core]).copy()
            )
    if seen != set(by_receipt):
        raise ValueError("saved case coverage differs")
    joined = {
        key: np.concatenate(values, axis=0 if key == "query_global_index" else 1)
        for key, values in arrays.items()
    }
    indices = joined.pop("query_global_index")
    order = np.argsort(indices, kind="stable")
    if (len(indices) != request["queries"]
            or not np.array_equal(indices[order], np.arange(request["queries"]))):
        raise ValueError("owned query coverage is not exactly complete")
    joined = {key: value[:, order] for key, value in joined.items()}
    joined_predictions = {
        name: np.concatenate(parts, axis=1)[:, order]
        for name, parts in predictions.items()
    }
    if (len(people) != request["people"]
            or len(set(people.astype(str).tolist())) != len(people)):
        raise ValueError("person count or identities differ")
    return {
        "flare": joined["baseline_probabilities"],
        "truth": joined["truth_state"],
        "observed": joined["truth_observed"],
        "cm": joined["query_cm"],
        "burden": joined["query_burden"],
        "people": people,
        "rare_allele": contract["rare_allele"],
        "training_receipt": request["training_receipt"],
        "predictions": joined_predictions,
    }
