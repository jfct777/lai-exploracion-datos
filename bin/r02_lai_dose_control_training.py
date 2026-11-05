"""Authenticate prepared whole-person LAI dose views before FIT updates.

This module adapts an already admitted NONE training link.  It does not admit
training, open SELECT/EVALUATE, derive a new permutation, or build new prefix
matrices.  Original-dose counts remain receipt provenance and never enter the
returned model batch for the removed condition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from r02_lai_control_views import array_sha256, sha256
from r02_lai_dose_controls import REMOVED, PERMUTED, regional_view


BINDING_SCHEMA = "r02_lai_personal_dose_training_binding_v1"
PRODUCT_SCHEMA = "r02_lai_personal_dose_preparation_v1"
PRODUCT_STATUS = "TWO_VIEWS_AND_FULL_CHROMOSOME_COUNTS_VERIFIED_PREPARE_ONLY"
SOURCE_MINOR = "DNABR2723_FIXED_SOURCE_MINOR"
CONDITIONS = (REMOVED, PERMUTED)
BLOCK = "lai_personal_dose_control"
_HEX = re.compile(r"[0-9a-f]{64}")
_SPEC_FIELDS = {"schema", "base_condition", "effective_condition", "role",
                "source_minor", "personal_dose_available", "products_by_case"}
_PRODUCT_FIELDS = {"receipt", "batch"}
_PRODUCT_BASE_ARRAYS = {
    "person_id", "event_valid", "event_cm", "event_global_index",
    "event_chromosome", "event_position_bp", "event_ref", "event_alt",
    "event_counted_allele", "event_is_alt", "event_source_index",
}
_PRODUCER_CODE = {"r02_lai_dose_controls.py", "r02_lai_dose_control_prepare.py",
                  "r02_lai_query_burden.py"}


def require(value, message):
    if not value:
        raise ValueError(message)


def _descriptor(value, name, authenticate=True):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value.get("path"), str) and value["path"]
            and _HEX.fullmatch(value.get("sha256", "")), name + " descriptor differs")
    path = Path(value["path"])
    if authenticate:
        require(path.is_file() and sha256(path) == value["sha256"], name + " differs")
    return path


def _json_descriptor(value, name):
    path = _descriptor(value, name)
    try:
        result = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(name + " is not valid JSON") from exc
    require(isinstance(result, dict), name + " JSON must be an object")
    return result


def validate_binding_spec(contract):
    """Validate declaration shape without treating PREPARE as admission.

    Descriptor contents are deliberately authenticated once, before use, by
    ``apply_product``/``authenticate_product`` or ``checkpoint_binding_identity``.
    This phase only validates descriptor format and the reviewed FIT inventory.
    """
    spec = contract.get(BLOCK)
    if spec is None:
        return None
    require(contract.get("objective") == "lai", "personal-dose views require LAI")
    require(isinstance(spec, dict) and set(spec) == _SPEC_FIELDS,
            "personal-dose binding fields differ")
    condition = spec.get("effective_condition")
    require(spec.get("schema") == BINDING_SCHEMA
            and spec.get("base_condition") == "NONE"
            and contract.get("information_condition") == "NONE"
            and condition in CONDITIONS and spec.get("role") == "FIT"
            and spec.get("source_minor") == contract.get("rare_allele") == SOURCE_MINOR,
            "personal-dose base/effective condition, role, or source-minor identity differs")
    expected_available = condition == PERMUTED
    require(type(spec.get("personal_dose_available")) is bool
            and spec["personal_dose_available"] is expected_available,
            "personal-dose availability is inconsistent with the effective condition")
    products = spec.get("products_by_case")
    require(isinstance(products, dict) and bool(products)
            and all(isinstance(case, str) and case for case in products),
            "nonempty FIT case products are required")
    require(set(products) == _declared_fit_cases(contract),
            "personal-dose products differ from the exact declared NONE FIT cases")
    for case, product in products.items():
        require(isinstance(product, dict) and set(product) == _PRODUCT_FIELDS,
                "personal-dose product fields differ: " + case)
        _descriptor(product["receipt"], case + " receipt", authenticate=False)
        _descriptor(product["batch"], case + " batch", authenticate=False)
    return spec


def _expected_arrays():
    names = set(_PRODUCT_BASE_ARRAYS)
    for condition in CONDITIONS:
        names.update({condition + suffix for suffix in (
            "_event_dosage", "_person_id", "_event_order", "_event_cm",
            "_event_chrom", "_prefix_counts")})
    return names


def _reference_identity(contract, case):
    by_case = contract.get("reference_identity_by_case")
    if isinstance(by_case, dict):
        return by_case.get(case)
    return contract.get("reference_identity")


def _declared_chromosome(contract):
    text = str(contract.get("lai_chromosome", ""))
    require(text.isdigit() and text == str(int(text)) and 1 <= int(text) <= 22,
            "personal-dose binding requires an integer autosome from 1 through 22")
    return int(text)


def _training_source_identity(entry, name, contract):
    """Authenticate the real training wrapper and its immutable PREPARE source."""
    wrapper = _json_descriptor(
        {"path": entry.get("receipt_path"), "sha256": entry.get("receipt_sha256")}, name)
    chromosome = str(_declared_chromosome(contract))
    require(wrapper.get("schema") ==
                "r02_lai_factorized_training_link_receipt_v1"
            and wrapper.get("status") ==
                "FACTORIZED_LINK_BOUND_TO_AUTHENTICATED_TRAINING_PROTOCOL"
            and wrapper.get("condition") == "NONE" and wrapper.get("role") == "FIT"
            and str(wrapper.get("chromosome")) == chromosome,
            "personal-dose source requires an exact NONE FIT training wrapper")
    source = _json_descriptor(wrapper.get("source_prepare_receipt"), name + " PREPARE source")
    require(source.get("schema") == "r02_lai_regional_truth_factorized_link_receipt_v1"
            and source.get("status") ==
                "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION"
            and source.get("training_authorized") is False
            and source.get("selection_authorized") is False
            and source.get("evaluate_opened") is False
            and all(source.get(key) == wrapper.get(key) for key in
                    ("case_id", "role", "condition", "direction", "region_id"))
            and str(source.get("chromosome")) == chromosome,
            "training wrapper and PREPARE source identity differ")
    return wrapper, source


def _declared_fit_cases(contract):
    """Read only FIT wrapper metadata to derive the reviewed case inventory."""
    batches = contract.get("batches")
    require(isinstance(batches, dict) and isinstance(batches.get("FIT"), list)
            and batches["FIT"], "personal-dose binding requires declared FIT batches")
    cases = set()
    for index, entry in enumerate(batches["FIT"]):
        require(isinstance(entry, dict), "FIT batch entry differs")
        wrapper, _source = _training_source_identity(
            entry, "FIT training-link receipt " + str(index), contract)
        case = wrapper.get("case_id")
        require(isinstance(case, str) and case
                and entry.get("lai_case_id", case) == case,
                "personal-dose FIT wrapper case differs")
        require(_reference_identity(contract, case) is not None,
                "FIT case lacks a reviewed reference identity")
        cases.add(case)
    require(cases, "personal-dose FIT case inventory is empty")
    return cases


def _authenticate_receipt(case, product, contract):
    receipt = _json_descriptor(product["receipt"], case + " dose receipt")
    batch_path = _descriptor(product["batch"], case + " dose batch")
    require(receipt.get("schema") == PRODUCT_SCHEMA
            and receipt.get("status") == PRODUCT_STATUS
            and receipt.get("training_authorized") is False
            and receipt.get("truth_loaded") is False
            and receipt.get("evaluate_opened") is False,
            "dose product is not the closed PREPARE product")
    require(receipt.get("batch_sha256") == product["batch"]["sha256"]
            and receipt.get("bytes") == batch_path.stat().st_size
            and set(receipt.get("arrays_sha256", {})) == _expected_arrays()
            and all(_HEX.fullmatch(value or "")
                    for value in receipt["arrays_sha256"].values()),
            "dose product batch or array inventory differs")
    metrics = receipt.get("metrics", {})
    require(metrics.get("algorithm") == "OBSERVED_PERSON_DOSE_SHA256_PCG64_V1"
            and metrics.get("role") == "FIT"
            and metrics.get("statistical_null_certified") is False
            and metrics.get("training_authorized") is False
            and metrics.get("evaluate_opened") is False
            and type(metrics.get("seed")) is int
            and all(_HEX.fullmatch(metrics.get(key, "")) for key in
                    ("universe_sha256", "original_dose_sha256", "observation_sha256")),
            "dose intervention provenance differs")
    require(set(receipt.get("code_sha256", {})) == _PRODUCER_CODE
            and all(_HEX.fullmatch(value or "") for value in receipt["code_sha256"].values()),
            "dose producer code identity differs")

    chromosome = str(_declared_chromosome(contract))
    link = _json_descriptor(receipt.get("source_link"), case + " source link")
    require(link.get("case_id") == case and link.get("role") == "FIT"
            and str(link.get("chromosome")) == chromosome and link.get("condition") == "NONE"
            and link.get("training_authorized") is False
            and link.get("selection_authorized") is False
            and link.get("evaluate_opened") is False,
            "dose source is not the matching NONE FIT link")
    factor = _json_descriptor(receipt.get("source_factorized_receipt"),
                              case + " factorized receipt")
    require(factor.get("schema") == "r02_lai_factorized_predictors_v1"
            and factor.get("status") == "PREDICTOR_INDEX_COMPLETE_NOT_TRAINING"
            and factor.get("case_id") == case and factor.get("role") == "FIT"
            and str(factor.get("chromosome")) == chromosome
            and factor.get("training_authorized") is False
            and factor.get("truth_arrays_loaded") is False
            and factor.get("evaluate_opened") is False
            and factor.get("outputs_sha256", {}).get("indices.npz") ==
                receipt.get("source_index_sha256"),
            "dose factorized source identity differs")
    reference = _reference_identity(contract, case)
    require(reference is not None and receipt.get("reference_identity") == reference
            and factor.get("reference_identity") == reference,
            "dose reference identity differs from the training contract")
    return receipt, link, factor, batch_path


def _load_product(case, product, contract):
    receipt, link, factor, path = _authenticate_receipt(case, product, contract)
    # This cap covers the declared compressed file and the sum of uncompressed
    # arrays retained from the NPZ. It is not a claim about total Python/NumPy
    # peak RSS, which is measured separately by bounded execution audits.
    maximum = contract.get("limits", {}).get("max_batch_bytes")
    require(type(maximum) is int and maximum > 0, "positive batch memory limit required")
    require(receipt["bytes"] <= maximum,
            "compressed dose product exceeds admitted batch memory limit")
    with zipfile.ZipFile(path) as archive:
        require(sum(item.file_size for item in archive.infolist()) <= maximum,
                "uncompressed dose product exceeds admitted batch memory limit")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            require(set(loaded.files) == _expected_arrays(), "dose NPZ array inventory differs")
            arrays = {name: loaded[name] for name in loaded.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError("dose NPZ cannot be loaded safely") from exc
    require(all(array_sha256(arrays[name], np) == digest
                for name, digest in receipt["arrays_sha256"].items()),
            "dose NPZ array hash differs")
    people = np.asarray(arrays["person_id"]).astype(str).tolist()
    universe = hashlib.sha256(json.dumps(sorted(people), separators=(",", ":")).encode()).hexdigest()
    metrics = receipt["metrics"]
    require(metrics.get("people") == len(people)
            and metrics.get("loci") == len(arrays["event_global_index"])
            and metrics.get("universe_sha256") == universe
            and metrics.get("observation_sha256") == array_sha256(arrays["event_valid"], np),
            "dose cohort, EVENT count, or observation identity differs")
    return arrays, receipt, link, factor


def _axis(data, name, width):
    value = np.asarray(data[name])
    if value.ndim == 1:
        require(value.shape == (width,), "regional shared axis shape differs: " + name)
        return value
    require(value.ndim == 2 and value.shape == (len(data["person_id"]), width)
            and np.array_equal(value, np.broadcast_to(value[:1], value.shape)),
            "regional shared axis differs between people: " + name)
    return value[0]


def _entry_identity(entry, contract):
    wrapper, source = _training_source_identity(entry, "base training-link receipt", contract)
    result = dict(wrapper)
    result["rare_radius_cm"] = source.get("rare_radius_cm")
    result["source_prepare_receipt_sha256"] = wrapper["source_prepare_receipt"]["sha256"]
    return result


def authenticate_product(spec, base_data, entry, contract, stage, resolved):
    """Authenticate one case product and its exact regional projection axes."""
    require(spec is not None and stage == "FIT", "personal-dose views are FIT-only")
    base = _entry_identity(entry, contract)
    case = base.get("case_id")
    require(case in spec["products_by_case"]
            and str(base.get("chromosome")) == str(contract.get("lai_chromosome")),
            "base training link is outside the declared dose cases/chromosome")
    arrays, receipt, link, factor = _load_product(
        case, spec["products_by_case"][case], contract)
    require(base.get("direction") == link.get("direction") == factor.get("direction"),
            "dose/base fold direction differs")
    people = np.asarray(base_data["person_id"]).astype(str)
    whole_people = np.asarray(arrays["person_id"]).astype(str)
    require(np.array_equal(people, whole_people), "dose/base person order differs")
    global_index = np.asarray(arrays["event_global_index"])
    require(global_index.ndim == 1 and np.issubdtype(global_index.dtype, np.integer)
            and len(set(global_index.tolist())) == len(global_index)
            and np.array_equal(np.sort(global_index), np.arange(len(global_index))),
            "dose product is not a complete unique chromosome EVENT axis")
    regional_index = np.asarray(base_data["event_global_index"])
    require(regional_index.ndim == 1 and len(regional_index) > 0
            and np.issubdtype(regional_index.dtype, np.integer),
            "regional EVENT global index differs")
    positions = {int(value): i for i, value in enumerate(global_index)}
    require(all(int(value) in positions for value in regional_index),
            "regional locus is outside the whole-chromosome dose product")
    cols = np.asarray([positions[int(value)] for value in regional_index], dtype=np.int64)
    width = len(regional_index)
    comparisons = {
        "event_cm": (arrays["event_cm"][cols], _axis(base_data, "event_cm", width)),
        "event_chromosome": (arrays["event_chromosome"][cols].astype(str),
                              _axis(base_data, "event_chrom", width).astype(str)),
        "event_position_bp": (arrays["event_position_bp"][cols].astype(str),
                              _axis(base_data, "event_pos_bp", width).astype(str)),
        "event_ref": (arrays["event_ref"][cols].astype(str),
                      _axis(base_data, "event_ref", width).astype(str)),
        "event_alt": (arrays["event_alt"][cols].astype(str),
                      _axis(base_data, "event_alt", width).astype(str)),
        "event_counted_allele": (arrays["event_counted_allele"][cols].astype(str),
                                  _axis(base_data, "event_counted_allele", width).astype(str)),
        "event_is_alt": (arrays["event_is_alt"][cols],
                         _axis(base_data, "event_is_alt", width)),
    }
    for name, (left, right) in comparisons.items():
        require(np.array_equal(left, right),
                "dose/base locus, allele, or source-minor orientation differs: " + name)
    require(np.asarray(base_data["event_valid"]).shape == (len(people), width)
            and np.asarray(base_data["event_dosage"]).shape == (len(people), width)
            and np.array_equal(arrays["event_valid"][:, cols], base_data["event_valid"]),
            "dose/base observation support differs")

    condition = spec["effective_condition"]
    dose = np.asarray(arrays[condition + "_event_dosage"])
    order = np.asarray(arrays[condition + "_event_order"])
    prefix = np.asarray(arrays[condition + "_prefix_counts"])
    require(dose.shape == arrays["event_valid"].shape
            and np.array_equal(arrays[condition + "_person_id"].astype(str), whole_people)
            and order.shape == (dose.shape[1],) and np.array_equal(np.sort(order), np.arange(dose.shape[1]))
            and np.array_equal(arrays[condition + "_event_cm"], arrays["event_cm"][order])
            and np.array_equal(arrays[condition + "_event_chrom"].astype(str),
                               arrays["event_chromosome"][order].astype(str))
            and prefix.shape == (dose.shape[0], dose.shape[1] + 1, 3)
            and np.issubdtype(prefix.dtype, np.integer) and (prefix >= 0).all(),
            "stored condition-specific whole-chromosome prefix axes differ")
    valid = np.asarray(arrays["event_valid"], dtype=bool)
    measured = np.where(valid, dose, 0)[:, order]
    increments = np.diff(prefix, axis=1)
    require(not prefix[:, 0, :].any(), "stored condition prefix origin is not zero")
    require(np.isin(dose[valid], (0, 1, 2)).all()
            and np.array_equal(increments[..., 0], measured > 0)
            and np.array_equal(increments[..., 1], measured)
            and np.array_equal(increments[..., 2], valid[:, order]),
            "stored condition prefix does not represent its authenticated effective dose")
    if condition == REMOVED:
        require(not dose.any() and not prefix[..., :2].any()
                and spec["personal_dose_available"] is False,
                "removed dose must be unavailable with zero effective counts")
    else:
        require(spec["personal_dose_available"] is True,
                "permuted effective dose must be marked available")
    radius = resolved.get("rare_radius_cm") if isinstance(resolved, dict) else None
    require(type(radius) in (int, float) and radius == base.get("rare_radius_cm"),
            "resolved rare radius differs from the base training link")
    return arrays, receipt, radius


def apply_product(base_data, entry, contract, stage, resolved):
    """Replace dose/count inputs before person microbatching; keep all other arrays."""
    spec = validate_binding_spec(contract)
    if spec is None:
        return base_data
    arrays, _receipt, radius = authenticate_product(
        spec, base_data, entry, contract, stage, resolved)
    views = {condition: arrays[condition + "_event_dosage"] for condition in CONDITIONS}
    prefixes = {condition: {
        "person_id": arrays[condition + "_person_id"],
        "event_order": arrays[condition + "_event_order"],
        "event_cm": arrays[condition + "_event_cm"],
        "event_chrom": arrays[condition + "_event_chrom"],
        "prefix_counts": arrays[condition + "_prefix_counts"],
    } for condition in CONDITIONS}
    declared_chromosome = _declared_chromosome(contract)
    whole_chrom = np.asarray(arrays["event_chromosome"]).astype(str)
    require(np.array_equal(whole_chrom,
                           np.full(whole_chrom.shape, str(declared_chromosome))),
            "whole product chromosome differs from the declared chromosome")
    canonical_chrom = np.full(whole_chrom.shape, declared_chromosome, dtype=np.int64)
    return regional_view(base_data, condition=spec["effective_condition"],
        views=views, prefixes=prefixes, whole_person_ids=arrays["person_id"],
        whole_observed=arrays["event_valid"],
        whole_event_global_index=arrays["event_global_index"],
        regional_event_global_index=base_data["event_global_index"],
        whole_event_cm=arrays["event_cm"], whole_event_chrom=canonical_chrom,
        radius_cm=radius, expected_radius_cm=radius)


def checkpoint_binding_identity(contract):
    """Return provenance that makes dose-view checkpoints/cache non-interchangeable."""
    spec = validate_binding_spec(contract)
    if spec is None:
        return None
    products = {}
    for case in sorted(spec["products_by_case"]):
        product = spec["products_by_case"][case]
        receipt, link, factor, _path = _authenticate_receipt(case, product, contract)
        condition = spec["effective_condition"]
        products[case] = {
            "receipt_sha256": product["receipt"]["sha256"],
            "batch_sha256": product["batch"]["sha256"],
            "source_link_sha256": receipt["source_link"]["sha256"],
            "source_factorized_receipt_sha256": receipt["source_factorized_receipt"]["sha256"],
            "source_index_sha256": receipt["source_index_sha256"],
            "reference_identity": receipt["reference_identity"],
            "condition_arrays_sha256": {name: digest for name, digest in
                receipt["arrays_sha256"].items() if name.startswith(condition + "_")
                or name in _PRODUCT_BASE_ARRAYS},
            "algorithm": receipt["metrics"]["algorithm"],
            "seed": receipt["metrics"]["seed"],
            "universe_sha256": receipt["metrics"]["universe_sha256"],
            "original_dose_sha256_ledger_only": receipt["metrics"]["original_dose_sha256"],
            "observation_sha256": receipt["metrics"]["observation_sha256"],
            "producer_code_sha256": receipt["code_sha256"],
            "case_id": link["case_id"], "task_id": factor["task_id"],
        }
    return {"schema": BINDING_SCHEMA, "base_condition": "NONE",
            "effective_condition": spec["effective_condition"],
            "role": "FIT", "source_minor": SOURCE_MINOR,
            "personal_dose_available": spec["personal_dose_available"],
            "products_by_case": products}
