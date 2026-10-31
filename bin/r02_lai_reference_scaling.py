#!/usr/bin/env python3
"""Versioned scaling of the six AC/AN LAI reference feature columns.

The 15 raw reference features remain the authenticated stored representation.
This module changes only the tensor presented to a LAI model, after the raw
features have passed their existing REF/ALT and receipt checks.
"""
import math
import csv
import hashlib
import json
from pathlib import Path
import re


SCHEMA = "r02_lai_reference_scaling_v1"
RAW_MODE = "RAW_REFERENCE_FEATURES_V1"
LOG1P_MODE = "LOG1P_AC_AN_BY_PANEL_TOTAL_V1"
CONTRACT_KEY = "lai_reference_scaling"
ANCESTRIES = ("African", "European", "Native_American")
FEATURES_PER_ANCESTRY = ("AC_0", "AN_0", "AF_0", "KNOWN", "FREQUENCY_DEFINED")
REFERENCE_FEATURE_NAMES = tuple(
    f"{ancestry}.{feature}"
    for ancestry in ANCESTRIES for feature in FEATURES_PER_ANCESTRY)
AC_AN_COLUMNS = tuple(
    index for index, name in enumerate(REFERENCE_FEATURE_NAMES)
    if name.endswith((".AC_0", ".AN_0")))
PRESERVED_COLUMNS = tuple(index for index in range(15) if index not in AC_AN_COLUMNS)
PANEL_RECEIPT_SCHEMA = "r02_lai_ref_provenance_support_v1"
COUNT_FIELD = "people"


def require(value, message):
    if not value:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_unique_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate reference panel receipt key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(
                          ValueError("nonfinite reference panel receipt value")))


def _authenticated_role_counts(provenance):
    path = Path(provenance["roles_path"])
    require(path.is_file() and _sha256(path) == provenance["roles_sha256"],
            "reference role table path or SHA-256 differs")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None
                and {"sample_id", "role", "ancestry"} <= set(reader.fieldnames),
                "reference role table columns differ")
        seen = set()
        counts = {ancestry: 0 for ancestry in ANCESTRIES}
        for row in reader:
            sample = row["sample_id"]
            require(sample and sample not in seen, "empty or duplicate panel sample identifier")
            seen.add(sample)
            if row["role"] == "REF_TRAIN":
                require(row["ancestry"] in counts, "REF_TRAIN ancestry differs")
                counts[row["ancestry"]] += 1
    return counts


def _reference_identities(contract):
    identity = contract.get("reference_identity")
    if isinstance(identity, dict):
        return (identity,)
    by_case = contract.get("reference_identity_by_case")
    require(isinstance(by_case, dict) and by_case,
            "scaled LAI reference features require authenticated reference identities")
    return tuple(by_case.values())


def contract_scaling(contract):
    """Return and validate the scaling specification; absence means exact raw mode."""
    value = contract.get(CONTRACT_KEY)
    if value is None:
        return {"schema": SCHEMA, "mode": RAW_MODE}
    require(isinstance(value, dict), "LAI reference scaling must be an object")
    require(value.get("schema") == SCHEMA, "unknown LAI reference scaling schema")
    mode = value.get("mode")
    require(mode in (RAW_MODE, LOG1P_MODE), "unknown LAI reference scaling mode")
    if mode == RAW_MODE:
        require(set(value) == {"schema", "mode"},
                "raw LAI reference scaling has no panel-size parameters")
        return dict(value)

    require(set(value) == {"schema", "mode", "reference_panel_people", "provenance"},
            "scaled LAI reference configuration fields differ")
    count = value["reference_panel_people"]
    require(type(count) is int and count > 0,
            "reference_panel_people must be a positive authenticated person count")
    provenance = value["provenance"]
    require(isinstance(provenance, dict) and set(provenance) == {
        "path", "sha256", "schema", "count_field", "reference_vcf_sha256",
        "roles_path", "roles_sha256", "reference_people_by_ancestry"},
        "reference panel count provenance fields differ")
    require(provenance["schema"] == PANEL_RECEIPT_SCHEMA
            and provenance["count_field"] == COUNT_FIELD
            and all(re.fullmatch(r"[0-9a-f]{64}", provenance[key] or "")
                    for key in ("sha256", "reference_vcf_sha256", "roles_sha256")),
            "reference panel count provenance is not an authenticated panel receipt")
    receipt_path = Path(provenance["path"])
    require(receipt_path.is_file() and _sha256(receipt_path) == provenance["sha256"],
            "reference panel receipt path or SHA-256 differs")
    receipt = _read_unique_json(receipt_path)
    require(receipt.get("schema") == PANEL_RECEIPT_SCHEMA
            and receipt.get("status") == "COMPLETE_COUNTS_AND_CELL_PROVENANCE_NOT_TRAINING"
            and receipt.get("role") == "REF_TRAIN"
            and type(receipt.get(COUNT_FIELD)) is int and receipt[COUNT_FIELD] == count,
            "reference panel count does not match the authenticated REF_TRAIN receipt")
    receipt_inputs = receipt.get("input_sha256")
    require(isinstance(receipt_inputs, dict)
            and receipt_inputs.get(provenance["roles_path"]) == provenance["roles_sha256"],
            "reference panel receipt does not authenticate the role table")
    declared_counts = provenance["reference_people_by_ancestry"]
    require(isinstance(declared_counts, dict) and set(declared_counts) == set(ANCESTRIES)
            and all(type(value) is int and value >= 0 for value in declared_counts.values()),
            "reference ancestry count configuration differs")
    actual_counts = _authenticated_role_counts(provenance)
    require(actual_counts == declared_counts and sum(actual_counts.values()) == count,
            "reference ancestry counts do not sum to the authenticated panel total")
    identities = _reference_identities(contract)
    require(all(isinstance(item, dict)
                and item.get("ref_train_receipt_sha256") == provenance["sha256"]
                and item.get("reference_vcf_sha256") == provenance["reference_vcf_sha256"]
                for item in identities),
            "scaling panel receipt or VCF and LAI reference identities differ")
    return {
        "schema": SCHEMA,
        "mode": LOG1P_MODE,
        "reference_panel_people": count,
        "provenance": dict(provenance),
    }


def scale_reference_features(features, specification, np):
    """Scale AC/AN only; retain AF, masks and every other feature exactly."""
    values = np.asarray(features)
    require(values.ndim >= 1 and values.shape[-1] == 15
            and np.issubdtype(values.dtype, np.floating)
            and np.isfinite(values).all(),
            "LAI reference features must be finite floating arrays with width 15")
    require(isinstance(specification, dict)
            and specification.get("schema") == SCHEMA
            and specification.get("mode") in (RAW_MODE, LOG1P_MODE),
            "validated LAI reference scaling specification required")
    if specification["mode"] == RAW_MODE:
        return values

    count = specification.get("reference_panel_people")
    require(type(count) is int and count > 0,
            "scaled LAI reference features require reference_panel_people")
    maximum_alleles = 2 * count
    ancestry_counts = specification["provenance"]["reference_people_by_ancestry"]
    for ancestry, offset in zip(ANCESTRIES, range(0, 15, len(FEATURES_PER_ANCESTRY))):
        ac, an, known = values[..., offset], values[..., offset + 1], values[..., offset + 3]
        require(np.isin(known, (0, 1)).all(), "LAI reference KNOWN must be binary")
        known = known.astype(bool)
        require((ac[known] >= 0).all() and (an[known] >= 0).all()
                and (ac[known] <= an[known]).all()
                and (an[known] <= 2 * ancestry_counts[ancestry]).all()
                and (ac[~known] == 0).all() and (an[~known] == 0).all(),
                "LAI reference AC/AN contradicts knownness or authenticated ancestry diploid panel bound")
    selected = values[..., AC_AN_COLUMNS]
    result = values.copy()
    denominator = math.log1p(maximum_alleles)
    result[..., AC_AN_COLUMNS] = (
        np.log1p(selected.astype(np.float64)) / denominator).astype(values.dtype)
    require(np.array_equal(result[..., PRESERVED_COLUMNS], values[..., PRESERVED_COLUMNS]),
            "LAI scaling changed AF, knownness, or frequency-defined columns")
    return result
