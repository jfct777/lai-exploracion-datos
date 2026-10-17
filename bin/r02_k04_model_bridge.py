#!/usr/bin/env python3
"""Authenticated K04 C/CQ/CQR inputs and the unweighted interval-solver bridge.

This module implements the reviewed conditional K04 design.  It materializes
FIT and SELECT predictors together so paired missing-value means and column
scaling are learned on FIT once.  It never reads an EVALUATE target, chooses a
penalty, or claims biological validation.

The robust solver owns the normalized pair weights.  ``solver_operator`` calls
``PairDesign.operator`` with unit row weights, preserving centered/scaled
columns without multiplying rows by sqrt(scientific_weight).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

import r02_k04_predictor_inputs as legacy
from r02_interval_ridge import solve_interval_ridge
from r02_pair_operator import PairDesign, check_weights


REQUEST_SCHEMA = "r02_k04_ccqcqr_request_v1"
OUTPUT_SCHEMA = "r02_k04_ccqcqr_inputs_v1"
OPERATION = "MATERIALIZE_REVIEWED_C_CQ_CQR_FIT_SELECT"
ROLES = legacy.ROLES
ARMS = ("C", "CQ", "CQR")
N_AXES = 1459
POINT_COLUMNS = {
    "C": [*(f"common_score_{i:04d}" for i in range(1, N_AXES + 1)),
          "common_mean_standardized_dosage", "common_callable_fraction",
          "rare_callable_fraction"],
    "CQ": [*(f"common_score_{i:04d}" for i in range(1, N_AXES + 1)),
           "common_mean_standardized_dosage", "common_callable_fraction",
           "rare_callable_fraction", "rare_carrier_sites", "rare_copy_count"],
}
POINT_COLUMNS["CQR"] = POINT_COLUMNS["CQ"]
EXTRA_COLUMNS = {
    "C": ["common_GRM", "common_joint_fraction", "rare_joint_fraction",
          "common_GRM_missing", "common_joint_fraction_missing",
          "rare_joint_fraction_missing"],
    "CQ": ["common_GRM", "common_joint_fraction", "rare_joint_fraction",
           "common_GRM_missing", "common_joint_fraction_missing",
           "rare_joint_fraction_missing", "rare_Jaccard_unavailable"],
    "CQR": ["common_GRM", "common_joint_fraction", "rare_joint_fraction",
            "common_GRM_missing", "common_joint_fraction_missing",
            "rare_joint_fraction_missing", "rare_Jaccard_unavailable",
            "rare_Jaccard"],
}
DESIGN_COLUMNS = {arm: 3 * len(POINT_COLUMNS[arm]) + len(EXTRA_COLUMNS[arm])
                  for arm in ARMS}
EXPECTED_DESIGN_COLUMNS = {"C": 4392, "CQ": 4399, "CQR": 4400}
SOURCE_ARRAYS = legacy.SOURCE_ARRAYS
RARE_ARRAYS = {"I", "U", "Q", "individual_carrier", "individual_callable", "n_sites"}
LIMIT_FIELDS = {"max_metadata_bytes", "max_input_bytes", "max_array_bytes",
                "max_memory_bytes", "max_output_bytes", "min_free_bytes",
                "rows_per_chunk", "max_working_mib", "workers"}
OPERATOR_BACKENDS = {"factorized", "factorized_admitted_pairs"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


sha256 = legacy.sha256
array_sha256 = legacy.array_sha256


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def implementation_sha256():
    root = Path(__file__).resolve().parent
    names = (Path(__file__).name, "r02_k04_predictor_inputs.py",
             "r02_pair_operator.py", "r02_interval_ridge.py")
    return {name: sha256(root / name) for name in names}


def _valid_hash(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value))


class PinResolver:
    """Resolve exact file descriptors directly or from a Nextflow staging root."""

    def __init__(self, staged_root=None):
        self.staged_root = None if staged_root is None else Path(staged_root).resolve()
        self._by_hash = None
        self.used = {}
        self._used_resolved = set()
        self.total_bytes = 0

    def _index(self):
        require(self.staged_root is not None and self.staged_root.is_dir(),
                "Staged input root is not a directory")
        result = {}
        for path in sorted(p for p in self.staged_root.rglob("*") if p.is_file()):
            digest = sha256(path)
            result.setdefault(digest, []).append(path.resolve())
        self._by_hash = result

    def pin(self, descriptor, maximum, label):
        require(isinstance(descriptor, dict) and set(descriptor) == {"path", "sha256"} and
                isinstance(descriptor["path"], str) and Path(descriptor["path"]).is_absolute() and
                _valid_hash(descriptor["sha256"]), "Exact absolute path/SHA256 required: " + label)
        original = Path(descriptor["path"])
        path = original.resolve() if original.is_file() else None
        if path is None or sha256(path) != descriptor["sha256"]:
            if self._by_hash is None:
                self._index()
            matches = self._by_hash.get(descriptor["sha256"], [])
            require(len(matches) == 1, "Pinned staged input is missing or ambiguous: " + label)
            path = matches[0]
        require(path.is_file() and path.stat().st_size <= maximum and
                sha256(path) == descriptor["sha256"],
                "Pinned input changed or exceeds bound: " + label)
        self.used[descriptor["path"]] = descriptor["sha256"]
        if path not in self._used_resolved:
            self._used_resolved.add(path)
            self.total_bytes += path.stat().st_size
        return path


def _npz_inventory(path, maximum, expected):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(members and len({member.filename for member in members}) == len(members) and
                all(not member.is_dir() and not (member.flag_bits & 1) for member in members) and
                {member.filename for member in members} == {name + ".npy" for name in expected} and
                sum(member.file_size for member in members) <= maximum,
                "NPZ inventory or decompression bound differs")


def _chromosome(value):
    text = str(value).removeprefix("chr")
    require(text.isdigit() and 1 <= int(text) <= 22, "Autosome 1..22 required")
    return str(int(text))


def _sample_hash(ids):
    return legacy.sample_hash(ids)


def _read_persons(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(rows and set(rows[0]) >= {"sample_id", "component", "role"} and
            len(rows) == sum(ROLES.values()) and len({row["sample_id"] for row in rows}) == len(rows) and
            {role: sum(row["role"] == role for row in rows) for role in ROLES} == ROLES,
            "Frozen 2,619-person roster/roles/components differ")
    components = {}
    for row in rows:
        require(row["role"] in ROLES and row["component"], "Person role/component missing")
        prior = components.setdefault(row["component"], row["role"])
        require(prior == row["role"], "A PC-Relate component crosses frozen roles")
    return rows


_ratio = legacy._ratio
_fit_impute = legacy._fit_impute


def validate_exclusion(target_chromosome, predictor_chromosomes):
    target = _chromosome(target_chromosome)
    expected = [str(i) for i in range(1, 23) if str(i) != target]
    actual = [_chromosome(value) for value in predictor_chromosomes]
    require(actual == expected, "Exactly ordered 21 whole non-target chromosomes required")
    return expected


def _pair_codes(left, right, n_people):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and left.ndim == 1 and left.dtype.kind in "iu" and
            right.dtype.kind in "iu" and np.all(left < right) and np.all(right < n_people),
            "Canonical unordered pair arrays required")
    code = left.astype(np.uint64) * np.uint64(n_people) + right
    require(np.all(code[1:] > code[:-1]), "Pair rows must be unique lexicographic")
    return code


def _aligned(source_code, target_code, label):
    positions = np.searchsorted(source_code, target_code)
    found = positions < len(source_code)
    found[found] &= source_code[positions[found]] == target_code[found]
    require(np.all(found), "A required pair is absent: " + label)
    return positions


def _load_target(role, descriptors, target, ids, resolver, limits):
    require(set(descriptors) == {"manifest", "person_i", "person_j", "lower", "upper", "opportunities"},
            "Exact interval-target file inventory required")
    manifest_path = resolver.pin(descriptors["manifest"], limits["max_metadata_bytes"], role + " target manifest")
    manifest = load_json(manifest_path)
    expected = validate_exclusion(target, manifest.get("predictor_chromosomes", []))
    require(manifest.get("schema") == "r02_k04_interval_target_v1" and
            manifest.get("status") == "COMPLETE_SOLVER_TARGET_ADAPTER_NOT_MODEL_FIT" and
            manifest.get("role") == role and _chromosome(manifest.get("chromosome")) == target and
            manifest.get("sample_order_sha256") == _sample_hash(ids) and
            manifest.get("evaluate_pairs_materialized") is False,
            "Authenticated FIT/SELECT interval target differs")
    arrays = {}
    for name in ("person_i", "person_j", "lower", "upper", "opportunities"):
        path = resolver.pin(descriptors[name], limits["max_input_bytes"], role + " target " + name)
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    n = manifest["n_pairs"]
    require(all(array.shape == (n,) for array in arrays.values()) and
            arrays["person_i"].dtype.kind == arrays["person_j"].dtype.kind == "u" and
            np.all(arrays["lower"] <= arrays["upper"]) and np.all(arrays["lower"] >= 0) and
            np.all(arrays["upper"] <= 1) and np.all(arrays["opportunities"] > 0),
            "Interval target array shape/domain differs")
    code = _pair_codes(arrays["person_i"], arrays["person_j"], len(ids))
    return manifest, arrays, code, expected


def _load_axis(source, target, ids, resolver, limits):
    require(set(source) == {"manifest", "array"}, "Exact common-axis file inventory required")
    manifest_path = resolver.pin(source["manifest"], limits["max_metadata_bytes"], "common axis manifest")
    manifest = load_json(manifest_path)
    intervals = manifest.get("allowed_input_intervals", [])
    require(all(isinstance(row, list) and len(row) == 3 and row[1] == 1 and
                type(row[2]) is int and row[2] > 0 for row in intervals),
            "Common axis must contain one complete interval per predictor chromosome")
    expected = validate_exclusion(target, [row[0] for row in intervals])
    require(manifest.get("schema") == "r02_structure_materialized_common_merge_v2" and
            manifest.get("stage") == "FULL_NUMERICAL_COMMON_SPECTRUM_NOT_PREDICTIVE_SELECTION" and
            manifest.get("exclusion_scope") == "WHOLE_CHROMOSOME" and
            manifest.get("numerical_rank") == N_AXES and manifest.get("targets_read") is False,
            "Full authenticated 1,459-axis common view required")
    path = resolver.pin(source["array"], limits["max_input_bytes"], "common axis array")
    require(manifest.get("outputs_sha256", {}).get("continuous.private.npz") == source["array"]["sha256"],
            "Common axis manifest/array binding differs")
    with np.load(path, allow_pickle=False) as archive:
        require("scores" in archive.files, "Common axis scores absent")
        scores = archive["scores"].copy()
    require(scores.shape == (len(ids), N_AXES) and scores.dtype == np.float64 and np.isfinite(scores).all(),
            "Common score dimensions/dtype/finite values differ")
    return scores, manifest, expected


def _load_partial(desc, chromosome, ids, source_contract_sha256, resolver, limits):
    require(set(desc) == {"manifest", "array"}, "Exact partial file inventory required")
    manifest_path = resolver.pin(desc["manifest"], limits["max_metadata_bytes"], "partial manifest chr" + chromosome)
    manifest = load_json(manifest_path)
    request = manifest.get("request", {})
    require(manifest.get("schema") == "r02_structure_materialize_partial_v1" and
            manifest.get("stage") == "COMPLETE_SLICE" and request.get("kind") == "features" and
            _chromosome(request.get("chromosome")) == chromosome and
            manifest.get("contract_sha256") == source_contract_sha256 and
            manifest.get("selection_used") == "FIT_ONLY" and
            manifest.get("evaluated_outcomes_for_selection") is False,
            "Authenticated complete FIT-selected partial differs")
    path = resolver.pin(desc["array"], limits["max_input_bytes"], "partial data chr" + chromosome)
    require(manifest.get("outputs_sha256", {}).get("data.npz") == desc["array"]["sha256"],
            "Partial manifest/array binding differs")
    _npz_inventory(path, limits["max_array_bytes"], SOURCE_ARRAYS)
    with np.load(path, allow_pickle=False) as archive:
        require(archive["sample_ids"].tolist() == ids, "Partial person roster/order differs")
        values = {name: archive[name].copy() for name in SOURCE_ARRAYS - {"sample_ids"}}
    pairs = values["pairs"]
    require(pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in "iu",
            "Partial pair array differs")
    code = _pair_codes(pairs[:, 0], pairs[:, 1], len(ids))
    return code, values


def _load_repair(source, target, ids, resolver, limits):
    require(set(source) == {"manifest", "array"}, "Exact v4-repair file inventory required")
    manifest_path = resolver.pin(source["manifest"], limits["max_metadata_bytes"], "v4 repair manifest")
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == "r02_k04_common_pair_repair_aggregate_v1" and
            manifest.get("stage") == "ALL_22_TARGET_EXCLUSIONS_REPAIRED" and
            manifest.get("n_people") == len(ids) and manifest.get("n_target_chromosomes") == 22 and
            manifest.get("target_exclusion") == "SUM_ALL_22_MINUS_TARGET_CHROMOSOME" and
            manifest.get("evaluate_outcomes_read") is False and
            manifest.get("human_c_cq_cqr_fit_admitted") is False,
            "Authenticated all-target v4 repair required; its own non-admission is preserved")
    path = resolver.pin(source["array"], limits["max_input_bytes"], "v4 repair array")
    require(manifest.get("outputs_sha256", {}).get("repair.npz") == source["array"]["sha256"],
            "Repair manifest/array binding differs")
    required = {"target_chromosomes", "missing_pairs", "missing_pair_role", "sample_ids",
                "common_joint_missing", "common_numerator_missing", "common_zsum",
                "common_called", "common_sites", "rare_copies"}
    _npz_inventory(path, limits["max_array_bytes"], required)
    with np.load(path, allow_pickle=False) as archive:
        require(archive["sample_ids"].tolist() == ids and
                archive["target_chromosomes"].tolist() == list(range(1, 23)),
                "Repair person/target order differs")
        at = int(target) - 1
        result = {name: archive[name][at].copy() for name in
                  ("common_joint_missing", "common_numerator_missing", "common_zsum",
                   "common_called", "common_sites", "rare_copies")}
        pairs = archive["missing_pairs"].copy()
        roles = archive["missing_pair_role"].copy()
    code = _pair_codes(pairs[:, 0], pairs[:, 1], len(ids))
    require(np.all(np.diff(code) > 0) and len(roles) == len(code), "Repair missing-pair order differs")
    result["code"] = code
    result["roles"] = roles
    return result, manifest


def _load_rare(desc, chromosomes, ids, resolver, limits, label):
    path = resolver.pin(desc, limits["max_input_bytes"], label)
    with zipfile.ZipFile(path) as archive:
        names = {member.filename.removesuffix(".npy") for member in archive.infolist()}
    require(RARE_ARRAYS | {"samples", "sample_hash", "chromosomes", "kind"} <= names,
            "Rare count array inventory differs: " + label)
    with np.load(path, allow_pickle=False) as archive:
        require(archive["samples"].tolist() == ids and str(archive["sample_hash"]) == _sample_hash(ids) and
                [str(x) for x in archive["chromosomes"].tolist()] == chromosomes and
                str(archive["kind"]) == "rare", "Rare count identity differs: " + label)
        result = {name: archive[name].copy() for name in RARE_ARRAYS}
    n = len(ids)
    require(result["I"].shape == result["U"].shape == result["Q"].shape == (n, n) and
            result["individual_carrier"].shape == result["individual_callable"].shape == (n,) and
            np.ndim(result["n_sites"]) == 0 and
            all(np.array_equal(result[name], result[name].T) for name in ("I", "U", "Q")),
            "Rare count dimensions/symmetry differ: " + label)
    return result


def _paired_common(target_code, aggregate, repair, role_code):
    positions = np.searchsorted(aggregate["code"], target_code)
    old = positions < len(aggregate["code"])
    old[old] &= aggregate["code"][positions[old]] == target_code[old]
    rpos = np.searchsorted(repair["code"], target_code[~old])
    require(np.all(rpos < len(repair["code"])) and
            np.array_equal(repair["code"][rpos], target_code[~old]) and
            np.all(repair["roles"][rpos] == role_code),
            "Target pair coverage is neither original nor exact v4 repair for its role")
    numerator = np.empty(len(target_code), np.float64)
    joint = np.empty(len(target_code), np.int64)
    numerator[old] = aggregate["common_numerator"][positions[old]]
    joint[old] = aggregate["common_joint"][positions[old]]
    numerator[~old] = repair["common_numerator_missing"][rpos]
    joint[~old] = repair["common_joint_missing"][rpos]
    return numerator, joint, int((~old).sum())


def _raw_role_features(left, right, code, common, repair, rare, common_sites, rare_sites, role_code):
    numerator, joint, repaired = _paired_common(code, common, repair, role_code)
    i, u, q = rare["I"][left, right], rare["U"][left, right], rare["Q"][left, right]
    require(np.all((0 <= i) & (i <= u) & (u <= q) & (q <= rare_sites)),
            "Rare I/U/Q invariant differs after exact target subtraction")
    raw = np.column_stack((_ratio(numerator, joint), joint / common_sites,
                           q / rare_sites, _ratio(i, u)))
    return raw, repaired


def solver_operator(design, scientific_weights, center, scale):
    """Return centered/scaled Z without embedding scientific row weights.

    ``PairDesign.operator`` ordinarily multiplies rows by sqrt(weights).  Unit
    weights make that multiplication the identity.  The reviewed scientific
    weights are returned normalized for the caller and are passed only to
    ``solve_interval_ridge``.
    """
    weights = check_weights(scientific_weights, design.shape[0])
    normalized = weights / weights.sum(dtype=np.float64)
    operator = design.operator(np.ones(design.shape[0], dtype=np.float64), center, scale)
    return operator, normalized


def solve_with_scientific_weights(design, lower, upper, scientific_weights, center, scale, **solver):
    operator, normalized = solver_operator(design, scientific_weights, center, scale)
    result = solve_interval_ridge(operator, lower, upper, scientific_weights, **solver)
    return result, {"row_weighting_in_pair_operator": "UNIT_IDENTITY",
                    "row_weighting_in_interval_solver": "NORMALIZED_ONCE",
                    "normalized_weights_sha256": array_sha256(normalized)}


def _make_design(points, left, right, extra, resources):
    return PairDesign(points, left, right, extra=extra,
                      rows_per_chunk=resources["rows_per_chunk"],
                      max_working_mib=resources["max_working_mib"],
                      backend=resources.get("operator_backend", "factorized"),
                      preprocessing_backend="moments",
                      workers=resources["workers"])


def materialize(request_path, request_sha256, output, *, staged_root=None):
    require(DESIGN_COLUMNS == EXPECTED_DESIGN_COLUMNS, "Internal reviewed design dimensions differ")
    request_path = Path(request_path).resolve()
    require(sha256(request_path) == request_sha256, "K04 request SHA256 differs")
    request = load_json(request_path)
    required = {"schema", "operation", "scientific_scope", "implementation_sha256",
                "scientific_schema", "target_chromosome", "ancestry", "source_contract",
                "persons", "common_axis", "common_repair", "partials", "rare_global",
                "rare_target", "targets", "resources"}
    require(set(request) == required and request["schema"] == REQUEST_SCHEMA and
            request["operation"] == OPERATION and
            request["scientific_scope"] == "CONDITIONAL_GNOMIX_COMMON_DISCORDANCE_OUTER_INTERVAL" and
            request["implementation_sha256"] == implementation_sha256() and
            request["ancestry"] in {"AFR", "EUR", "NAM"},
            "Exact reviewed K04 request/current implementation required")
    resources = request["resources"]
    require(set(resources) in (LIMIT_FIELDS, LIMIT_FIELDS | {"operator_backend"}) and
            all(type(resources[key]) is int and resources[key] > 0 for key in LIMIT_FIELDS) and
            resources.get("operator_backend", "factorized") in OPERATOR_BACKENDS,
            "Explicit positive K04 resource limits required")
    resolver = PinResolver(staged_root)
    scientific = resolver.pin(request["scientific_schema"], resources["max_metadata_bytes"], "scientific schema")
    schema = load_json(scientific)
    require(schema.get("schema") == "r02_k04_scientific_design_v1" and
            schema.get("human_launch_admitted") is False and schema.get("evaluate_allowed") is False and
            {arm: schema["arms"][arm]["n_design_columns"] for arm in ARMS} == EXPECTED_DESIGN_COLUMNS and
            schema.get("parameter_changes_this_delivery") == [],
            "Reviewed scientific schema or non-admission state differs")
    target = _chromosome(request["target_chromosome"])
    contract = resolver.pin(request["source_contract"], resources["max_metadata_bytes"], "source contract")
    persons_path = resolver.pin(request["persons"], resources["max_metadata_bytes"], "persons")
    persons = _read_persons(persons_path); ids = [row["sample_id"] for row in persons]
    contract_value = load_json(contract)
    require(contract_value.get("roles_state") == "EXPLICITLY_FROZEN" and
            contract_value.get("metadata_sha256", {}).get("persons.tsv") == request["persons"]["sha256"],
            "Source contract/person roster/role binding differs")
    scores, axis_manifest, axis_chromosomes = _load_axis(request["common_axis"], target, ids, resolver, resources)
    require(axis_manifest.get("source_contract_sha256") == sha256(contract),
            "Common axis/source contract binding differs")
    repair, repair_manifest = _load_repair(request["common_repair"], target, ids, resolver, resources)
    expected = validate_exclusion(target, request["partials"].keys())
    require(expected == axis_chromosomes, "Common axis and partial exclusion territories differ")
    aggregate = None
    for chrom in expected:
        code, values = _load_partial(request["partials"][chrom], chrom, ids, sha256(contract), resolver, resources)
        if aggregate is None:
            aggregate = {"code": code, "common_numerator": values["common_numerator"].astype(np.float64),
                         "common_joint": values["common_joint"].astype(np.int64)}
        else:
            require(np.array_equal(code, aggregate["code"]), "Original partial pair order differs")
            aggregate["common_numerator"] += values["common_numerator"]
            aggregate["common_joint"] += values["common_joint"]
    global_rare = _load_rare(request["rare_global"], [str(i) for i in range(1, 23)], ids,
                             resolver, resources, "rare global")
    target_rare = _load_rare(request["rare_target"], [target], ids, resolver, resources, "rare target")
    rare = {name: np.asarray(global_rare[name], dtype=np.int64) -
                  np.asarray(target_rare[name], dtype=np.int64) for name in RARE_ARRAYS}
    del global_rare, target_rare
    require(all(np.all(value >= 0) for value in rare.values()), "Rare global-minus-target count is negative")
    common_sites = int(repair["common_sites"]); rare_sites = int(rare["n_sites"])
    require(common_sites > 0 and rare_sites > 0 and
            np.all((repair["common_called"] >= 0) & (repair["common_called"] <= common_sites)) and
            np.all((rare["individual_callable"] >= 0) & (rare["individual_callable"] <= rare_sites)),
            "Non-target catalogue denominators/counts differ")
    base_points = np.column_stack((scores,
        _ratio(repair["common_zsum"], repair["common_called"]),
        repair["common_called"] / common_sites,
        rare["individual_callable"] / rare_sites))
    base_points[:, N_AXES] = np.nan_to_num(base_points[:, N_AXES], nan=0.0)
    quantity_points = np.column_stack((base_points, rare["individual_carrier"], repair["rare_copies"]))
    require(base_points.shape[1] == len(POINT_COLUMNS["C"]) and
            quantity_points.shape[1] == len(POINT_COLUMNS["CQ"]) and
            np.isfinite(base_points).all() and np.isfinite(quantity_points).all(),
            "Reviewed individual predictor columns differ")
    require(set(request["targets"]) == {"FIT", "SELECT"}, "Exactly FIT and SELECT targets required")
    targets = {role: _load_target(role, request["targets"][role], target, ids, resolver, resources)
               for role in ("FIT", "SELECT")}
    require(targets["FIT"][3] == targets["SELECT"][3] == expected,
            "Target and predictor chromosome exclusions differ")
    raw = {}; repaired_counts = {}; component_pair_counts = {}
    for role, role_code in (("FIT", 0), ("SELECT", 1)):
        arrays, code = targets[role][1], targets[role][2]
        require(np.all([persons[int(i)]["role"] == role for i in arrays["person_i"]]) and
                np.all([persons[int(i)]["role"] == role for i in arrays["person_j"]]),
                "Target pair crosses or differs from frozen role")
        within = sum(persons[int(i)]["component"] == persons[int(j)]["component"]
                     for i, j in zip(arrays["person_i"], arrays["person_j"]))
        component_pair_counts[role] = {"eligible": len(code), "within_component": within,
                                       "between_component": len(code) - within}
        raw[role], repaired_counts[role] = _raw_role_features(
            arrays["person_i"], arrays["person_j"], code, aggregate, repair, rare,
            common_sites, rare_sites, role_code)
    means, fit_values, select_values = _fit_impute(raw["FIT"], raw["SELECT"])
    role_values = {"FIT": fit_values, "SELECT": select_values}
    require(resolver.total_bytes <= resources["max_input_bytes"], "Combined K04 input byte bound exceeded")
    memory_admission = (scores.nbytes + base_points.nbytes + quantity_points.nbytes +
                        sum(np.asarray(value).nbytes for value in rare.values()) +
                        sum(np.asarray(value).nbytes for value in repair.values() if isinstance(value, np.ndarray)) +
                        sum(np.asarray(value).nbytes for value in aggregate.values()) +
                        sum(np.asarray(value).nbytes for role in role_values for value in role_values[role]))
    require(memory_admission <= resources["max_memory_bytes"], "K04 resident-array memory admission failed")
    output = Path(output)
    require(not output.exists() and output.parent.is_dir(), "Fresh K04 output directory required")
    require(shutil.disk_usage(output.parent).free >= resources["min_free_bytes"], "K04 disk reserve exhausted")
    output.mkdir(mode=0o700)
    outputs = {}
    for name, array in (("points_C.npy", base_points), ("points_CQ.npy", quantity_points)):
        np.save(output / name, np.asarray(array, dtype="<f8"), allow_pickle=False)
        outputs[name] = sha256(output / name)
    preprocessing = {}
    for role in ("FIT", "SELECT"):
        role_dir = output / role; role_dir.mkdir(mode=0o700)
        values, missing = role_values[role]
        extra_full = np.column_stack((values[:, :3], missing[:, :3], missing[:, 3], values[:, 3]))
        require(extra_full.shape[1] == len(EXTRA_COLUMNS["CQR"]) and
                np.array_equal(extra_full[:, 6], (raw[role][:, 3] != raw[role][:, 3]).astype(float)),
                "Jaccard unavailable indicator differs from U=0/missing")
        for name, array in (("person_i.npy", targets[role][1]["person_i"]),
                            ("person_j.npy", targets[role][1]["person_j"]),
                            ("lower.npy", targets[role][1]["lower"]),
                            ("upper.npy", targets[role][1]["upper"]),
                            ("opportunities.npy", targets[role][1]["opportunities"]),
                            ("extra_full.npy", extra_full)):
            np.save(role_dir / name, array, allow_pickle=False)
            outputs[str((role_dir / name).relative_to(output))] = sha256(role_dir / name)
    fit_arrays = targets["FIT"][1]
    equal_weights = np.ones(len(fit_arrays["person_i"]), dtype=np.float64)
    for arm in ARMS:
        points = base_points if arm == "C" else quantity_points
        extra = role_values["FIT"][0]
        fit_extra_full = np.column_stack((extra[:, :3], role_values["FIT"][1][:, :3],
                                          role_values["FIT"][1][:, 3], extra[:, 3]))
        fit_extra = fit_extra_full[:, :len(EXTRA_COLUMNS[arm])]
        design = _make_design(points, fit_arrays["person_i"], fit_arrays["person_j"], fit_extra, resources)
        require(design.shape[1] == EXPECTED_DESIGN_COLUMNS[arm], "PairDesign arm width differs")
        center, scale = design.fit_preprocessing(equal_weights)
        quadratic_sum = np.zeros(design.shape[1], dtype=np.float64)
        for _, values in design.chunks():
            values -= center
            values /= scale
            quadratic_sum += np.sum(values*values, axis=0, dtype=np.float64)
        quadratic_diagonal = quadratic_sum / design.shape[0]
        require(quadratic_diagonal.shape == center.shape and np.isfinite(quadratic_diagonal).all() and
                np.all(quadratic_diagonal >= 0), "FIT quadratic diagonal differs")
        for kind, array in (("center", center), ("scale", scale),
                            ("quadratic_diagonal", quadratic_diagonal)):
            name = f"preprocessing_{arm}_{kind}.npy"
            np.save(output / name, np.asarray(array, dtype="<f8"), allow_pickle=False)
            outputs[name] = sha256(output / name)
        preprocessing[arm] = {"center_sha256": array_sha256(center), "scale_sha256": array_sha256(scale),
                              "quadratic_diagonal_sha256": array_sha256(quadratic_diagonal),
                              "L": float(np.max(quadratic_diagonal)),
                              "U": float(math.fsum(map(float, quadratic_diagonal))),
                              "n_columns": design.shape[1], "fit_weighting": "EQUAL_ELIGIBLE_PAIRS"}
    total_output = sum((output / name).stat().st_size for name in outputs)
    require(total_output <= resources["max_output_bytes"], "K04 output byte bound exceeded")
    shared_rows = {role: array_sha256(np.column_stack((targets[role][1]["person_i"],
                                                       targets[role][1]["person_j"])))
                   for role in ("FIT", "SELECT")}
    manifest = dict(schema=OUTPUT_SCHEMA,
        status="COMPLETE_REVIEWED_MODEL_INPUTS_NOT_MODEL_SELECTION_OR_BIOLOGICAL_VALIDATION",
        request_sha256=request_sha256, implementation_sha256=implementation_sha256(),
        scientific_schema_sha256=sha256(scientific), target_chromosome=target,
        ancestry=request["ancestry"], predictor_chromosomes=expected,
        target_chromosome_excluded_from_every_channel=True,
        cohort=dict(n_people=len(ids), role_counts=ROLES, sample_order_sha256=_sample_hash(ids)),
        arms={arm: dict(points=POINT_COLUMNS[arm], extra=EXTRA_COLUMNS[arm],
                        n_design_columns=EXPECTED_DESIGN_COLUMNS[arm],
                        row_identity_sha256=shared_rows, preprocessing=preprocessing[arm]) for arm in ARMS},
        points_storage={"C": "points_C.npy", "CQ": "points_CQ.npy", "CQR": "points_CQ.npy"},
        extra_storage={role: dict(path=f"{role}/extra_full.npy",
            arm_prefix_columns={arm: len(EXTRA_COLUMNS[arm]) for arm in ARMS}) for role in ("FIT", "SELECT")},
        pair_imputation_means=means.tolist(), pair_imputation="UNWEIGHTED_FINITE_FIT_MEAN",
        paired_missing_indicators_preserved=True, jaccard_zero_distinct_from_unavailable=True,
        rows_identical_across_arms=True, weights_identical_across_arms=True,
        solver_bridge=dict(pair_design_operator_row_weights="UNIT_IDENTITY",
                           interval_solver_weights="EQUAL_ELIGIBLE_PAIRS_NORMALIZED_IN_SOLVER",
                           double_weighting=False,
                           operator_backend=resources.get("operator_backend", "factorized")),
        common_axis_source_sha256=request["common_axis"], common_repair_source_sha256=request["common_repair"],
        rare_sources_sha256={"global": request["rare_global"], "target": request["rare_target"]},
        target_sources_sha256={role: request["targets"][role] for role in ("FIT", "SELECT")},
        source_files_sha256=resolver.used, common_sites=common_sites, rare_sites=rare_sites,
        repaired_target_pairs=repaired_counts, component_pair_counts=component_pair_counts,
        input_bytes=resolver.total_bytes, memory_admission_bytes=memory_admission,
        output_sha256=outputs, output_bytes=total_output,
        evaluate_target_read=False, evaluate_allowed=False, penalty_chosen=False,
        model_fit=False, model_selection=False, biological_validation=False,
        limitations=["Rows are dependent pairs and do not establish population generalization.",
                     "Targets are conditional on uncalibrated Gnomix hard calls.",
                     "CQR adds global rare Jaccard, not rare haplotype, local context or rare ancestry assignment."])
    write_json(output / "manifest.json", manifest)
    require(sha256(request_path) == request_sha256 and request["implementation_sha256"] == implementation_sha256(),
            "Request or implementation changed during materialization")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--staged-root")
    args = parser.parse_args(); os.umask(0o077)
    result = materialize(args.request, args.request_sha256, args.output,
                         staged_root=args.staged_root)
    print(json.dumps({key: result[key] for key in ("schema", "status", "target_chromosome", "ancestry")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
