#!/usr/bin/env python3
"""Materialize the historical B0/B1/B2 compatibility predictors.

This is an engineering compatibility producer, not the admitted K04 C/CQ/CQR
producer. Frozen K04 requires every eligible common axis; historical B0 has
only a scalar common summary and therefore cannot be relabelled C. A request
contains both FIT and SELECT targets so
that every missing-value mean is learned from FIT and then frozen for SELECT.
The program never reads an EVALUATE target and never chooses a penalty.

Each source partial is one complete non-target chromosome produced by
``r02_structure_materialize.py``.  This intentionally requires its original
``data.npz``: the later rare-count cache has carrier-site counts but lacks the
source-minor copy count required by B1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np


SCHEMA = "r02_k04_legacy_b012_predictor_inputs_request_v1"
OUTPUT_SCHEMA = "r02_k04_legacy_b012_predictor_inputs_v1"
OPERATION = "MATERIALIZE_LEGACY_B0_B1_B2_COMPATIBILITY_ONLY"
ROLES = {"FIT": 1460, "SELECT": 476, "EVALUATE": 683}
ARMS = ("B0", "B1", "B2")
SOURCE_ARRAYS = {
    "pairs", "common_numerator", "common_joint", "common_zsum",
    "common_called", "common_sites", "rare_I", "rare_U", "rare_Q",
    "rare_sites", "rare_carriers", "rare_called", "rare_copies",
    "common_rare_overlap_sites", "sample_ids",
}
TARGET_FILES = ("person_i.npy", "person_j.npy")
LIMIT_FIELDS = {
    "max_metadata_bytes", "max_input_bytes", "max_array_bytes",
    "max_memory_bytes", "max_output_bytes", "min_free_bytes",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps(list(value.shape)).encode())
    digest.update(value.dtype.str.encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def implementation_sha256():
    return sha256(__file__)


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def pin(desc, maximum, label):
    require(isinstance(desc, dict) and set(desc) == {"path", "sha256"},
            "Exact path/SHA256 descriptor required: " + label)
    path = Path(desc["path"])
    require(path.is_absolute() and path.is_file() and path.resolve() == path and
            isinstance(desc["sha256"], str) and len(desc["sha256"]) == 64 and
            all(c in "0123456789abcdef" for c in desc["sha256"]) and
            path.stat().st_size <= maximum and sha256(path) == desc["sha256"],
            "Pinned input changed, is noncanonical or exceeds bound: " + label)
    return path


def npz_bytes(path, maximum):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(members and len({m.filename for m in members}) == len(members) and
                all(not m.is_dir() and not (m.flag_bits & 1) for m in members),
                "Invalid NPZ inventory")
        total = sum(m.file_size for m in members)
    require(total <= maximum, "NPZ decompression bound exceeded")
    return total


def chromosome(value):
    text = str(value).removeprefix("chr")
    require(text.isdigit() and 1 <= int(text) <= 22, "Autosome 1..22 required")
    return str(int(text))


def read_persons(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(len(rows) == sum(ROLES.values()) and set(rows[0]) >=
            {"sample_id", "component", "role"}, "Frozen person roster differs")
    require(len({r["sample_id"] for r in rows}) == len(rows) and
            {role: sum(r["role"] == role for r in rows) for role in ROLES} == ROLES,
            "Frozen sample IDs or role counts differ")
    return rows


def sample_hash(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def load_target(desc, role, target_chromosome, ids_hash, limits):
    require(set(desc) == {"directory", "manifest_sha256"} and
            Path(desc["directory"]).is_absolute(), "Exact target descriptor required")
    root = Path(desc["directory"]).resolve()
    manifest_path = pin({"path": str(root / "manifest.json"),
                         "sha256": desc["manifest_sha256"]},
                        limits["max_metadata_bytes"], role + " target manifest")
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == "r02_k04_interval_target_v1" and
            manifest.get("status") == "COMPLETE_SOLVER_TARGET_ADAPTER_NOT_MODEL_FIT" and
            manifest.get("role") == role and
            chromosome(manifest.get("chromosome")) == target_chromosome and
            manifest.get("sample_order_sha256") == ids_hash and
            manifest.get("evaluate_pairs_materialized") is False and
            manifest.get("predictor_chromosomes") ==
            [str(i) for i in range(1, 23) if str(i) != target_chromosome],
            "K04 target role/chromosome/cohort/exclusion differs")
    require(set(manifest.get("outputs_sha256", {})) >= set(TARGET_FILES),
            "K04 target pair arrays missing")
    arrays = []
    for name in TARGET_FILES:
        path = pin({"path": str(root / name), "sha256": manifest["outputs_sha256"][name]},
                   limits["max_input_bytes"], role + " target " + name)
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        require(value.ndim == 1 and value.dtype.kind == "u", "Invalid K04 target pair array")
        arrays.append(value)
    left, right = arrays
    require(left.shape == right.shape == (manifest["n_pairs"],) and
            np.all(left < right), "K04 target pair dimensions/order differ")
    encoded = left.astype(np.uint64) * np.uint64(sum(ROLES.values())) + right
    require(np.all(encoded[1:] > encoded[:-1]), "K04 target pairs must be unique lexicographic")
    return manifest_path, manifest, left, right, encoded


def load_partial(desc, expected_chromosome, expected_end, ids,
                 source_contract_sha256, limits):
    require(set(desc) == {"directory", "manifest_sha256"} and
            Path(desc["directory"]).is_absolute(), "Exact partial descriptor required")
    root = Path(desc["directory"]).resolve()
    manifest_path = pin({"path": str(root / "manifest.json"),
                         "sha256": desc["manifest_sha256"]},
                        limits["max_metadata_bytes"], "source partial manifest")
    manifest = load_json(manifest_path)
    request = manifest.get("request", {})
    intervals = request.get("intervals")
    require(manifest.get("schema") == "r02_structure_materialize_partial_v1" and
            manifest.get("stage") == "COMPLETE_SLICE" and
            manifest.get("contract_sha256") == source_contract_sha256 and
            manifest.get("selection_used") == "FIT_ONLY" and
            manifest.get("evaluated_outcomes_for_selection") is False and
            manifest.get("roles_state") == "EXPLICITLY_FROZEN" and
            request.get("kind") == "features" and isinstance(intervals, list) and
            len(intervals) == 1 and isinstance(intervals[0], list) and
            len(intervals[0]) == 2 and intervals[0] == [1, expected_end] and
            chromosome(request.get("chromosome")) == expected_chromosome and
            manifest.get("source_intervals") == [[expected_chromosome, 1, intervals[0][1]]],
            "Complete full-chromosome FIT-selected source partial required")
    require(set(manifest.get("outputs_sha256", {})) >= {"data.npz"},
            "Original partial data.npz with rare copies required")
    data_path = pin({"path": str(root / "data.npz"),
                     "sha256": manifest["outputs_sha256"]["data.npz"]},
                    limits["max_input_bytes"], "source partial data")
    npz_bytes(data_path, limits["max_array_bytes"])
    with np.load(data_path, allow_pickle=False) as archive:
        require(set(archive.files) == SOURCE_ARRAYS and archive["sample_ids"].tolist() == ids,
                "Original partial array inventory or roster differs")
        values = {name: archive[name].copy() for name in SOURCE_ARRAYS - {"sample_ids"}}
    pairs = values["pairs"]
    require(pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in "iu" and
            np.all(pairs[:, 0] < pairs[:, 1]), "Source pair keys differ")
    code = pairs[:, 0].astype(np.uint64) * np.uint64(len(ids)) + pairs[:, 1]
    require(np.all(code[1:] > code[:-1]), "Source pairs must be unique lexicographic")
    require(all(values[k].shape == (len(pairs),) for k in
                ("common_numerator", "common_joint", "rare_I", "rare_U", "rare_Q")) and
            all(values[k].shape == (len(ids),) for k in
                ("common_zsum", "common_called", "rare_carriers", "rare_called", "rare_copies")),
            "Source sufficient-statistic dimensions differ")
    require(np.all((0 <= values["rare_I"]) & (values["rare_I"] <= values["rare_U"])) and
            np.all(values["rare_U"] <= values["rare_Q"]), "Rare I/U/Q invariant differs")
    return manifest_path, data_path, code, values


def _ratio(numerator, denominator):
    result = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=np.asarray(denominator) > 0)
    return result


def _fit_impute(fit, select):
    fit = np.asarray(fit, dtype=np.float64)
    select = np.asarray(select, dtype=np.float64)
    require(fit.ndim == select.ndim == 2 and fit.shape[1] == select.shape[1],
            "FIT/SELECT paired feature width differs")
    observed = np.isfinite(fit)
    require(np.all(observed.any(axis=0)), "A paired predictor has no FIT observation")
    means = np.asarray([fit[observed[:, k], k].mean() for k in range(fit.shape[1])])
    def apply(value):
        missing = ~np.isfinite(value)
        out = np.where(missing, means, value)
        require(np.isfinite(out).all(), "Nonfinite predictor after FIT-only imputation")
        return out, missing.astype(np.float64)
    return means, apply(fit), apply(select)


def _aligned_positions(source_code, target_code, role):
    positions = np.searchsorted(source_code, target_code)
    require(np.all(positions < len(source_code)) and
            np.array_equal(source_code[positions], target_code),
            "A supported K04 target pair is absent from original B0/B1/B2 source coverage: " + role)
    return positions


def _paired_arms(values, missing):
    values, missing = np.asarray(values), np.asarray(missing)
    require(values.ndim == missing.ndim == 2 and values.shape == missing.shape and
            values.shape[1] == 4 and np.isfinite(values).all() and
            np.all((missing == 0) | (missing == 1)), "Invalid imputed paired predictors/masks")
    return {
        "B0": np.column_stack((values[:, :3], missing[:, :3])),
        "B1": np.column_stack((values[:, :3], missing[:, :3], missing[:, 3])),
        "B2": np.column_stack((values[:, :3], missing[:, :3], missing[:, 3], values[:, 3])),
    }


def materialize(request_path, request_sha256, output, *, allow_legacy_compatibility=False):
    require(allow_legacy_compatibility is True,
            "Legacy B0/B1/B2 is not admitted as K04 C/CQ/CQR; explicit compatibility opt-in required")
    request_path = Path(request_path).resolve()
    require(sha256(request_path) == request_sha256, "Predictor request SHA256 differs")
    request = load_json(request_path)
    require(set(request) == {"schema", "operation", "scientific_scope", "implementation_sha256",
            "target_chromosome", "source_contract", "persons", "partials", "targets", "resources"} and
            request["schema"] == SCHEMA and request["operation"] == OPERATION and
            request["scientific_scope"] == "LEGACY_B0_B1_B2_ENGINEERING_COMPATIBILITY_NOT_K04_MODEL" and
            request["implementation_sha256"] == implementation_sha256(),
            "Exact K04 predictor request/current implementation required")
    limits = request["resources"]
    require(set(limits) == LIMIT_FIELDS and all(type(v) is int and v > 0 for v in limits.values()),
            "Explicit positive predictor resource limits required")
    target_chromosome = chromosome(request["target_chromosome"])
    contract_path = pin(request["source_contract"], limits["max_metadata_bytes"], "source contract")
    contract = load_json(contract_path)
    require(contract.get("roles_state") == "EXPLICITLY_FROZEN" and
            contract.get("metadata_sha256", {}).get("persons.tsv") == request["persons"]["sha256"] and
            isinstance(contract.get("sources"), dict),
            "Frozen source roles/person binding differs")
    persons_path = pin(request["persons"], limits["max_metadata_bytes"], "persons")
    persons = read_persons(persons_path); ids = [r["sample_id"] for r in persons]
    ids_hash = sample_hash(ids)
    require(set(request["targets"]) == {"FIT", "SELECT"}, "FIT and SELECT targets required; EVALUATE forbidden")
    targets = {role: load_target(request["targets"][role], role, target_chromosome, ids_hash, limits)
               for role in ("FIT", "SELECT")}
    partials = request["partials"]
    expected = [str(i) for i in range(1, 23) if str(i) != target_chromosome]
    require(isinstance(partials, dict) and list(partials) == expected,
            "Exactly the ordered 21 non-target chromosome partials required")

    aggregate = None; source_code = None; sources = {}
    decoded_peak = 0; input_bytes = 0
    for chrom in expected:
        manifest_path, data_path, code, values = load_partial(
            partials[chrom], chrom, contract["sources"][chrom]["length_bp"], ids,
            sha256(contract_path), limits)
        input_bytes += manifest_path.stat().st_size + data_path.stat().st_size
        require(input_bytes <= limits["max_input_bytes"], "Combined source byte bound exceeded")
        if aggregate is None:
            source_code = code
            aggregate = {k: np.asarray(v, dtype=np.float64 if k in
                {"common_numerator", "common_zsum"} else np.int64).copy()
                for k, v in values.items() if k not in {"pairs", "common_rare_overlap_sites"}}
        else:
            require(np.array_equal(code, source_code), "Pair coverage/order differs among source chromosomes")
            for key in aggregate:
                aggregate[key] += values[key]
        decoded_peak = max(decoded_peak, sum(v.nbytes for v in values.values()))
        sources[chrom] = {"manifest": sha256(manifest_path), "data": sha256(data_path)}
    require(target_chromosome not in sources and list(sources) == expected,
            "Whole target chromosome was not excluded from every predictor channel")

    n = len(ids); memory_bound = decoded_peak + sum(v.nbytes for v in aggregate.values())
    memory_bound += sum(int(t[1]["n_pairs"]) * 8 * 24 for t in targets.values())
    require(memory_bound <= limits["max_memory_bytes"], "Predictor memory admission failed")
    common_sites = int(aggregate["common_sites"]); rare_sites = int(aggregate["rare_sites"])
    require(common_sites > 0 and rare_sites > 0, "Nonempty non-target common and rare catalogues required")
    x0 = np.column_stack((_ratio(aggregate["common_zsum"], aggregate["common_called"]),
                          aggregate["common_called"] / common_sites,
                          aggregate["rare_called"] / rare_sites))
    # The original v3 materializer pairs a zero common mean with its explicit
    # callability column; it never turns an unknown rare identity into zero.
    x0[:, 0] = np.nan_to_num(x0[:, 0], nan=0.0)
    x1 = np.column_stack((x0, aggregate["rare_carriers"], aggregate["rare_copies"]))
    require(np.isfinite(x0).all() and np.isfinite(x1).all(), "Nonfinite individual predictor")
    source_pos = {role: _aligned_positions(source_code, target[4], role)
                  for role, target in targets.items()}

    raw = {}
    for role, positions in source_pos.items():
        common = _ratio(aggregate["common_numerator"][positions], aggregate["common_joint"][positions])
        joint_common = aggregate["common_joint"][positions] / common_sites
        joint_rare = aggregate["rare_Q"][positions] / rare_sites
        jaccard = _ratio(aggregate["rare_I"][positions], aggregate["rare_U"][positions])
        raw[role] = np.column_stack((common, joint_common, joint_rare, jaccard))
    means, fit_values, select_values = _fit_impute(raw["FIT"], raw["SELECT"])

    output = Path(output)
    require(not output.exists(), "Predictor output exists; no overwrite allowed")
    require(shutil.disk_usage(output.parent).free >= limits["min_free_bytes"], "Predictor disk reserve exhausted")
    output.mkdir(mode=0o700)
    outputs = {}
    for role, ((values, missing), target) in {
            "FIT": (fit_values, targets["FIT"]), "SELECT": (select_values, targets["SELECT"])}.items():
        role_dir = output / role; role_dir.mkdir(mode=0o700)
        points = {"B0": x0, "B1": x1, "B2": x1}
        # B0 has three paired values and their technical missing indicators.
        # B1 adds the rare-pattern availability indicator.  B2 adds J itself.
        paired = _paired_arms(values, missing)
        for arm in ARMS:
            for kind, array in (("points", points[arm]), ("paired", paired[arm])):
                path = role_dir / f"{arm}_{kind}.npy"
                np.save(path, np.asarray(array, dtype="<f8"), allow_pickle=False)
                outputs[str(path.relative_to(output))] = sha256(path)
        for name, value in (("person_i", target[2]), ("person_j", target[3])):
            path = role_dir / (name + ".npy")
            np.save(path, np.asarray(value, dtype="<u4"), allow_pickle=False)
            outputs[str(path.relative_to(output))] = sha256(path)
    total_output = sum((output / name).stat().st_size for name in outputs)
    require(total_output <= limits["max_output_bytes"], "Predictor output bound exceeded")
    manifest = dict(schema=OUTPUT_SCHEMA,
        status="COMPLETE_LEGACY_COMPATIBILITY_NOT_ADMITTED_FOR_K04_FULL_COMPARISON",
        request_sha256=request_sha256, implementation_sha256=implementation_sha256(),
        target_chromosome=target_chromosome, predictor_chromosomes=expected,
        target_chromosome_excluded_from_all_features=True,
        cohort=dict(n_people=n, role_counts=ROLES, sample_order_sha256=ids_hash),
        arms=["B0", "B1", "B2"], k04_arm_mapping_admitted=False,
        k04_non_admission_reason="B0 retains one common summary; frozen C requires every eligible common axis",
        pair_imputation_means=means.tolist(),
        columns=dict(
            individual_B0=["common_mean_standardized_dosage", "common_callable_fraction", "rare_callable_fraction"],
            individual_B1_B2=["individual_B0", "rare_carrier_sites", "rare_copy_count"],
            paired_B0=["common_GRM", "common_joint_fraction", "rare_joint_fraction", "missing_indicators_for_preceding_3"],
            paired_B1=["paired_B0", "rare_Jaccard_missing_indicator"],
            paired_B2=["paired_B1", "rare_Jaccard"]),
        source_contract_sha256=sha256(contract_path), source_partials_sha256=sources,
        targets={role: sha256(value[0]) for role, value in targets.items()},
        n_pairs={role: int(value[1]["n_pairs"]) for role, value in targets.items()},
        source_pair_count=len(source_code), common_sites=common_sites, rare_sites=rare_sites,
        memory_admission_bytes=memory_bound, output_bytes=total_output,
        output_sha256=outputs, evaluate_target_read=False, penalty_chosen=False,
        biological_validation=False, k04_full_comparison_admitted=False,
        limitations=["Rows are dependent pairs and do not establish population generalization.",
                     "Rare ascertainment remains fixed in the source cohort.",
                     "Targets remain conditional on uncalibrated Gnomix hard calls."])
    write_json(output / "manifest.json", manifest)
    require(sha256(request_path) == request_sha256 and implementation_sha256() == request["implementation_sha256"],
            "Request or implementation changed during materialization")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-legacy-compatibility", action="store_true")
    args = parser.parse_args(); os.umask(0o077)
    result = materialize(args.request, args.request_sha256, args.output,
                         allow_legacy_compatibility=args.allow_legacy_compatibility)
    print(json.dumps({k: result[k] for k in ("schema", "status", "target_chromosome", "n_pairs")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
