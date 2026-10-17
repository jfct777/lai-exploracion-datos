#!/usr/bin/env python3
"""Fit and evaluate a symmetric Ridge comparator on authenticated prepared features.

Implements PROTOCOLO_CRITERIOS_20261003 §§2–4, not a genomic materializer or a
population classifier. Input lineage is checked, not independently reconstructed
from VCFs. Statistical intervals are conditional on the selected fitted models.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import shutil
import zipfile
import importlib.util
import sys

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARMS = ("B0", "B1", "B2")
ROLES = ("FIT", "SELECT", "EVALUATE")
KINDS = {"B0": {"common", "measurement"}, "B1": {"rare_count"}, "B2": {"rare_pattern"}}
DATA_FILES = ("persons.tsv", "blocks.tsv", "markers.tsv", "targets.npz", "features.npz", "lineage.json")
FILES = (*DATA_FILES, "upstream_verification.json")
PAIR_FILES = ("pair_features.npz", "pair_lineage.json")
BASE_CHECKS = ("target_exclusion", "mask_preservation", "common_information_matched", "role_separation")
PAIR_CHECKS = ("paired_key_coverage", "paired_order_invariance", "paired_mask_preservation")
SCOPED_SCHEMA = "r02_structure_increment_v3"
BLOCK_SCHEMA = "r02_structure_prepared_block_v1"
STREAM_SCHEMA = "r02_structure_increment_v4"
SELECT_SCHEMA = 'r02_structure_select_checkpoint_v1'
SELECT_STATUS = 'SELECT_COMPLETE_AWAITING_REVIEW'
SCORE_SCHEMA = 'r02_structure_pair_score_checkpoint_v1'
SCORE_STATUS = 'FIT_SELECT_SCORES_COMPLETE'
REVIEW_SCHEMA = 'r02_structure_selection_review_v1'
NUMERICAL_SCHEMA = 'r02_structure_pair_operator_numerics_v1'
LEGACY_INDICATOR_SCOPE = "LEGACY_ALL_B2_INDICATORS_SHARED"
INDICATOR_ARMS = {"measurement": ARMS, "rare_pattern": ("B1", "B2")}


def paired_contract(contract):
    return contract["schema"] in {"r02_structure_increment_v2", SCOPED_SCHEMA, BLOCK_SCHEMA}


def verification_schema(contract):
    version = 'v3' if contract['schema'] == BLOCK_SCHEMA else contract['schema'].rsplit('_',1)[1]
    return "r02_structure_prepared_features_verification_" + version


def verification_checks(contract):
    return (BASE_CHECKS + (PAIR_CHECKS if paired_contract(contract) else ())
            + (("paired_indicator_scope",) if contract["schema"] in {SCOPED_SCHEMA,BLOCK_SCHEMA} else ()))


def validate_pair_indicator_scope(contract, lineage):
    """Separate technical availability from rare-derived availability, explicitly.

    The consumer authenticates this classification; it cannot reconstruct the
    biological cause of NaN from a tensor. The producer must audit that cause.
    """
    scope = contract.get("pair_missing_indicator_scope")
    if contract["schema"] == "r02_structure_increment_v2":
        require(scope == LEGACY_INDICATOR_SCOPE,
                "Historical v2 requires explicit LEGACY_ALL_B2_INDICATORS_SHARED declaration")
        return
    require(isinstance(scope, dict) and set(scope) == set(lineage),
            "Paired indicator scope must cover exactly every block")
    for name, arms in lineage.items():
        columns = arms["B2"]
        require(isinstance(scope[name], dict) and set(scope[name]) == {c["id"] for c in columns},
                "Paired indicator scope must cover exactly every B2 column")
        for col in columns:
            label = scope[name][col["id"]]
            require(isinstance(label, str) and label in INDICATOR_ARMS, "Unknown paired indicator scope")
            if col["kind"] in {"common", "measurement"}:
                require(label == "measurement", "Common/technical availability must use measurement scope")
            if col["kind"] == "rare_pattern":
                require(label == "rare_pattern", "Rare-pattern availability cannot enter common B0")


def data_files(contract):
    return DATA_FILES + (PAIR_FILES if paired_contract(contract) else ())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "Nonfinite JSON"))


def table(path, required):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        require(reader.fieldnames == required, f"Invalid columns: {Path(path).name}")
        result = list(reader)
    require(result and all(None not in r and all(v is not None and v != "" for v in r.values()) for r in result),
            f"Empty/malformed table: {Path(path).name}")
    return result


def assign_component_roles(persons, fractions, seed):
    """Deterministic component split proposal; never changes a supplied DNABR split.

Counts of components, not individuals, approximate the supplied proportions.
Representativeness and ancestry/batch balance require a separate review.
"""
    require(set(fractions) == set(ROLES) and all(math.isfinite(v) and v > 0 for v in fractions.values())
            and math.isclose(sum(fractions.values()), 1), "Three positive role fractions summing to one required")
    require(type(seed) is int and seed >= 0, "Invalid split seed")
    ids = [p["sample_id"] for p in persons]
    require(len(ids) == len(set(ids)) and all(p["component"] for p in persons), "Invalid persons/components")
    groups = sorted({p["component"] for p in persons},
                    key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    require(len(groups) >= 3, "At least three components required to propose three roles")
    counts = {r: 1 for r in ROLES}
    for _ in range(len(groups) - 3):
        role = max(ROLES, key=lambda r: (fractions[r] * len(groups) - counts[r], -ROLES.index(r)))
        counts[role] += 1
    mapping, start = {}, 0
    for role in ROLES:
        mapping.update({c: role for c in groups[start:start + counts[role]]})
        start += counts[role]
    return [{**p, "role": mapping[p["component"]]} for p in persons]


def validate_roles(persons):
    ids = [r["sample_id"] for r in persons]
    require(len(ids) == len(set(ids)), "Duplicate sample ID")
    groups = defaultdict(set)
    for row in persons:
        require(row["role"] in ROLES and row["component"].strip() == row["component"], "Invalid role/component")
        groups[row["component"]].add(row["role"])
    require(all(len(v) == 1 for v in groups.values()), "Related component crosses FIT/SELECT/EVALUATE")
    for role in ROLES:
        require(len({p["component"] for p in persons if p["role"] == role}) >= 2,
                f"{role} needs at least two distinct components to form between-component pairs")
    return {role: np.array([i for i, p in enumerate(persons) if p["role"] == role], dtype=int) for role in ROLES}


def overlap(a, b):
    return a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2]


def valid_interval(value):
    return (isinstance(value, list) and len(value) == 3 and isinstance(value[0], str)
            and value[0] in {str(c) for c in range(1, 23)} and type(value[1]) is int
            and type(value[2]) is int and 1 <= value[1] <= value[2])


def validate_column(col, arm, fit_ids_hash, excluded, contract, is_pair=False):
    fields = {"id", "kind", "source_intervals", "fit_sample_ids_sha256", "derivation"}
    require(set(col) == fields | ({"symmetry"} if is_pair else set()), "Invalid lineage fields")
    if is_pair:
        require(col["symmetry"] == "PAIR_ORDER_INVARIANT", "Paired derivation must be symmetric")
    require(col["kind"] in KINDS[arm], "Arm contains undeclared information")
    require(col["fit_sample_ids_sha256"] == fit_ids_hash, "Feature fit statistics not restricted to FIT")
    require(isinstance(col["derivation"], str) and len(col["derivation"].strip()) >= 10, "Feature derivation required")
    intervals = col["source_intervals"]
    require(isinstance(intervals, list) and all(valid_interval(v) for v in intervals), "Invalid source intervals")
    require(intervals or col["kind"] == "measurement", "Genetic feature without locus provenance")
    require(not any(overlap(v, excluded) for v in intervals), "Target/LD-margin leakage into feature sources")
    if contract["exclusion_scope"] == "WHOLE_CHROMOSOME":
        require(all(v[0] != excluded[0] for v in intervals), "Target chromosome leaked into features")


def validate_lineage(lineage, blocks, persons, features, contract, pair_features=None, pair_lineage=None):
    require(set(lineage) == {b["block_id"] for b in blocks}, "Feature lineage block coverage mismatch")
    if pair_features is not None:
        require(set(pair_lineage) == set(lineage), "Paired lineage block coverage mismatch")
    fit_ids_hash = hashlib.sha256(("\n".join(sorted(p["sample_id"] for p in persons if p["role"] == "FIT")) + "\n").encode()).hexdigest()
    for b in blocks:
        name, previous, previous_pair = b["block_id"], [], []
        require(set(lineage[name]) == set(ARMS), "All three arms require lineage")
        if pair_features is not None:
            require(set(pair_lineage[name]) == set(ARMS), "All three paired arms require lineage")
        excluded = [b["chromosome"], int(b["excluded_start_bp"]), int(b["excluded_end_bp"])]
        for arm in ARMS:
            columns = lineage[name][arm]
            x = features[f"{name}_{arm}"]
            require(x.ndim == 2 and x.shape[0] == len(persons) and x.shape[1] == len(columns)
                    and len(columns) >= len(previous) and np.isfinite(x).all(), "Invalid finite feature tensor/columns")
            if pair_features is None:
                require(len(columns) > len(previous), "Invalid finite feature tensor/columns")
            require(columns[:len(previous)] == previous, "B0/B1/B2 feature lineage is not nested")
            if previous:
                previous_arm = ARMS[ARMS.index(arm)-1]
                require(np.array_equal(x[:, :len(previous)], features[f"{name}_{previous_arm}"]),
                        "Common/measurement/count values changed across arms")
            require(len({c["id"] for c in columns}) == len(columns), "Duplicate feature ID")
            if arm == "B0":
                require({c["kind"] for c in columns} == {"common", "measurement"},
                        "B0 must retain common information AND measurement/missingness features")
            for col in columns[len(previous):]:
                validate_column(col, arm, fit_ids_hash, excluded, contract)
            if pair_features is not None:
                paired_columns = pair_lineage[name][arm]
                w = pair_features[name][arm]
                require(w.shape[1] == len(paired_columns), "Paired lineage column count mismatch")
                require(paired_columns[:len(previous_pair)] == previous_pair, "Paired B0/B1/B2 lineage is not nested")
                if arm != "B0":
                    prior = pair_features[name][ARMS[ARMS.index(arm)-1]]
                    require(np.array_equal(w[:, :len(previous_pair)], prior, equal_nan=True),
                            "Paired common/measurement/count values or missingness changed across arms")
                require(len(columns) + len(paired_columns) > len(previous) + len(previous_pair),
                        "Combined individual/paired arms must add information columns")
                require(len({c["id"] for c in columns + paired_columns}) == len(columns) + len(paired_columns),
                        "Duplicate combined feature ID")
                for col in paired_columns[len(previous_pair):]:
                    validate_column(col, arm, fit_ids_hash, excluded, contract, is_pair=True)
                previous_pair = paired_columns
            previous = columns


def load_pair_features(bundle, blocks, persons, contract):
    """Sparse keyed rows, not an N-by-N tensor; canonical sorted integer lookup.

    Missing values are NaN, never an implicit biological zero. Keys may arrive
    reversed, but one unordered pair can occur only once in each block.
    """
    lineage = load_json(bundle/"pair_lineage.json")
    validate_pair_indicator_scope(contract, lineage)
    n = len(persons)
    require(n*n <= np.iinfo(np.int64).max, "Paired key range exceeds int64")
    expected = {f"{b['block_id']}_{suffix}" for b in blocks for suffix in ("pairs", *ARMS)}
    result, allocated = {}, 0
    roles = np.array([p["role"] for p in persons])
    components = np.array([p["component"] for p in persons])
    with np.load(bundle/"pair_features.npz", allow_pickle=False) as z:
        require(set(z.files) == expected | {"sample_ids"}, "Unexpected paired tensor keys")
        require(z["sample_ids"].tolist() == [p["sample_id"] for p in persons], "Paired sample order mismatch")
        for b in blocks:
            name = b["block_id"]
            pairs = z[f"{name}_pairs"]
            require(pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in "iu"
                    and len(pairs) <= len(ROLES)*contract["max_pairs"], "Invalid paired keys/row budget")
            require(((pairs >= 0) & (pairs < n)).all(), "Paired key outside sample roster")
            left, right = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1])
            require((left != right).all(), "Paired diagonal is not allowed")
            require((roles[left] == roles[right]).all(), "Paired key crosses FIT/SELECT/EVALUATE")
            require((components[left] != components[right]).all(), "Paired key is within one component")
            keys = left.astype(np.int64)*n + right.astype(np.int64)
            order = np.argsort(keys)
            keys = keys[order]
            require(len(keys) < 2 or (keys[1:] != keys[:-1]).all(), "Duplicate or reversed duplicate paired key")
            result[name] = {"keys": keys, "n_people": n}
            column_ids = [c["id"] for c in lineage[name]["B2"]]
            legacy = contract["schema"] == "r02_structure_increment_v2"
            scopes = (["measurement"]*len(column_ids) if legacy else
                      [contract["pair_missing_indicator_scope"][name][c] for c in column_ids])
            result[name].update(indicator_column_ids=column_ids, indicator_scopes=scopes,
                indicator_indices={arm: [k for k, label in enumerate(scopes) if arm in INDICATOR_ARMS[label]]
                                   for arm in ARMS}, legacy_indicator_scope=legacy)
            allocated += keys.nbytes
            for arm in ARMS:
                values = np.array(z[f"{name}_{arm}"], dtype=float)
                require(values.ndim == 2 and values.shape[0] == len(keys) and not np.isinf(values).any(),
                        "Paired features must be numeric finite values or NaN with matching rows")
                allocated += values.nbytes
                require(6*allocated <= contract["max_matrix_mb"]*2**20,
                        "Paired inputs exceed conservative memory budget")
                result[name][arm] = values[order]
    return result


def validate_bundle(bundle, contract, with_pair_features=False, data_only=False):
    require(contract["schema"] in {"r02_structure_increment_v1", "r02_structure_increment_v2", SCOPED_SCHEMA, BLOCK_SCHEMA}, "Unknown contract schema")
    require(data_only or contract['schema'] != BLOCK_SCHEMA, 'Prepared data require a separate evaluation contract before fitting')
    paired = paired_contract(contract)
    required_data = data_files(contract)
    require(set(contract["sha256"]) == set(required_data) | {"upstream_verification.json"}, "Exact input files required")
    for name, digest in contract["sha256"].items():
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and sha(bundle/name) == digest, "Input SHA256 mismatch")
    require(contract["exclusion_scope"] in {"BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME"}, "Missing exclusion scope")
    if not data_only:
        for name in ("range_rationale", "ld_margin_rationale", "dependence_rationale", "capacity_comparison", "historical_exposure"):
            require(isinstance(contract.get(name), str) and len(contract[name].strip()) >= 20, f"Required scientific declaration: {name}")
        require(contract["capacity_comparison"] != "unverified", "Capacity comparison required")
    require(contract["feature_materializer_verified"] is True, "Feature materialization has not passed its independent checks")
    for key in ("materializer_code_sha256", "materializer_verification_sha256", "scientific_protocol_sha256"):
        require(isinstance(contract.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", contract[key]),
                f"Authenticated upstream artifact required: {key}")
    upstream = load_json(bundle/"upstream_verification.json")
    require(upstream.get("schema") == verification_schema(contract)
            and upstream.get("status") == "COMPLETE_FEATURE_EXCLUSION_AND_SPLIT_CHECK",
            "Missing successful upstream materialization verification")
    require(sha(bundle/"upstream_verification.json") == contract["materializer_verification_sha256"]
            and upstream.get("materializer_code_sha256") == contract["materializer_code_sha256"]
            and upstream.get("scientific_protocol_sha256") == contract["scientific_protocol_sha256"]
            and upstream.get("input_sha256") == {k: contract["sha256"][k] for k in required_data},
            "Upstream materialization receipt is not bound to this exact input bundle")
    require(upstream.get("checks") == {k: True for k in verification_checks(contract)},
            "Upstream materialization scientific checks incomplete")
    if contract["schema"] in {SCOPED_SCHEMA, BLOCK_SCHEMA}:
        require(upstream.get("pair_missing_indicator_scope") == contract.get("pair_missing_indicator_scope"),
                "Upstream paired indicator scope is not bound to this contract")
    if paired:
        require(contract.get("pair_missing_policy") == "FIT_UNWEIGHTED_MEAN_WITH_PER_COLUMN_INDICATOR",
                "Explicit paired missing-value policy required")
    if not data_only:
        require(contract["model_family"] == "SYMMETRIC_STANDARDIZED_RIDGE", "Unapproved model family")
    require(contract["rare_allele_convention"] == "SOURCE_MINOR_M02_1", "Rare allele convention is not the source minor allele")
    require(contract["unphased_rare_policy"] == "DIPLOID_NO_HAPLOTYPE_ASSIGNMENT", "Unphased raras cannot become haplotypes")
    if not data_only:
        require(contract["evaluation_claim"] == "CONDITIONAL_PREDICTIVE_INCREMENT_NOT_POPULATIONS", "Unsupported biological claim")
        alphas = contract["ridge_alphas"]
        require(isinstance(alphas, list) and alphas and len(alphas) == len(set(alphas))
                and all(type(a) in (int, float) and math.isfinite(a) and a > 0 for a in alphas), "Invalid prespecified ridge alphas")
    integer_limits = ("max_pairs", "max_matrix_mb") if data_only else ("min_fit_called", "min_pair_markers", "bootstrap_replicates", "seed", "max_pairs", "max_prediction_records", "max_matrix_mb")
    for key in integer_limits:
        require(type(contract[key]) is int and contract[key] >= (0 if key == "seed" else 1), f"Invalid {key}")
    if not data_only:
        require(contract["bootstrap_replicates"] >= 100, "At least 100 resamples needed even for numerical interval diagnostics")
    archive_budget = 0
    for name in ("targets.npz", "features.npz") + (("pair_features.npz",) if paired else ()):
        with zipfile.ZipFile(bundle/name) as archive:
            archive_bytes = sum(f.file_size for f in archive.infolist())
            require(archive_bytes <= contract["max_matrix_mb"]*2**20,
                    "Uncompressed tensor archive exceeds declared memory budget")
            archive_budget += archive_bytes*(6 if name == "pair_features.npz" else 1)
    if paired:
        require(archive_budget <= contract["max_matrix_mb"]*2**20,
                "Combined input archives exceed conservative paired memory budget")
    if not data_only:
        for key, minimum in (("min_evaluation_components", 3), ("min_evaluation_chromosomes", 2)):
            require(type(contract[key]) is int and contract[key] >= minimum, f"Invalid prespecified {key}")
    persons = table(bundle/"persons.tsv", ["sample_id", "component", "role"])
    roles = validate_roles(persons)
    blocks = table(bundle/"blocks.tsv", ["block_id", "chromosome", "start_bp", "end_bp", "excluded_start_bp", "excluded_end_bp"])
    require(len({b["block_id"] for b in blocks}) == len(blocks), "Duplicate block ID")
    for b in blocks:
        require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", b["block_id"]), "Unsafe block ID")
        interval = [b["chromosome"], int(b["start_bp"]), int(b["end_bp"])]
        require(valid_interval(interval) and 1 <= int(b["excluded_start_bp"]) <= interval[1]
                <= interval[2] <= int(b["excluded_end_bp"]), "Invalid block/margin coordinates")
    for i, b in enumerate(blocks):
        require(not any(overlap([b["chromosome"], int(b["start_bp"]), int(b["end_bp"])],
                                [c["chromosome"], int(c["start_bp"]), int(c["end_bp"])]) for c in blocks[i+1:]),
                "Target blocks overlap and would reuse markers")
    markers = table(bundle/"markers.tsv", ["variant_id", "chromosome", "position_bp", "counted_allele"])
    require(len({m["variant_id"] for m in markers}) == len(markers), "Duplicate target variant")
    require(all(m["counted_allele"] == "ALT" and valid_interval([m["chromosome"], int(m["position_bp"]), int(m["position_bp"])]) for m in markers), "Targets require diploid ALT with valid positions")
    with np.load(bundle/"targets.npz", allow_pickle=False) as z:
        require(set(z.files) == {"genotypes", "sample_ids", "variant_ids"}, "Unknown target tensor keys")
        require(z["sample_ids"].tolist() == [p["sample_id"] for p in persons], "Target sample order mismatch")
        require(z["variant_ids"].tolist() == [m["variant_id"] for m in markers], "Target variant order mismatch")
        genotypes = np.array(z["genotypes"], dtype=float)
    require(genotypes.shape == (len(persons), len(markers)) and np.isin(genotypes[np.isfinite(genotypes)], [0, 1, 2]).all()
            and not np.isinf(genotypes).any(), "Targets must be diploid GT 0/1/2 or NaN, not dosage/imputation")
    expected_keys = {f"{b['block_id']}_{arm}" for b in blocks for arm in ARMS}
    with np.load(bundle/"features.npz", allow_pickle=False) as z:
        require(set(z.files) == expected_keys | {"sample_ids"}, "Unexpected feature tensor keys")
        require(z["sample_ids"].tolist() == [p["sample_id"] for p in persons], "Feature sample order mismatch")
        features = {k: np.array(z[k], dtype=float) for k in expected_keys}
    pair_features = load_pair_features(bundle, blocks, persons, contract) if paired else None
    pair_bytes = (sum(block["keys"].nbytes + sum(block[a].nbytes for a in ARMS)
                      for block in pair_features.values()) if paired else 0)
    require(genotypes.nbytes + sum(x.nbytes for x in features.values()) + 6*pair_bytes <= contract["max_matrix_mb"] * 2**20,
            "Input arrays exceed declared memory budget")
    validate_lineage(load_json(bundle/"lineage.json"), blocks, persons, features, contract,
                     pair_features, load_json(bundle/"pair_lineage.json") if paired else None)
    result = (persons, roles, blocks, markers, genotypes, features)
    return (*result, pair_features) if with_pair_features else result


def validate_prepared_shard(bundle, contract, with_pair_features=False):
    """Authenticate feature preparation without selecting a model or its parameters."""
    require(contract.get('schema') == BLOCK_SCHEMA, 'Prepared block schema required')
    return validate_bundle(bundle, contract, with_pair_features, data_only=True)


def target_pairs(genotypes, fit_rows, role_rows, components, min_fit_called, min_pair_markers, max_pairs):
    """C_ij and joint-called denominator; p estimated from FIT only, no GT imputation."""
    require(len(role_rows)*(len(role_rows)-1)//2 <= max_pairs, "Pair count exceeds declared allocation limit")
    gt_fit = genotypes[fit_rows]
    called = np.isfinite(gt_fit).sum(axis=0)
    p = np.nansum(gt_fit, axis=0) / np.maximum(2*called, 1)
    eligible = (called >= min_fit_called) & (p > 0) & (p < 1)
    require(eligible.any(), "No eligible target markers with FIT frequency")
    gt = genotypes[np.ix_(role_rows, eligible)]
    p = p[eligible]
    z = (gt - 2*p) / np.sqrt(2*p*(1-p))
    ii, jj, yy, nn, excluded = [], [], [], [], Counter()
    # Row-vector processing avoids an n_people² × n_markers intermediate.
    for a in range(len(role_rows)):
        for b in range(a+1, len(role_rows)):
            i, j = int(role_rows[a]), int(role_rows[b])
            if components[i] == components[j]:
                excluded["within_component"] += 1
                continue
            valid = np.isfinite(z[a]) & np.isfinite(z[b])
            n = int(valid.sum())
            if n < min_pair_markers:
                excluded["insufficient_joint_markers"] += 1
                continue
            ii.append(i); jj.append(j); yy.append(float(np.dot(z[a, valid], z[b, valid])/n)); nn.append(n)
    require(yy, "No evaluable between-component pairs")
    return (np.array(ii), np.array(jj), np.array(yy), np.array(nn),
            {"eligible_markers": int(eligible.sum()), "excluded_pairs": dict(excluded)})


def component_pair_weights(i, j, components):
    labels = [tuple(sorted((components[a], components[b]))) for a, b in zip(i, j)]
    counts = Counter(labels)
    weights = np.array([1/counts[key] for key in labels], dtype=float)
    return weights/weights.mean(), labels


def symmetric_features(x, i, j, max_matrix_mb):
    require(len(i)*x.shape[1]*3*8*6 <= max_matrix_mb*2**20, "Symmetric design exceeds conservative 6-copy memory budget")
    a, b = x[i], x[j]
    return np.concatenate(((a+b)/2, np.abs(a-b), a*b), axis=1)


def paired_rows(paired, arm, i, j, column_indices=None):
    """Lookup requested unordered pairs; no missing row is invented or filled."""
    ii, jj = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
    keys = np.minimum(ii, jj)*paired["n_people"] + np.maximum(ii, jj)
    rows = np.searchsorted(paired["keys"], keys)
    require((rows < len(paired["keys"])).all(), "Missing required paired feature row")
    require(np.array_equal(paired["keys"][rows], keys), "Missing required paired feature row")
    if column_indices is not None:
        # Select both axes together: B0 must not allocate all rare B2 values
        # merely to obtain its technical indicators under a smaller budget.
        return paired[arm][np.ix_(rows, np.asarray(column_indices, dtype=int))]
    return paired[arm][rows]


def design_features(x, i, j, max_matrix_mb, paired=None, arm=None, require_fit_support=False):
    if paired is None:
        return symmetric_features(x, i, j, max_matrix_mb)
    width = paired[arm].shape[1]
    indicator_indices = paired["indicator_indices"][arm]
    mask_width = len(indicator_indices)
    require(len(i)*(3*x.shape[1]+width+mask_width)*8*6 <= max_matrix_mb*2**20,
            "Combined individual/paired design exceeds conservative memory budget")
    values = paired_rows(paired, arm, i, j)
    if require_fit_support:
        require(np.isfinite(values).any(axis=0).all(), "Paired column has no observed support in FIT")
    individual = symmetric_features(x, i, j, max_matrix_mb)
    # Technical indicators are shared by all arms; rare-derived availability
    # (e.g. J undefined when U=0) belongs only to B1/B2, never common B0.
    # B1 controls that availability in the B2-minus-B1 value contrast. Legacy
    # v2 sharing is possible only through its explicit compatibility contract.
    # NaN values survive until the FIT-only library imputer; observed zero stays
    # zero with indicator 0. No indicator infers a biological absence.
    missing = np.isnan(paired_rows(paired, "B2", i, j, column_indices=indicator_indices)).astype(float)
    return np.concatenate((individual, values, missing), axis=1)


def paired_support(paired, i, j):
    if paired is None:
        return {}
    values = paired_rows(paired, "B2", i, j)
    return dict(paired_observed_by_B2_column=np.isfinite(values).sum(axis=0).tolist(),
                paired_missing_by_B2_column=np.isnan(values).sum(axis=0).tolist(),
                paired_input_rows_total=len(paired["keys"]))


def ridge_model(alpha, paired=False):
    steps = ([("impute", SimpleImputer(strategy="mean", keep_empty_features=True))] if paired else [])
    return Pipeline(steps + [("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha, solver="svd"))])


def load_pair_operator_backend(config_path, config_digest, code_path, code_digest,
                               preconditioner_path, preconditioner_digest):
    """Load a hash-bound matrix-free backend; it cannot change model semantics."""
    _file_pin(config_path, config_digest, 'Pair-operator numerical configuration')
    _file_pin(code_path, code_digest, 'Pair-operator code')
    _file_pin(preconditioner_path, preconditioner_digest, 'Pair preconditioner code')
    config = load_json(config_path)
    fields = {'schema','backend','operator_backend','preprocessing_backend','rows_per_chunk',
              'max_working_mib','workers','atol','btol','max_iter','gradient_tolerance',
              'solver','max_preconditioner_mib','parameter_basis','compatibility_evidence'}
    require(set(config) in (fields,fields|{'max_total_iter'}) and config['schema'] == NUMERICAL_SCHEMA and
            config['backend'] == 'PAIRDESIGN_LSQR' and
            config['operator_backend'] == 'factorized_admitted_pairs' and
            config['preprocessing_backend'] == 'moments' and
            config['solver'] in {'LSQR','EXACT_MEAN_BLOCK_LSQR'},
            'Exact reviewed pair-operator numerical configuration required')
    require(type(config['rows_per_chunk']) is int and config['rows_per_chunk'] > 0 and
            type(config['max_working_mib']) in (int,float) and config['max_working_mib'] > 0 and
            type(config['workers']) is int and config['workers'] > 0 and
            type(config['max_iter']) is int and config['max_iter'] > 0 and
            (('max_total_iter' not in config) or
             (type(config['max_total_iter']) is int and config['max_total_iter']>=config['max_iter'])) and
            type(config['max_preconditioner_mib']) in (int,float) and
            ((config['solver']=='LSQR' and config['max_preconditioner_mib']==0) or
             (config['solver']=='EXACT_MEAN_BLOCK_LSQR' and config['max_preconditioner_mib']>0)) and
            all(type(config[k]) in (int,float) and math.isfinite(config[k]) and 0 < config[k] < 1
                for k in ('atol','btol','gradient_tolerance')) and
            isinstance(config['parameter_basis'],str) and len(config['parameter_basis'].strip()) >= 20 and
            isinstance(config['compatibility_evidence'],str) and len(config['compatibility_evidence'].strip()) >= 20,
            'Invalid or unjustified pair-operator numerical limits')
    module_name='r02_structure_pinned_pair_operator'
    spec = importlib.util.spec_from_file_location(module_name, code_path)
    require(spec is not None and spec.loader is not None, 'Pair-operator module cannot be loaded')
    module = importlib.util.module_from_spec(spec);sys.modules[module_name]=module
    # The preconditioner imports this canonical name; bind it to the exact
    # already hash-checked module rather than resolving another file.
    sys.modules['r02_pair_operator']=module;spec.loader.exec_module(module)
    pre_name='r02_structure_pinned_pair_preconditioner'
    pre_spec=importlib.util.spec_from_file_location(pre_name,preconditioner_path)
    require(pre_spec is not None and pre_spec.loader is not None,'Pair preconditioner module cannot be loaded')
    preconditioner=importlib.util.module_from_spec(pre_spec);sys.modules[pre_name]=preconditioner
    pre_spec.loader.exec_module(preconditioner)
    return PairOperatorBackend(module, config, config_digest, code_digest,
        preconditioner,preconditioner_digest)


class PairOperatorBackend:
    """Matrix-free equivalent of impute -> weighted scale -> weighted Ridge.

    Individual symmetric columns stay factorized. Paired values are small in
    width and preserve SimpleImputer's FIT-only *unweighted* column means;
    missingness indicators are computed before imputation.
    """
    def __init__(self,module,config,config_digest,code_digest,preconditioner=None,preconditioner_digest=None):
        self.module,self.config=module,config
        self.config_digest,self.code_digest=config_digest,code_digest
        self.preconditioner,self.preconditioner_digest=preconditioner,preconditioner_digest

    def _extra(self,paired,arm,i,j,imputation=None,require_fit_support=False):
        values=np.asarray(paired_rows(paired,arm,i,j),dtype=np.float64)
        if require_fit_support:
            require(np.isfinite(values).any(axis=0).all(),'Paired column has no observed support in FIT')
        if imputation is None:
            require(require_fit_support,'Only FIT may derive paired imputation statistics')
            imputation=np.nanmean(values,axis=0)
        imputation=np.asarray(imputation,dtype=np.float64)
        require(imputation.shape==(values.shape[1],) and np.isfinite(imputation).all(),
                'Invalid FIT paired imputation statistics')
        missing=np.isnan(paired_rows(paired,'B2',i,j,
            column_indices=paired['indicator_indices'][arm])).astype(np.float64)
        values=np.where(np.isnan(values),imputation[None,:],values)
        extra=np.column_stack((values,missing))
        require(np.isfinite(extra).all(),'Nonfinite pair-operator extra columns')
        return extra,imputation

    def design(self,x,i,j,paired,arm,role_rows,*,imputation=None,require_fit_support=False):
        role_rows=np.asarray(role_rows,dtype=np.int64)
        require(role_rows.ndim==1 and len(role_rows)>1 and np.all(role_rows[1:]>role_rows[:-1]),
                'Sorted unique role rows required for pair operator')
        left,right=np.searchsorted(role_rows,i),np.searchsorted(role_rows,j)
        require(np.all(left<len(role_rows)) and np.all(right<len(role_rows)) and
                np.array_equal(role_rows[left],i) and np.array_equal(role_rows[right],j),
                'Pair endpoints cross the requested FIT/SELECT/EVALUATE role')
        extra,imputation=self._extra(paired,arm,i,j,imputation,require_fit_support)
        c=self.config
        design=self.module.PairDesign(np.asarray(x)[role_rows],left,right,extra=extra,
            rows_per_chunk=c['rows_per_chunk'],max_working_mib=c['max_working_mib'],
            backend=c['operator_backend'],preprocessing_backend=c['preprocessing_backend'],
            workers=c['workers'])
        return design,imputation

    def fit(self,design,response,weights,alpha):
        return self.fit_path(design,response,weights,[alpha])[float(alpha)]

    def fit_path(self,design,response,weights,alphas,*,checkpoint_path=None,checkpoint_identity=None):
        c=self.config
        require((checkpoint_path is None)==(checkpoint_identity is None),
                'Solver checkpoint path and identity are required together')
        arguments=dict(atol=c['atol'],btol=c['btol'],max_iter=c['max_iter'],
            gradient_tolerance=c['gradient_tolerance'])
        alpha_values=[float(a) for a in alphas]
        recovered={};fingerprint=None;pending=None;partial_path=None
        if checkpoint_path is not None:
            identity=json.dumps(dict(caller=checkpoint_identity,evaluator_sha256=sha(__file__),
                numerical_config_sha256=self.config_digest,pair_operator_sha256=self.code_digest,
                pair_preconditioner_sha256=self.preconditioner_digest,solver=c.get('solver','LSQR')),
                sort_keys=True,separators=(',',':'))
            settings=dict(solver=c.get('solver','LSQR'),atol=c['atol'],btol=c['btol'],
                max_iter=c['max_iter'],max_total_iter=c.get('max_total_iter',c['max_iter']),
                gradient_tolerance=c['gradient_tolerance'],
                max_preconditioner_mib=c['max_preconditioner_mib'])
            fingerprint=self.module.ridge_path_fingerprint(
                design,response,weights,alpha_values,settings,identity)
            saved=self.module.load_ridge_path_checkpoint(
                checkpoint_path,fingerprint,alpha_values,design.shape[1])
            if saved is not None:
                recovered=saved['models']
                if len(recovered)==len(alpha_values):return {a:recovered[a] for a in alpha_values}
            partial_path=checkpoint_path.with_name(checkpoint_path.name+'.lsqr.npz')
            pending=self.module.load_lsqr_partial_checkpoint(
                partial_path,fingerprint,alpha_values,design.shape[1])
            # A crash can occur after the accepted candidate checkpoint is
            # atomically replaced but before its older status-7 sidecar is
            # removed.  The sidecar was authenticated above.  If its alpha is
            # already present in the accepted checkpoint it is stale, not a
            # second candidate to solve or a checkpoint conflict.
            if pending is not None and pending['alpha'] in recovered:
                self.module.clear_lsqr_partial_checkpoint(partial_path)
                pending=None
        def save(models):
            if checkpoint_path is not None:
                self.module.write_ridge_path_checkpoint(checkpoint_path,fingerprint,0,
                    np.zeros(design.shape[1]),[],[],len(models),models,alpha_values)
                self.module.clear_lsqr_partial_checkpoint(partial_path)
        def save_partial(state):
            if partial_path is not None:
                self.module.write_lsqr_partial_checkpoint(partial_path,fingerprint,
                    state['alpha'],state['x0'],state['iterations'],state['segments'],state['status'])
        if c.get('solver','LSQR')=='EXACT_MEAN_BLOCK_LSQR':
            require(self.preconditioner is not None,'Exact mean-block solver code is required')
            return self.preconditioner.fit_mean_preconditioned_path(design,response,weights,alphas,
                max_preconditioner_mib=c['max_preconditioner_mib'],completed_models=recovered,
                candidate_callback=save,max_total_iter=c.get('max_total_iter',c['max_iter']),
                pending_state=pending,partial_callback=save_partial,**arguments)
        models=dict(recovered)
        for alpha in alpha_values:
            if alpha not in models:
                models[alpha]=self.module.fit_ridge(design,response,weights,alpha,**arguments)
                save(models)
        return {a:models[a] for a in alpha_values}

    def metadata(self):
        parameters={k:self.config[k] for k in
            ('rows_per_chunk','max_working_mib','workers','atol','btol','max_iter','gradient_tolerance')}
        if 'max_preconditioner_mib' in self.config:parameters['max_preconditioner_mib']=self.config['max_preconditioner_mib']
        if 'max_total_iter' in self.config:parameters['max_total_iter']=self.config['max_total_iter']
        return dict(numerical_backend=self.config['backend'],operator_backend=self.config['operator_backend'],
            preprocessing_backend=self.config['preprocessing_backend'],numerical_parameters=parameters,
            solver=self.config.get('solver','LSQR'),numerical_config_sha256=self.config_digest,
            pair_operator_sha256=self.code_digest,pair_preconditioner_sha256=self.preconditioner_digest,
            parameter_basis=self.config['parameter_basis'],compatibility_evidence=self.config['compatibility_evidence'])


def macro_loss(y, pred, weights):
    require(np.isfinite(pred).all(), "Nonfinite model prediction")
    return float(np.average((y-pred)**2, weights=weights))


def contrasts(loss):
    return {"delta21": loss["B1"]-loss["B2"], "delta20": loss["B0"]-loss["B2"], "delta10": loss["B0"]-loss["B1"]}


def grouped_losses(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["block_id"], r["chromosome"], r["component_a"], r["component_b"])].append(r)
    return [(key, {a: float(np.mean([(r["target"]-r[a])**2 for r in rows])) for a in ARMS})
            for key, rows in sorted(grouped.items())]


def weighted_group_loss(grouped, component_counts=None, chromosome_counts=None):
    by_block = defaultdict(list)
    for (block, chrom, ca, cb), losses in grouped:
        w = 1 if component_counts is None else component_counts[ca]*component_counts[cb]
        if w and (chromosome_counts is None or chromosome_counts[chrom]):
            by_block[(block, chrom)].append((w, losses))
    if not by_block:
        return None
    denom, sums = 0, dict.fromkeys(ARMS, 0.0)
    for (_, chrom), entries in by_block.items():
        weight = 1 if chromosome_counts is None else chromosome_counts[chrom]
        total = sum(w for w, _ in entries)
        denom += weight
        for arm in ARMS:
            sums[arm] += weight*sum(w*loss[arm] for w, loss in entries)/total
    return {a: sums[a]/denom for a in ARMS}


def evaluate(records, contract):
    grouped = grouped_losses(records)
    components = sorted({r[f"component_{s}"] for r in records for s in ("a", "b")})
    chromosomes = sorted({r["chromosome"] for r in records})
    block_keys = sorted({(k[0], k[1]) for k, _ in grouped})
    component_index = {c: i for i, c in enumerate(components)}
    chromosome_index = {c: i for i, c in enumerate(chromosomes)}
    block_index = {b: i for i, b in enumerate(block_keys)}
    ca = np.array([component_index[k[2]] for k, _ in grouped])
    cb = np.array([component_index[k[3]] for k, _ in grouped])
    bi = np.array([block_index[k[:2]] for k, _ in grouped])
    chrom_of_block = np.array([chromosome_index[c] for _, c in block_keys])
    error = np.array([[v[a] for a in ARMS] for _, v in grouped])
    losses = weighted_group_loss(grouped)
    # Secondary estimand: each person, not each component pair, gets equal
    # weight within a block. This is not a second independent confirmation.
    person_errors = defaultdict(lambda: np.zeros(4, dtype=float))
    for row in records:
        loss = np.array([(row["target"]-row[a])**2 for a in ARMS])
        for side in ("a", "b"):
            entry = person_errors[(row["block_id"], row[f"sample_{side}"])]
            entry[:3] += loss
            entry[3] += 1
    person_blocks = defaultdict(list)
    for (block, _), errors in person_errors.items():
        person_blocks[block].append(errors[:3]/errors[3])
    person_loss = np.mean([np.mean(v, axis=0) for v in person_blocks.values()], axis=0)
    person_losses = dict(zip(ARMS, map(float, person_loss)))
    rng = np.random.default_rng(contract["seed"])
    draws, invalid = [], 0
    for _ in range(contract["bootstrap_replicates"]):
        cc = rng.multinomial(len(components), np.repeat(1/len(components), len(components)))
        gc = rng.multinomial(len(chromosomes), np.repeat(1/len(chromosomes), len(chromosomes)))
        # Endpoints share one component draw; all blocks of one chromosome share
        # one chromosome draw. Dyads and neighboring windows are NOT iid draws.
        pair_weight = cc[ca]*cc[cb]
        denominator = np.bincount(bi, weights=pair_weight, minlength=len(block_keys))
        block_weight = gc[chrom_of_block]
        valid = (denominator > 0) & (block_weight > 0)
        if not valid.any():
            invalid += 1
        else:
            value = {}
            for a, arm in enumerate(ARMS):
                numerator = np.bincount(bi, weights=pair_weight*error[:, a], minlength=len(block_keys))
                value[arm] = float(np.average(numerator[valid]/denominator[valid], weights=block_weight[valid]))
            draws.append(contrasts(value))
    enough = (len(components) >= contract["min_evaluation_components"]
              and len(chromosomes) >= contract["min_evaluation_chromosomes"])
    status = "CONDITIONAL_ESTIMATE" if enough and invalid == 0 else "IMPRECISE_DEPENDENCE_SUPPORT"
    intervals = {k: (np.quantile([d[k] for d in draws], [.025, .975]).tolist() if enough and invalid == 0 else None)
                 for k in contrasts(losses)}
    influences = []
    for unit, values in (("component", components), ("chromosome", chromosomes)):
        for dropped in values:
            sub = [(k, v) for k, v in grouped if (dropped not in k[2:] if unit == "component" else k[1] != dropped)]
            value = weighted_group_loss(sub)
            influences.append({"unit": unit, "excluded": dropped, "deltas": contrasts(value) if value else None})
    conclusions = {k: ("IMPRECISE_DEPENDENCE_SUPPORT" if ci is None else
                       "DETECTABLE_PREDICTIVE_INCREMENT" if ci[0] > 0 else
                       "DETECTABLE_PREDICTIVE_WORSENING" if ci[1] < 0 else "INCONCLUSIVE_SIGN")
                   for k, ci in intervals.items()}
    return dict(status=status, losses=losses, deltas=contrasts(losses), percentile_ci95=intervals,
        contrast_interpretation=conclusions,
        person_macro_losses=person_losses, person_macro_deltas=contrasts(person_losses),
        n_components=len(components), n_chromosomes=len(chromosomes), n_blocks=len({r["block_id"] for r in records}),
        n_people=len({r[f"sample_{s}"] for r in records for s in ("a", "b")}),
        n_distinct_person_pairs=len({(r["sample_a"], r["sample_b"]) for r in records}),
        n_pair_block_observations=len(records), bootstrap_attempted=contract["bootstrap_replicates"],
        bootstrap_valid=len(draws), bootstrap_invalid=invalid, influence=influences,
        uncertainty_scope="Conditional on fitted/selected models; no refit or historical-selection uncertainty",
        bootstrap_method="Crossed endpoint-component and chromosome multinomial weights; pair errors averaged within component pairs",
        limitations=["Component cutoff does not certify independence; residual dependence can invalidate the interval",
                     "Minimum counts are a prespecified admission rule, not proof of statistical precision",
                     "More feature columns change model dimension; this comparator alone does not validate capacity adequacy",
                     "Feature lineage is authenticated supplied provenance, not a reconstruction of VCF materialization",
                     "No population labels, demographic event, ancestry-specific attribution or K03 conclusion"])


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


EVALUATION_FIELDS = frozenset((
    'range_rationale','ld_margin_rationale','dependence_rationale','capacity_comparison','historical_exposure',
    'model_family','evaluation_claim','ridge_alphas','min_fit_called','min_pair_markers','bootstrap_replicates',
    'seed','max_prediction_records','min_evaluation_components','min_evaluation_chromosomes'))


class BlockBundles:
    """Authenticated block catalogue; at most one prepared tensor shard is resident."""
    def __init__(self,bundle,contract,evaluation):
        self.bundle,self.contract,self.evaluation=bundle,contract,evaluation
        require(contract.get('schema')==STREAM_SCHEMA,'Streamed v4 bundle required')
        require(contract.get('roles_state')=='EXPLICITLY_FROZEN',
            'Prepared roles are not explicitly frozen; feature preparation does not authorize fitting')
        require(set(contract['sha256'])=={'persons.tsv','blocks.tsv','block_index.json','protocol.md'},'Exact v4 catalogue required')
        self.verify()
        require(sha(bundle/'protocol.md')==contract['scientific_protocol_sha256'],'Protocol bytes differ')
        require(evaluation.get('schema')=='r02_structure_evaluation_config_v1','Separate evaluation configuration required')
        require(set(evaluation)==EVALUATION_FIELDS|{'schema','prepared_contract_sha256','scientific_protocol_sha256','max_work_bytes'},
            'Exact separate evaluator fields required; no defaults or data overrides')
        require(evaluation['scientific_protocol_sha256']==contract['scientific_protocol_sha256'],'Evaluation/data protocol mismatch')
        require(type(evaluation['max_work_bytes']) is int and evaluation['max_work_bytes']>0,'Positive scratch-byte bound required')
        self.persons=table(bundle/'persons.tsv',['sample_id','component','role']);self.roles=validate_roles(self.persons)
        self.blocks=table(bundle/'blocks.tsv',['block_id','chromosome','start_bp','end_bp','excluded_start_bp','excluded_end_bp'])
        self.index=load_json(bundle/'block_index.json')
        require(self.index.get('schema')=='r02_structure_block_index_v1','Unknown block index')
        self.entries=self.index['blocks']
        require(bool(self.entries),'Empty prepared block catalogue')
        require(all(set(item)=={'block_id','path','contract_sha256','manifest_sha256'} for item in self.entries),
            'Exact block descriptor fields required')
        require([r['block_id'] for r in self.entries]==[b['block_id'] for b in self.blocks] and
            len({b['block_id'] for b in self.blocks})==len(self.blocks),'Ordered block inventory differs')
        self.loaded_contracts={}
        for b in self.blocks:
            interval=[b['chromosome'],int(b['start_bp']),int(b['end_bp'])]
            require(valid_interval(interval),'Invalid block interval')
        for k,b in enumerate(self.blocks):
            require(not any(overlap([b['chromosome'],int(b['start_bp']),int(b['end_bp'])],
                [c['chromosome'],int(c['start_bp']),int(c['end_bp'])]) for c in self.blocks[k+1:]),'Overlapping global target blocks')

    def verify(self):
        for name,digest in self.contract['sha256'].items():require(sha(self.bundle/name)==digest,'V4 catalogue changed')

    def path(self,item):
        rel=Path(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe shard path')
        path=(self.bundle/rel).resolve(strict=True)
        require(path.is_relative_to(self.bundle.resolve()),'Shard escaped bundle')
        require(sha(path/'contract.json')==item['contract_sha256'] and
                sha(path/'manifest.json')==item['manifest_sha256'],'Shard contract/manifest mismatch')
        return path

    def load(self,k):
        b,item=self.blocks[k],self.entries[k];path=self.path(item);c=load_json(path/'contract.json')
        require(c.get('schema') in {BLOCK_SCHEMA,SCOPED_SCHEMA},'Unsupported prepared shard schema')
        require(c.get('roles_state')==self.contract['roles_state'],'Shard role-freezing state differs')
        receipt=load_json(path/'manifest.json')
        expected=set(DATA_FILES+PAIR_FILES)|{'pair_support.private.npz','upstream_verification.json','contract.json'}
        require(set(receipt.get('outputs_sha256',{}))==expected,'Prepared shard manifest inventory incomplete')
        for name,digest in receipt['outputs_sha256'].items():
            require(sha(path/name)==digest,'Prepared shard artifact SHA256 mismatch: '+name)
        require(sha(path/'persons.tsv')==self.contract['sha256']['persons.tsv'],'Shard persons/roles differ')
        require(c['scientific_protocol_sha256']==self.contract['scientific_protocol_sha256'],'Shard protocol differs')
        # Scientific data semantics cannot be overridden by model selection.
        for key in ('exclusion_scope','rare_allele_convention','unphased_rare_policy','max_pairs','max_matrix_mb'):
            if key in self.contract:require(c[key]==self.contract[key],'Shard global data contract differs: '+key)
        if 'pair_missing_indicator_scope' in self.contract:
            require(c['pair_missing_indicator_scope']=={b['block_id']:self.contract['pair_missing_indicator_scope'][b['block_id']]},
                'Shard missing-indicator scope differs from global contract')
        require(table(path/'blocks.tsv',['block_id','chromosome','start_bp','end_bp','excluded_start_bp','excluded_end_bp'])==[b],
            'Shard block differs from global target/margin')
        c={**c,**{key:self.evaluation[key] for key in EVALUATION_FIELDS},'schema':SCOPED_SCHEMA}
        values=validate_bundle(path,c,with_pair_features=True)
        self.loaded_contracts[k]=(path,c)
        return values,c

    def verify_shards(self):
        self.verify()
        for item in self.entries:
            path=self.path(item);c=load_json(path/'contract.json')
            for name,digest in c['sha256'].items():require(sha(path/name)==digest,'Shard data changed during evaluation')
            for name,digest in load_json(path/'manifest.json')['outputs_sha256'].items():
                require(sha(path/name)==digest,'Shard audit support changed during evaluation')


class BlockErrors:
    """Persist component-pair means, not Python dictionaries for every prediction.

    Bootstrap draws retain the original endpoint-component/chromosome scheme.
    Numeric summation can differ at floating-point roundoff, not by estimand.
    """
    def __init__(self,root,persons,contract):
        self.root,self.persons,self.contract=root,persons,contract
        self.components=sorted({p['component'] for p in persons if p['role']=='EVALUATE'})
        self.cindex={c:i for i,c in enumerate(self.components)}
        self.person_components=np.array([self.cindex.get(p['component'],-1) for p in persons],dtype=np.int64)
        self.entries=[];self.total=0;self.person_macro=[]
        self.seen_people=np.zeros(len(persons),bool)
        require(len(persons)**2<=contract['max_matrix_mb']*2**20,'Pair denominator bitmap exceeds memory bound')
        self.seen_pairs=np.zeros((len(persons),len(persons)),bool)

    def add(self,b,i,j,y,predictions):
        self.total+=len(y)
        require(self.total<=self.contract['max_prediction_records'],'Prediction record budget exceeded')
        errors=np.column_stack([(y-predictions[a])**2 for a in ARMS])
        a,bcomp=self.person_components[i],self.person_components[j]
        require((a>=0).all() and (bcomp>=0).all(),'Non-evaluation component reached statistics')
        left,right=np.minimum(a,bcomp),np.maximum(a,bcomp)
        keys,inverse=np.unique(left*len(self.components)+right,return_inverse=True)
        count=np.bincount(inverse)
        means=np.column_stack([np.bincount(inverse,weights=errors[:,k])/count for k in range(3)])
        indices=np.column_stack((keys//len(self.components),keys%len(self.components)))
        stem=self.root/f'group_{len(self.entries):04d}'
        if 'max_work_bytes' in self.contract:
            _scratch_guard(self.root,self.contract['max_work_bytes'],indices.nbytes+means.nbytes+1024)
        np.save(str(stem)+'_keys.npy',indices);np.save(str(stem)+'_errors.npy',means)
        self.entries.append(dict(block_id=b['block_id'],chromosome=b['chromosome'],stem=stem))
        sums=np.zeros((len(self.persons),3));n=np.zeros(len(self.persons),np.int64)
        for side in (i,j):np.add.at(sums,side,errors);np.add.at(n,side,1)
        self.person_macro.append((sums[n>0]/n[n>0,None]).mean(axis=0))
        self.seen_people[i]=True;self.seen_people[j]=True;self.seen_pairs[np.minimum(i,j),np.maximum(i,j)]=True

    def evaluate(self, audit_dir=None, draw_reducer=None):
        chromosomes=sorted({r['chromosome'] for r in self.entries})
        used=np.zeros(len(self.components),bool)
        for item in self.entries:
            ids=np.load(str(item['stem'])+'_keys.npy',mmap_mode='r');used[ids.reshape(-1)]=True
        # Legacy evaluate only resamples components actually represented.
        components=[c for c,ok in zip(self.components,used) if ok]
        compact=np.full(len(self.components),-1,dtype=np.int64);compact[used]=np.arange(len(components))
        rng=np.random.default_rng(self.contract['seed']);rep=self.contract['bootstrap_replicates']
        require(rep*(len(components)+len(chromosomes)+4)*8<=self.contract['max_matrix_mb']*2**20,
            'Bootstrap multiplicities exceed memory bound')
        cc=np.empty((rep,len(components)),dtype=np.int64);gc=np.empty((rep,len(chromosomes)),dtype=np.int64)
        for n in range(rep):
            cc[n]=rng.multinomial(len(components),np.repeat(1/len(components),len(components)))
            gc[n]=rng.multinomial(len(chromosomes),np.repeat(1/len(chromosomes),len(chromosomes)))
        draw_sum=np.zeros((rep,3));draw_den=np.zeros(rep)
        component_sum=np.zeros((len(components),3));component_den=np.zeros(len(components))
        block_losses=[]
        for item in self.entries:
            ids=np.load(str(item['stem'])+'_keys.npy',mmap_mode='r');error=np.load(str(item['stem'])+'_errors.npy',mmap_mode='r')
            ca,cb=compact[ids[:,0]],compact[ids[:,1]];chrom=chromosomes.index(item['chromosome'])
            base_loss=error.mean(axis=0);block_losses.append((item['chromosome'],base_loss))
            incident=np.bincount(ca,minlength=len(components))+np.bincount(cb,minlength=len(components))
            remain=len(error)-incident;good=remain>0
            losses=np.zeros((len(components),3))
            for k in range(3):
                removed=np.bincount(ca,weights=error[:,k],minlength=len(components))+np.bincount(cb,weights=error[:,k],minlength=len(components))
                losses[good,k]=(error[:,k].sum()-removed[good])/remain[good]
            component_sum[good]+=losses[good];component_den[good]+=1
            if draw_reducer is not None:
                numerator,denominator=draw_reducer(error,ca,cb,cc,gc[:,chrom])
                require(numerator.shape==draw_sum.shape and denominator.shape==draw_den.shape
                    and np.isfinite(numerator).all() and np.isfinite(denominator).all(),
                    'Invalid bounded bootstrap reduction')
                draw_sum+=numerator;draw_den+=denominator
            else:
                for r in range(rep):
                    if not gc[r,chrom]:continue
                    weight=cc[r,ca]*cc[r,cb];den=weight.sum()
                    if den:
                        draw_sum[r]+=gc[r,chrom]*np.sum(weight[:,None]*error,axis=0)/den
                        draw_den[r]+=gc[r,chrom]
        loss_values=np.mean([v for _,v in block_losses],axis=0);losses=dict(zip(ARMS,map(float,loss_values)))
        person_losses=dict(zip(ARMS,map(float,np.mean(self.person_macro,axis=0))))
        valid=draw_den>0;invalid=int((~valid).sum())
        draw_losses=np.zeros_like(draw_sum)
        draw_losses[valid]=draw_sum[valid]/draw_den[valid,None]
        draws=[contrasts(dict(zip(ARMS,row))) for row in draw_losses[valid]]
        enough=len(components)>=self.contract['min_evaluation_components'] and len(chromosomes)>=self.contract['min_evaluation_chromosomes']
        if audit_dir is not None:
            write_bootstrap_audit(audit_dir,components,chromosomes,cc,gc,draw_losses,valid,self.contract,
                intervals_admitted=enough and invalid==0)
        intervals={k:np.quantile([d[k] for d in draws],[.025,.975]).tolist() if enough and invalid==0 else None for k in contrasts(losses)}
        influences=[]
        for k,c in enumerate(components):
            value=dict(zip(ARMS,component_sum[k]/component_den[k])) if component_den[k] else None
            influences.append(dict(unit='component',excluded=c,deltas=contrasts(value) if value else None))
        for c in chromosomes:
            values=[x for chrom,x in block_losses if chrom!=c]
            value=dict(zip(ARMS,np.mean(values,axis=0))) if values else None
            influences.append(dict(unit='chromosome',excluded=c,deltas=contrasts(value) if value else None))
        conclusions={k:'IMPRECISE_DEPENDENCE_SUPPORT' if ci is None else 'DETECTABLE_PREDICTIVE_INCREMENT' if ci[0]>0 else
            'DETECTABLE_PREDICTIVE_WORSENING' if ci[1]<0 else 'INCONCLUSIVE_SIGN' for k,ci in intervals.items()}
        return dict(status='CONDITIONAL_ESTIMATE' if enough and invalid==0 else 'IMPRECISE_DEPENDENCE_SUPPORT',
            losses=losses,deltas=contrasts(losses),percentile_ci95=intervals,contrast_interpretation=conclusions,
            person_macro_losses=person_losses,person_macro_deltas=contrasts(person_losses),n_components=len(components),
            n_chromosomes=len(chromosomes),n_blocks=len(self.entries),n_people=int(self.seen_people.sum()),
            n_distinct_person_pairs=int(self.seen_pairs.sum()),n_pair_block_observations=self.total,
            bootstrap_attempted=rep,bootstrap_valid=int(valid.sum()),bootstrap_invalid=invalid,influence=influences,
            uncertainty_scope='Conditional on fitted/selected models; no refit or historical-selection uncertainty',
            bootstrap_method='Crossed endpoint-component and chromosome multinomial weights; pair errors averaged within component pairs',
            limitations=['Component cutoff does not certify independence; residual dependence can invalidate the interval',
                'Minimum counts are a prespecified admission rule, not proof of statistical precision',
                'More feature columns change model dimension; this comparator alone does not validate capacity adequacy',
                'Feature lineage is authenticated supplied provenance, not a reconstruction of VCF materialization',
                'No population labels, demographic event, ancestry-specific attribution or K03 conclusion'])


def write_bootstrap_audit(output,components,chromosomes,component_counts,chromosome_counts,losses,valid,contract,*,
                         intervals_admitted=False):
    """Retain every resampling draw and its finite-simulation precision audit.

    The binomial order-statistic bands quantify Monte Carlo uncertainty in the
    conditional bootstrap quantiles. They do not establish coverage of the
    biological interval under residual genealogical or ancestral dependence.
    Invalid draws remain present with an explicit validity mask, never dropped
    from the recorded draw indices. Zero loss on those rows is only a sentinel.
    """
    import scipy
    from scipy.stats import binom

    output=Path(output)
    rep=len(valid)
    require(component_counts.shape==(rep,len(components)) and
        chromosome_counts.shape==(rep,len(chromosomes)) and losses.shape==(rep,len(ARMS)),
        'Bootstrap audit dimensions differ')
    require(np.isfinite(losses).all(),'Nonfinite bootstrap audit loss')
    keys=list(contrasts(dict.fromkeys(ARMS,0.0)))
    values=np.zeros((rep,len(keys)))
    for i in np.flatnonzero(valid):
        values[i]=list(contrasts(dict(zip(ARMS,losses[i]))).values())
    payload=dict(component_ids=np.asarray(components,dtype=str),chromosomes=np.asarray(chromosomes,dtype=str),
        arm_names=np.asarray(ARMS,dtype=str),contrast_names=np.asarray(keys,dtype=str),
        component_counts=component_counts,chromosome_counts=chromosome_counts,
        arm_losses=losses,contrast_values=values,valid=valid,draw_index=np.arange(rep,dtype=np.int64))
    bytes_required=sum(a.nbytes for a in payload.values())+16384
    require(bytes_required<=contract['max_matrix_mb']*2**20,'Bootstrap audit exceeds configured memory-size bound')
    require(shutil.disk_usage(output).free>=bytes_required,'Insufficient disk for retained bootstrap audit')
    with (output/'bootstrap_draws.private.npz').open('xb') as handle:
        np.savez(handle,**payload)
    n=int(valid.sum())
    bands={};interpretations={}
    for k,name in enumerate(keys):
        ordered=np.sort(values[valid,k])
        endpoints=[]
        for p in (.025,.975):
            # If N observations lie below the population p-quantile, then
            # X_(L) <= q_p <= X_(U) when L <= N < U. For continuous draws,
            # N~Binomial(n,p); ties give conservative bounds. Ranks 0/n+1
            # indicate unbounded sides, represented by null, not clipped.
            lower=int(binom.ppf(.025,n,p)) if n else 0
            upper=int(binom.ppf(.975,n,p))+1 if n else 1
            endpoints.append(dict(probability=p,estimate=float(np.quantile(ordered,p)) if n else None,
                order_statistic_ranks_1based=[lower,upper],
                monte_carlo_ci95=[float(ordered[lower-1]) if lower>0 else None,
                    float(ordered[upper-1]) if upper<=n else None]))
        lower_band,upper_band=(e['monte_carlo_ci95'] for e in endpoints)
        if n!=rep:
            sign='INVALID_DRAWS_NO_SIGN_ASSERTION'
        elif lower_band[0] is not None and lower_band[0]>0:
            sign='POSITIVE_ENDPOINT_SIGN_STABLE_AT_MC95'
        elif upper_band[1] is not None and upper_band[1]<0:
            sign='NEGATIVE_ENDPOINT_SIGN_STABLE_AT_MC95'
        elif lower_band[1] is not None and lower_band[1]<0 and upper_band[0] is not None and upper_band[0]>0:
            sign='INTERVAL_SPANS_ZERO_AT_MC95'
        else:
            sign='MONTE_CARLO_ENDPOINT_SIGN_UNRESOLVED'
        bands[name]=dict(endpoints=endpoints,sign_audit=sign)
        interpretations[name]=('IMPRECISE_DEPENDENCE_SUPPORT' if not intervals_admitted or n!=rep else
            'DETECTABLE_PREDICTIVE_INCREMENT' if sign=='POSITIVE_ENDPOINT_SIGN_STABLE_AT_MC95' else
            'DETECTABLE_PREDICTIVE_WORSENING' if sign=='NEGATIVE_ENDPOINT_SIGN_STABLE_AT_MC95' else
            'INCONCLUSIVE_SIGN' if sign=='INTERVAL_SPANS_ZERO_AT_MC95' else
            'MONTE_CARLO_PRECISION_INSUFFICIENT')
    write_json(output/'bootstrap_monte_carlo.private.json',dict(schema='r02_bootstrap_audit_v1',
        seed=contract['seed'],attempted=rep,valid=n,invalid=rep-n,
        scipy_version=scipy.__version__,numpy_version=np.__version__,
        draws_file='bootstrap_draws.private.npz',draws_sha256=sha(output/'bootstrap_draws.private.npz'),
        conditional_quantile_mc_bands=bands,
        interval_support_admitted=bool(intervals_admitted),
        contrast_interpretation_after_mc=interpretations,
        method='Binomial order-statistic pointwise MC95 bands; no simultaneous-contrast coverage claim',
        scope='Finite bootstrap simulation only; does not validate biological interval coverage',
        decision_rule='Do not assert a detectable increment/worsening when its endpoint sign is MC-unresolved; '
            'no reseeding, retuning or selective increase after EVALUATE'))


def _scratch_guard(path,limit,additional_bytes=0):
    require(sum(p.stat().st_size for p in path.rglob('*') if p.is_file())+additional_bytes<=limit,
            'Streamed evaluator scratch budget exceeded')
    require(shutil.disk_usage(path).free>=additional_bytes,'Insufficient local scratch capacity')


def _block_genotypes(block,markers,genotypes):
    """Same target coordinates as the legacy evaluator, even for a wider shard."""
    mask=np.array([m['chromosome']==block['chromosome'] and
        int(block['start_bp'])<=int(m['position_bp'])<=int(block['end_bp']) for m in markers])
    require(mask.any(),'Block has no target markers')
    return genotypes if mask.all() else genotypes[:,mask]


def _stream_select(store,k,work,support,selection,*,target_function=None,path_factory=None,
                   pair_backend=None,arms=ARMS,solver_checkpoint_dir=None,
                   solver_checkpoint_identity=None):
    values,c=store.load(k)
    persons,roles,blocks,markers,gt,features,pairs=values
    b=blocks[0];name=b['block_id'];paired=pairs[name];components=[p['component'] for p in persons]
    gt=_block_genotypes(b,markers,gt)
    targets={};target_function=target_pairs if target_function is None else target_function
    for role in ('FIT','SELECT'):
        i,j,y,n,info=target_function(gt,roles['FIT'],roles[role],components,c['min_fit_called'],c['min_pair_markers'],c['max_pairs'])
        w,_=component_pair_weights(i,j,components);targets[role]=(i,j,y,n,w)
        support.append(dict(block_id=name,role=role,n_pairs=len(y),min_joint_markers=int(n.min()),
            max_joint_markers=int(n.max()),**info,**paired_support(paired,i,j)))
    fi,fj,fy,fn,fw=targets['FIT'];si,sj,sy,_,sw=targets['SELECT']
    _scratch_guard(work,store.evaluation['max_work_bytes'],sum(x.nbytes for x in (fi,fj,fy,fn,fw))+4096)
    fit_cache=work/f'fit_{k:04d}.npz'
    if fit_cache.exists():
        with np.load(fit_cache,allow_pickle=False) as saved:
            require(set(saved.files)=={'i','j','y','n','w'} and
                    all(np.array_equal(saved[key],value) for key,value in
                        zip(('i','j','y','n','w'),(fi,fj,fy,fn,fw))),
                    'Existing resumable FIT cache differs')
    else:
        np.savez(fit_cache,i=fi,j=fj,y=fy,n=fn,w=fw)
    _scratch_guard(work,store.evaluation['max_work_bytes'])
    require(tuple(arms) and all(a in ARMS for a in arms) and len(set(arms))==len(tuple(arms)),
            'Distinct known score arms required')
    for arm in arms:
        x=features[f'{name}_{arm}']
        if pair_backend is None:
            xf=design_features(x,fi,fj,c['max_matrix_mb'],paired,arm,require_fit_support=True)
            xs=design_features(x,si,sj,c['max_matrix_mb'],paired,arm)
            path=path_factory(xf,fy,fw,paired=paired[arm].shape[1]>0) if path_factory else None
        else:
            xf,imputation=pair_backend.design(x,fi,fj,paired,arm,roles['FIT'],require_fit_support=True)
            xs,_=pair_backend.design(x,si,sj,paired,arm,roles['SELECT'],imputation=imputation)
            checkpoint_path=None;identity=None
            if solver_checkpoint_dir is not None:
                identity=json.dumps(dict(scope=solver_checkpoint_identity,block_id=name,arm=arm),
                    sort_keys=True,separators=(',',':'))
                checkpoint_name=hashlib.sha256(identity.encode()).hexdigest()+'.npz'
                checkpoint_path=Path(solver_checkpoint_dir)/checkpoint_name
            path=pair_backend.fit_path(xf,fy,fw,sorted(c['ridge_alphas']),
                checkpoint_path=checkpoint_path,checkpoint_identity=identity)
        for alpha in sorted(c['ridge_alphas']):
            if pair_backend is not None:
                prediction=path[float(alpha)].predict(xs)
            elif path is None:
                model=ridge_model(alpha,paired=paired[arm].shape[1]>0)
                model.fit(xf,fy,scale__sample_weight=fw,ridge__sample_weight=fw)
                prediction=model.predict(xs)
            else:
                prediction=path.predict(xs,alpha)
            selection.append(dict(block_id=name,arm=arm,alpha=alpha,loss=macro_loss(sy,prediction,sw),
                n_symmetric_features=xf.shape[1]))


def _stream_predict(store,k,work,chosen,support,fitted,writer,statistics,metadata,fit_root=None,*,
                    target_function=None,pair_backend=None):
    values,c=store.load(k)
    persons,roles,blocks,markers,gt,features,pairs=values
    b=blocks[0];name=b['block_id'];paired=pairs[name];components=[p['component'] for p in persons]
    gt=_block_genotypes(b,markers,gt)
    with np.load((fit_root if fit_root is not None else work)/f'fit_{k:04d}.npz',allow_pickle=False) as fit:
        fi,fj,fy,fw=(fit[key] for key in ('i','j','y','w'))
    target_function=target_pairs if target_function is None else target_function
    ei,ej,ey,en,info=target_function(gt,roles['FIT'],roles['EVALUATE'],components,c['min_fit_called'],c['min_pair_markers'],c['max_pairs'])
    support.append(dict(block_id=name,role='EVALUATE',n_pairs=len(ey),min_joint_markers=int(en.min()),
        max_joint_markers=int(en.max()),**info,**paired_support(paired,ei,ej)))
    predictions={}
    for arm in ARMS:
        x=features[f'{name}_{arm}']
        if pair_backend is None:
            model=ridge_model(chosen[arm],paired=paired[arm].shape[1]>0)
            model.fit(design_features(x,fi,fj,c['max_matrix_mb'],paired,arm,require_fit_support=True),fy,
                scale__sample_weight=fw,ridge__sample_weight=fw)
            predictions[arm]=model.predict(design_features(x,ei,ej,c['max_matrix_mb'],paired,arm))
            mean,scale,coef,intercept=(model['scale'].mean_,model['scale'].scale_,model['ridge'].coef_,
                float(model['ridge'].intercept_))
            diagnostics=None
            imputation_statistics=model['impute'].statistics_.tolist() if 'impute' in model.named_steps else None
        else:
            fit_design,imputation=pair_backend.design(x,fi,fj,paired,arm,roles['FIT'],require_fit_support=True)
            evaluate_design,_=pair_backend.design(x,ei,ej,paired,arm,roles['EVALUATE'],imputation=imputation)
            model=pair_backend.fit(fit_design,fy,fw,chosen[arm]);predictions[arm]=model.predict(evaluate_design)
            mean,scale,coef,intercept=(model.center,model.scale,model.coefficient,float(model.response_mean))
            diagnostics=model.diagnostics
            imputation_statistics=imputation.tolist()
        require(np.isfinite(predictions[arm]).all(),'Nonfinite evaluation prediction')
        indices=paired['indicator_indices'][arm]
        record=dict(block_id=name,arm=arm,alpha=chosen[arm],mean=mean.tolist(),
            scale=scale.tolist(),coef=coef.tolist(),intercept=intercept,
            n_individual_features=x.shape[1],n_paired_features=paired[arm].shape[1],
            n_shared_paired_missing_indicators=len(paired['indicator_indices']['B0']),n_paired_missing_indicators=len(indices),
            paired_missing_indicator_columns=[paired['indicator_column_ids'][i] for i in indices],
            design_order='individual_mean,individual_abs_difference,individual_product,paired_value,scoped_B2_paired_missing_indicators',
            paired_missing_policy=c['pair_missing_policy'],
            imputation_statistics=imputation_statistics)
        if diagnostics is not None:record['numerical_diagnostics']=diagnostics
        fitted.append(record)
    statistics.add(b,ei,ej,ey,predictions);_scratch_guard(work,store.evaluation['max_work_bytes'])
    for row,(i,j) in enumerate(zip(ei,ej)):
        ca,cb=sorted((components[i],components[j]))
        writer.writerow(dict(block_id=name,chromosome=b['chromosome'],sample_a=persons[i]['sample_id'],
            sample_b=persons[j]['sample_id'],component_a=ca,component_b=cb,target=float(ey[row]),
            n_joint_markers=int(en[row]),**{arm:float(predictions[arm][row]) for arm in ARMS}))
    metadata[name]=dict(shared=paired['indicator_indices']['B0']==paired['indicator_indices']['B2'],
        indicators={a:[paired['indicator_column_ids'][i] for i in paired['indicator_indices'][a]] for a in ARMS},
        columns={a:paired[a].shape[1] for a in ARMS},scope=c['pair_missing_indicator_scope'][name])


def _choose_ridge(selection,alphas):
    """Original global block-mean SELECT loss, exact ties prefer larger alpha."""
    return {arm:min(((float(np.mean([r['loss'] for r in selection if r['arm']==arm and r['alpha']==a])),a)
        for a in alphas),key=lambda t:(t[0],-t[1]))[1] for arm in ARMS}


def _selection_record(chosen,trials,contract_hash,evaluation_digest):
    return dict(schema='r02_structure_ridge_selection_v1',source='SELECT_ONLY',chosen=chosen,trials=trials,
        contract_sha256=contract_hash,evaluation_contract_sha256=evaluation_digest,frozen_before_evaluation=True)


def _file_pin(path,digest,label):
    require(path is not None and isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest)
        and sha(path)==digest,label+' SHA256 mismatch')


def _select_manifest(output,store,contract_hash,evaluation_digest,code_digest,*,extra_metadata=None):
    store.verify_shards()
    require(sha(__file__)==code_digest,'Evaluator code changed during SELECT')
    manifest=dict(schema=SELECT_SCHEMA,status=SELECT_STATUS,scientific_status='NOT_EVALUATED',
        claim=store.evaluation['evaluation_claim'],input_schema=STREAM_SCHEMA,
        source_contract_sha256=contract_hash,evaluation_contract_sha256=evaluation_digest,
        sources_sha256=store.contract['sha256'],code_sha256=code_digest,
        versions={'numpy':np.__version__,'sklearn':sklearn.__version__},
        block_ids=[b['block_id'] for b in store.blocks],evaluate_targets_computed=False,
        evaluate_predictions_computed=False,approval_automatically_generated=False,
        input_read_scope='Prepared shards include all roles; structural checks read arrays, but no EVALUATE target is computed',
        outputs_sha256={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    if extra_metadata is not None:
        require(not set(extra_metadata)&set(manifest),'Selection metadata cannot override scientific identity')
        manifest.update(extra_metadata)
    write_json(output/'manifest.json',manifest)
    write_json(output/'complete.json',dict(status=SELECT_STATUS,manifest_sha256=sha(output/'manifest.json')))
    return manifest


def _score_manifest(output,store,contract_hash,evaluation_digest,code_digest,*,arm,extra_metadata=None):
    """Bind one independently resumable block score without choosing an alpha."""
    store.verify_shards()
    require(len(store.blocks)==1,'Distributed score checkpoints require exactly one block')
    require(sha(__file__)==code_digest,'Evaluator code changed during SCORE')
    manifest=dict(schema=SCORE_SCHEMA,status=SCORE_STATUS,scientific_status='NOT_EVALUATED',
        claim=store.evaluation['evaluation_claim'],input_schema=STREAM_SCHEMA,
        source_contract_sha256=contract_hash,evaluation_contract_sha256=evaluation_digest,
        sources_sha256=store.contract['sha256'],code_sha256=code_digest,
        versions={'numpy':np.__version__,'sklearn':sklearn.__version__},
        block_id=store.blocks[0]['block_id'],arm=arm,
        exclusion_scope=store.loaded_contracts[0][1]['exclusion_scope'],
        evaluate_targets_computed=False,evaluate_predictions_computed=False,
        alpha_selected=False,approval_automatically_generated=False,
        outputs_sha256={p.relative_to(output).as_posix():sha(p)
                        for p in sorted(output.rglob('*')) if p.is_file()})
    amendment=store.contract.get('run_scope_amendment_sha256')
    if amendment is not None:
        require(isinstance(amendment,str) and re.fullmatch('[0-9a-f]{64}',amendment),
            'Run-scope amendment SHA256 is invalid')
        manifest['run_scope_amendment_sha256']=amendment
    if extra_metadata is not None:
        require(not set(extra_metadata)&set(manifest),'Score metadata cannot override scientific identity')
        manifest.update(extra_metadata)
    write_json(output/'manifest.json',manifest)
    write_json(output/'complete.json',dict(status=SCORE_STATUS,manifest_sha256=sha(output/'manifest.json')))
    return manifest


def _admit_selection(store,checkpoint,checkpoint_sha,review_path,review_sha,contract_hash,evaluation_digest,code_digest,
                     pair_backend=None):
    """Admit all saved inputs before any EVALUATE target or model prediction.

    ACCEPT is an explicit review record, not a cryptographic human signature.
    Neither selection nor this reader generates one or revises the candidate.
    """
    require(checkpoint is not None,'EVALUATE requires a SELECT checkpoint')
    checkpoint=Path(checkpoint).resolve(strict=True)
    _file_pin(checkpoint/'manifest.json',checkpoint_sha,'SELECT checkpoint')
    m=load_json(checkpoint/'manifest.json')
    require(m.get('schema')==SELECT_SCHEMA and m.get('status')==SELECT_STATUS and
        m.get('input_schema')==STREAM_SCHEMA,'Incomplete or incompatible SELECT checkpoint')
    require(load_json(checkpoint/'complete.json')==dict(status=SELECT_STATUS,manifest_sha256=checkpoint_sha),
        'SELECT completion marker mismatch')
    require(m.get('source_contract_sha256')==contract_hash and m.get('evaluation_contract_sha256')==evaluation_digest and
        m.get('sources_sha256')==store.contract['sha256'],'SELECT data/configuration changed')
    require(m.get('code_sha256')==code_digest and m.get('versions')=={'numpy':np.__version__,'sklearn':sklearn.__version__},
        'SELECT evaluator code/library versions changed')
    backend_metadata={} if pair_backend is None else pair_backend.metadata()
    require(all(m.get(k)==v for k,v in backend_metadata.items()) and
            ((pair_backend is not None)==('numerical_backend' in m)),
        'SELECT numerical backend/code/configuration changed')
    require(m.get('block_ids')==[b['block_id'] for b in store.blocks] and
        m.get('evaluate_targets_computed') is False and m.get('evaluate_predictions_computed') is False,
        'SELECT block coverage or reserved-role state differs')
    expected={'frozen_selection.json','support.json'}|{f'fit_targets/fit_{k:04d}.npz' for k in range(len(store.blocks))}
    require(set(m.get('outputs_sha256',{}))==expected,'Exact SELECT output inventory required')
    require(not any(p.is_symlink() for p in checkpoint.rglob('*')),'SELECT checkpoint contains symlinks')
    require({p.relative_to(checkpoint).as_posix() for p in checkpoint.rglob('*') if p.is_file()}==
        expected|{'manifest.json','complete.json'},'Unexpected/missing SELECT files')
    for name in sorted(expected):_file_pin(checkpoint/name,m['outputs_sha256'][name],'SELECT artifact')
    selected=load_json(checkpoint/'frozen_selection.json');support=load_json(checkpoint/'support.json')
    require(selected.get('schema')=='r02_structure_ridge_selection_v1' and selected.get('source')=='SELECT_ONLY' and
        selected.get('contract_sha256')==contract_hash and selected.get('evaluation_contract_sha256')==evaluation_digest and
        selected.get('frozen_before_evaluation') is True,'Selection record identity differs')
    trials=selected['trials'];grid={(b['block_id'],a,alpha) for b in store.blocks for a in ARMS for alpha in store.evaluation['ridge_alphas']}
    require(len(trials)==len(grid) and {(r['block_id'],r['arm'],r['alpha']) for r in trials}==grid and
        all(math.isfinite(r['loss']) and r['loss']>=0 for r in trials),'SELECT trial grid/loss differs')
    require(selected['chosen']==_choose_ridge(trials,store.evaluation['ridge_alphas']),'Candidate differs from saved SELECT trials')
    require(len(support)==2*len(store.blocks) and {(s['block_id'],s['role']) for s in support}==
        {(b['block_id'],r) for b in store.blocks for r in ('FIT','SELECT')},'SELECT support includes wrong roles/blocks')
    _file_pin(review_path,review_sha,'Explicit review')
    review=load_json(review_path)
    links=dict(selection_manifest_sha256=checkpoint_sha,selection_sha256=m['outputs_sha256']['frozen_selection.json'],
        prepared_contract_sha256=contract_hash,evaluation_contract_sha256=evaluation_digest,code_sha256=code_digest,
        chosen=selected['chosen'])
    require(set(review)==set(links)|{'schema','decision','review_scope','reviewer','reviewed_at_utc','rationale'},
        'Exact explicit review fields required')
    require(review['schema']==REVIEW_SCHEMA and review['decision']=='ACCEPT' and
        review['review_scope']=='SELECT_ONLY_NO_EVALUATE_RESULTS' and all(review[k]==v for k,v in links.items()),
        'Explicit ACCEPT is not bound to this selection/data/configuration/code')
    require(isinstance(review['reviewer'],str) and review['reviewer'].strip() and
        isinstance(review['rationale'],str) and len(review['rationale'].strip())>=20,'Explicit reviewer and rationale required')
    stamp=datetime.fromisoformat(review['reviewed_at_utc'].replace('Z','+00:00'))
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds()==0,'Review UTC timestamp required')
    # Hash every shard before scoring the first block, including later shards.
    store.verify_shards()
    for k,item in enumerate(store.entries):
        c=load_json(store.path(item)/'contract.json');path=checkpoint/f'fit_targets/fit_{k:04d}.npz'
        with zipfile.ZipFile(path) as archive:
            require(sum(x.file_size for x in archive.infolist())<=c['max_matrix_mb']*2**20,'FIT cache exceeds memory admission')
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)=={'i','j','y','n','w'},'FIT cache keys differ')
            i,j,y,n,w=(z[key] for key in ('i','j','y','n','w'))
            require(i.ndim==1 and 0<len(i)<=c['max_pairs'] and all(x.shape==i.shape for x in (j,y,n,w)),
                'FIT cache dimensions differ')
            require(i.dtype.kind in 'iu' and j.dtype.kind in 'iu' and n.dtype.kind in 'iu' and
                np.isin(i,store.roles['FIT']).all() and np.isin(j,store.roles['FIT']).all() and (i<j).all(),
                'FIT cache contains non-FIT or invalid pairs')
            require(np.isfinite(y).all() and np.isfinite(w).all() and (n>=store.evaluation['min_pair_markers']).all() and
                (w>0).all(),'Invalid FIT targets/weights/denominators')
    return selected,support


def run_streamed(bundle,contract_path,output,evaluation_path,evaluation_digest,mode='legacy',
                 selection_checkpoint=None,selection_checkpoint_sha256=None,review=None,review_sha256=None,
                 pair_backend=None,score_arm=None,solver_checkpoint_dir=None):
    """Legacy two-pass execution or explicit SELECT / reviewed EVALUATE modes."""
    require(mode in {'legacy','score','select','evaluate'},'Unknown streamed execution mode')
    require(evaluation_path is not None and evaluation_digest is not None,
        'Prepared v4 features require separately reviewed evaluation configuration; no training defaults')
    require(re.fullmatch('[0-9a-f]{64}',evaluation_digest) and sha(evaluation_path)==evaluation_digest,
        'Evaluation configuration hash mismatch')
    evaluation=load_json(evaluation_path);contract=load_json(contract_path);contract_hash=sha(contract_path)
    code_digest=sha(__file__)
    require(evaluation.get('prepared_contract_sha256')==contract_hash,'Evaluation config belongs to another prepared catalogue')
    store=BlockBundles(bundle,contract,evaluation)
    selection,support,fitted,metadata=[],[],[],{}
    selected=None
    if solver_checkpoint_dir is not None:
        require(mode in {'score','select'} and pair_backend is not None,
                'Solver checkpoints require SCORE/SELECT pair backend')
        solver_checkpoint_dir=Path(solver_checkpoint_dir)
        require(solver_checkpoint_dir.is_absolute(),
                'Solver checkpoint directory must be absolute')
        solver_checkpoint_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
        require(not solver_checkpoint_dir.is_symlink(),
                'Solver checkpoint directory must be absolute and not a symlink')
        checkpoint_scope=dict(source_contract_sha256=contract_hash,
            evaluation_contract_sha256=evaluation_digest,code_sha256=code_digest,
            mode=mode,score_arm=score_arm,evaluate_allowed=False)
    else:
        checkpoint_scope=None
    if mode=='evaluate':
        selected,support=_admit_selection(store,selection_checkpoint,selection_checkpoint_sha256,review,review_sha256,
            contract_hash,evaluation_digest,code_digest,pair_backend)
        # All FIT-cache identities were checked above; no SELECT loop is run.
        fit_root=Path(selection_checkpoint)/'fit_targets'
    output.mkdir(parents=True,mode=0o700,exist_ok=solver_checkpoint_dir is not None)
    if mode in {'score','select'}:
        require((mode=='score')==(score_arm in ARMS),'SCORE requires exactly one explicit arm')
        work=output/'fit_targets';work.mkdir(mode=0o700,exist_ok=solver_checkpoint_dir is not None)
        arms=(score_arm,) if mode=='score' else ARMS
        for k in range(len(store.blocks)):_stream_select(store,k,work,support,selection,
            pair_backend=pair_backend,arms=arms,solver_checkpoint_dir=solver_checkpoint_dir,
            solver_checkpoint_identity=checkpoint_scope)
        if mode=='score':
            require(len(store.blocks)==1,'Distributed SCORE requires a one-block prepared bundle')
            write_json(output/'candidate_scores.json',dict(schema=SCORE_SCHEMA,status=SCORE_STATUS,
                source='SELECT_ONLY_NO_ALPHA_CHOSEN',block_id=store.blocks[0]['block_id'],
                exclusion_scope=store.loaded_contracts[0][1]['exclusion_scope'],arm=score_arm,trials=selection,
                ridge_alphas=evaluation['ridge_alphas'],
                contract_sha256=contract_hash,evaluation_contract_sha256=evaluation_digest,
                evaluate_targets_computed=False,alpha_selected=False))
            write_json(output/'support.json',support)
            require(sha(contract_path)==contract_hash and sha(evaluation_path)==evaluation_digest,
                    'SCORE input contract changed')
            return _score_manifest(output,store,contract_hash,evaluation_digest,code_digest,
                arm=score_arm,extra_metadata=pair_backend.metadata() if pair_backend is not None else None)
        chosen=_choose_ridge(selection,evaluation['ridge_alphas'])
        write_json(output/'frozen_selection.json',_selection_record(chosen,selection,contract_hash,evaluation_digest))
        write_json(output/'support.json',support)
        require(sha(contract_path)==contract_hash and sha(evaluation_path)==evaluation_digest,'SELECT input contract changed')
        return _select_manifest(output,store,contract_hash,evaluation_digest,code_digest,
            extra_metadata=pair_backend.metadata() if pair_backend is not None else None)
    with tempfile.TemporaryDirectory(prefix='r02-increment-',dir=output.parent) as temporary:
        work=Path(temporary)
        if mode=='legacy':
            for k in range(len(store.blocks)):_stream_select(store,k,work,support,selection,pair_backend=pair_backend)
            chosen=_choose_ridge(selection,evaluation['ridge_alphas']);fit_root=work
            selected=_selection_record(chosen,selection,contract_hash,evaluation_digest)
            matrix_mb=store.loaded_contracts[0][1]['max_matrix_mb'];reserved=0
        else:
            chosen=selected['chosen']
            matrix_mb=load_json(store.path(store.entries[0])/'contract.json')['max_matrix_mb']
            reserved=sum(p.stat().st_size for p in fit_root.iterdir())
            require(reserved<evaluation['max_work_bytes'],'Retained FIT caches exhaust scratch budget')
        write_json(output/'frozen_selection.json',selected)
        statistics=BlockErrors(work,store.persons,{**evaluation,'max_matrix_mb':matrix_mb,
            'max_work_bytes':evaluation['max_work_bytes']-reserved})
        fields=['block_id','chromosome','sample_a','sample_b','component_a','component_b','target','n_joint_markers',*ARMS]
        with (output/'predictions.private.tsv').open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t');writer.writeheader()
            for k in range(len(store.blocks)):
                _stream_predict(store,k,work,chosen,support,fitted,writer,statistics,metadata,fit_root=fit_root,
                    pair_backend=pair_backend)
                _scratch_guard(work,evaluation['max_work_bytes'],reserved)
        result=statistics.evaluate(audit_dir=output)
    write_json(output/'models.json',fitted);write_json(output/'support.json',support)
    write_json(output/'evaluation.private.json',result)
    store.verify_shards()
    require(sha(contract_path)==contract_hash and sha(evaluation_path)==evaluation_digest,'Evaluation/input contract changed')
    require(sha(__file__)==code_digest,'Evaluator code changed during execution')
    if mode=='evaluate':
        _admit_selection(store,selection_checkpoint,selection_checkpoint_sha256,review,review_sha256,
            contract_hash,evaluation_digest,code_digest,pair_backend)
    manifest=dict(schema='r02_structure_increment_result_v1',status='COMPLETE_PREPARED_FEATURE_COMPARATOR',
        input_schema=STREAM_SCHEMA,scientific_status=result['status'],claim=evaluation['evaluation_claim'],
        genomic_materialization_performed=False,deep_learning_performed=False,K03_discrete_structure_evaluated=False,
        source_contract_sha256=contract_hash,evaluation_contract_sha256=evaluation_digest,sources_sha256=contract['sha256'],
        code_sha256=sha(__file__),versions={'numpy':np.__version__,'sklearn':sklearn.__version__},
        paired_missing_policy='FIT_UNWEIGHTED_MEAN_WITH_PER_COLUMN_INDICATOR',
        paired_missing_indicator_scope={k:v['scope'] for k,v in metadata.items()},
        paired_indicators_shared_across_arms=all(v['shared'] for v in metadata.values()),
        paired_missing_indicator_columns_by_block={k:v['indicators'] for k,v in metadata.items()},
        paired_columns_by_block={k:v['columns'] for k,v in metadata.items()},
        storage='One prepared shard resident; predictions streamed; component-pair means mapped per block',
        interpretation_required_read='bootstrap_monte_carlo.private.json:contrast_interpretation_after_mc',
        legacy_evaluation_interpretation_scope='Before Monte Carlo precision check; not the final interpretation',
        outputs_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    if pair_backend is not None:manifest.update(pair_backend.metadata())
    if mode=='evaluate':manifest.update(execution_mode='REVIEWED_EVALUATE',
        selection_checkpoint_sha256=selection_checkpoint_sha256,review_sha256=review_sha256,
        select_recomputed=False,fit_target_cache_reused=True)
    write_json(output/'manifest.json',manifest)
    write_json(output/'complete.json',dict(status=manifest['status'],manifest_sha256=sha(output/'manifest.json')))
    return manifest


def run(bundle, contract_path, output, evaluation_contract=None, evaluation_contract_sha256=None,*,mode='legacy',
        selection_checkpoint=None,selection_checkpoint_sha256=None,review=None,review_sha256=None,
        numerical_contract=None,numerical_contract_sha256=None,pair_operator_code=None,pair_operator_code_sha256=None,
        pair_preconditioner_code=None,pair_preconditioner_code_sha256=None,score_arm=None,
        solver_checkpoint_dir=None):
    if output.exists():
        require(solver_checkpoint_dir is not None and output.is_dir() and not output.is_symlink() and
                not (output/'complete.json').exists(),
                'Output exists; only an incomplete solver-checkpoint attempt may resume')
        fit_targets=output/'fit_targets';allowed={fit_targets}
        require(set(output.iterdir())<=allowed and (not fit_targets.exists() or fit_targets.is_dir()) and
                all(not p.is_symlink() and p.is_file() and
                re.fullmatch(r'fit_[0-9]{4}\.npz',p.name) for p in
                (fit_targets.iterdir() if fit_targets.is_dir() else [])),
                'Incomplete output contains non-resumable artifacts')
    require(mode in {'legacy','score','select','evaluate'},'Unknown execution mode')
    require((mode=='score')==(score_arm in ARMS),'SCORE requires exactly one explicit arm')
    gates=(selection_checkpoint,selection_checkpoint_sha256,review,review_sha256)
    require(all(v is not None for v in gates) if mode=='evaluate' else all(v is None for v in gates),
        'Checkpoint and explicit review are required only for EVALUATE mode')
    contract = load_json(contract_path)
    numerical_args=(numerical_contract,numerical_contract_sha256,pair_operator_code,pair_operator_code_sha256,
                    pair_preconditioner_code,pair_preconditioner_code_sha256)
    require(all(x is None for x in numerical_args) or all(x is not None for x in numerical_args),
        'Pair-operator numerical config/code and both SHA256 values are required together')
    pair_backend=(None if numerical_contract is None else load_pair_operator_backend(
        numerical_contract,numerical_contract_sha256,pair_operator_code,pair_operator_code_sha256,
        pair_preconditioner_code,pair_preconditioner_code_sha256))
    require(contract.get('schema')==STREAM_SCHEMA or pair_backend is None,
        'Pair-operator backend applies only to streamed v4 bundles')
    if contract.get('schema') == STREAM_SCHEMA:
        return run_streamed(bundle,contract_path,output,evaluation_contract,evaluation_contract_sha256,mode,
            selection_checkpoint,selection_checkpoint_sha256,review,review_sha256,pair_backend,score_arm,
            solver_checkpoint_dir)
    require(mode=='legacy','Separated SCORE/SELECT/EVALUATE requires v4 prepared features')
    require(evaluation_contract is None and evaluation_contract_sha256 is None,'Separate evaluation contract applies only to v4')
    contract_hash = sha(contract_path)
    persons, roles, blocks, markers, gt, features, pair_features = validate_bundle(bundle, contract, with_pair_features=True)
    components = [p["component"] for p in persons]
    targets, selection, support = {}, [], []
    for b in blocks:
        name = b["block_id"]
        paired = pair_features[name] if pair_features is not None else None
        mask = np.array([m["chromosome"] == b["chromosome"] and int(b["start_bp"]) <= int(m["position_bp"]) <= int(b["end_bp"]) for m in markers])
        require(mask.any(), "Block has no target markers")
        targets[name] = {}
        for role in ("FIT", "SELECT"):
            data = target_pairs(gt[:, mask], roles["FIT"], roles[role], components,
                                contract["min_fit_called"], contract["min_pair_markers"], contract["max_pairs"])
            i, j, y, n, audit = data
            weights, _ = component_pair_weights(i, j, components)
            targets[name][role] = (i, j, y, n, weights)
            support.append(dict(block_id=name, role=role, n_pairs=len(y), min_joint_markers=int(n.min()),
                                max_joint_markers=int(n.max()), **audit, **paired_support(paired, i, j)))
        fi, fj, fy, _, fw = targets[name]["FIT"]
        si, sj, sy, _, sw = targets[name]["SELECT"]
        for arm in ARMS:
            x = features[f"{name}_{arm}"]
            xf = design_features(x, fi, fj, contract["max_matrix_mb"], paired, arm, require_fit_support=True)
            xs = design_features(x, si, sj, contract["max_matrix_mb"], paired, arm)
            for alpha in sorted(contract["ridge_alphas"]):
                model = ridge_model(alpha, paired=paired is not None and paired[arm].shape[1] > 0)
                model.fit(xf, fy, scale__sample_weight=fw, ridge__sample_weight=fw)
                selection.append(dict(block_id=name, arm=arm, alpha=alpha,
                                      loss=macro_loss(sy, model.predict(xs), sw), n_symmetric_features=xf.shape[1]))
    # Same candidate set/budget for each arm. Alpha is selected across all SELECT
    # blocks; exact loss ties prefer stronger regularization deterministically.
    chosen = {}
    for arm in ARMS:
        candidates = [(float(np.mean([r["loss"] for r in selection if r["arm"] == arm and r["alpha"] == a])), a)
                      for a in contract["ridge_alphas"]]
        chosen[arm] = min(candidates, key=lambda t: (t[0], -t[1]))[1]
    output.mkdir(parents=True)
    write_json(output/"frozen_selection.json", dict(schema="r02_structure_ridge_selection_v1", source="SELECT_ONLY",
               chosen=chosen, trials=selection, contract_sha256=contract_hash, frozen_before_evaluation=True))
    records, fitted = [], []
    for b in blocks:
        name = b["block_id"]
        paired = pair_features[name] if pair_features is not None else None
        fi, fj, fy, _, fw = targets[name]["FIT"]
        mask = np.array([m["chromosome"] == b["chromosome"] and int(b["start_bp"]) <= int(m["position_bp"]) <= int(b["end_bp"]) for m in markers])
        ei, ej, ey, en, audit = target_pairs(gt[:, mask], roles["FIT"], roles["EVALUATE"], components,
            contract["min_fit_called"], contract["min_pair_markers"], contract["max_pairs"])
        support.append(dict(block_id=name, role="EVALUATE", n_pairs=len(ey),
                            min_joint_markers=int(en.min()), max_joint_markers=int(en.max()), **audit,
                            **paired_support(paired, ei, ej)))
        require(len(records)+len(ey) <= contract["max_prediction_records"], "Prediction record budget exceeded")
        require((len(records)+len(ey))*2048 <= contract["max_matrix_mb"]*2**20,
                "Prediction records exceed conservative memory admission estimate")
        predictions = {}
        for arm in ARMS:
            x = features[f"{name}_{arm}"]
            model = ridge_model(chosen[arm], paired=paired is not None and paired[arm].shape[1] > 0)
            model.fit(design_features(x, fi, fj, contract["max_matrix_mb"], paired, arm, require_fit_support=True), fy,
                      scale__sample_weight=fw, ridge__sample_weight=fw)
            predictions[arm] = model.predict(design_features(x, ei, ej, contract["max_matrix_mb"], paired, arm))
            require(np.isfinite(predictions[arm]).all(), "Nonfinite evaluation prediction")
            # JSON coefficients avoid executable pickle model artifacts.
            fitted.append(dict(block_id=name, arm=arm, alpha=chosen[arm],
                mean=model["scale"].mean_.tolist(), scale=model["scale"].scale_.tolist(),
                coef=model["ridge"].coef_.tolist(), intercept=float(model["ridge"].intercept_)))
            if paired is not None:
                indices = paired["indicator_indices"][arm]
                fitted[-1].update(n_individual_features=x.shape[1], n_paired_features=paired[arm].shape[1],
                    n_shared_paired_missing_indicators=len(paired["indicator_indices"]["B0"]),
                    n_paired_missing_indicators=len(indices),
                    paired_missing_indicator_columns=[paired["indicator_column_ids"][k] for k in indices],
                    design_order="individual_mean,individual_abs_difference,individual_product,paired_value,scoped_B2_paired_missing_indicators",
                    paired_missing_policy=contract["pair_missing_policy"],
                    imputation_statistics=(model["impute"].statistics_.tolist() if "impute" in model.named_steps else None))
        for row, (i, j) in enumerate(zip(ei, ej)):
            ca, cb = sorted((components[i], components[j]))
            records.append(dict(block_id=name, chromosome=b["chromosome"], sample_a=persons[i]["sample_id"],
                sample_b=persons[j]["sample_id"], component_a=ca, component_b=cb,
                target=float(ey[row]), n_joint_markers=int(en[row]), **{a: float(predictions[a][row]) for a in ARMS}))
    result = evaluate(records, contract)
    with (output/"predictions.private.tsv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(records)
    write_json(output/"models.json", fitted)
    write_json(output/"support.json", support)
    write_json(output/"evaluation.private.json", result)
    for name, digest in contract["sha256"].items():
        require(sha(bundle/name) == digest, "Input changed during evaluation")
    require(sha(contract_path) == contract_hash, "Contract changed during evaluation")
    manifest = dict(schema="r02_structure_increment_result_v1", status="COMPLETE_PREPARED_FEATURE_COMPARATOR",
        scientific_status=result["status"], claim=contract["evaluation_claim"],
        genomic_materialization_performed=False, deep_learning_performed=False,
        K03_discrete_structure_evaluated=False, source_contract_sha256=contract_hash,
        sources_sha256=contract["sha256"], code_sha256=sha(__file__),
        versions={"numpy": np.__version__, "sklearn": sklearn.__version__},
        outputs_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    if pair_features is not None:
        manifest.update(input_schema=contract["schema"],
            paired_missing_policy=contract["pair_missing_policy"],
            pair_missing_indicator_scope=contract["pair_missing_indicator_scope"],
            paired_indicators_shared_across_arms=all(value["indicator_indices"]["B0"] == value["indicator_indices"]["B2"]
                                                   for value in pair_features.values()),
            paired_missing_indicator_columns_by_block={name: {a: [value["indicator_column_ids"][k]
                for k in value["indicator_indices"][a]] for a in ARMS} for name, value in pair_features.items()},
            paired_columns_by_block={name: {a: value[a].shape[1] for a in ARMS}
                                     for name, value in pair_features.items()})
    write_json(output/"manifest.json", manifest)
    write_json(output/"complete.json", dict(status=manifest["status"], manifest_sha256=sha(output/"manifest.json")))
    return manifest


def preflight_streamed_score(bundle, contract_path, output, evaluation_path, evaluation_digest,
        numerical_contract, numerical_contract_sha256, pair_operator_code, pair_operator_code_sha256,
        pair_preconditioner_code, pair_preconditioner_code_sha256, score_arm):
    """Exercise the exact streamed SCORE loader/design path without fitting or EVALUATE."""
    require(score_arm in ARMS, 'Preflight SCORE requires one explicit arm')
    require(not output.exists(), 'Output exists; will not overwrite')
    require(re.fullmatch('[0-9a-f]{64}', evaluation_digest) and sha(evaluation_path)==evaluation_digest,
        'Evaluation configuration hash mismatch')
    contract=load_json(contract_path);evaluation=load_json(evaluation_path)
    require(contract.get('schema')==STREAM_SCHEMA,'Preflight SCORE requires streamed v4 bundle')
    require(evaluation.get('prepared_contract_sha256')==sha(contract_path),
        'Evaluation configuration belongs to another prepared catalogue')
    backend=load_pair_operator_backend(numerical_contract,numerical_contract_sha256,
        pair_operator_code,pair_operator_code_sha256,pair_preconditioner_code,
        pair_preconditioner_code_sha256)
    store=BlockBundles(bundle,contract,evaluation)
    require(len(store.blocks)==1,'Distributed SCORE preflight requires one block')
    values,scoped=store.load(0)
    persons,roles,blocks,markers,gt,features,pairs=values
    block=blocks[0];name=block['block_id'];paired=pairs[name]
    components=[p['component'] for p in persons];gt=_block_genotypes(block,markers,gt)
    targets={}
    for role in ('FIT','SELECT'):
        i,j,y,n,info=target_pairs(gt,roles['FIT'],roles[role],components,
            scoped['min_fit_called'],scoped['min_pair_markers'],scoped['max_pairs'])
        weights,_=component_pair_weights(i,j,components)
        targets[role]=(i,j,y,n,weights,info)
    fi,fj,fy,fn,fw,fit_info=targets['FIT'];si,sj,sy,sn,sw,select_info=targets['SELECT']
    x=features[f'{name}_{score_arm}']
    fit_design,imputation=backend.design(x,fi,fj,paired,score_arm,roles['FIT'],require_fit_support=True)
    select_design,reused=backend.design(x,si,sj,paired,score_arm,roles['SELECT'],imputation=imputation)
    require(np.array_equal(reused,imputation),'FIT imputation was not frozen for SELECT')
    output.mkdir(parents=True,mode=0o700)
    receipt=dict(schema='r02_pair_score_cli_preflight_v1',status='COMPLETE_NO_MODEL_FIT',
        view_id=name+'__'+scoped['exclusion_scope'],arm=score_arm,
        roles={key:int(len(value)) for key,value in roles.items()},ridge_alphas=scoped['ridge_alphas'],
        fit_pairs=int(len(fy)),select_pairs=int(len(sy)),fit_eligible_markers=int(fit_info['eligible_markers']),
        select_eligible_markers=int(select_info['eligible_markers']),fit_shape=list(fit_design.shape),
        select_shape=list(select_design.shape),fit_weight_sum=float(fw.sum()),
        select_weight_sum=float(sw.sum()),imputation_columns=int(len(imputation)),
        operator_working_bytes_bound=int(fit_design.working_bytes_bound),models_fitted=0,
        evaluate_targets_computed=False,evaluate_allowed=False,contract_sha256=sha(contract_path),
        evaluation_sha256=evaluation_digest,**backend.metadata())
    write_json(output/'preflight.json',receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument('--evaluation-contract',type=Path)
    parser.add_argument('--evaluation-contract-sha256')
    parser.add_argument('--mode',choices=('legacy','score','select','evaluate'),default='legacy')
    parser.add_argument('--score-arm',choices=ARMS)
    parser.add_argument('--selection-checkpoint',type=Path)
    parser.add_argument('--selection-checkpoint-sha256')
    parser.add_argument('--review',type=Path)
    parser.add_argument('--review-sha256')
    parser.add_argument('--numerical-contract',type=Path)
    parser.add_argument('--numerical-contract-sha256')
    parser.add_argument('--pair-operator-code',type=Path)
    parser.add_argument('--pair-operator-code-sha256')
    parser.add_argument('--pair-preconditioner-code',type=Path)
    parser.add_argument('--pair-preconditioner-code-sha256')
    parser.add_argument('--solver-checkpoint-dir',type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    # The monolithic Python API remains for explicit compatibility tests, not
    # the production v4 entry point. It must never be selected accidentally.
    require(load_json(args.contract).get('schema')!=STREAM_SCHEMA or args.mode in {'score','select','evaluate'},
        'Production v4 requires explicit score/select/evaluate mode; monolithic API is compatibility-only')
    if args.preflight_only:
        require(args.mode=='score' and all(x is None for x in (args.selection_checkpoint,
            args.selection_checkpoint_sha256,args.review,args.review_sha256,args.solver_checkpoint_dir)),
            'Preflight is SCORE-only and cannot consume EVALUATE gates')
        result=preflight_streamed_score(args.bundle,args.contract,args.output_dir,args.evaluation_contract,
            args.evaluation_contract_sha256,args.numerical_contract,args.numerical_contract_sha256,
            args.pair_operator_code,args.pair_operator_code_sha256,args.pair_preconditioner_code,
            args.pair_preconditioner_code_sha256,args.score_arm)
    else:
        result = run(args.bundle,args.contract,args.output_dir,args.evaluation_contract,args.evaluation_contract_sha256,
            mode=args.mode,selection_checkpoint=args.selection_checkpoint,selection_checkpoint_sha256=args.selection_checkpoint_sha256,
            review=args.review,review_sha256=args.review_sha256,numerical_contract=args.numerical_contract,
            numerical_contract_sha256=args.numerical_contract_sha256,pair_operator_code=args.pair_operator_code,
            pair_operator_code_sha256=args.pair_operator_code_sha256,pair_preconditioner_code=args.pair_preconditioner_code,
            pair_preconditioner_code_sha256=args.pair_preconditioner_code_sha256,score_arm=args.score_arm,
            solver_checkpoint_dir=args.solver_checkpoint_dir)
    print(json.dumps({k:result[k] for k in ('status','scientific_status','claim') if k in result}))


if __name__ == "__main__":
    main()
