#!/usr/bin/env python3
"""Train one authenticated, bounded contextual development task.

This consumes individual genotype tensors, not K02/K07 pair summaries. An
explicit task/feature admission is mandatory; no dataset is inferred from a
directory. FIT and SELECT are distinct family/donor components. A fixed locus
catalogue, measurement masks and common context are identical in all arms.
The contract limits each batch; oversized genomes must be tiled upstream,
never truncated here. This executable does not declare population structure.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
import zipfile
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from genomic_context_runtime import (ContextConfig, ContextualModel, LOCUS_DESCRIPTOR_CHROM22,
                                     LOCUS_DESCRIPTOR_LEGACY, Sites, require)
from r02_context_autosomal_pool import (AutosomalPool, AutosomalPoolResult,
    AutosomalSufficientStatistics, RARE_POOLING_MEAN_WITH_EXPLICIT_BURDEN,
    RARE_POOLING_SUM, finalize_rare_sum, finalize_statistics)
from r02_context_checkpoint import CheckpointStore, identity_digest
from r02_context_update_checkpoint import (PartialUpdateStore,
    burden_identity, capture_parameter_gradients, load_completed_budgets, optimizer_start_digest,
    restore_parameter_gradients, update_identity)


SCHEMA = "r02_context_training_v1"
ARMS = ("NONE", "COUNTS", "RARE")
LAI_CONDITIONS = ("NONE", "NO_RARE", "COUNT_MATCHED", "DENSITY_MATCHED", "IDENTITY_SHAM")
SINGLE_CHROMOSOME = "SINGLE_CHROMOSOME_BATCH_V1"
AUTOSOMAL_POOL = "AUTOSOMAL_POOL_V1"
AUTOSOMAL_PERSON_BATCHES = "AUTOSOMAL_PERSON_BATCHES_V1"
ADAM_STATE_PRECISIONS = ("fp32", "fp64")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def configure_torch_cache(path):
    """Set PyTorch's cache without consulting a passwd entry for the host UID."""
    value = os.fspath(path)
    require(bool(value), "nonempty PyTorch cache path required")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", value)
    return os.environ["TORCHINDUCTOR_CACHE_DIR"]


def read_json(path, expected):
    require(sha(path) == expected, "JSON SHA256 mismatch")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tensor_sites(data, prefix, features=None):
    return Sites(torch.from_numpy(data[prefix + "_features"] if features is None else features).float(),
                 torch.from_numpy(data[prefix + "_cm"]).double(),
                 torch.from_numpy(data[prefix + "_chrom"]).long(),
                 torch.from_numpy(data[prefix + "_valid"]))


def event_burden_counts(data):
    """Untransformed carrier/copy/exposure totals for one owned core."""
    mask, dose = data["event_valid"], data["event_dosage"]
    measured = np.where(mask, dose, 0).astype(np.float32)
    core = data["event_core"] & mask
    core_dose = np.where(core, measured, 0)
    # Keep integer totals exact across tiles; conversion to float happens only
    # once after the autosomal sum, matching a single concatenated catalogue.
    return np.stack(((core_dose > 0).sum(1), core_dose.sum(1), core.sum(1)), -1).astype(np.int64)


def blinded_events(data, arm, burden_counts=None):
    """The event grid never depends on a person's carrier pattern.

    Metadata (allele, locus, source frequency) is constant across people. All
    arms know measurement availability. Only RARE sees dosage by locus; COUNTS
    and RARE see total carrier/copy counts and exposure, broadcast uniformly.
    NONE uses the same encoder and local common context with those values zero.
    Counts refer exactly to this admitted batch's locus scope, not implicitly
    to a whole genome. They must match the frozen scientific comparison.
    """
    require(arm in ARMS, "unknown information arm")
    mask, dose = data["event_valid"], data["event_dosage"]
    measured = np.where(mask, dose, 0).astype(np.float32)
    burden = event_burden_counts(data) if burden_counts is None else np.asarray(burden_counts)
    require(burden.shape == (mask.shape[0], 3) and np.isfinite(burden).all() and (burden >= 0).all(),
            "global event burden shape/value differs")
    burden = np.log1p(burden).astype(np.float32)
    if arm == "NONE":
        burden[:, :2] = 0
    actual = measured if arm == "RARE" else np.zeros_like(measured)
    features = np.concatenate((data["event_metadata"], actual[..., None],
                               np.broadcast_to(burden[:, None], (*mask.shape, 3))), -1)
    return tensor_sites(data, "event", features)


def lai_events(data):
    """Build LAI EVENT input; per-query counts are transported separately."""
    mask, dose = data["event_valid"], data["event_dosage"]
    measured = np.where(mask, dose, 0).astype(np.float32)
    references = np.asarray(data.get("event_reference_features"))
    require(references.shape == (*mask.shape, 15) and np.isfinite(references).all()
            and np.array_equal(references, np.broadcast_to(references[:1], references.shape)),
            "LAI EVENT reference features differ")
    features = np.concatenate((data["event_metadata"], measured[..., None], references), -1)
    return tensor_sites(data, "event", features)


def validate_batch(data, contract, stage, action="TRAIN"):
    require(action in ("PREPARE", "TRAIN"), "unknown batch validation action")
    persons = data["person_id"].astype(str).tolist()
    groups = data["component_id"].astype(str).tolist()
    require(len(persons) > 0 and len(set(persons)) == len(persons), "duplicate/empty batch person IDs")
    require(len(groups) == len(persons) and all(groups), "component identifiers required")
    roles = contract["persons"]
    require(all(p in roles and roles[p]["role"] == stage and roles[p]["component"] == g
                for p, g in zip(persons, groups)), "batch role/component differs from frozen manifest")
    if "role" in data:
        require(data["role"].shape == (len(persons),) and set(data["role"].astype(str)) == {stage},
                "serialized role differs from frozen manifest")
    for prefix in ("common", "event"):
        mask = data[prefix + "_valid"]
        require(mask.dtype == bool and mask.ndim == 2 and mask.shape[0] == len(persons), "measurement mask shape/type")
        require(0 < mask.shape[1] <= contract["limits"]["max_" + prefix + "_sites"], "site bound exceeded")
        cm, chrom = data[prefix + "_cm"], data[prefix + "_chrom"]
        require(cm.shape == mask.shape and chrom.shape == mask.shape, "coordinate shape")
        require(np.issubdtype(chrom.dtype, np.integer) and np.isfinite(cm).all()
                and np.isin(chrom, np.arange(1, 23)).all(), "invalid mapped autosomal coordinates")
        require(np.array_equal(cm, np.broadcast_to(cm[:1], cm.shape))
                and np.array_equal(chrom, np.broadcast_to(chrom[:1], chrom.shape)),
                "all people must share the same fixed locus catalogue")
        bp, core = data[prefix + "_pos_bp"], data[prefix + "_core"]
        require(bp.shape == mask.shape and np.issubdtype(bp.dtype, np.integer) and (bp > 0).all()
                and np.array_equal(bp, np.broadcast_to(bp[:1], bp.shape)), "fixed physical positions required")
        require(core.shape == mask.shape and core.dtype == bool, "core ownership mask required")
        require(np.array_equal(core, np.broadcast_to(core[:1], core.shape)),
                "core ownership must be fixed across people, not carrier-dependent")
        padding = data.get(prefix + "_padding")
        if padding is not None:
            require(padding.shape == mask.shape and padding.dtype == bool
                    and np.array_equal(padding, np.broadcast_to(padding[:1], padding.shape)),
                    "padding mask must be fixed across people")
            require(not (padding & (mask | core)).any(), "padding cannot be measured or core-owned")
        keys = list(zip(chrom[0].tolist(), bp[0].tolist()))
        require(len(keys) == len(set(keys)), "duplicate site coordinates: normalize upstream")
        if contract["objective"] == "structure":
            for interval in contract["excluded_intervals"]:
                if "start_bp" in interval:
                    excluded = (chrom == interval["chrom"]) & (bp >= interval["start_bp"]) & (bp <= interval["end_bp"])
                else:
                    excluded = (chrom == interval["chrom"]) & (cm >= interval["start_cm"]) & (cm <= interval["end_cm"])
                require(not (excluded & mask).any(), "withheld target/context leaked into an input grid")
    metadata = data["event_metadata"]
    require(metadata.ndim == 3 and metadata.shape[:2] == data["event_valid"].shape
            and metadata.shape[2] == contract["event_metadata_features"], "event metadata shape")
    require(np.isfinite(metadata).all() and np.array_equal(metadata, np.broadcast_to(metadata[:1], metadata.shape)),
            "event metadata must not contain person-specific carrier information")
    if contract["objective"] == "lai":
        from r02_lai_query_burden import BOUNDARY, COUNT_NAMES, SCOPE
        burden = np.asarray(data.get("query_burden"))
        require(contract.get("event_reference_features") == 15
                and contract.get("event_reference_feature_schema") == "r02_lai_event_reference_features_v1"
                and contract.get("event_reference_ancestry_order") ==
                    ["African", "European", "Native_American"]
                and (isinstance(contract.get("reference_identity"), dict)
                     or isinstance(contract.get("reference_identity_by_case"), dict)),
                "LAI reference specification differs")
        if action == "TRAIN" or "query_burden" in data:
            require(contract.get("lai_burden_scope") == SCOPE
                    and contract.get("lai_burden_boundary") == BOUNDARY
                    and contract.get("lai_burden_count_names") == list(COUNT_NAMES)
                    and contract.get("lai_burden_transform") == "LOG1P_IN_MODEL"
                    and burden.shape == (*data["query_valid"].shape, 3)
                    and np.issubdtype(burden.dtype, np.integer)
                    and (burden >= 0).all(),
                    "LAI query-local burden specification/array differs")
        lai_events(data).check(contract["event_metadata_features"] + 1 + 15,
                               contract["limits"]["max_event_sites"])
    dose, mask = data["event_dosage"], data["event_valid"]
    require(dose.shape == mask.shape and np.isin(dose[mask], (0, 1, 2)).all(), "invalid counted-allele dosage")
    tensor_sites(data, "common").check(contract["common_features"], contract["limits"]["max_common_sites"])
    require((data["common_core"] & data["common_valid"]).any()
            or (data["event_core"] & data["event_valid"]).any(), "entire input core is unavailable")
    if contract["objective"] == "structure":
        pairs, target = data["pairs"], data["target"]
        require(pairs.ndim == 2 and pairs.shape[1] == 2 and np.issubdtype(pairs.dtype, np.integer), "pair indices")
        require(len(pairs) > 0 and len(pairs) <= contract["limits"]["max_targets"]
                and (pairs >= 0).all() and (pairs < len(persons)).all(), "pair index bound")
        require((pairs[:, 0] < pairs[:, 1]).all() and len(set(map(tuple, pairs))) == len(pairs), "noncanonical duplicate pairs")
        require(all(groups[a] != groups[b] for a, b in pairs), "within-component target pairs forbidden")
        require(target.shape == (len(pairs),) and np.isfinite(target).all(), "finite target per pair required")
    else:
        queries = tensor_sites(data, "query")
        queries.check(1, contract["limits"]["max_targets"])
        truth = data["truth_state"]
        # v2 batches keep technical query support independent of label
        # availability.  Falling back to query_valid preserves the exact v1
        # interpretation used by frozen M39 inputs.
        truth_observed = data.get("truth_observed", data["query_valid"])
        require(truth_observed.shape == queries.valid.shape and truth_observed.dtype == bool,
                "truth_observed must be a boolean mask matching query_valid")
        require(not (truth_observed & ~data["query_valid"]).any(),
                "truth_observed cannot extend beyond technical query support")
        require(truth.shape == queries.valid.shape and np.issubdtype(truth.dtype, np.integer)
                and np.isin(truth[truth_observed], np.arange(6)).all(), "invalid six-state truth")
        require(data["query_valid"].any(1).all(), "each person requires measurable query positions")
        require(lai_loss_mask(data).any(1).all(),
                "each person requires observed truth in its owned query core")
    return persons, groups


def validate_batch_receipt(data, entry, contract, resolved=None, action="PREPARE"):
    """Bind excluded views and prepared context halos to the admitted task.

    Raw-array hashes alone do not say which core owns a site or how much
    neighboring context was extracted. The producer receipt fixes those facts;
    a wider search candidate must rebuild its input view instead of silently
    treating unextracted neighboring sites as absent.
    """
    require(action in ("PREPARE", "TRAIN"), "unknown batch-reading action")
    fields = ("receipt_path", "receipt_sha256")
    declared = any(key in entry for key in fields)
    require(declared or contract["scope"] == "SYNTHETIC_CODE_TEST",
            "development batch requires an authenticated producer receipt")
    if not declared:
        return
    require(all(entry.get(key) for key in fields), "incomplete producer receipt reference")
    receipt = read_json(entry["receipt_path"], entry["receipt_sha256"])
    if (contract["objective"] == "lai" and
            receipt.get("schema") == "r02_lai_factorized_training_link_receipt_v1"):
        from r02_lai_training_campaign import validate_loaded_training_link
        stages = set(data["role"].astype(str).tolist())
        require(len(stages) == 1, "training-link batch contains multiple roles")
        validate_loaded_training_link(data, entry, receipt, contract, resolved,
                                      next(iter(stages)), action)
        return
    if contract["objective"] == "lai" and receipt.get("schema") == "r02_lai_regional_truth_factorized_link_receipt_v1":
        from r02_lai_regional_channel_bridge import array_sha256
        from r02_lai_query_burden import BOUNDARY, COUNT_NAMES, SCOPE
        require(entry.get("format") == "LAI_FACTORIZED_LINK_V1"
                and receipt.get("request_sha256") == entry["sha256"]
                and receipt.get("status") == "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION"
                and receipt.get("arrays") == sorted(data)
                and receipt.get("reconstructed_arrays_sha256") ==
                    {name: array_sha256(value, np) for name, value in data.items()},
                "factorized LAI receipt does not authenticate these arrays")
        require(receipt.get("lai_burden_scope") == contract.get("lai_burden_scope") == SCOPE
                and receipt.get("lai_burden_boundary") == contract.get("lai_burden_boundary") == BOUNDARY
                and receipt.get("lai_burden_count_names") == contract.get("lai_burden_count_names") == list(COUNT_NAMES)
                and resolved is not None
                and receipt.get("rare_radius_cm") == resolved.get("rare_radius_cm"),
                "factorized LAI scope/radius differs")
        require(action != "TRAIN", "factorized PREPARE link cannot authorize TRAIN")
        return
    if contract["objective"] == "lai" and receipt.get("schema") == "r02_lai_regional_truth_join_receipt_v2":
        from r02_lai_query_burden import BOUNDARY, COUNT_NAMES, SCOPE
        require(receipt.get("status") == "REGIONAL_TRUTH_CONTROL_AND_QUERY_BURDEN_PREPARED_NO_TRAINING"
                and receipt.get("batch_sha256") == entry["sha256"]
                and receipt.get("arrays") == sorted(data)
                and "query_burden" in data, "LAI query-burden receipt does not authenticate this batch")
        require(receipt.get("lai_burden_scope") == contract.get("lai_burden_scope") == SCOPE
                and receipt.get("lai_burden_boundary") == contract.get("lai_burden_boundary") == BOUNDARY
                and receipt.get("lai_burden_count_names") == contract.get("lai_burden_count_names") == list(COUNT_NAMES)
                and resolved is not None
                and receipt.get("rare_radius_cm") == resolved.get("rare_radius_cm"),
                "LAI query-burden scope/radius differs")
        require(action != "TRAIN",
                "PREPARE query-burden receipt cannot train; admitted materialized-control receipt required")
        return
    if contract["objective"] == "lai" and receipt.get("schema") == "r02_lai_neural_input_batch_v1":
        require(receipt.get("status") == "ASSEMBLED_NOT_TRAINED_OR_EVALUATED"
                and receipt.get("batch_sha256") == entry["sha256"]
                and receipt.get("arrays") == sorted(data),
                "LAI input receipt does not authenticate this batch")
        expected_role = receipt.get("expected_role")
        require(expected_role in ("FIT", "SELECT")
                and set(data["role"].astype(str).tolist()) == {expected_role}
                and receipt.get("n_people") == len(data["person_id"])
                and set(receipt.get("controls", [])) == {
                    "NONE", "NO_RARE", "COUNT_MATCHED", "DENSITY_MATCHED", "IDENTITY_SHAM"},
                "LAI input people, role, or registered controls differ")
        require(receipt.get("controls_materialized") is False
                and receipt.get("training_authorized") is False
                and receipt.get("model_selection_authorized") is False
                and receipt.get("evaluate_opened") is False,
                "LAI assembly receipt may not claim training, selection, evaluation, or materialized controls")
        require(action != "TRAIN",
                "LAI assembly-only receipt cannot enter training; materialized-control batch receipt required")
        require(receipt.get("reference_identity") == contract.get("reference_identity")
                and receipt.get("reference_join_key") == ["chromosome", "position_bp", "REF", "ALT"]
                and receipt.get("reference_orientation_fields") == ["event_counted_allele", "event_is_alt"],
                "LAI receipt reference identity, join key, or orientation fields differ")
        from r02_lai_reference_features import reference_axes_identity, reference_features
        rebuilt = reference_features(data, np)
        require(np.array_equal(data["event_reference_features"], rebuilt),
                "LAI reference features differ from the explicit REF/ALT join")
        axes_identity = reference_axes_identity(data, np)
        require(all(receipt.get(key) == value for key, value in axes_identity.items()),
                "LAI receipt person/role or locus/allele axes differ")
        return
    expected_schema = ("r02_context_reserved_batch_v1" if contract.get("evaluation_stage") == "EVALUATE"
                       else "r02_genomic_event_training_batch_v1")
    require(receipt.get("schema") == expected_schema
            and receipt.get("arrays_sha256") == entry["sha256"], "producer receipt does not authenticate this batch")
    target_hash = contract.get("target_contract_sha256", "")
    require(isinstance(target_hash, str) and bool(re.fullmatch(r"[0-9a-f]{64}", target_hash))
            and receipt.get("target_contract_sha256") == target_hash,
            "producer target-definition contract differs from the admitted task")
    require(receipt.get("missing_policy") == "VALID_MASK_NOT_CARRIER"
            and receipt.get("n_people") == len(data["person_id"]), "producer person/missing policy differs")
    require(len(receipt.get("common_features", [])) == contract["common_features"]
            and len(receipt.get("event_metadata_features", [])) == contract["event_metadata_features"],
            "producer feature definitions differ from batch dimensions")
    lineage = receipt.get("input_lineage", {})
    require(all(isinstance(lineage.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", lineage[key])
                for key in ("chromosome_manifest_sha256", "event_contract_sha256", "source_contract_sha256")),
            "producer source lineage is incomplete")
    bounds = receipt.get("core_bp", [])
    require(len(bounds) == 2 and all(type(x) is int for x in bounds)
            and 1 <= bounds[0] <= bounds[1], "producer physical core is invalid")
    chromosomes = set()
    for prefix in ("common", "event"):
        padding = data.get(prefix + "_padding")
        require(padding is not None, "producer padding mask is required")
        chromosomes.update(data[prefix + "_chrom"].reshape(-1).tolist())
        bp = data[prefix + "_pos_bp"]
        expected = (bp >= bounds[0]) & (bp <= bounds[1]) & ~padding
        require(np.array_equal(expected, data[prefix + "_core"]),
                "core ownership differs from authenticated physical core")
    require(len(chromosomes) == 1, "producer receipt authenticates only one chromosome")
    if contract["objective"] == "structure":
        require(receipt.get("n_pairs") == len(data["pairs"]), "producer pair count differs")
        intervals = [x for x in contract["excluded_intervals"] if x["chrom"] in chromosomes]
        require(len(intervals) <= 1 and all("start_bp" in x and "end_bp" in x for x in intervals),
                "producer requires a single exact physical exclusion per chromosome")
        expected_exclusion = [intervals[0]["start_bp"], intervals[0]["end_bp"]] if intervals else None
        require(receipt.get("excluded_bp") == expected_exclusion,
                "producer exclusion differs from admitted physical exclusion")
    common_radius = receipt.get("common_radius_cm")
    rare_radius, depth = receipt.get("rare_radius_cm"), receipt.get("attention_depth")
    require(type(common_radius) in (int, float) and math.isfinite(common_radius) and common_radius > 0,
            "producer common halo radius is invalid")
    require(type(depth) is int and depth >= 0
            and ((rare_radius is None and depth == 0)
                 or (type(rare_radius) in (int, float) and math.isfinite(rare_radius)
                     and rare_radius > 0 and depth > 0)), "producer rare halo radius/depth is invalid")
    if resolved is not None:
        config = ContextConfig.from_recipe(resolved)
        # Deep Sets depth counts dense layers, not genomic message-passing hops.
        requested_rare = 0 if config.family == "structure_deep_sets" else config.depth * config.rare_radius_cm
        available_rare = depth * (rare_radius or 0)
        require(requested_rare <= available_rare
                and requested_rare + config.common_radius_cm <= available_rare + common_radius,
                "resolved radii/depth exceed prepared context halos; re-extract the view")


def load_batch(entry, contract, stage, resolved=None, action="PREPARE"):
    path = Path(entry["path"])
    if entry.get("format") == "LAI_FACTORIZED_TRAINING_LINK_V1":
        from r02_lai_training_campaign import load_training_link
        require(action == "TRAIN" and stage in ("FIT", "SELECT"),
                "admitted factorized training links permit TRAIN for FIT/SELECT only")
        require(contract["objective"] == "lai", "factorized training links require LAI")
        data = load_training_link(entry, contract, resolved, stage,
                                  max_uncompressed_bytes=contract["limits"]["max_batch_bytes"])
        validate_batch(data, contract, stage, action)
        validate_batch_receipt(data, entry, contract, resolved, action)
        return data
    if entry.get("format") == "LAI_FACTORIZED_LINK_V1":
        from r02_lai_neural_input import load_factorized_link
        require(action == "PREPARE" and stage in ("FIT", "SELECT"),
                "factorized links only permit PREPARE for FIT/SELECT; EVALUATE is closed")
        require(contract["objective"] == "lai", "factorized regional links require LAI")
        data = load_factorized_link(path, entry["sha256"], entry["receipt_path"],
                                   entry["receipt_sha256"], action=action,
                                   max_uncompressed_bytes=contract["limits"]["max_batch_bytes"])
        validate_batch(data, contract, stage, action)
        validate_batch_receipt(data, entry, contract, resolved, action)
        return data
    require(sha(path) == entry["sha256"], "genotype batch SHA256 mismatch")
    with zipfile.ZipFile(path) as archive:
        require(sum(item.file_size for item in archive.infolist()) <= contract["limits"]["max_batch_bytes"],
                "uncompressed batch exceeds admitted memory budget")
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    validate_batch(data, contract, stage, action)
    validate_batch_receipt(data, entry, contract, resolved, action)
    return data


def load_pair_target(entry, contract, stage, request):
    """Delegate the K03 target and weighting specification to its sole helper."""
    from r02_k03_neural_target import load_pair_target as authenticated_target
    return authenticated_target(entry, contract, stage, request)


def dataset_mode(contract):
    """Keep historical single-chromosome contracts byte-compatible."""
    return contract.get("dataset_mode", SINGLE_CHROMOSOME)


def autosomal_tile(view):
    """Adapt one authenticated EventChromosome view without copying semantics."""
    data = {key: np.asarray(view[key]) for key in ("person_id", "component_id", "role")}
    for source, prefix in (("common", "common"), ("events", "event")):
        for field in ("cm", "chrom", "valid"):
            data[prefix + "_" + field] = np.asarray(view[source][field])
        for field in ("pos_bp", "core", "padding"):
            data[prefix + "_" + field] = np.asarray(view[source + "_" + field])
    data["common_features"] = np.asarray(view["common"]["features"])
    data["event_metadata"] = np.asarray(view["events"]["features"][:, :, 1:])
    data["event_dosage"] = np.asarray(view["events"]["features"][:, :, 0] * 2)
    return data


def open_autosomal_entry(entry, contract, stage, resolved, authenticate_windows=False,
                         phase_metrics=None):
    """Bind an autosomal request to the already authenticated target batch."""
    # Keep the historical single-chromosome trainer free of the VCF/materializer
    # dependency chain. The autosomal reader is loaded only for this explicit mode.
    from r02_context_autosomal_dataset import open_autosomal
    required = ("autosomal_request_path", "autosomal_request_sha256")
    require(all(entry.get(key) for key in required), "explicit autosomal request path/hash required")
    dataset = open_autosomal(entry["autosomal_request_path"], entry["autosomal_request_sha256"])
    expected_request_schema = ("r02_context_autosomal_person_batches_v1"
                               if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES
                               else "r02_context_autosomal_dataset_request_v1")
    request_schema = dataset.request.get("schema")
    require(request_schema == expected_request_schema or
            (request_schema is None and contract["scope"] == "SYNTHETIC_CODE_TEST" and
             dataset_mode(contract) == AUTOSOMAL_POOL),
            "dataset mode and autosomal request schema/lineage differ")
    target=(load_pair_target(entry,contract,stage,dataset.request)
            if dataset_mode(contract)==AUTOSOMAL_PERSON_BATCHES else load_batch(entry,contract,stage,resolved))
    require(tuple(target["person_id"].astype(str)) == dataset.ids and
            tuple(target["component_id"].astype(str)) == dataset.components and
            tuple(target.get("role", np.full(len(dataset.ids), stage)).astype(str)) == dataset.roles,
            "autosomal request and target person/role/component order differ")
    config = ContextConfig.from_recipe(resolved)
    require(contract["objective"] == "structure" and config.family in
            ("structure_deep_sets", "structure_local_attention"),
            "autosomal inputs require a structure model")
    request = dataset.request
    expected_depth = config.depth if config.family == "structure_local_attention" else 0
    require(request["common_radius_cm"] == config.common_radius_cm and
            request["rare_radius_cm"] == config.rare_radius_cm and
            request["attention_depth"] == expected_depth,
            "autosomal request radii/depth differ from resolved structure configuration")
    if dataset_mode(contract) != AUTOSOMAL_PERSON_BATCHES and all(entry.get(key) for key in ("receipt_path", "receipt_sha256")):
        receipt = read_json(entry["receipt_path"], entry["receipt_sha256"])
        lineage = receipt.get("input_lineage", {})
        require(receipt.get("block_id") == request["block_id"] and
                receipt.get("exclusion_scope") == request["exclusion_scope"] and
                (lineage.get("event_contract_sha256"), lineage.get("source_contract_sha256")) == dataset.lineage,
                "autosomal request and supervised target lineage/block/exclusion differ")
    if contract.get("dose_relocation"):
        from r02_structure_identity import RelocatedAutosomalDataset
        record = contract["dose_relocation"]["maps_by_request"][entry["autosomal_request_sha256"]]
        dataset = RelocatedAutosomalDataset(dataset, entry["autosomal_request_sha256"],
                                             record["path"], record["sha256"])
    if authenticate_windows:
        seen = 0
        for batch_index in range(_dataset_batch_count(dataset)):
            for view in _iter_dataset_batch_measured(
                    dataset, batch_index, 0, phase_metrics,
                    "AUTOSOMAL_PREFLIGHT_VALIDATION"):
                tile = autosomal_tile(view)
                tensor_sites(tile, "common").check(contract["common_features"], contract["limits"]["max_common_sites"])
                blinded_events(tile, "NONE").check(contract["event_metadata_features"] + 4,
                                                    contract["limits"]["max_event_sites"])
                seen += 1
        require(seen > 0, "empty autosomal request")
    return target, dataset


def _dataset_batch_count(dataset):
    return int(getattr(dataset, "n_person_batches", 1))


def _dataset_batch_axis(dataset, batch_index):
    batches = getattr(dataset, "person_batches", None)
    if batches is None:
        require(batch_index == 0, "legacy autosomal dataset has one person axis")
        return tuple(dataset.ids), tuple(dataset.roles), tuple(dataset.components)
    return tuple(batches[batch_index][0]), tuple(batches[batch_index][1]), tuple(batches[batch_index][2])


def _iter_dataset_batch(dataset, batch_index, start_tile_index=0):
    if hasattr(dataset, "iter_person_batch"):
        if start_tile_index:
            require(hasattr(dataset, "tiles_per_person_batch"),
                    "dataset cannot resume from a flat autosomal tile cursor")
            return dataset.iter_person_batch(batch_index,start_tile_index=start_tile_index)
        return dataset.iter_person_batch(batch_index)
    require(batch_index == 0 and start_tile_index == 0,
            "legacy autosomal dataset cannot resume inside its person batch")
    return iter(dataset)


def _process_io_bytes():
    """Return Linux process I/O counters when exposed; otherwise unknown."""
    try:
        rows = {}
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            rows[key] = int(value.strip())
        return rows.get("read_bytes"), rows.get("rchar")
    except (OSError, ValueError):
        return None, None


def _array_bytes(value):
    """Count arrays delivered by one view; this is useful materialization, not disk I/O."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_array_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(item) for item in value)
    return 0


def _record_phase_metric(metrics, phase, seconds, views, materialized_bytes=0,
                         process_read_bytes=0, process_read_chars=0,
                         source_bytes_hashed=0, cache_hit=False):
    """Accumulate process-local timing and emitted-view counts for one phase."""
    if metrics is None:
        return
    require(isinstance(metrics, dict) and isinstance(phase, str) and phase and
            isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds >= 0 and
            type(views) is int and views >= 0 and
            all(type(value) is int and value >= 0 for value in
                (materialized_bytes, process_read_bytes, process_read_chars,
                 source_bytes_hashed)) and type(cache_hit) is bool,
            "invalid autosomal phase metric")
    row = metrics.setdefault(phase, {"seconds": 0.0, "views": 0,
        "materialized_bytes": 0, "process_read_bytes": 0,
        "process_read_chars": 0, "source_bytes_hashed": 0,
        "invocations": 0, "cache_hits": 0})
    require(set(row) == {"seconds", "views", "materialized_bytes",
            "process_read_bytes", "process_read_chars", "source_bytes_hashed",
            "invocations", "cache_hits"},
            "autosomal phase metric fields differ")
    row["seconds"] += float(seconds)
    row["views"] += views
    row["materialized_bytes"] += materialized_bytes
    row["process_read_bytes"] += process_read_bytes
    row["process_read_chars"] += process_read_chars
    row["source_bytes_hashed"] += source_bytes_hashed
    row["invocations"] += 1
    row["cache_hits"] += int(cache_hit)


def _iter_dataset_batch_measured(dataset, batch_index, start_tile_index, metrics, phase):
    """Measure the complete consumer phase, including work between yields."""
    started = time.monotonic(); views = materialized_bytes = 0
    read_before, chars_before = _process_io_bytes()
    cache = getattr(dataset, "chunk_cache", None)
    hashed_before = int(getattr(cache, "bytes_hashed", 0))
    try:
        for view in _iter_dataset_batch(dataset, batch_index, start_tile_index):
            views += 1
            materialized_bytes += _array_bytes(view)
            yield view
    finally:
        read_after, chars_after = _process_io_bytes()
        hashed_after = int(getattr(cache, "bytes_hashed", hashed_before))
        _record_phase_metric(metrics, phase, time.monotonic() - started, views,
            materialized_bytes,
            max(0, read_after - read_before) if None not in (read_before, read_after) else 0,
            max(0, chars_after - chars_before) if None not in (chars_before, chars_after) else 0,
            max(0, hashed_after - hashed_before))


def _dataset_batch_tile_count(dataset, batch_index):
    count=getattr(dataset,"tiles_per_person_batch",None)
    if count is not None:
        require(type(count)is int and count>0,"invalid autosomal tile inventory")
        return count
    require(batch_index==0,"person-batch dataset must expose its tile inventory")
    return None


def _concatenate_pool_results(results, expected_ids):
    require(results and sum(len(ids) for ids, _ in results) == len(expected_ids),
            "person batches do not cover the complete frozen axis")
    ids = tuple(person for batch_ids, _ in results for person in batch_ids)
    require(ids == tuple(expected_ids), "person batches reordered or omitted people")
    values = []
    for field in AutosomalPoolResult.__dataclass_fields__:
        parts = [getattr(result, field) for _, result in results]
        if field == "chromosomes":
            require(all(value == parts[0] for value in parts), "person batches cover different autosomes")
            values.append(parts[0])
        else:
            values.append(torch.cat(parts, 0))
    return AutosomalPoolResult(*values)


def validate_training_entry(entry, contract, stage, resolved=None, authenticate_windows=False,
                            phase_metrics=None):
    if dataset_mode(contract) == SINGLE_CHROMOSOME:
        return load_batch(entry, contract, stage, resolved, action="TRAIN"), None
    require(resolved is not None, "autosomal entry validation requires resolved configuration")
    return open_autosomal_entry(entry, contract, stage, resolved, authenticate_windows,
                                phase_metrics)


def _entry_burden_cache_key(entry):
    """In-process key for counts authenticated by ``PartialUpdateStore``."""
    fields = ("path", "sha256", "receipt_sha256", "autosomal_request_sha256")
    require(all(entry.get(field) for field in fields),
            "autosomal burden cache requires exact target/request identities")
    return tuple(str(entry[field]) for field in fields)


def predict_training_entry(model, entry, contract, stage, resolved, arm,
                           burden_cache=None, phase_metrics=None):
    data, dataset = validate_training_entry(entry, contract, stage, resolved)
    if dataset is None:
        return model(data, arm), data, None
    prediction, pooled = model.forward_autosomal(
        dataset, data, arm, burden_cache=burden_cache, phase_metrics=phase_metrics)
    return prediction, data, pooled


class TaskModel(nn.Module):
    def __init__(self, resolved, contract):
        super().__init__()
        config = ContextConfig.from_recipe(resolved)
        self.objective = contract["objective"]
        require(config.family.startswith(self.objective + "_"), "wrong model family")
        limits = contract["limits"]
        cnn_query_config = resolved.get("cnn_query_config") if config.family == "lai_cnn" else None
        if cnn_query_config is not None:
            require(resolved.get("query_max_gap_cm") == contract.get("query_max_gap_cm"),
                    "CNN resolved query gap differs from training specification")
        event_width = contract["event_metadata_features"] + (1 + contract.get("event_reference_features", 0)
            if self.objective == "lai" else 4)
        self.encoder = ContextualModel(config, contract["common_features"], event_width,
            max_common_sites=limits["max_common_sites"], max_events=limits["max_event_sites"],
            max_queries=limits["max_targets"], query_max_gap_cm=contract.get("query_max_gap_cm"),
            context_execution=contract.get("context_execution"), cnn_query_config=cnn_query_config,
            lai_query_burden_features=3 if self.objective == "lai" else 0)
        self.head = nn.Linear(2 * config.width, 1) if self.objective == "structure" else None

    def structure_prediction(self, embedding, data):
        pairs = torch.from_numpy(data["pairs"]).long()
        left, right = embedding[pairs[:, 0]], embedding[pairs[:, 1]]
        return self.head(torch.cat(((left - right).abs(), left * right), -1)).squeeze(-1)

    def forward(self, data, arm):
        common = tensor_sites(data, "common")
        events = lai_events(data) if self.objective == "lai" else blinded_events(data, arm)
        if self.objective == "structure":
            # Halos supply context only. Pool every owned site exactly once.
            encoded = self.encoder.common_values(common)
            common_core = torch.from_numpy(data["common_core"]) & common.valid
            baseline = torch.where(common_core[..., None], encoded, 0).sum(1) / common_core.sum(1).clamp_min(1)[:, None]
            values = self.encoder.event_values(common, encoded, events)
            event_core = torch.from_numpy(data["event_core"]) & events.valid
            rare = finalize_rare_sum(torch.where(event_core[..., None], values, 0).sum(1),
                                     event_core.sum(1), self.encoder.person_rho,
                                     self.encoder.config.rare_pooling)
            z = baseline + rare
            return self.structure_prediction(z, data)
        return self.encoder.predict_lai(common, tensor_sites(data, "query"),
            torch.from_numpy(data["baseline_probabilities"]).float(), events, include_rare=True,
            query_burden=torch.from_numpy(data["query_burden"]).float())

    @classmethod
    def _validated_cached_burdens(cls, dataset, cached):
        """Reject cached counts unless every person batch and tile is identical."""
        required = {"ids_by_batch", "tiles_per_batch", "totals", "identities"}
        require(isinstance(cached, dict) and set(cached) == required,
                "cached autosomal burden fields differ")
        n_batches = _dataset_batch_count(dataset)
        expected_ids = tuple(_dataset_batch_axis(dataset, index)[0]
                             for index in range(n_batches))
        expected_tiles = tuple(_dataset_batch_tile_count(dataset, index)
                               for index in range(n_batches))
        require(all(value is not None for value in expected_tiles) and
                tuple(tuple(map(str, ids)) for ids in cached["ids_by_batch"]) == expected_ids and
                tuple(cached["tiles_per_batch"]) == expected_tiles and
                len(cached["totals"]) == len(cached["identities"]) == n_batches,
                "cached autosomal burden person/tile inventory differs")
        request = getattr(dataset, "request", None)
        require(isinstance(request, dict) and isinstance(request.get("tiles"), dict),
                "cached burdens require an explicit autosomal request")
        normalized_identities = []
        for batch_index, (ids, count, totals, identities) in enumerate(zip(
                expected_ids, expected_tiles, cached["totals"], cached["identities"])):
            totals = np.asarray(totals)
            require(totals.shape == (len(ids), 3) and
                    np.issubdtype(totals.dtype, np.integer) and
                    np.isfinite(totals).all() and (totals >= 0).all(),
                    "cached autosomal burden total shape/value differs")
            expected = []
            for chrom in tuple(str(value) for value in range(1, 23)):
                tiles = request["tiles"].get(chrom)
                require(isinstance(tiles, list) and tiles,
                        "cached autosomal burden request tile inventory differs")
                expected.extend((chrom, index, index + 1 == len(tiles), tuple(core),
                                 batch_index, n_batches)
                                for index, core in enumerate(tiles))
            expected = tuple(expected)
            actual = tuple(identities)
            require(len(expected) == count and actual == expected,
                    "cached autosomal burden tile identity differs")
            normalized_identities.append(actual)
        return tuple(cached["totals"]), tuple(normalized_identities)

    def forward_autosomal(self, dataset, target_data, arm, burden_cache=None,
                           phase_metrics=None):
        """Encode all owned autosomal cores and apply the existing head once.

        The first pass obtains the same global burden features as concatenating
        every owned event locus. The second pass builds the ordinary reference
        graph. Training uses ``backward_autosomal`` below to bound that graph.
        """
        require(self.objective == "structure" and self.encoder.config.family in
                ("structure_deep_sets", "structure_local_attention"),
                "autosomal pooling requires a structure model")
        cached = (None if burden_cache is None else
                  self._validated_cached_burdens(dataset, burden_cache))
        results = []
        for batch_index in range(_dataset_batch_count(dataset)):
            ids, roles, _ = _dataset_batch_axis(dataset, batch_index)
            if cached is None:
                totals = self._autosomal_burdens(
                    dataset, batch_index, phase_metrics, "AUTOSOMAL_SCORE_BURDENS")
                expected_identities = None
            else:
                totals = cached[0][batch_index]
                expected_identities = cached[1][batch_index]
                _record_phase_metric(phase_metrics, "AUTOSOMAL_SCORE_BURDENS",
                                     0.0, 0, cache_hit=True)
            pool = AutosomalPool(ids, roles)
            seen = 0
            for view in _iter_dataset_batch_measured(
                    dataset, batch_index, 0, phase_metrics, "AUTOSOMAL_SCORE_FORWARD"):
                if expected_identities is not None:
                    require(self._autosomal_identity(view) == expected_identities[seen],
                            "autosomal view changed since burden counts were authenticated")
                seen += 1
                tile = autosomal_tile(view)
                common, events = tensor_sites(tile, "common"), blinded_events(tile, arm, totals)
                encoded = self.encoder.common_values(common)
                event_values = self.encoder.event_values(common, encoded, events)
                pool.add(int(view["chromosome"]), ids, roles,
                         encoded, torch.from_numpy(tile["common_core"]), common.valid,
                         event_values, torch.from_numpy(tile["event_core"]), events.valid,
                         torch.from_numpy(tile["event_dosage"]),
                         tile_index=view["autosomal_tile_index"], last_tile=view["autosomal_last_tile"],
                         core_bp=view["core_bp"])
            require(expected_identities is None or seen == len(expected_identities),
                    "autosomal score omitted cached burden views")
            results.append((ids, pool.finalize(self.encoder.person_rho,
                                               self.encoder.config.rare_pooling)))
        result = _concatenate_pool_results(results, dataset.ids)
        return self.structure_prediction(result.embedding, target_data), result

    @staticmethod
    def _autosomal_identity(view):
        return (str(view["chromosome"]), int(view["autosomal_tile_index"]),
                bool(view["autosomal_last_tile"]), tuple(view["core_bp"]),
                int(view.get("person_batch_index", 0)), int(view.get("person_batch_count", 1)))

    @staticmethod
    def _autosomal_burdens(dataset, batch_index=0, phase_metrics=None,
                            phase="AUTOSOMAL_BURDENS"):
        totals = None
        for view in _iter_dataset_batch_measured(
                dataset, batch_index, 0, phase_metrics, phase):
            value = event_burden_counts(autosomal_tile(view))
            totals = value if totals is None else totals + value
        require(totals is not None, "empty autosomal request")
        return totals

    def _autosomal_tile_values(self, view, arm, totals):
        tile = autosomal_tile(view)
        common, events = tensor_sites(tile, "common"), blinded_events(tile, arm, totals)
        encoded = self.encoder.common_values(common)
        event_values = self.encoder.event_values(common, encoded, events)
        common_core = torch.from_numpy(tile["common_core"])
        event_core = torch.from_numpy(tile["event_core"])
        cmask, rmask = common_core & common.valid, event_core & events.valid
        csum = torch.where(cmask[..., None], encoded, 0).sum(1)
        rsum = torch.where(rmask[..., None], event_values, 0).sum(1)
        return tile, common, events, encoded, event_values, csum, rsum

    def _aggregate_autosomal_backward(self, dataset, arm, phase_metrics=None):
        """Rebuild deterministic sufficient sums and per-tile RNG identities."""
        records=[]
        with torch.no_grad():
            for batch_index in range(_dataset_batch_count(dataset)):
                ids,roles,_=_dataset_batch_axis(dataset,batch_index)
                totals=self._autosomal_burdens(dataset,batch_index,phase_metrics);pool=AutosomalPool(ids,roles)
                rng_states=[];identities=[]
                for view in _iter_dataset_batch_measured(
                        dataset,batch_index,0,phase_metrics,"AUTOSOMAL_AGGREGATION"):
                    rng_states.append(torch.get_rng_state().clone());identities.append(self._autosomal_identity(view))
                    tile,common,events,encoded,event_values,_,_=self._autosomal_tile_values(view,arm,totals)
                    pool.add(int(view["chromosome"]),ids,roles,encoded,torch.from_numpy(tile["common_core"]),common.valid,
                             event_values,torch.from_numpy(tile["event_core"]),events.valid,torch.from_numpy(tile["event_dosage"]),
                             tile_index=view["autosomal_tile_index"],last_tile=view["autosomal_last_tile"],core_bp=view["core_bp"])
                require(bool(rng_states),"empty autosomal person batch")
                records.append((ids,totals,rng_states,identities,pool.finalize_sums()))
        require(tuple(p for ids,*_ in records for p in ids)==tuple(dataset.ids),
                "person batches reordered or omitted people")
        statistics=AutosomalSufficientStatistics(*[
            (records[0][4].chromosomes if field=="chromosomes" else
             torch.cat([getattr(record[4],field) for record in records],0))
            for field in AutosomalSufficientStatistics.__dataclass_fields__])
        require(all(record[4].chromosomes==statistics.chromosomes for record in records),
                "person batches cover different autosomes")
        return records, statistics

    @staticmethod
    def _statistics_state(statistics):
        return {field:(getattr(statistics,field).detach() if
                       torch.is_tensor(getattr(statistics,field)) else getattr(statistics,field))
                for field in AutosomalSufficientStatistics.__dataclass_fields__}

    @staticmethod
    def _statistics_from_state(state):
        require(isinstance(state,dict)and
                set(state)==set(AutosomalSufficientStatistics.__dataclass_fields__),
                "partial autosomal statistic inventory differs")
        return AutosomalSufficientStatistics(**state)

    @staticmethod
    def _pool_accumulator_state(pool):
        return {"person_ids":pool.person_ids,"roles":pool.roles,"expected":pool.expected,
                "seen":tuple(pool.seen),"current":pool.current,"next_tile":pool.next_tile,
                "last_core_end":pool.last_core_end,
                "common_sum":pool.common_sum,"rare_sum":pool.rare_sum,
                "common_valid":pool.common_valid,"common_missing":pool.common_missing,
                "rare_valid":pool.rare_valid,"rare_missing":pool.rare_missing,
                "rare_zero":pool.rare_zero}

    @staticmethod
    def _pool_accumulator_from_state(ids,roles,state):
        required={"person_ids","roles","expected","seen","current","next_tile","last_core_end",
                  "common_sum","rare_sum","common_valid","common_missing","rare_valid",
                  "rare_missing","rare_zero"}
        require(isinstance(state,dict)and set(state)==required and
                tuple(state["person_ids"])==tuple(ids)and tuple(state["roles"])==tuple(roles),
                "partial autosomal accumulator identity differs")
        pool=AutosomalPool(ids,roles)
        require(tuple(state["expected"])==pool.expected and not state.get("finalized",False),
                "partial autosomal accumulator chromosome inventory differs")
        pool.seen=list(state["seen"]);pool.current=state["current"]
        pool.next_tile=state["next_tile"];pool.last_core_end=state["last_core_end"]
        for name in ("common_sum","rare_sum","common_valid","common_missing",
                     "rare_valid","rare_missing","rare_zero"):
            value=state[name]
            require(value is None or(torch.is_tensor(value)and bool(torch.isfinite(value).all())),
                    "partial autosomal accumulator contains invalid values")
            setattr(pool,name,value)
        return pool

    @classmethod
    def _record_state(cls,record):
        ids,totals,rng_states,identities,statistics=record
        return {"ids":ids,"totals":totals,"rng_states":rng_states,
                "identities":identities,"statistics":cls._statistics_state(statistics)}

    @classmethod
    def _record_from_state(cls,state):
        require(isinstance(state,dict)and
                set(state)=={"ids","totals","rng_states","identities","statistics"},
                "partial autosomal record inventory differs")
        return (tuple(state["ids"]),state["totals"],list(state["rng_states"]),
                list(state["identities"]),cls._statistics_from_state(state["statistics"]))

    @staticmethod
    def _detached_pool(pooled):
        return AutosomalPoolResult(*(value.detach() if torch.is_tensor(value) else value for value in (
            pooled.embedding, pooled.common_mean, pooled.rare_embedding,
            pooled.common_valid_loci, pooled.common_missing_loci, pooled.rare_valid_loci,
            pooled.rare_missing_loci, pooled.rare_valid_zero_dose_loci, pooled.chromosomes)))

    @staticmethod
    def _pool_state(pooled):
        return {field: (getattr(pooled, field).detach() if
                        torch.is_tensor(getattr(pooled, field)) else getattr(pooled, field))
                for field in AutosomalPoolResult.__dataclass_fields__}

    @staticmethod
    def _pool_from_state(state):
        require(isinstance(state, dict) and
                set(state)==set(AutosomalPoolResult.__dataclass_fields__),
                "partial pooled-statistic inventory differs")
        return AutosomalPoolResult(**state)

    def backward_autosomal(self, dataset, target_data, arm, update_checkpoint=None,
                            optimizer=None, resumed=None,
                            interrupt_after_recomputed=None,
                            interrupt_after_burdened=None,
                            interrupt_after_aggregated=None, phase_metrics=None):
        """Exact pooled loss/backward with at most one tile encoder graph live.

        A no-grad aggregation records the CPU Torch RNG state before each tile.
        The pooled head is differentiated once through leaf sufficient sums.
        Each tile is then recomputed with its original RNG state and receives
        the corresponding sum gradient. Restoring the post-forward RNG makes
        this trajectory identical to the ordinary reference forward, including
        dropout, while releasing every tile graph before reading the next tile.
        """
        require(self.training and self.objective == "structure" and self.encoder.config.family in
                ("structure_deep_sets", "structure_local_attention"),
                "bounded autosomal backward requires a training structure model")
        require((update_checkpoint is None)==(optimizer is None) and
                (resumed is None or update_checkpoint is not None),
                "partial update store and optimizer must be supplied together")
        interruptions=(interrupt_after_burdened,interrupt_after_aggregated,
                       interrupt_after_recomputed)
        require(all(value is None or(type(value)is int and value>0)for value in interruptions)and
                (all(value is None for value in interruptions)or update_checkpoint is not None),
                "test interruption requires a positive cursor and checkpoint store")
        if update_checkpoint is None:
            aggregation_start_rng=torch.get_rng_state().clone()
            records,statistics=self._aggregate_autosomal_backward(dataset,arm,phase_metrics)
        else:
            n_batches=_dataset_batch_count(dataset)
            tiles_per_batch=tuple(_dataset_batch_tile_count(dataset,index)
                                  for index in range(n_batches))
            require(all(value is not None for value in tiles_per_batch),
                    "recoverable autosomal update requires explicit tile inventory")
            tile_count=sum(tiles_per_batch)
            if resumed is None:
                aggregation_start_rng=torch.get_rng_state().clone()
                attempts=1
                cached=update_checkpoint.restore_burdens(
                    burden_identity(update_checkpoint.identity))
                if cached is None:
                    phase="AUTOSOMAL_BURDENS";burden_totals=[];burden_identities=[]
                    burden_current=None;next_batch=next_tile=0
                    records=[];current_pool=None;current_rng=[];current_identities=[]
                else:
                    require(isinstance(cached,dict)and
                            set(cached)=={"ids_by_batch","tiles_per_batch","totals","identities"}and
                            tuple(cached["tiles_per_batch"])==tiles_per_batch and
                            tuple(tuple(x)for x in cached["ids_by_batch"])==tuple(
                                _dataset_batch_axis(dataset,index)[0]for index in range(n_batches))and
                            len(cached["totals"])==len(cached["identities"])==n_batches and
                            all(len(x)==count for x,count in
                                zip(cached["identities"],tiles_per_batch)),
                            "cached autosomal burden inventory differs")
                    phase="AUTOSOMAL_AGGREGATION"
                    burden_totals=list(cached["totals"])
                    burden_identities=[list(x)for x in cached["identities"]]
                    burden_current=None;next_batch=next_tile=0
                    records=[];current_pool=None;current_rng=[];current_identities=[]
            else:
                progress,state=resumed
                require(progress.get("phase")in("AUTOSOMAL_BURDENS","AUTOSOMAL_AGGREGATION",
                                                 "AUTOSOMAL_RECOMPUTATION")and
                        tuple(progress.get("tiles_per_person_batch",()))==tiles_per_batch and
                        progress.get("total_recompute_tiles")==tile_count,
                        "partial update phase/tile inventory differs")
                phase=progress["phase"];attempts=progress.get("attempts_durable_lower_bound",0)+1
                aggregation_start_rng=state["aggregation_start_rng"].clone()
                burden_totals=list(state["burden_totals"])
                burden_identities=[list(x)for x in state["burden_identities"]]
                burden_current=state.get("burden_current")
                next_batch=progress["next_person_batch_index"]
                next_tile=progress["next_tile_index"]
                records=[self._record_from_state(value)for value in state.get("records",[])]
                current_pool=(None if state.get("current_pool")is None else
                    self._pool_accumulator_from_state(*_dataset_batch_axis(dataset,next_batch)[:2],
                                                      state["current_pool"]))
                current_rng=list(state.get("current_rng_states",[]))
                current_identities=list(state.get("current_identities",[]))

            def progress_for(current_phase,batch,tile,**updates):
                base={"phase":current_phase,"next_person_batch_index":batch,
                      "next_tile_index":tile,"tiles_per_person_batch":tiles_per_batch,
                      "total_person_batches":n_batches,"total_recompute_tiles":tile_count,
                      "attempts_durable_lower_bound":attempts,
                      "burden_tiles_durable_unique":(
                          sum(tiles_per_batch[:len(burden_totals)])+tile
                          if current_phase=="AUTOSOMAL_BURDENS" else tile_count),
                      "aggregated_tiles_durable_unique":(
                          0 if current_phase=="AUTOSOMAL_BURDENS" else
                          (sum(len(record[3])for record in records)+tile
                           if current_phase=="AUTOSOMAL_AGGREGATION" else tile_count)),
                      "recomputed_tiles_durable_unique":0,
                      "next_recompute_index":0,
                      "aggregated_tiles_durable_lower_bound":(
                          0 if current_phase=="AUTOSOMAL_BURDENS" else
                          (sum(len(record[3])for record in records)+tile
                           if current_phase=="AUTOSOMAL_AGGREGATION" else tile_count))}
                base.update(updates);return base

            def save_phase(progress,**extra):
                state={"aggregation_start_rng":aggregation_start_rng,
                       "burden_totals":burden_totals,
                       "burden_identities":burden_identities,
                       "burden_current":burden_current,
                       "records":[self._record_state(value)for value in records],
                       "current_pool":(None if current_pool is None else
                                       self._pool_accumulator_state(current_pool)),
                       "current_rng_states":current_rng,
                       "current_identities":current_identities}
                state.update(extra)
                update_checkpoint.save(self,optimizer,progress,state)

            if phase=="AUTOSOMAL_BURDENS":
                for batch_index in range(next_batch,n_batches):
                    ids,_,_=_dataset_batch_axis(dataset,batch_index)
                    start=next_tile if batch_index==next_batch else 0
                    total=burden_current if batch_index==next_batch else None
                    identities=(burden_identities[batch_index]if
                                batch_index<len(burden_identities)else [])
                    require(len(identities)==start,"burden cursor/identity count differs")
                    for flat_index,view in enumerate(_iter_dataset_batch_measured(
                            dataset,batch_index,start,phase_metrics,"AUTOSOMAL_BURDENS"),start=start):
                        identity=self._autosomal_identity(view)
                        value=event_burden_counts(autosomal_tile(view))
                        total=value if total is None else total+value
                        identities.append(identity);done=flat_index+1
                        burden_current=total
                        if batch_index==len(burden_identities):
                            burden_identities.append(identities)
                        durable=sum(tiles_per_batch[:batch_index])+done
                        if done==tiles_per_batch[batch_index]or durable==interrupt_after_burdened:
                            if done==tiles_per_batch[batch_index]:
                                burden_totals.append(total);burden_current=None
                                nb,nt=batch_index+1,0
                            else:nb,nt=batch_index,done
                            save_phase(progress_for("AUTOSOMAL_BURDENS",nb,nt))
                        if durable==interrupt_after_burdened:
                            raise RuntimeError("synthetic interruption after durable burden cursor")
                    require(len(identities)==tiles_per_batch[batch_index],
                            "autosomal burden tile coverage differs")
                    next_tile=0
                require(len(burden_totals)==n_batches,"autosomal burden person coverage differs")
                update_checkpoint.save_burdens(burden_identity(update_checkpoint.identity),{
                    "ids_by_batch":[_dataset_batch_axis(dataset,index)[0]for index in range(n_batches)],
                    "tiles_per_batch":tiles_per_batch,"totals":burden_totals,
                    "identities":burden_identities})
                phase="AUTOSOMAL_AGGREGATION";next_batch=next_tile=0
                records=[];current_pool=None;current_rng=[];current_identities=[]
                save_phase(progress_for(phase,0,0))

            if phase=="AUTOSOMAL_AGGREGATION":
                torch.set_rng_state(torch.get_rng_state() if resumed is not None and
                                    resumed[0].get("phase")=="AUTOSOMAL_AGGREGATION" else
                                    aggregation_start_rng)
                for batch_index in range(next_batch,n_batches):
                    ids,roles,_=_dataset_batch_axis(dataset,batch_index)
                    start=next_tile if batch_index==next_batch else 0
                    pool=(current_pool if batch_index==next_batch and current_pool is not None
                          else AutosomalPool(ids,roles))
                    rng_states=(current_rng if batch_index==next_batch else [])
                    identities=(current_identities if batch_index==next_batch else [])
                    require(len(rng_states)==len(identities)==start,
                            "aggregation cursor/state count differs")
                    totals=burden_totals[batch_index]
                    for flat_index,view in enumerate(_iter_dataset_batch_measured(
                            dataset,batch_index,start,phase_metrics,"AUTOSOMAL_AGGREGATION"),start=start):
                        identity=self._autosomal_identity(view)
                        require(identity==burden_identities[batch_index][flat_index],
                                "autosomal identity changed after burden counting")
                        rng_states.append(torch.get_rng_state().clone());identities.append(identity)
                        with torch.no_grad():
                            tile,common,events,encoded,event_values,_,_=self._autosomal_tile_values(
                                view,arm,totals)
                            pool.add(int(view["chromosome"]),ids,roles,encoded,
                                torch.from_numpy(tile["common_core"]),common.valid,event_values,
                                torch.from_numpy(tile["event_core"]),events.valid,
                                torch.from_numpy(tile["event_dosage"]),
                                tile_index=view["autosomal_tile_index"],
                                last_tile=view["autosomal_last_tile"],core_bp=view["core_bp"])
                        done=flat_index+1;current_pool=pool
                        current_rng=rng_states;current_identities=identities
                        durable=sum(tiles_per_batch[:batch_index])+done
                        if done==tiles_per_batch[batch_index]or durable==interrupt_after_aggregated:
                            if done==tiles_per_batch[batch_index]:
                                records.append((ids,totals,rng_states,identities,pool.finalize_sums()))
                                current_pool=None;current_rng=[];current_identities=[];nb,nt=batch_index+1,0
                            else:nb,nt=batch_index,done
                            save_phase(progress_for("AUTOSOMAL_AGGREGATION",nb,nt))
                        if durable==interrupt_after_aggregated:
                            raise RuntimeError("synthetic interruption after durable aggregation cursor")
                    next_tile=0
                require(len(records)==n_batches,"autosomal aggregation person coverage differs")
                phase="AUTOSOMAL_RECOMPUTATION"

            require(phase=="AUTOSOMAL_RECOMPUTATION","unknown autosomal recovery phase")
            if resumed is not None and resumed[0].get("phase")==phase:
                state=resumed[1]
                statistics=self._statistics_from_state(state["statistics"])
                records=[self._record_from_state(value)for value in state["records"]]
                progress=dict(resumed[0],attempts_durable_lower_bound=attempts)
                common_gradient=state["common_gradient"];rare_gradient=state["rare_gradient"]
                restore_parameter_gradients(self,state["parameter_gradients"])
                value=state["loss"].detach();prediction=state["prediction"].detach()
                detached_pool=self._pool_from_state(state["pooled"])
                post_forward_rng=state["post_forward_rng"].clone()
            else:
                statistics=AutosomalSufficientStatistics(*[
                    (records[0][4].chromosomes if field=="chromosomes" else
                     torch.cat([getattr(record[4],field)for record in records],0))
                    for field in AutosomalSufficientStatistics.__dataclass_fields__])
                require(all(record[4].chromosomes==statistics.chromosomes for record in records),
                        "person batches cover different autosomes")
                common_sum=statistics.common_sum.detach().requires_grad_(True)
                rare_sum=statistics.rare_sum.detach().requires_grad_(True)
                pooled=finalize_statistics(replace(statistics,common_sum=common_sum,rare_sum=rare_sum),
                                           self.encoder.person_rho,self.encoder.config.rare_pooling)
                prediction=self.structure_prediction(pooled.embedding,target_data)
                value=training_loss(prediction,target_data,self.objective)
                post_forward_rng=torch.get_rng_state().clone()
                require(bool(torch.isfinite(value)),"nonfinite training loss");value.backward()
                require(common_sum.grad is not None and rare_sum.grad is not None,
                        "pooled sufficient-sum gradient is disconnected")
                common_gradient,rare_gradient=common_sum.grad.detach(),rare_sum.grad.detach()
                detached_pool=self._detached_pool(pooled)
                progress=progress_for(phase,0,0)
                save_phase(progress,statistics=self._statistics_state(statistics),
                           common_gradient=common_gradient,rare_gradient=rare_gradient,
                           parameter_gradients=capture_parameter_gradients(self),
                           loss=value.detach(),prediction=prediction.detach(),
                           pooled=self._pool_state(detached_pool),post_forward_rng=post_forward_rng)

        identities=tuple(tuple(record[3])for record in records)
        tile_count=sum(len(record[3])for record in records)

        def cursor_coordinates(flat_cursor):
            remaining=flat_cursor
            for batch_index,record in enumerate(records):
                if remaining<len(record[3]):return batch_index,remaining
                remaining-=len(record[3])
            require(remaining==0,"partial recomputation cursor exceeds tile inventory")
            return len(records),0
        if update_checkpoint is None:
            common_sum=statistics.common_sum.detach().requires_grad_(True)
            rare_sum=statistics.rare_sum.detach().requires_grad_(True)
            pooled=finalize_statistics(replace(statistics,common_sum=common_sum,rare_sum=rare_sum),
                                       self.encoder.person_rho,self.encoder.config.rare_pooling)
            prediction=self.structure_prediction(pooled.embedding,target_data)
            value=training_loss(prediction,target_data,self.objective)
            post_forward_rng=torch.get_rng_state().clone()
            require(bool(torch.isfinite(value)),"nonfinite training loss")
            value.backward()
            require(common_sum.grad is not None and rare_sum.grad is not None,
                    "pooled sufficient-sum gradient is disconnected")
            common_gradient,rare_gradient=common_sum.grad.detach(),rare_sum.grad.detach()
            detached_pool=self._detached_pool(pooled)
            progress={"phase":"AUTOSOMAL_RECOMPUTATION","next_recompute_index":0,
                      "next_person_batch_index":0,"next_tile_index":0,
                      "total_recompute_tiles":tile_count,
                      "attempts_durable_lower_bound":1,
                      "aggregated_tiles_durable_lower_bound":tile_count,
                      "recomputed_tiles_durable_unique":0}
        elif not (resumed is not None and resumed[0].get("phase")=="AUTOSOMAL_RECOMPUTATION"):
            pass
        else:
            state=resumed[1]
            require(progress.get("phase")=="AUTOSOMAL_RECOMPUTATION" and
                    progress.get("total_recompute_tiles")==tile_count and
                    0<=progress.get("next_recompute_index",-1)<=tile_count,
                    "partial update progress/tile count differs")
            require((progress.get("next_person_batch_index"),progress.get("next_tile_index"))==
                    cursor_coordinates(progress["next_recompute_index"]) and
                    progress.get("recomputed_tiles_durable_unique")==progress["next_recompute_index"] and
                    type(progress.get("attempts_durable_lower_bound"))is int and
                    progress["attempts_durable_lower_bound"]>=1 and
                    type(progress.get("aggregated_tiles_durable_lower_bound"))is int and
                    progress["aggregated_tiles_durable_lower_bound"]>=tile_count,
                    "partial update cursor/budget fields differ")
            require(state.get("tile_identities")==identities,
                    "autosomal request changed since partial checkpoint")
            common_gradient=state["common_gradient"]
            rare_gradient=state["rare_gradient"]
            require(torch.is_tensor(common_gradient) and torch.is_tensor(rare_gradient) and
                    common_gradient.shape==statistics.common_sum.shape and
                    rare_gradient.shape==statistics.rare_sum.shape and
                    bool(torch.isfinite(common_gradient).all()) and
                    bool(torch.isfinite(rare_gradient).all()),
                    "partial sufficient-sum gradients differ")
            restore_parameter_gradients(self,state["parameter_gradients"])
            value=state["loss"].detach();prediction=state["prediction"].detach()
            detached_pool=self._pool_from_state(state["pooled"])
            post_forward_rng=state["post_forward_rng"].clone()
            progress=dict(progress,attempts_durable_lower_bound=attempts)
        cursor=progress["next_recompute_index"]

        def checkpoint(next_batch,next_tile):
            if update_checkpoint is None:return
            current=dict(progress,next_recompute_index=cursor,
                         next_person_batch_index=next_batch,next_tile_index=next_tile,
                         recomputed_tiles_durable_unique=cursor)
            state={"aggregation_start_rng":aggregation_start_rng,
                   "burden_totals":burden_totals,"burden_identities":burden_identities,
                   "burden_current":None,"records":[self._record_state(value)for value in records],
                   "current_pool":None,"current_rng_states":[],"current_identities":[],
                   "statistics":self._statistics_state(statistics),
                   "post_forward_rng":post_forward_rng,"tile_identities":identities,
                   "common_gradient":common_gradient,"rare_gradient":rare_gradient,
                   "parameter_gradients":capture_parameter_gradients(self),
                   "loss":value.detach(),"prediction":prediction.detach(),
                   "pooled":self._pool_state(detached_pool)}
            update_checkpoint.save(self,optimizer,current,state)
            progress.update(current)

        checkpoint(progress.get("next_person_batch_index",0),progress.get("next_tile_index",0))
        recomputed=cursor;offset=0
        try:
            for batch_index,(ids,totals,rng_states,record_identities,_) in enumerate(records):
                batch_gradient=slice(offset,offset+len(ids));offset+=len(ids)
                batch_start=(progress["next_tile_index"] if
                             batch_index==progress["next_person_batch_index"] else 0)
                if batch_index<progress["next_person_batch_index"]:continue
                for index,view in enumerate(_iter_dataset_batch_measured(
                                            dataset,batch_index,batch_start,phase_metrics,
                                            "AUTOSOMAL_RECOMPUTATION"),
                                            start=batch_start):
                    require(index<len(rng_states)and self._autosomal_identity(view)==record_identities[index],
                            "autosomal request changed between aggregation and recomputation")
                    torch.set_rng_state(rng_states[index])
                    _,_,_,_,_,csum,rsum=self._autosomal_tile_values(view,arm,totals)
                    torch.autograd.backward((csum,rsum),(common_gradient[batch_gradient],rare_gradient[batch_gradient]))
                    recomputed+=1;cursor+=1
                    next_batch,next_tile=(batch_index,index+1)
                    if index+1==len(rng_states):next_batch,next_tile=batch_index+1,0
                    if (next_tile==0 or cursor==tile_count or
                            cursor==interrupt_after_recomputed):checkpoint(next_batch,next_tile)
                    if interrupt_after_recomputed is not None and cursor==interrupt_after_recomputed:
                        raise RuntimeError("synthetic interruption after durable recomputation cursor")
                require(index+1==len(rng_states),"autosomal recomputation tile count differs")
            require(offset==len(dataset.ids) and cursor==tile_count,
                    "autosomal recomputation coverage differs")
        finally:
            torch.set_rng_state(post_forward_rng)
        return value.detach(),prediction.detach(),detached_pool,recomputed


def lai_loss_mask(data):
    """Keep context queries visible but score each owned query only once.

    Legacy whole-region batches have no query_core and retain their support.
    Regional v2 batches must not train on a neighbouring core's halo labels.
    """
    observed = np.asarray(data.get("truth_observed", data["query_valid"]))
    require(observed.dtype == bool and observed.shape == data["query_valid"].shape
            and not (observed & ~data["query_valid"]).any(),
            "LAI truth support differs from valid queries")
    if "query_core" in data:
        core = np.asarray(data["query_core"])
        require(core.dtype == bool and core.shape == observed.shape,
                "query ownership mask differs from LAI truth support")
        observed = observed & core
    require(observed.any(1).all(), "LAI person has no observed owned query")
    return observed


def losses(prediction, data, objective):
    if objective == "structure":
        return (prediction - torch.from_numpy(data["target"]).float()).square()
    # In v2, truth_observed is used only here.  Model inputs and predictions
    # continue to use query_valid.  Legacy batches omit truth_observed and keep
    # their previous behavior byte-for-byte at this decision point.
    observed = lai_loss_mask(data)
    mask = torch.from_numpy(observed)
    truth = torch.from_numpy(np.where(observed, data["truth_state"], 0)).long()
    probability = prediction[mask].gather(1, truth[mask][:, None]).squeeze(-1)
    # No clipping in the training objective: valid model probabilities must be positive.
    require(bool((probability > 0).all()), "zero true-state probability; reject unstable training loss")
    flat = torch.zeros_like(truth, dtype=prediction.dtype)
    flat[mask] = -probability.log()
    return flat.sum(1) / mask.sum(1)


def training_loss(prediction, data, objective):
    """Use one global objective after all technical person batches are joined."""
    if objective == "structure" and "pair_weight" in data:
        from r02_k03_neural_target import weighted_pair_mse
        return weighted_pair_mse(prediction, data)
    return losses(prediction, data, objective).mean()


def person_losses(values, data, objective):
    if objective == "lai":
        return [(i, float(x)) for i, x in enumerate(values)]
    rows = defaultdict(list)
    for pair, value in zip(data["pairs"], values):
        for index in pair:
            rows[int(index)].append(float(value))
    return [(i, sum(errors) / len(errors)) for i, errors in rows.items()]


def score_batches(model, contract, stage, resolved, arm, output,
                  burden_caches=None, phase_metrics=None):
    """Same descriptive macro loss for SELECT/EVALUATE; no independence claim."""
    model.eval()
    errors, group_by_person, predictions = defaultdict(list), {}, []
    query_score = None
    if contract["objective"] == "lai" and contract.get("lai_training_protocol"):
        from r02_lai_training_campaign import validate_protocol
        spec = contract["lai_training_protocol"]
        protocol, _ = validate_protocol(spec["path"], spec["sha256"])
        from r02_lai_query_score import QueryScore, RULE
        if protocol["training_unit"]["selection_loss_aggregation"] == RULE:
            query_score = QueryScore()
    weighted_numerator, weighted_denominator, weighted_entries = 0.0, 0.0, 0
    with torch.inference_mode():
        for index, entry in enumerate(contract["batches"][stage]):
            cached = (None if burden_caches is None else
                      burden_caches.get(_entry_burden_cache_key(entry)))
            prediction, data, _ = predict_training_entry(
                model, entry, contract, stage, resolved, arm,
                burden_cache=cached, phase_metrics=phase_metrics)
            value = losses(prediction, data, contract["objective"]).numpy()
            require(np.isfinite(value).all(), "nonfinite evaluation loss")
            if query_score is not None:
                receipt = read_json(entry["receipt_path"], entry["receipt_sha256"])
                query_score.add(receipt["case_id"], data, prediction.numpy(),
                                lai_loss_mask(data))
            if "pair_weight" in data:
                weight = np.asarray(data["pair_weight"], dtype=np.float64)
                weighted_numerator += float(np.dot(weight, value.astype(np.float64)))
                weighted_denominator += float(weight.sum())
                weighted_entries += 1
            for person_index, error in person_losses(value, data, contract["objective"]):
                person = str(data["person_id"][person_index])
                errors[person].append(error)
                group_by_person[person] = str(data["component_id"][person_index])
            path = Path(output) / f"predictions-{index:05d}.npy"
            np.save(path, prediction.numpy(), allow_pickle=False)
            predictions.append({"path": str(path.resolve()), "sha256": sha(path), "input": entry})
    components = defaultdict(list)
    for person, items in errors.items():
        components[group_by_person[person]].append(sum(items) / len(items))
    require(bool(components), "no evaluable components")
    if query_score is not None:
        metric, errors, components = query_score.result()
    elif weighted_denominator:
        require(contract["objective"] == "structure" and
                weighted_entries == len(contract["batches"][stage]),
                "cannot mix weighted and unweighted selection targets")
        metric = weighted_numerator / weighted_denominator
    else:
        metric = sum(sum(v) / len(v) for v in components.values()) / len(components)
    write_json(Path(output) / "predictions.json", predictions)
    return metric, errors, components


def validate_contract(contract):
    require(contract["schema"] == SCHEMA and contract["objective"] in ("structure", "lai"), "unsupported training contract")
    require(contract["scope"] in ("SYNTHETIC_CODE_TEST", "DEVELOPMENT"), "final evaluation is forbidden")
    if contract["scope"] == "DEVELOPMENT":
        if contract["objective"] == "lai":
            require(contract.get("controls_materialized") is True
                    and contract.get("training_authorized") is True,
                    "LAI preparation is not complete admitted training: controls and admission required")
        from r02_neural_training_eligibility import require_contract_eligibility
        require_contract_eligibility(contract)
        if contract["objective"] == "lai":
            from r02_lai_separation import require_training_separation
            require_training_separation(contract)
    require(contract["admission"]["status"] == ("CODE_TEST_ONLY" if contract["scope"] == "SYNTHETIC_CODE_TEST"
                                                else "APPROVED_DEVELOPMENT"), "task has not been admitted")
    require(contract["target_definition"] and contract["count_scope"] and contract["feature_definition"],
            "explicit biological target, allele/feature definitions and count scope are required")
    if contract["objective"] == "lai":
        from r02_lai_query_burden import BOUNDARY, COUNT_NAMES, SCOPE
        identity = contract.get("reference_identity")
        identity_by_case = contract.get("reference_identity_by_case")
        identity_fields = {"event_catalogue_sha256", "ref_train_receipt_sha256",
                           "ref_support_by_provenance_sha256", "reference_vcf_sha256",
                           "reference_roles_sha256"}
        valid_identity = (isinstance(identity, dict) and set(identity) == identity_fields
                          and all(re.fullmatch(r"[0-9a-f]{64}", value or "")
                                  for value in identity.values()))
        valid_identity_by_case = (identity is None and isinstance(identity_by_case, dict)
            and bool(identity_by_case)
            and all(isinstance(case, str) and case and isinstance(value, dict)
                    and set(value) == identity_fields
                    and all(re.fullmatch(r"[0-9a-f]{64}", digest or "")
                            for digest in value.values())
                    for case, value in identity_by_case.items()))
        require(contract.get("event_metadata_features") == 12
                and contract.get("event_reference_features") == 15
                and contract.get("event_reference_feature_schema") == "r02_lai_event_reference_features_v1"
                and contract.get("event_reference_feature_formula") ==
                    "PER_ANCESTRY(AC_0,AN_0,AF_0,KNOWN,FREQUENCY_DEFINED);AF_0=AC/AN_IFF_KNOWN_AND_AN_GT_0"
                and contract.get("event_reference_ancestry_order") ==
                    ["African", "European", "Native_American"]
                and (valid_identity or valid_identity_by_case),
                "exact LAI reference feature specification required")
        require(contract.get("lai_burden_scope") == SCOPE
                and contract.get("lai_burden_boundary") == BOUNDARY
                and contract.get("lai_burden_count_names") == list(COUNT_NAMES)
                and contract.get("lai_burden_transform") == "LOG1P_IN_MODEL"
                and contract.get("rare_allele") == "DNABR2723_FIXED_SOURCE_MINOR"
                and contract.get("rare_phase") == "DIPLOID_UNPHASED_NO_HAPLOTYPE_ASSIGNMENT",
                "exact LAI query-local burden specification required; legacy fragment counts are rejected")
    mode = dataset_mode(contract)
    require(mode in (SINGLE_CHROMOSOME, AUTOSOMAL_POOL, AUTOSOMAL_PERSON_BATCHES), "unsupported explicit dataset mode")
    declared = contract.get("dataset_declared_scope")
    if mode in (AUTOSOMAL_POOL, AUTOSOMAL_PERSON_BATCHES):
        require(declared == "AUTOSOMAL_22" and contract["objective"] == "structure",
                "AUTOSOMAL_POOL_V1 requires declared AUTOSOMAL_22 structure data")
        require(all(all(entry.get(key) for key in ("autosomal_request_path", "autosomal_request_sha256"))
                    for batches in contract["batches"].values() for entry in batches),
                "each autosomal training entry requires an authenticated request")
        if mode == AUTOSOMAL_PERSON_BATCHES and contract["scope"] == "DEVELOPMENT":
            from r02_k03_neural_target import LOSS_POLICY
            require(contract.get("pair_loss_policy") == LOSS_POLICY,
                    "v2 human training requires the reviewed K03 weighted-pair loss policy")
    else:
        require(declared is None or declared == "BOUNDED_CONTEXT_PILOT",
                "single-chromosome mode cannot claim AUTOSOMAL_22")
    if contract["scope"] == "DEVELOPMENT":
        require(((mode == SINGLE_CHROMOSOME and declared == "BOUNDED_CONTEXT_PILOT" and
                  contract.get("purpose") == "PREPARATION_OR_EXPLICIT_PILOT_NOT_PRIMARY_DELIVERY") or
                 (mode in (AUTOSOMAL_POOL, AUTOSOMAL_PERSON_BATCHES) and declared == "AUTOSOMAL_22" and
                  contract.get("purpose") == "AUTOSOMAL_STRUCTURE_DEVELOPMENT")),
                "real training purpose and declared data mode disagree")
        admission = contract["admission"]
        review = read_json(admission["path"], admission["sha256"])
        require(review["status"] == "APPROVED_DEVELOPMENT" and review["scope"] == "DEVELOPMENT"
                and review["target_definition"] == contract["target_definition"]
                and review["feature_definition"] == contract["feature_definition"]
                and review["persons"] == contract["persons"]
                and review["batches"] == contract["batches"]
                and review["count_scope"] == contract["count_scope"]
                and review.get("target_contract_sha256") == contract.get("target_contract_sha256")
                and review.get("dataset_declared_scope") == declared
                and review.get("dataset_mode", SINGLE_CHROMOSOME) == mode
                and review.get("purpose") == contract.get("purpose")
                and review.get("pair_loss_policy") == contract.get("pair_loss_policy")
                and review.get("dose_relocation") == contract.get("dose_relocation")
                and review.get("eligibility_manifest") == contract.get("eligibility_manifest")
                and review["excluded_intervals"] == contract.get("excluded_intervals", []),
                "review does not admit these exact people, features, exclusions and batches")
        if contract["objective"] == "lai":
            expected_identity = ({"reference_identity_by_case": identity_by_case}
                                 if identity_by_case is not None else
                                 {"reference_identity": identity})
            require(all(review.get(key) == value for key, value in expected_identity.items())
                    and review.get("controls_materialized") is True
                    and review.get("training_authorized") is True,
                    "LAI review does not admit these references and materialized controls")
    roles = defaultdict(set)
    for person, spec in contract["persons"].items():
        require(person and spec["component"] and spec["role"] in ("FIT", "SELECT"), "only explicit development people allowed")
        roles[spec["component"]].add(spec["role"])
    require(all(len(v) == 1 for v in roles.values()), "family/donor component crosses FIT and SELECT")
    require(set(contract["batches"]) == {"FIT", "SELECT"}
            and all(contract["batches"].values()), "both development datasets required")
    if (contract["objective"] == "lai" and any(
            entry.get("format") == "LAI_FACTORIZED_TRAINING_LINK_V1"
            for batches in contract["batches"].values() for entry in batches)):
        from r02_lai_training_campaign import validate_campaign_contract
        validate_campaign_contract(contract)
    require(type(contract["steps"]) is int and contract["steps"] >= len(contract["batches"]["FIT"]),
            "fixed budget must expose every FIT batch at least once")
    require(type(contract["threads"]) is int and 1 <= contract["threads"] <= 8, "explicit bounded CPU threads required")
    for limit in contract["limits"].values():
        require(type(limit) is int and limit > 0, "positive integer resource limits required")
    if contract["objective"] == "structure":
        require(bool(contract["excluded_intervals"]), "withheld-genetic-similarity task requires excluded intervals")
        for region in contract["excluded_intervals"]:
            require(type(region["chrom"]) is int and 1 <= region["chrom"] <= 22
                    and all(math.isfinite(region[x]) for x in ("start_cm", "end_cm"))
                    and region["start_cm"] <= region["end_cm"], "invalid excluded region")
            if contract["scope"] == "DEVELOPMENT":
                require(type(region.get("start_bp")) is int and type(region.get("end_bp")) is int
                        and 1 <= region["start_bp"] <= region["end_bp"], "frozen physical exclusions required in real data")
    return contract


def build_optimizer(parameters, learning_rate, adam_state_precision="fp32"):
    """Construct the explicit Adam numerical implementation for this task."""
    require(adam_state_precision in ADAM_STATE_PRECISIONS,
            "unsupported Adam state precision")
    if adam_state_precision == "fp32":
        return torch.optim.Adam(parameters, lr=learning_rate)
    from r02_adam_fp64_state import AdamFP64State
    return AdamFP64State(parameters, lr=learning_rate, foreach=False)


def aggregate_norm_diagnostics(model, prediction, pooled, before):
    """Return module/person-vector norm summaries without person-level rows."""
    require(isinstance(before, dict), "diagnostic parameter snapshot required")
    groups = {}
    for name, parameter in model.named_parameters():
        prefix = ("common_encoder" if name.startswith("encoder.common_encoder") else
                  "event_encoder" if name.startswith("encoder.event_encoder") else
                  "event_attention" if name.startswith("encoder.event_attention") else
                  "person_rho" if name.startswith("encoder.person_rho") else
                  "pair_head" if name.startswith("head") else "other")
        values = groups.setdefault(prefix, {"parameter_sq": 0.0, "gradient_sq": 0.0, "update_sq": 0.0})
        values["parameter_sq"] += float(parameter.detach().double().square().sum())
        if parameter.grad is not None:
            values["gradient_sq"] += float(parameter.grad.detach().double().square().sum())
        values["update_sq"] += float((parameter.detach() - before[name]).double().square().sum())
    modules = {name: {key.replace("_sq", "_l2"): math.sqrt(value)
                      for key, value in values.items()} for name, values in sorted(groups.items())}

    def summary(values):
        values = values.detach().double().reshape(-1)
        require(bool(values.numel()) and bool(torch.isfinite(values).all()),
                "nonfinite/empty aggregate norm diagnostic")
        quantiles = torch.quantile(values, torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0), dtype=torch.float64))
        return {"count": int(values.numel()), "mean": float(values.mean()),
                "min": float(quantiles[0]), "q25": float(quantiles[1]),
                "median": float(quantiles[2]), "q75": float(quantiles[3]),
                "max": float(quantiles[4])}

    result = {"modules": modules, "prediction": summary(prediction),
              "person_level_rows_emitted": False}
    if pooled is not None:
        common = torch.linalg.vector_norm(pooled.common_mean.detach().double(), dim=-1)
        rare = torch.linalg.vector_norm(pooled.rare_embedding.detach().double(), dim=-1)
        ratio = rare / common.clamp_min(torch.finfo(torch.float64).tiny)
        result["representation"] = {"common_l2": summary(common), "rare_l2": summary(rare),
                                    "rare_to_common_l2": summary(ratio)}
    return result


def checkpoint_identity(contract, contract_sha256, resolved_sha256, arm, seed,
                        adam_state_precision="fp32", rare_pooling=RARE_POOLING_SUM,
                        locus_descriptor=LOCUS_DESCRIPTOR_LEGACY):
    """Bind resumable state to exact inputs, configuration, code and arm."""
    source = Path(__file__)
    inputs = []
    for stage in ("FIT", "SELECT"):
        for entry in contract["batches"][stage]:
            inputs.append({"stage": stage, "path": entry["path"], "sha256": entry["sha256"],
                           "receipt_sha256": entry.get("receipt_sha256"),
                           "autosomal_request_sha256": entry.get("autosomal_request_sha256")})
    code = {name: sha(source.with_name(name)) for name in (
        "r02_context_train.py", "r02_context_checkpoint.py", "r02_context_update_checkpoint.py",
        "r02_adam_fp64_state.py",
        "genomic_context_runtime.py", "r02_lai_cnn_config.py", "r02_local_attention_blocks.py",
        "r02_context_autosomal_dataset.py", "r02_context_autosomal_pool.py",
        "r02_genomic_event_inputs.py",
        "r02_neural_training_eligibility.py", "r02_lai_reference_features.py",
        "r02_lai_event_annotations.py", "r02_lai_separation.py",
        "r02_lai_target_dependencies.py", "r02_lai_query_burden.py",
        "r02_lai_training_campaign.py")}
    if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES:
        code["r02_event_mmap_reader.py"] = sha(source.with_name("r02_event_mmap_reader.py"))
        code["r02_k03_neural_target.py"] = sha(source.with_name("r02_k03_neural_target.py"))
    if contract.get("dose_relocation"):
        code["r02_structure_identity.py"] = sha(source.with_name("r02_structure_identity.py"))
    require(rare_pooling in (RARE_POOLING_SUM, RARE_POOLING_MEAN_WITH_EXPLICIT_BURDEN),
            "unknown rare pooling checkpoint identity")
    require(locus_descriptor in (LOCUS_DESCRIPTOR_LEGACY, LOCUS_DESCRIPTOR_CHROM22),
            "unknown locus descriptor checkpoint identity")
    return {"schema": "r02_context_checkpoint_identity_v1", "contract_sha256": contract_sha256,
            "resolved_sha256": resolved_sha256, "objective": contract["objective"],
            "lai_burden_scope": contract.get("lai_burden_scope"),
            "lai_burden_count_names": contract.get("lai_burden_count_names"),
            "dataset_mode": dataset_mode(contract), "arm": arm, "seed": seed,
            "steps": contract["steps"], "adam_state_precision": adam_state_precision,
            "rare_pooling": rare_pooling,
            "locus_descriptor": locus_descriptor,
            "inputs": inputs, "code_sha256": code,
            "torch_version": torch.__version__, "numpy_version": np.__version__, "device": "cpu"}


def train(contract_path, contract_sha256, resolved_path, resolved_sha256, arm, seed, output,
          checkpoint_dir=None, checkpoint_every_steps=1, adam_state_precision="fp32",
          fit_only=False, stop_after_steps=None, record_aggregate_norms=False):
    output = Path(output)
    configure_torch_cache(output.parent / f".{output.name}.torch-cache")
    # Force any PyTorch startup/import failure before opening genotype batches.
    torch.use_deterministic_algorithms(True)
    contract = validate_contract(read_json(contract_path, contract_sha256))
    resolved = read_json(resolved_path, resolved_sha256)
    if contract.get("lai_training_protocol") is not None:
        from r02_lai_training_campaign import FIT_ONLY_INITIAL, descriptor, validate_protocol
        protocol_spec = contract["lai_training_protocol"]
        protocol, _ = validate_protocol(
            descriptor(protocol_spec, "training protocol"), protocol_spec["sha256"])
        require(protocol["execution_mode"] != FIT_ONLY_INITIAL or fit_only,
                "FIT_ONLY_INITIAL protocol requires fit_only before any batch is loaded")
        allowed_seeds = protocol["optimization"]["training_seeds"]
        if not fit_only:
            allowed_seeds = sorted(set(allowed_seeds +
                                       protocol["optimization"]["stability_seeds"]))
        require(contract["scope"] == protocol["scope"]
                and arm == contract.get("information_condition")
                and seed in allowed_seeds
                and checkpoint_every_steps ==
                    protocol["optimization"]["checkpoint_every_steps"]
                and adam_state_precision ==
                    protocol["optimization"]["adam_state_precision"],
                "LAI arm, seed, scope, checkpoint cadence, or Adam precision differs from protocol")
    if contract.get("dose_relocation"):
        from r02_structure_identity import validate_selected_control
        require(dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES,
                "dose relocation requires the authenticated 22-autosome reader")
        validate_selected_control(contract["dose_relocation"], contract, resolved_sha256, arm, seed)
    allowed_conditions = ARMS if contract["objective"] == "structure" else LAI_CONDITIONS
    require(arm in allowed_conditions and type(seed) is int and seed >= 0,
            "invalid information condition/seed")
    require(type(checkpoint_every_steps) is int and checkpoint_every_steps >= 1,
            "checkpoint frequency must be a positive number of completed optimizer steps")
    require(adam_state_precision in ADAM_STATE_PRECISIONS,
            "unsupported Adam state precision")
    require(type(fit_only) is bool and (stop_after_steps is None or fit_only),
            "partial training requires FIT_ONLY; SELECT cannot score an unfinished budget")
    require(type(record_aggregate_norms) is bool, "aggregate norm diagnostic flag must be boolean")
    final_step = contract["steps"] if stop_after_steps is None else stop_after_steps
    require(type(final_step) is int and 1 <= final_step <= contract["steps"],
            "FIT stop must lie inside the frozen training ceiling")
    phase_metrics = {}
    # Authenticate all data before any optimizer step. Load one bounded batch at a time.
    for stage, batches in contract["batches"].items():
        if fit_only and stage != "FIT":
            continue
        for entry in batches:
            validate_training_entry(entry, contract, stage, resolved,
                authenticate_windows=dataset_mode(contract) in
                    (AUTOSOMAL_POOL,AUTOSOMAL_PERSON_BATCHES),
                phase_metrics=phase_metrics)
    if output.exists():
        require(output.is_dir() and not any(output.iterdir()),
                "existing training output must be an empty directory left before scoring")
    else:
        output.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(contract["threads"])
    model = TaskModel(resolved, contract)
    require(resolved.get("optimizer", "Adam") == "Adam", "unsupported optimizer; no silent substitution")
    optimizer = build_optimizer(model.parameters(), model.encoder.config.learning_rate,
                                adam_state_precision)
    identity = checkpoint_identity(contract, contract_sha256, resolved_sha256, arm, seed,
                                   adam_state_precision, model.encoder.config.rare_pooling,
                                   model.encoder.config.locus_descriptor)
    checkpoint_dir = (Path(checkpoint_dir) if checkpoint_dir is not None
                      else output.parent / f".{output.name}.internal-checkpoints")
    partial_root=checkpoint_dir/"partial-updates"
    store = CheckpointStore(checkpoint_dir, identity)
    norm_diagnostics = []
    score_burden_caches = {}
    try:
        start_step, history = store.restore(model, optimizer)
        require(start_step <= final_step, "checkpoint is beyond the requested safe stopping point")
        for step in range(start_step, final_step):
            model.train()
            fit_batch_index=step%len(contract["batches"]["FIT"])
            entry=contract["batches"]["FIT"][fit_batch_index]
            update_store=None;resumed_update=None
            if dataset_mode(contract) in (AUTOSOMAL_POOL,AUTOSOMAL_PERSON_BATCHES):
                update_id=update_identity(identity,step,fit_batch_index,entry,history,
                                          optimizer_start_digest(model,optimizer))
                update_store=PartialUpdateStore(
                    partial_root/f"update-{step:012d}",update_id)
                resumed_update=update_store.restore(model,optimizer)
            if resumed_update is None:
                optimizer.zero_grad(set_to_none=True)
            before=({name:parameter.detach().clone() for name,parameter in model.named_parameters()}
                    if record_aggregate_norms else None)
            data, dataset = validate_training_entry(entry, contract, "FIT", resolved)
            try:
                if dataset is None:
                    prediction=model(data,arm);pooled=None
                    value=training_loss(prediction,data,contract["objective"])
                    require(bool(torch.isfinite(value)),"nonfinite training loss")
                    value.backward()
                else:
                    value,prediction,pooled,_=model.backward_autosomal(
                        dataset,data,arm,update_store,optimizer,resumed_update,
                        phase_metrics=phase_metrics)
                    cached = update_store.restore_burdens(
                        burden_identity(update_store.identity))
                    require(cached is not None,
                            "completed autosomal update omitted authenticated burden counts")
                    score_burden_caches[_entry_burden_cache_key(entry)] = cached
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"nonfinite gradient")
                optimizer.step()
                if record_aggregate_norms:
                    diagnostic=aggregate_norm_diagnostics(model,prediction,pooled,before)
                    diagnostic.update(completed_step=step+1,loss=float(value.detach()))
                    norm_diagnostics.append(diagnostic)
                history.append(float(value.detach()));completed=step+1
                if update_store is not None:update_store.save_budget()
                # An autosomal partial update is anchored by a completed Adam
                # generation before its own files are cleared. This preserves
                # recovery even when the requested completed-step cadence is >1.
                if (dataset is not None or completed%checkpoint_every_steps==0 or
                        completed==final_step):
                    store.save(model,optimizer,completed,history)
                if update_store is not None:update_store.clear()
            finally:
                if update_store is not None:update_store.close()
    finally:
        store.close()
    score_role = "FIT" if fit_only else "SELECT"
    metric, errors, components = score_batches(
        model, contract, score_role, resolved, arm, output,
        burden_caches=(score_burden_caches if score_role == "FIT" and
                       dataset_mode(contract) in (AUTOSOMAL_POOL, AUTOSOMAL_PERSON_BATCHES)
                       else None),
        phase_metrics=phase_metrics)
    checkpoint = output / "weights.npz"
    np.savez(checkpoint, **{k: v.detach().numpy() for k, v in model.state_dict().items()})
    metric_name = ("k03_component_pair_weighted_mse"
                   if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES else
                   ("component_mse" if contract["objective"] == "structure" else "component_log_loss"))
    partial_update_budgets=(load_completed_budgets(
        partial_root,identity_digest(identity),final_step)
        if dataset_mode(contract) in (AUTOSOMAL_POOL,AUTOSOMAL_PERSON_BATCHES) else [])
    receipt = {"schema": SCHEMA, "status": "FIT_CHECKPOINT" if fit_only else "COMPLETE",
        "score_role": score_role, "select_accessed": not fit_only,
        "frozen_training_ceiling_steps": contract["steps"],
        "scope": contract["scope"], "arm": arm, "seed": seed,
        "contract_sha256": contract_sha256, "resolved_sha256": resolved_sha256,
        "dataset_declared_scope": contract.get("dataset_declared_scope", "SYNTHETIC_UNSPECIFIED"),
        "dataset_mode": dataset_mode(contract),
        "rare_pooling": model.encoder.config.rare_pooling,
        "locus_descriptor": model.encoder.config.locus_descriptor,
        **({"dose_relocation": contract["dose_relocation"],
            "comparison_use": "POST_SELECTION_MECHANISTIC_ABLATION_NOT_INDEPENDENT_CONFIRMATION"}
           if contract.get("dose_relocation") else {}),
        "eligibility_manifest_sha256": contract.get("eligibility_manifest", {}).get("sha256"),
        "trainer_sha256": sha(__file__), "runtime_sha256": sha(Path(__file__).with_name("genomic_context_runtime.py")),
        "autosomal_dataset_sha256": sha(Path(__file__).with_name("r02_context_autosomal_dataset.py")),
        "autosomal_pool_sha256": sha(Path(__file__).with_name("r02_context_autosomal_pool.py")),
        "event_mmap_reader_sha256": (sha(Path(__file__).with_name("r02_event_mmap_reader.py"))
                                      if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES else None),
        "k03_neural_target_sha256": (sha(Path(__file__).with_name("r02_k03_neural_target.py"))
                                      if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES else None),
        "metrics": {metric_name: metric}, "training_losses": history,
        "autosomal_phase_metrics": phase_metrics,
        "steps": final_step, "persons_evaluated": len(errors), "n_evaluated": len(components),
        "internal_checkpoint": {"schema": "r02_context_internal_checkpoint_v1",
                                "identity_sha256": identity_digest(identity),
                                "adam_state_precision": adam_state_precision,
                                "restored_steps": start_step,
                                "frequency_completed_steps": checkpoint_every_steps,
                                "requested_frequency_completed_steps": checkpoint_every_steps,
                                "effective_frequency_completed_steps": (1 if
                                    dataset_mode(contract) in (AUTOSOMAL_POOL,AUTOSOMAL_PERSON_BATCHES)
                                    else checkpoint_every_steps),
                                "final_confirmed_step": final_step,
                                "partial_update_schema": "r02_context_partial_update_v1",
                                "partial_update_budgets": partial_update_budgets},
        "evaluation_unit": ("authenticated_k03_component_pair_weights"
                            if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES else "component_macro"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha(checkpoint), "prediction_sha256": sha(output / "predictions.json"),
        "torch_version": torch.__version__, "device": "cpu", "scientific_conclusion": "NOT_AUTOMATICALLY_ESTABLISHED",
        **({"aggregate_norm_diagnostics": norm_diagnostics} if record_aggregate_norms else {}),
        "limitations": (["autosomal_training_reads_authenticated_tiles_three_times_for_global_counts_aggregation_and_recomputation",
                         "autosomal_backward_retains_one_tile_encoder_graph_plus_pooled_head"] +
                        (["person_partition_is_technical_only_complete_pair_loss_is_global",
                          "zip_stored_mmap_backend_is_explicit_and_source_verified"]
                         if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES else [])
                        if dataset_mode(contract) in (AUTOSOMAL_POOL,AUTOSOMAL_PERSON_BATCHES) else
                        ["bounded_batches_not_whole_genome_streaming_training"]) + [
                        "development_selection_not_independent_evaluation",
                        "matched_nominal_capacity_not_a_mechanistic_identification",
                        "equal_batch_then_person_then_component_loss_not_genetic_length_weighting"]}
    write_json(output / "training.receipt.json", receipt)
    return receipt


def validate_reserved(plan_path, plan_sha256):
    """Authenticate reserved data/model admission before Nextflow cache lookup.

    One frozen fold is supported: this is NOT nested evaluation or autosomal
    inference. EVALUATE is never admitted by train() or the search bridge.
    The caller must supply the separate authorized evaluation plan by hash.
    """
    plan = read_json(plan_path, plan_sha256)
    require(plan["schema"] == "r02_context_reserved_evaluation_v1" and
            plan["status"] in ("CODE_TEST_ONLY", "APPROVED_RESERVED_PILOT_EVALUATION"),
            "separate reserved-evaluation admission required")
    contract = plan["evaluation_contract"]
    require(contract["objective"] == "structure" and contract["evaluation_stage"] == "EVALUATE",
            "only explicit reserved structure pilot supported")
    require(contract["dataset_declared_scope"] == "BOUNDED_CONTEXT_PILOT",
            "AUTOSOMAL_22 aggregation/evaluation not implemented")
    development = plan["development_contract"]
    dev = validate_contract(read_json(development["path"], development["sha256"]))
    require((plan["status"] == "CODE_TEST_ONLY") == (dev["scope"] == "SYNTHETIC_CODE_TEST") and
            contract["scope"] == dev["scope"], "synthetic and real evaluation cannot be relabeled")
    for key in ("objective", "target_definition", "feature_definition", "count_scope", "target_contract_sha256",
                "excluded_intervals", "common_features", "event_metadata_features", "limits"):
        require(contract.get(key) == dev.get(key), "evaluation task changed: " + key)
    require(set(contract["batches"]) == {"EVALUATE"} and contract["batches"]["EVALUATE"],
            "only reserved batches allowed")
    require(contract["persons"] and all(p and s["component"] and s["role"] == "EVALUATE"
                for p, s in contract["persons"].items()), "explicit reserved people required")
    require(not (set(contract["persons"]) & set(dev["persons"])) and
            not ({s["component"] for s in contract["persons"].values()} &
                 {s["component"] for s in dev["persons"].values()}), "reserved person/component leaked into development")
    selected = plan["frozen_selection"]
    selection = read_json(selected["path"], selected["sha256"])
    require(selection["objective"] == "structure" and selection["final_evaluation"] == "NOT_ACCESSED" and
            selection["actual_data_scope"] == dev["scope"], "selection is not frozen on matching development scope")
    recipe = selection["recipe"]
    resolved = read_json(plan["resolved"]["path"], plan["resolved"]["sha256"])
    require(recipe["config"] == resolved, "resolved model differs from frozen selection")
    expected = {(t["fold"], t["seed"], t["arm"]) for t in recipe["tasks"]}
    tasks = {(t["fold"], t["seed"], t["arm"]): t for t in recipe["tasks"]}
    require(len(expected) == len(recipe["tasks"]) and len({t[0] for t in expected}) == 1 and
            {t[2] for t in expected} == set(ARMS) and
            len(expected) == 3*len({t[1] for t in expected}), "all paired arms/seeds of one frozen fold required")
    require(type(plan["max_checkpoint_bytes"]) is int and plan["max_checkpoint_bytes"] > 0,
            "checkpoint memory limit required")
    models, seen = [], set()
    for item in plan["models"]:
        key = (item["fold"], item["seed"], item["arm"])
        require(key in expected and key not in seen, "unexpected/duplicate model; no favorable seed selection")
        seen.add(key)
        trained = read_json(item["receipt_path"], item["receipt_sha256"])
        require(trained["status"] == "COMPLETE" and trained["scope"] == dev["scope"] and
                trained["contract_sha256"] == development["sha256"] and
                trained["resolved_sha256"] == plan["resolved"]["sha256"] and
                trained.get("rare_pooling", RARE_POOLING_SUM) ==
                resolved.get("rare_pooling", RARE_POOLING_SUM) and
                trained.get("locus_descriptor", LOCUS_DESCRIPTOR_LEGACY) ==
                resolved.get("locus_descriptor", LOCUS_DESCRIPTOR_LEGACY) and
                trained["steps"] == tasks[key]["checkpoint_step"] and
                (trained["arm"], trained["seed"]) == (item["arm"], item["seed"]) and
                trained["trainer_sha256"] == sha(__file__) and
                trained["runtime_sha256"] == sha(Path(__file__).with_name("genomic_context_runtime.py")),
                "training receipt/code/checkpoint does not belong to selected task")
        checkpoint = Path(item["checkpoint_path"])
        require(sha(checkpoint) == trained["checkpoint_sha256"], "checkpoint hash differs")
        with zipfile.ZipFile(checkpoint) as archive:
            require(sum(x.file_size for x in archive.infolist()) <= plan["max_checkpoint_bytes"],
                    "checkpoint allocation exceeds budget")
        models.append((item, checkpoint))
    require(seen == expected, "missing paired checkpoint; evaluation does not select a favorable subset")
    # Only now open EVALUATE tensors: every selected model was authenticated.
    for entry in contract["batches"]["EVALUATE"]:
        load_batch(entry, contract, "EVALUATE", resolved)
    return plan, contract, dev, selected, resolved, models


def evaluate_reserved(plan_path, plan_sha256, output):
    """No optimizer or model selection; all frozen paired checkpoints are scored."""
    plan, contract, dev, selected, resolved, models = validate_reserved(plan_path, plan_sha256)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(dev["threads"])
    rows = []
    for index, (item, checkpoint) in enumerate(models):
        model = TaskModel(resolved, dev)
        with np.load(checkpoint, allow_pickle=False) as archive:
            state = model.state_dict()
            require(set(archive.files) == set(state), "checkpoint parameter names differ")
            weights = {}
            for k, tensor in state.items():
                array = archive[k]
                require(array.shape == tuple(tensor.shape) and np.isfinite(array).all() and
                        array.dtype == tensor.numpy().dtype, "checkpoint shape/dtype/nonfinite parameter")
                weights[k] = torch.from_numpy(array.copy())
            model.load_state_dict(weights, strict=True)
        model.requires_grad_(False)
        folder = output / f"model-{index:04d}"
        folder.mkdir()
        metric, errors, components = score_batches(model, contract, "EVALUATE", resolved, item["arm"], folder)
        rows.append(dict(fold=item["fold"], seed=item["seed"], arm=item["arm"], component_mse=metric,
            persons_evaluated=len(errors), n_evaluated=len(components), checkpoint_sha256=sha(checkpoint),
            prediction_sha256=sha(folder / "predictions.json")))
    require(len({(r["persons_evaluated"], r["n_evaluated"]) for r in rows}) == 1, "unpaired evaluation support")
    result = dict(schema="r02_context_reserved_evaluation_receipt_v1", status="COMPLETE_PILOT_ONLY",
        plan_sha256=plan_sha256, frozen_selection_sha256=selected["sha256"], trainer_sha256=sha(__file__),
        dataset_declared_scope="BOUNDED_CONTEXT_PILOT", data_scope=dev["scope"], models=rows,
        optimizer_steps=0, checkpoint_selection_using_evaluate=False, evaluation_unit="component_macro",
        scientific_conclusion="NOT_PRIMARY_AUTOSOMAL_RESULT_OR_BIOLOGICAL_VALIDATION",
        limitations=["No uncertainty estimate: components/windows/pairs are not independent replicates.",
                     "A frozen subset pilot does not replace all-autosome aggregation or K02 L_cov."])
    write_json(output / "evaluation.receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "evaluate", "verify-evaluation"), default="train")
    parser.add_argument("--evaluation-plan")
    parser.add_argument("--evaluation-plan-sha256")
    for key in ("contract", "contract-sha256", "resolved", "resolved-sha256", "output"):
        parser.add_argument("--" + key, required=key == "output")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-every-steps", type=int, default=1)
    parser.add_argument("--adam-state-precision", choices=ADAM_STATE_PRECISIONS, default="fp32")
    parser.add_argument("--fit-only", action="store_true",
                        help="Train/score FIT only; receipt cannot enter SELECT model selection")
    parser.add_argument("--stop-after-steps", type=int,
                        help="Absolute safe stopping point within the immutable training ceiling")
    parser.add_argument("--record-aggregate-norms", action="store_true",
                        help="Record aggregate module/representation norms; never person rows")
    args = parser.parse_args()
    if args.stage in ("evaluate", "verify-evaluation"):
        require(not args.fit_only and args.stop_after_steps is None,
                "FIT continuation options do not apply to reserved evaluation")
        require(not args.record_aggregate_norms,
                "training norm diagnostics do not apply to reserved evaluation")
        require(args.evaluation_plan and args.evaluation_plan_sha256 and
                all(getattr(args, x) is None for x in ("contract", "contract_sha256", "resolved", "resolved_sha256", "arm", "seed",
                                                       "checkpoint_dir")),
                "evaluation accepts only its separate frozen plan; no training overrides")
        if args.stage == "evaluate":
            result = evaluate_reserved(args.evaluation_plan, args.evaluation_plan_sha256, args.output)
        else:
            plan, _, _, _, _, models = validate_reserved(args.evaluation_plan, args.evaluation_plan_sha256)
            require(not Path(args.output).exists(), "verification output exists")
            result = dict(status="AUTHENTICATED_RESERVED_PILOT_ONLY", plan_sha256=args.evaluation_plan_sha256,
                          trainer_sha256=sha(__file__), models=len(models),
                          dataset_declared_scope=plan["evaluation_contract"]["dataset_declared_scope"])
            write_json(args.output, result)
    else:
        require(all(getattr(args, x) is not None for x in ("contract", "contract_sha256", "resolved", "resolved_sha256", "arm", "seed"))
                and args.evaluation_plan is None and args.evaluation_plan_sha256 is None, "complete training arguments required")
        result = train(args.contract, args.contract_sha256, args.resolved, args.resolved_sha256,
                       args.arm, args.seed, args.output, args.checkpoint_dir,
                       args.checkpoint_every_steps, args.adam_state_precision,
                       args.fit_only, args.stop_after_steps, args.record_aggregate_norms)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
