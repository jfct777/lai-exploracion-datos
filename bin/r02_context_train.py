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
import zipfile
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from genomic_context_runtime import ContextConfig, ContextualModel, Sites, require
from r02_context_autosomal_pool import (AutosomalPool, AutosomalPoolResult,
                                         AutosomalSufficientStatistics, finalize_statistics)
from r02_context_checkpoint import CheckpointStore, identity_digest


SCHEMA = "r02_context_training_v1"
ARMS = ("NONE", "COUNTS", "RARE")
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


def validate_batch(data, contract, stage):
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
        require(truth_observed.any(1).all(), "each person requires observed truth for a defined loss")
    return persons, groups


def validate_batch_receipt(data, entry, contract, resolved=None):
    """Bind excluded views and prepared context halos to the admitted task.

    Raw-array hashes alone do not say which core owns a site or how much
    neighboring context was extracted. The producer receipt fixes those facts;
    a wider search candidate must rebuild its input view instead of silently
    treating unextracted neighboring sites as absent.
    """
    fields = ("receipt_path", "receipt_sha256")
    declared = any(key in entry for key in fields)
    require(declared or contract["scope"] == "SYNTHETIC_CODE_TEST",
            "development batch requires an authenticated producer receipt")
    if not declared:
        return
    require(all(entry.get(key) for key in fields), "incomplete producer receipt reference")
    receipt = read_json(entry["receipt_path"], entry["receipt_sha256"])
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


def load_batch(entry, contract, stage, resolved=None):
    path = Path(entry["path"])
    require(sha(path) == entry["sha256"], "genotype batch SHA256 mismatch")
    with zipfile.ZipFile(path) as archive:
        require(sum(item.file_size for item in archive.infolist()) <= contract["limits"]["max_batch_bytes"],
                "uncompressed batch exceeds admitted memory budget")
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    validate_batch(data, contract, stage)
    validate_batch_receipt(data, entry, contract, resolved)
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


def open_autosomal_entry(entry, contract, stage, resolved, authenticate_windows=False):
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
    require(contract["objective"] == "structure" and config.family == "structure_deep_sets",
            "AUTOSOMAL_POOL_V1 supports only the existing structure_deep_sets route")
    request = dataset.request
    require(request["common_radius_cm"] == config.common_radius_cm and
            request["rare_radius_cm"] is None and request["attention_depth"] == 0,
            "autosomal request radii/depth differ from structure_deep_sets configuration")
    if dataset_mode(contract) != AUTOSOMAL_PERSON_BATCHES and all(entry.get(key) for key in ("receipt_path", "receipt_sha256")):
        receipt = read_json(entry["receipt_path"], entry["receipt_sha256"])
        lineage = receipt.get("input_lineage", {})
        require(receipt.get("block_id") == request["block_id"] and
                receipt.get("exclusion_scope") == request["exclusion_scope"] and
                (lineage.get("event_contract_sha256"), lineage.get("source_contract_sha256")) == dataset.lineage,
                "autosomal request and supervised target lineage/block/exclusion differ")
    if authenticate_windows:
        seen = 0
        for batch_index in range(_dataset_batch_count(dataset)):
            for view in _iter_dataset_batch(dataset, batch_index):
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


def _iter_dataset_batch(dataset, batch_index):
    if hasattr(dataset, "iter_person_batch"):
        return dataset.iter_person_batch(batch_index)
    require(batch_index == 0, "legacy autosomal dataset has one person batch")
    return iter(dataset)


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


def validate_training_entry(entry, contract, stage, resolved=None, authenticate_windows=False):
    if dataset_mode(contract) == SINGLE_CHROMOSOME:
        return load_batch(entry, contract, stage, resolved), None
    require(resolved is not None, "autosomal entry validation requires resolved configuration")
    return open_autosomal_entry(entry, contract, stage, resolved, authenticate_windows)


def predict_training_entry(model, entry, contract, stage, resolved, arm):
    data, dataset = validate_training_entry(entry, contract, stage, resolved)
    if dataset is None:
        return model(data, arm), data, None
    prediction, pooled = model.forward_autosomal(dataset, data, arm)
    return prediction, data, pooled


class TaskModel(nn.Module):
    def __init__(self, resolved, contract):
        super().__init__()
        config = ContextConfig.from_recipe(resolved)
        self.objective = contract["objective"]
        require(config.family.startswith(self.objective + "_"), "wrong model family")
        limits = contract["limits"]
        self.encoder = ContextualModel(config, contract["common_features"], contract["event_metadata_features"] + 4,
            max_common_sites=limits["max_common_sites"], max_events=limits["max_event_sites"],
            max_queries=limits["max_targets"], query_max_gap_cm=contract.get("query_max_gap_cm"))
        self.head = nn.Linear(2 * config.width, 1) if self.objective == "structure" else None

    def structure_prediction(self, embedding, data):
        pairs = torch.from_numpy(data["pairs"]).long()
        left, right = embedding[pairs[:, 0]], embedding[pairs[:, 1]]
        return self.head(torch.cat(((left - right).abs(), left * right), -1)).squeeze(-1)

    def forward(self, data, arm):
        common, events = tensor_sites(data, "common"), blinded_events(data, arm)
        if self.objective == "structure":
            # Halos supply context only. Pool every owned site exactly once.
            encoded = self.encoder.common_values(common)
            common_core = torch.from_numpy(data["common_core"]) & common.valid
            baseline = torch.where(common_core[..., None], encoded, 0).sum(1) / common_core.sum(1).clamp_min(1)[:, None]
            values = self.encoder.event_values(common, encoded, events)
            event_core = torch.from_numpy(data["event_core"]) & events.valid
            rare = self.encoder.person_rho(torch.where(event_core[..., None], values, 0).sum(1))
            z = baseline + torch.where(event_core.any(1)[:, None], rare, 0)
            return self.structure_prediction(z, data)
        return self.encoder.predict_lai(common, tensor_sites(data, "query"),
            torch.from_numpy(data["baseline_probabilities"]).float(), events, include_rare=True)

    def forward_autosomal(self, dataset, target_data, arm):
        """Encode all owned autosomal cores and apply the existing head once.

        The first pass obtains the same global burden features as concatenating
        every owned event locus. The second pass builds the ordinary reference
        graph. Training uses ``backward_autosomal`` below to bound that graph.
        """
        require(self.objective == "structure" and self.encoder.config.family == "structure_deep_sets",
                "autosomal pooling currently supports only structure_deep_sets")
        results = []
        for batch_index in range(_dataset_batch_count(dataset)):
            ids, roles, _ = _dataset_batch_axis(dataset, batch_index)
            totals = self._autosomal_burdens(dataset, batch_index)
            pool = AutosomalPool(ids, roles)
            for view in _iter_dataset_batch(dataset, batch_index):
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
            results.append((ids, pool.finalize(self.encoder.person_rho)))
        result = _concatenate_pool_results(results, dataset.ids)
        return self.structure_prediction(result.embedding, target_data), result

    @staticmethod
    def _autosomal_identity(view):
        return (str(view["chromosome"]), int(view["autosomal_tile_index"]),
                bool(view["autosomal_last_tile"]), tuple(view["core_bp"]),
                int(view.get("person_batch_index", 0)), int(view.get("person_batch_count", 1)))

    @staticmethod
    def _autosomal_burdens(dataset, batch_index=0):
        totals = None
        for view in _iter_dataset_batch(dataset, batch_index):
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

    def backward_autosomal(self, dataset, target_data, arm):
        """Exact pooled loss/backward with at most one tile encoder graph live.

        A no-grad aggregation records the CPU Torch RNG state before each tile.
        The pooled head is differentiated once through leaf sufficient sums.
        Each tile is then recomputed with its original RNG state and receives
        the corresponding sum gradient. Restoring the post-forward RNG makes
        this trajectory identical to the ordinary reference forward, including
        dropout, while releasing every tile graph before reading the next tile.
        """
        require(self.training and self.objective == "structure" and
                self.encoder.config.family == "structure_deep_sets",
                "bounded autosomal backward requires training structure_deep_sets")
        records=[]
        with torch.no_grad():
            for batch_index in range(_dataset_batch_count(dataset)):
                ids,roles,_=_dataset_batch_axis(dataset,batch_index)
                totals=self._autosomal_burdens(dataset,batch_index);pool=AutosomalPool(ids,roles)
                rng_states=[];identities=[]
                for view in _iter_dataset_batch(dataset,batch_index):
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
        common_sum=statistics.common_sum.detach().requires_grad_(True);rare_sum=statistics.rare_sum.detach().requires_grad_(True)
        pooled=finalize_statistics(replace(statistics,common_sum=common_sum,rare_sum=rare_sum),self.encoder.person_rho)
        prediction = self.structure_prediction(pooled.embedding, target_data)
        value = training_loss(prediction, target_data, self.objective)
        post_forward_rng = torch.get_rng_state().clone()
        require(bool(torch.isfinite(value)), "nonfinite training loss")
        value.backward()
        require(common_sum.grad is not None and rare_sum.grad is not None,
                "pooled sufficient-sum gradient is disconnected")
        common_gradient, rare_gradient = common_sum.grad.detach(), rare_sum.grad.detach()
        recomputed=0;offset=0
        try:
            for batch_index,(ids,totals,rng_states,identities,_) in enumerate(records):
                batch_gradient=slice(offset,offset+len(ids));offset+=len(ids)
                for index,view in enumerate(_iter_dataset_batch(dataset,batch_index)):
                    require(index<len(rng_states)and self._autosomal_identity(view)==identities[index],
                            "autosomal request changed between aggregation and recomputation")
                    torch.set_rng_state(rng_states[index])
                    _,_,_,_,_,csum,rsum=self._autosomal_tile_values(view,arm,totals)
                    torch.autograd.backward((csum,rsum),(common_gradient[batch_gradient],rare_gradient[batch_gradient]))
                    recomputed+=1
                require(index+1==len(rng_states),"autosomal recomputation tile count differs")
            require(offset==len(dataset.ids),"autosomal recomputation person coverage differs")
        finally:
            torch.set_rng_state(post_forward_rng)
        detached = AutosomalPoolResult(*(value.detach() if torch.is_tensor(value) else value for value in (
            pooled.embedding, pooled.common_mean, pooled.rare_embedding,
            pooled.common_valid_loci, pooled.common_missing_loci, pooled.rare_valid_loci,
            pooled.rare_missing_loci, pooled.rare_valid_zero_dose_loci, pooled.chromosomes)))
        return value.detach(), prediction.detach(), detached, recomputed


def losses(prediction, data, objective):
    if objective == "structure":
        return (prediction - torch.from_numpy(data["target"]).float()).square()
    # In v2, truth_observed is used only here.  Model inputs and predictions
    # continue to use query_valid.  Legacy batches omit truth_observed and keep
    # their previous behavior byte-for-byte at this decision point.
    observed = data.get("truth_observed", data["query_valid"])
    mask = torch.from_numpy(observed)
    truth = torch.from_numpy(np.where(observed, data["truth_state"], 0)).long()
    probability = prediction[mask].gather(1, truth[mask][:, None]).squeeze(-1)
    # No clipping in the training objective: valid model probabilities must be positive.
    require(bool((probability > 0).all()), "zero true-state probability; reject unstable training loss")
    flat = torch.zeros_like(truth, dtype=torch.float32)
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


def score_batches(model, contract, stage, resolved, arm, output):
    """Same descriptive macro loss for SELECT/EVALUATE; no independence claim."""
    model.eval()
    errors, group_by_person, predictions = defaultdict(list), {}, []
    weighted_numerator, weighted_denominator, weighted_entries = 0.0, 0.0, 0
    with torch.inference_mode():
        for index, entry in enumerate(contract["batches"][stage]):
            prediction, data, _ = predict_training_entry(model, entry, contract, stage, resolved, arm)
            value = losses(prediction, data, contract["objective"]).numpy()
            require(np.isfinite(value).all(), "nonfinite evaluation loss")
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
    if weighted_denominator:
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
        from r02_neural_training_eligibility import require_contract_eligibility
        require_contract_eligibility(contract)
        if contract["objective"] == "lai":
            from r02_lai_separation import require_training_separation
            require_training_separation(contract)
    require(contract["admission"]["status"] == ("CODE_TEST_ONLY" if contract["scope"] == "SYNTHETIC_CODE_TEST"
                                                else "APPROVED_DEVELOPMENT"), "task has not been admitted")
    require(contract["target_definition"] and contract["count_scope"] and contract["feature_definition"],
            "explicit biological target, allele/feature definitions and count scope are required")
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
                and review.get("eligibility_manifest") == contract.get("eligibility_manifest")
                and review["excluded_intervals"] == contract.get("excluded_intervals", []),
                "review does not admit these exact people, features, exclusions and batches")
    roles = defaultdict(set)
    for person, spec in contract["persons"].items():
        require(person and spec["component"] and spec["role"] in ("FIT", "SELECT"), "only explicit development people allowed")
        roles[spec["component"]].add(spec["role"])
    require(all(len(v) == 1 for v in roles.values()), "family/donor component crosses FIT and SELECT")
    require(set(contract["batches"]) == {"FIT", "SELECT"}
            and all(contract["batches"].values()), "both development datasets required")
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


def checkpoint_identity(contract, contract_sha256, resolved_sha256, arm, seed,
                        adam_state_precision="fp32"):
    """Bind resumable state to exact inputs, configuration, code and arm."""
    source = Path(__file__)
    inputs = []
    for stage in ("FIT", "SELECT"):
        for entry in contract["batches"][stage]:
            inputs.append({"stage": stage, "path": entry["path"], "sha256": entry["sha256"],
                           "receipt_sha256": entry.get("receipt_sha256"),
                           "autosomal_request_sha256": entry.get("autosomal_request_sha256")})
    code = {name: sha(source.with_name(name)) for name in (
        "r02_context_train.py", "r02_context_checkpoint.py", "r02_adam_fp64_state.py",
        "genomic_context_runtime.py",
        "r02_context_autosomal_dataset.py", "r02_context_autosomal_pool.py",
        "r02_neural_training_eligibility.py", "r02_lai_separation.py",
        "r02_lai_target_dependencies.py")}
    if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES:
        code["r02_event_mmap_reader.py"] = sha(source.with_name("r02_event_mmap_reader.py"))
        code["r02_k03_neural_target.py"] = sha(source.with_name("r02_k03_neural_target.py"))
    return {"schema": "r02_context_checkpoint_identity_v1", "contract_sha256": contract_sha256,
            "resolved_sha256": resolved_sha256, "objective": contract["objective"],
            "dataset_mode": dataset_mode(contract), "arm": arm, "seed": seed,
            "steps": contract["steps"], "adam_state_precision": adam_state_precision,
            "inputs": inputs, "code_sha256": code,
            "torch_version": torch.__version__, "numpy_version": np.__version__, "device": "cpu"}


def train(contract_path, contract_sha256, resolved_path, resolved_sha256, arm, seed, output,
          checkpoint_dir=None, checkpoint_every_steps=1, adam_state_precision="fp32"):
    contract = validate_contract(read_json(contract_path, contract_sha256))
    resolved = read_json(resolved_path, resolved_sha256)
    require(arm in ARMS and type(seed) is int and seed >= 0, "invalid arm/seed")
    require(type(checkpoint_every_steps) is int and checkpoint_every_steps >= 1,
            "checkpoint frequency must be a positive number of completed optimizer steps")
    require(adam_state_precision in ADAM_STATE_PRECISIONS,
            "unsupported Adam state precision")
    # Authenticate all data before any optimizer step. Load one bounded batch at a time.
    for stage, batches in contract["batches"].items():
        for entry in batches:
            validate_training_entry(entry, contract, stage, resolved,
                                    authenticate_windows=dataset_mode(contract) in (AUTOSOMAL_POOL,AUTOSOMAL_PERSON_BATCHES))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Containers can run under a host UID absent from /etc/passwd. PyTorch's
    # default compiler cache consults that account even without compilation.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output / "torch-cache"))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(contract["threads"])
    torch.use_deterministic_algorithms(True)
    model = TaskModel(resolved, contract)
    require(resolved.get("optimizer", "Adam") == "Adam", "unsupported optimizer; no silent substitution")
    optimizer = build_optimizer(model.parameters(), model.encoder.config.learning_rate,
                                adam_state_precision)
    identity = checkpoint_identity(contract, contract_sha256, resolved_sha256, arm, seed,
                                   adam_state_precision)
    checkpoint_dir = (Path(checkpoint_dir) if checkpoint_dir is not None
                      else output.parent / f".{output.name}.internal-checkpoints")
    store = CheckpointStore(checkpoint_dir, identity)
    try:
        start_step, history = store.restore(model, optimizer)
        require(start_step <= contract["steps"], "checkpoint is beyond the fixed training budget")
        for step in range(start_step, contract["steps"]):
            model.train()
            entry = contract["batches"]["FIT"][step % len(contract["batches"]["FIT"])]
            optimizer.zero_grad(set_to_none=True)
            data, dataset = validate_training_entry(entry, contract, "FIT", resolved)
            if dataset is None:
                prediction = model(data, arm)
                value = training_loss(prediction, data, contract["objective"])
                require(bool(torch.isfinite(value)), "nonfinite training loss")
                value.backward()
            else:
                value, _, _, _ = model.backward_autosomal(dataset, data, arm)
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "nonfinite gradient")
            optimizer.step()
            history.append(float(value.detach()))
            completed = step + 1
            if completed % checkpoint_every_steps == 0 or completed == contract["steps"]:
                store.save(model, optimizer, completed, history)
    finally:
        store.close()
    metric, errors, components = score_batches(model, contract, "SELECT", resolved, arm, output)
    checkpoint = output / "weights.npz"
    np.savez(checkpoint, **{k: v.detach().numpy() for k, v in model.state_dict().items()})
    metric_name = ("k03_component_pair_weighted_mse"
                   if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES else
                   ("component_mse" if contract["objective"] == "structure" else "component_log_loss"))
    receipt = {"schema": SCHEMA, "status": "COMPLETE", "scope": contract["scope"], "arm": arm, "seed": seed,
        "contract_sha256": contract_sha256, "resolved_sha256": resolved_sha256,
        "dataset_declared_scope": contract.get("dataset_declared_scope", "SYNTHETIC_UNSPECIFIED"),
        "dataset_mode": dataset_mode(contract),
        "eligibility_manifest_sha256": contract.get("eligibility_manifest", {}).get("sha256"),
        "trainer_sha256": sha(__file__), "runtime_sha256": sha(Path(__file__).with_name("genomic_context_runtime.py")),
        "autosomal_dataset_sha256": sha(Path(__file__).with_name("r02_context_autosomal_dataset.py")),
        "autosomal_pool_sha256": sha(Path(__file__).with_name("r02_context_autosomal_pool.py")),
        "event_mmap_reader_sha256": (sha(Path(__file__).with_name("r02_event_mmap_reader.py"))
                                      if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES else None),
        "k03_neural_target_sha256": (sha(Path(__file__).with_name("r02_k03_neural_target.py"))
                                      if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES else None),
        "metrics": {metric_name: metric}, "training_losses": history,
        "steps": contract["steps"], "persons_evaluated": len(errors), "n_evaluated": len(components),
        "internal_checkpoint": {"schema": "r02_context_internal_checkpoint_v1",
                                "identity_sha256": identity_digest(identity),
                                "adam_state_precision": adam_state_precision,
                                "restored_steps": start_step,
                                "frequency_completed_steps": checkpoint_every_steps,
                                "final_confirmed_step": contract["steps"]},
        "evaluation_unit": ("authenticated_k03_component_pair_weights"
                            if dataset_mode(contract) == AUTOSOMAL_PERSON_BATCHES else "component_macro"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha(checkpoint), "prediction_sha256": sha(output / "predictions.json"),
        "torch_version": torch.__version__, "device": "cpu", "scientific_conclusion": "NOT_AUTOMATICALLY_ESTABLISHED",
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
    args = parser.parse_args()
    if args.stage in ("evaluate", "verify-evaluation"):
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
                       args.checkpoint_every_steps, args.adam_state_precision)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
