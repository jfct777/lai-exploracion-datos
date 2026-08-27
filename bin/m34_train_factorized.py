#!/usr/bin/env python3
"""Train one M34 task directly from compact factors.

The expanded rare-context representation is built one sample/marker shard at
a time and is discarded after use.  RD and RE are derived together from the
same factors; they differ only in the rare-value channels declared by the M34
materializer.  Only FIT and VALID truth are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m33_safe_bridge_core as core
import m34_adaptive_sweep as sweep
import m34_materialize as materialize
import m34_models as models
import m34_train_packed as packed_train


MANIFEST_MEMBERS = {
    "schema_version", "ancestry_names", "haplotypes", "rotation", "splits",
}
FACTOR_ROW_MEMBERS = {
    "selected_variant", "target", "reference", "f0", "marker_cm", "truth",
}
TRUTH_MEMBERS = {"sample_key_sha256", "marker_pos", "labels"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a factor manifest that cannot name a TEST split."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(set(payload) == MANIFEST_MEMBERS, "factorized manifest members differ")
    names = materialize.normalize_ancestry_names(payload["ancestry_names"])
    require(type(payload["haplotypes"]) is int and payload["haplotypes"] == 2,
            "M34 requires two phased haplotypes")
    require(isinstance(payload["rotation"], str) and payload["rotation"],
            "manifest rotation is empty")
    require(set(payload["splits"]) == {"FIT", "VALID"},
            "factorized manifest must contain FIT and VALID only")
    base = path.parent.resolve()
    normalized: dict[str, list[dict[str, Path]]] = {"FIT": [], "VALID": []}
    for split in normalized:
        rows = payload["splits"][split]
        require(isinstance(rows, list) and rows, f"{split} factor list is empty")
        for row in rows:
            require(set(row) == FACTOR_ROW_MEMBERS, f"{split} factor members differ")
            paths = {name: _resolved(base, str(value)) for name, value in row.items()}
            require(all(value.is_file() and not value.is_symlink()
                        for value in paths.values()), f"{split} factor input is missing")
            require(len(set(paths.values())) == len(paths),
                    f"{split} factor roles must use distinct files")
            normalized[split].append(paths)
    payload["ancestry_names"] = names
    payload["splits"] = normalized
    return payload


class FactorCache:
    """Reuse compact factors shared across roots without duplicating arrays."""

    def __init__(self) -> None:
        self._items: dict[tuple[Path, tuple[str, ...]], dict[str, np.ndarray]] = {}

    def load(self, path: Path, expected: set[str]) -> dict[str, np.ndarray]:
        key = (path.resolve(), tuple(sorted(expected)))
        if key not in self._items:
            self._items[key] = materialize._load_npz(path, expected)
        return self._items[key]


@dataclass(frozen=True)
class FactorBundle:
    split: str
    index: int
    paths: Mapping[str, Path]
    selected: Mapping[str, np.ndarray]
    target: Mapping[str, np.ndarray]
    reference: Mapping[str, np.ndarray]
    f0: Mapping[str, np.ndarray]
    marker_cm: np.ndarray
    labels: np.ndarray
    transitions: np.ndarray
    dimensions: Mapping[str, int]


@dataclass(frozen=True)
class ShardSpec:
    bundle_index: int
    sample_start: int
    sample_end_exclusive: int
    marker_start: int
    marker_end_exclusive: int


@dataclass(frozen=True)
class ContextIndex:
    """Validated locus bounds for one factor and one genetic radius."""

    radius_cm: float
    starts: np.ndarray
    stops: np.ndarray


def _load_truth(path: Path, cache: FactorCache) -> dict[str, np.ndarray]:
    return cache.load(path, TRUTH_MEMBERS)


def load_factor_bundle(split: str, index: int, paths: Mapping[str, Path],
                       ancestry_names: Sequence[str], haplotypes: int,
                       cache: FactorCache) -> FactorBundle:
    selected = cache.load(paths["selected_variant"],
                          materialize.PRODUCTIVE_MEMBERS["selected"])
    target = cache.load(paths["target"], materialize.PRODUCTIVE_MEMBERS["target"])
    reference = cache.load(paths["reference"],
                           materialize.PRODUCTIVE_MEMBERS["reference"])
    f0 = cache.load(paths["f0"], materialize.PRODUCTIVE_MEMBERS["f0"])
    marker_cm = cache.load(paths["marker_cm"], {"marker_cM"})["marker_cM"]
    truth = _load_truth(paths["truth"], cache)
    dimensions = materialize.validate_inputs(
        selected, target, reference, f0, marker_cm, ancestry_names,
    )
    require(np.array_equal(truth["sample_key_sha256"], target["sample_key_sha256"]),
            f"{split} factor/truth sample axes differ")
    require(np.array_equal(truth["marker_pos"], f0["marker_pos"]),
            f"{split} factor/truth marker axes differ")
    labels = np.ascontiguousarray(truth["labels"], dtype=np.int64)
    require(labels.shape == (dimensions["sample_count"], haplotypes,
                             dimensions["marker_count"]),
            f"{split} truth dimensions differ")
    require(np.all((labels >= 0) & (labels < dimensions["ancestry_count"])),
            f"{split} truth labels differ")
    transitions = packed_train.transition_mask(labels)
    return FactorBundle(
        split=split, index=index, paths=paths, selected=selected, target=target,
        reference=reference, f0=f0, marker_cm=np.ascontiguousarray(marker_cm),
        labels=labels, transitions=transitions, dimensions=dimensions,
    )


def load_bundles(manifest: Mapping[str, Any]) -> dict[str, list[FactorBundle]]:
    cache = FactorCache()
    result: dict[str, list[FactorBundle]] = {"FIT": [], "VALID": []}
    for split in result:
        result[split] = [
            load_factor_bundle(
                split, index, paths, manifest["ancestry_names"],
                int(manifest["haplotypes"]), cache,
            )
            for index, paths in enumerate(manifest["splits"][split])
        ]
    validate_bundle_collection(result)
    return result


def validate_bundle_collection(bundles: Mapping[str, Sequence[FactorBundle]]) -> None:
    """Validate split isolation and one common marker output axis."""
    require(set(bundles) == {"FIT", "VALID"}, "bundle splits differ")
    split_keys: dict[str, set[bytes]] = {}
    for split in ("FIT", "VALID"):
        observed: set[bytes] = set()
        for bundle in bundles[split]:
            keys = [bytes(value) for value in bundle.target["sample_key_sha256"]]
            require(len(keys) == len(set(keys)), f"duplicate sample inside {split} factor")
            require(observed.isdisjoint(keys), f"sample repeated across {split} factors")
            observed.update(keys)
        split_keys[split] = observed
    require(split_keys["FIT"].isdisjoint(split_keys["VALID"]),
            "FIT and VALID sample axes overlap")
    first = bundles["FIT"][0]
    marker_members = ("marker_chrom", "marker_pos", "marker_ref", "marker_alt")
    for split in ("FIT", "VALID"):
        for bundle in bundles[split]:
            require(all(np.array_equal(bundle.f0[name], first.f0[name])
                        for name in marker_members) and
                    np.array_equal(bundle.marker_cm, first.marker_cm),
                    "factor bundles do not share one marker/genetic-map axis")


def plan_shards(bundles: Sequence[FactorBundle], sample_shard_size: int,
                marker_shard_size: int) -> list[ShardSpec]:
    require(sample_shard_size > 0 and marker_shard_size > 0,
            "sample and marker shard sizes must be positive")
    result: list[ShardSpec] = []
    for bundle_index, bundle in enumerate(bundles):
        samples = materialize.plan_sample_shards(
            int(bundle.dimensions["sample_count"]), sample_shard_size,
        )
        markers = int(bundle.dimensions["marker_count"])
        for sample in samples:
            for marker_start in range(0, markers, marker_shard_size):
                result.append(ShardSpec(
                    bundle_index=bundle_index,
                    sample_start=sample["sample_start"],
                    sample_end_exclusive=sample["sample_end_exclusive"],
                    marker_start=marker_start,
                    marker_end_exclusive=min(marker_start + marker_shard_size, markers),
                ))
    require(result, "no factorized shards were planned")
    return result


def build_context_index(bundle: FactorBundle, radius_cm: float) -> ContextIndex:
    require(radius_cm in materialize.m33.RADII,
            "radius differs from the audited context geometry")
    intervals = materialize.m33.build_interval_table(
        np.asarray(bundle.selected["cM"], dtype="<f8"), bundle.marker_cm,
    )
    radius_index = materialize.m33.RADII.index(radius_cm)
    starts = np.ascontiguousarray(intervals["context_start"][radius_index], dtype="<u8")
    stops = np.ascontiguousarray(intervals["context_stop"][radius_index], dtype="<u8")
    require(starts.shape == stops.shape == bundle.marker_cm.shape and
            np.all(starts <= stops), "context interval index differs")
    return ContextIndex(radius_cm=radius_cm, starts=starts, stops=stops)


def _base_channels_slice(bundle: FactorBundle, ancestry_names: Sequence[str],
                         max_callable_an: Mapping[str, int], sample_start: int,
                         sample_end: int, locus_start: int,
                         locus_end: int) -> np.ndarray:
    names = materialize.normalize_ancestry_names(ancestry_names)
    require(tuple(max_callable_an) == names and
            all(type(value) is int and value > 0 for value in max_callable_an.values()),
            "FIT callable normalization ancestry order differs")
    dosage = np.asarray(
        bundle.target["minor_dosage"][sample_start:sample_end, locus_start:locus_end],
        dtype="<f8",
    )
    observed = np.asarray(
        bundle.target["observed_mask"][sample_start:sample_end, locus_start:locus_end],
        dtype="<f8",
    )
    values = np.empty((*dosage.shape, 2 + 3 * len(names)), dtype="<f4")
    values[:, :, 0] = np.clip(dosage / 2.0, 0.0, 1.0)
    values[:, :, 1] = observed
    for ancestry_index, ancestry in enumerate(names):
        offset = 2 + 3 * ancestry_index
        values[:, :, offset] = np.clip(
            bundle.reference["minor_af"][ancestry_index, locus_start:locus_end],
            0.0, 1.0,
        )
        denominator = math.log1p(max_callable_an[ancestry])
        values[:, :, offset + 1] = np.clip(
            np.log1p(bundle.reference["callable_an"][
                ancestry_index, locus_start:locus_end
            ]) / denominator,
            0.0, 1.0,
        )
        values[:, :, offset + 2] = bundle.reference["observed_mask"][
            ancestry_index, locus_start:locus_end
        ]
    return np.ascontiguousarray(values)


def build_pair(bundle: FactorBundle, shard: ShardSpec,
               ancestry_names: Sequence[str], max_callable_an: Mapping[str, int],
               radius_cm: float, context_index: ContextIndex | None = None
               ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                          np.ndarray, np.ndarray]:
    """Build RD/RE from validated factors without rescanning unused loci."""
    require(shard.bundle_index == bundle.index, "shard/factor indexes differ")
    require(0 <= shard.sample_start < shard.sample_end_exclusive <=
            int(bundle.dimensions["sample_count"]), "sample shard bounds differ")
    require(0 <= shard.marker_start < shard.marker_end_exclusive <=
            int(bundle.dimensions["marker_count"]), "marker shard bounds differ")
    index = context_index or build_context_index(bundle, radius_cm)
    require(index.radius_cm == radius_cm and
            index.starts.shape == index.stops.shape == bundle.marker_cm.shape,
            "context index does not match the factor/radius")
    starts = index.starts[shard.marker_start:shard.marker_end_exclusive]
    stops = index.stops[shard.marker_start:shard.marker_end_exclusive]
    locus_start = int(starts.min(initial=int(bundle.dimensions["locus_count"])))
    locus_end = int(stops.max(initial=0))
    base = _base_channels_slice(
        bundle, ancestry_names, max_callable_an,
        shard.sample_start, shard.sample_end_exclusive, locus_start, locus_end,
    )
    sample_count = shard.sample_end_exclusive - shard.sample_start
    expected_tokens = sample_count * int(np.sum(stops - starts, dtype="<u8"))
    require(expected_tokens <= materialize.m33.TOKEN_BUDGET,
            "paired shard exceeds the audited token budget")
    rare_cm = np.asarray(bundle.selected["cM"], dtype="<f8")
    token_blocks: list[np.ndarray] = []
    locus_blocks: list[np.ndarray] = []
    row_ptr = [0]
    row_samples: list[int] = []
    row_markers: list[int] = []
    for local_sample in range(sample_count):
        for local_marker, marker_index in enumerate(
                range(shard.marker_start, shard.marker_end_exclusive)):
            left, right = int(index.starts[marker_index]), int(index.stops[marker_index])
            block = np.empty(
                (right - left, int(bundle.dimensions["channel_count"])), dtype="<f4",
            )
            if right > left:
                block[:, :-2] = base[local_sample,
                                     left - locus_start:right - locus_start]
                block[:, -2] = np.clip(
                    (rare_cm[left:right] - bundle.marker_cm[marker_index]) / radius_cm,
                    -1.0, 1.0,
                )
                delta = np.empty(right - left, dtype="<f8")
                delta[0] = 0.0
                if right - left > 1:
                    delta[1:] = np.diff(rare_cm[left:right])
                block[:, -1] = np.clip(delta / radius_cm, 0.0, 2.0)
            token_blocks.append(block)
            locus_blocks.append(np.arange(left, right, dtype="<u8"))
            row_ptr.append(row_ptr[-1] + right - left)
            row_samples.append(local_sample)
            row_markers.append(local_marker)
    require(row_ptr[-1] == expected_tokens, "paired-shard token accounting differs")
    sample_slice = slice(shard.sample_start, shard.sample_end_exclusive)
    marker_slice = slice(shard.marker_start, shard.marker_end_exclusive)
    re_view = {
        "sample_key_sha256": np.ascontiguousarray(
            bundle.target["sample_key_sha256"][sample_slice]),
        "marker_chrom": np.ascontiguousarray(bundle.f0["marker_chrom"][marker_slice]),
        "marker_pos": np.ascontiguousarray(bundle.f0["marker_pos"][marker_slice]),
        "marker_ref": np.ascontiguousarray(bundle.f0["marker_ref"][marker_slice]),
        "marker_alt": np.ascontiguousarray(bundle.f0["marker_alt"][marker_slice]),
        "marker_cM": np.ascontiguousarray(bundle.marker_cm[marker_slice], dtype="<f8"),
        "radius_cM": np.asarray([radius_cm], dtype="<f4"),
        "rare_tokens": (np.concatenate(token_blocks, axis=0) if token_blocks
                        else np.empty((0, int(bundle.dimensions["channel_count"])),
                                      dtype="<f4")),
        "rare_mask": np.ones(expected_tokens, dtype="|u1"),
        "rare_locus_index": (np.concatenate(locus_blocks) if locus_blocks
                             else np.empty(0, dtype="<u8")),
        "row_ptr": np.asarray(row_ptr, dtype="<u8"),
        "row_sample_index": np.asarray(row_samples, dtype="<u4"),
        "row_marker_index": np.asarray(row_markers, dtype="<u4"),
        "F0": np.ascontiguousarray(bundle.f0["F0"][sample_slice, :, marker_slice, :],
                                    dtype="<f4"),
    }
    rd_view = {name: np.ascontiguousarray(value.copy())
               for name, value in re_view.items()}
    rd_view["rare_tokens"][:, materialize.rare_value_channel_indices(
        len(ancestry_names))] = np.float32(0.0)
    materialize.validate_control_pair(rd_view, re_view, ancestry_names)
    labels = np.ascontiguousarray(bundle.labels[sample_slice, :, marker_slice])
    transitions = np.ascontiguousarray(bundle.transitions[sample_slice, :, marker_slice])
    return rd_view, re_view, labels, transitions


def selected_arm(rd_view: Mapping[str, np.ndarray], re_view: Mapping[str, np.ndarray],
                 arm: str) -> Mapping[str, np.ndarray]:
    require(arm in {"RD", "RE"}, "task arm differs")
    return rd_view if arm == "RD" else re_view


def _batch_to_device(batch: Sequence[torch.Tensor], device: torch.device
                     ) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device=device, non_blocking=False) for value in batch)


def weighted_loss(model: torch.nn.Module, packed: Mapping[str, np.ndarray],
                  labels: np.ndarray, transitions: np.ndarray, beta: float,
                  maximum_rows: int, maximum_tokens: int, device: torch.device,
                  backward: bool) -> tuple[float, float]:
    numerator_total = 0.0
    denominator_total = 0.0
    batches = packed_train.plan_row_batches(
        packed["row_ptr"], maximum_rows, maximum_tokens,
    )
    denominator = float(labels.size + beta * int(transitions.sum()))
    require(denominator > 0, "factorized shard has no weighted labels")
    for start, end in batches:
        values = _batch_to_device(
            packed_train.dense_batch(
                packed, labels, transitions, start, end, beta,
            ), device,
        )
        tokens, mask, baseline, target, weights = values
        probabilities = model(tokens, mask, baseline)
        chosen = probabilities.gather(2, target.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        numerator = (-torch.log(chosen) * weights).sum()
        require(torch.isfinite(numerator).item(), "factorized loss is non-finite")
        if backward:
            (numerator / denominator).backward()
        numerator_total += float(numerator.detach().cpu())
        denominator_total += float(weights.sum().detach().cpu())
    return numerator_total, denominator_total


def validation_loss(model: torch.nn.Module, bundles: Sequence[FactorBundle],
                    context_indexes: Sequence[ContextIndex],
                    shards: Sequence[ShardSpec], ancestry_names: Sequence[str],
                    max_callable_an: Mapping[str, int], task: Mapping[str, Any],
                    beta: float, maximum_rows: int, maximum_tokens: int,
                    device: torch.device) -> float:
    numerator = denominator = 0.0
    model.eval()
    with torch.inference_mode():
        for shard in shards:
            rd_view, re_view, labels, transitions = build_pair(
                bundles[shard.bundle_index], shard, ancestry_names,
                max_callable_an, float(task["radius_cM"]),
                context_indexes[shard.bundle_index],
            )
            packed = selected_arm(rd_view, re_view, str(task["arm"]))
            value, weight = weighted_loss(
                model, packed, labels, transitions, beta, maximum_rows,
                maximum_tokens, device, backward=False,
            )
            numerator += value
            denominator += weight
    require(denominator > 0, "VALID has no weighted labels")
    return numerator / denominator


def _prediction_rows(model: torch.nn.Module, packed: Mapping[str, np.ndarray],
                     labels: np.ndarray, transitions: np.ndarray,
                     maximum_rows: int, maximum_tokens: int,
                     device: torch.device) -> np.ndarray:
    samples, haplotypes, markers = labels.shape
    ancestries = packed["F0"].shape[-1]
    flat = np.empty((samples * markers, haplotypes, ancestries), dtype=np.float32)
    for start, end in packed_train.plan_row_batches(
            packed["row_ptr"], maximum_rows, maximum_tokens):
        values = _batch_to_device(
            packed_train.dense_batch(
                packed, labels, transitions, start, end, 0.0,
            ), device,
        )
        tokens, mask, baseline = values[:3]
        flat[start:end] = model(tokens, mask, baseline).cpu().numpy()
    return np.transpose(
        flat.reshape(samples, markers, haplotypes, ancestries), (0, 2, 1, 3),
    )


def predict_validation(model: torch.nn.Module, bundles: Sequence[FactorBundle],
                       context_indexes: Sequence[ContextIndex],
                       shards: Sequence[ShardSpec], ancestry_names: Sequence[str],
                       max_callable_an: Mapping[str, int], task: Mapping[str, Any],
                       maximum_rows: int, maximum_tokens: int,
                       device: torch.device) -> dict[str, np.ndarray]:
    marker_count = int(bundles[0].dimensions["marker_count"])
    ancestry_count = len(ancestry_names)
    offsets = np.cumsum([0] + [int(bundle.dimensions["sample_count"])
                              for bundle in bundles])
    probabilities = np.empty(
        (int(offsets[-1]), 2, marker_count, ancestry_count), dtype=np.float32,
    )
    coverage = np.zeros((int(offsets[-1]), marker_count), dtype=np.uint8)
    model.eval()
    with torch.inference_mode():
        for shard in shards:
            bundle = bundles[shard.bundle_index]
            rd_view, re_view, labels, transitions = build_pair(
                bundle, shard, ancestry_names, max_callable_an,
                float(task["radius_cM"]), context_indexes[shard.bundle_index],
            )
            packed = selected_arm(rd_view, re_view, str(task["arm"]))
            values = _prediction_rows(
                model, packed, labels, transitions, maximum_rows, maximum_tokens, device,
            )
            sample_start = int(offsets[shard.bundle_index]) + shard.sample_start
            sample_end = int(offsets[shard.bundle_index]) + shard.sample_end_exclusive
            probabilities[sample_start:sample_end, :,
                          shard.marker_start:shard.marker_end_exclusive, :] = values
            coverage[sample_start:sample_end,
                     shard.marker_start:shard.marker_end_exclusive] += 1
    require(np.all(coverage == 1), "VALID prediction shards overlap or contain gaps")
    sample_keys = np.concatenate([
        np.asarray(bundle.target["sample_key_sha256"]) for bundle in bundles
    ])
    return {
        "sample_key_sha256": np.ascontiguousarray(sample_keys),
        "marker_pos": np.ascontiguousarray(bundles[0].f0["marker_pos"], dtype="<i8"),
        "marker_cM": np.ascontiguousarray(bundles[0].marker_cm, dtype="<f8"),
        "ancestry_names": np.asarray(tuple(ancestry_names), dtype="|S32"),
        "probabilities": np.ascontiguousarray(probabilities),
    }


def _input_hashes(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for split in ("FIT", "VALID"):
        for index, row in enumerate(manifest["splits"][split]):
            result[f"{split}.{index}"] = {
                name: sha256_file(path) for name, path in sorted(row.items())
            }
    return result


def _state_to_cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _configure_device(name: str, threads: int) -> torch.device:
    require(threads > 0, "thread count must be positive")
    require(name in {"cpu", "cuda"}, "device must be cpu or cuda")
    if name == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.set_num_threads(threads)
    return torch.device(name)


def _training_units(shards: Sequence[ShardSpec], seed: int,
                    epoch: int) -> list[ShardSpec]:
    result = list(shards)
    random.Random(seed + epoch * 1009).shuffle(result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    require(not args.outdir.exists(), "refusing to overwrite the output directory")
    require(args.sample_shard_size > 0 and args.marker_shard_size > 0,
            "factor shard sizes must be positive")
    require(args.maximum_rows_per_batch > 0 and args.maximum_tokens_per_batch > 0,
            "dense batch limits must be positive")
    require(args.validation_every > 0, "validation interval must be positive")
    contract = sweep.validate_contract(sweep.strict_json(args.contract))
    manifest = load_manifest(args.manifest)
    task = packed_train.load_task(args.task, contract)
    require(task["rotation"] == manifest["rotation"],
            "task and factorized manifest rotations differ")
    require(tuple(contract["scope"]["ancestries"]) == manifest["ancestry_names"],
            "sweep and factorized ancestry axes differ")
    require(
        args.sample_shard_size == int(contract["training"]["batch_people"]),
        "sample shard size must equal the frozen batch_people value; only the final shard may be partial",
    )
    require(
        args.marker_shard_size == int(contract["training"]["marker_shard_size"]),
        "marker shard size differs from the frozen logical optimizer block",
    )
    require(
        args.maximum_rows_per_batch
        == int(contract["training"]["maximum_rows_per_microbatch"]),
        "maximum rows per microbatch differ from the frozen training contract",
    )
    require(
        args.maximum_tokens_per_batch
        == int(contract["training"]["maximum_padded_tokens_per_microbatch"]),
        "maximum padded tokens per microbatch differ from the frozen training contract",
    )
    require(
        args.validation_every
        == int(packed_train.training_parameter(
            contract, task, "validation_every_updates")),
        "validation cadence differs from the frozen training contract",
    )
    input_hashes = _input_hashes(manifest)
    contract_hash = sha256_file(args.contract)
    manifest_hash = sha256_file(args.manifest)
    task_hash = sha256_file(args.task)
    device = _configure_device(args.device, args.threads)
    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(contract["training"]["deterministic_algorithms"]),
    )
    boundary_contract = contract.get("boundary_loss", {})
    require(boundary_contract.get("provenance") == "M33_PRE4B" and
            boundary_contract.get("formula") ==
            "beta=q*(1-p)/(p*(1-q))-1_fit_truth_only",
            "boundary-loss contract differs from M33 PRE4B")

    bundles = load_bundles(manifest)
    max_callable_an = materialize.derive_fit_max_callable(
        [bundle.reference for bundle in bundles["FIT"]], manifest["ancestry_names"],
    )
    fit_shards = plan_shards(
        bundles["FIT"], args.sample_shard_size, args.marker_shard_size,
    )
    valid_shards = plan_shards(
        bundles["VALID"], args.sample_shard_size, args.marker_shard_size,
    )
    context_indexes = {
        split: [build_context_index(bundle, float(task["radius_cM"]))
                for bundle in bundles[split]]
        for split in ("FIT", "VALID")
    }
    boundary = packed_train.boundary_parameters(
        [bundle.labels for bundle in bundles["FIT"]],
        float(boundary_contract["target_transition_weight_share"]),
    )
    beta = float(boundary["beta"])
    channels = int(bundles["FIT"][0].dimensions["channel_count"])
    specification = packed_train.model_spec(
        contract, task, channels, len(manifest["ancestry_names"]),
    )
    model = models.build_model(specification).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(task["learning_rate"]),
        weight_decay=float(task["weight_decay"]),
    )
    maximum_updates = int(task["maximum_updates"])
    warmup = min(int(packed_train.training_parameter(
        contract, task, "warmup_updates")), maximum_updates)
    best_loss = float("inf")
    best_update = 0
    best_state = _state_to_cpu(model.state_dict())
    update = epoch = 0
    while update < maximum_updates:
        for shard in _training_units(fit_shards, seed, epoch):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rd_view, re_view, labels, transitions = build_pair(
                bundles["FIT"][shard.bundle_index], shard,
                manifest["ancestry_names"], max_callable_an,
                float(task["radius_cM"]),
                context_indexes["FIT"][shard.bundle_index],
            )
            packed = selected_arm(rd_view, re_view, str(task["arm"]))
            weighted_loss(
                model, packed, labels, transitions, beta,
                args.maximum_rows_per_batch, args.maximum_tokens_per_batch,
                device, backward=True,
            )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(contract["training"]["gradient_clip_norm"]),
            )
            update += 1
            learning_rate = packed_train.schedule(
                float(task["learning_rate"]), update, maximum_updates, warmup,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            if update % args.validation_every == 0 or update == maximum_updates:
                value = validation_loss(
                    model, bundles["VALID"], context_indexes["VALID"], valid_shards,
                    manifest["ancestry_names"], max_callable_an, task, beta,
                    args.maximum_rows_per_batch, args.maximum_tokens_per_batch, device,
                )
                if value < best_loss:
                    best_loss = value
                    best_update = update
                    best_state = _state_to_cpu(model.state_dict())
            if update >= maximum_updates:
                break
        epoch += 1
    require(math.isfinite(best_loss) and best_update > 0,
            "no finite VALID checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device)
    prediction = predict_validation(
        model, bundles["VALID"], context_indexes["VALID"], valid_shards,
        manifest["ancestry_names"],
        max_callable_an, task, args.maximum_rows_per_batch,
        args.maximum_tokens_per_batch, device,
    )
    require(_input_hashes(manifest) == input_hashes and
            sha256_file(args.contract) == contract_hash and
            sha256_file(args.manifest) == manifest_hash and
            sha256_file(args.task) == task_hash,
            "factorized inputs or contracts changed during training")

    args.outdir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.outdir / "model.pt"
    torch.save({
        "model_state_dict": best_state, "model_spec": asdict(specification),
        "ancestry_names": tuple(manifest["ancestry_names"]), "task": task,
        "fit_max_callable_an": dict(max_callable_an),
    }, checkpoint_path)
    prediction_path = args.outdir / "valid.prediction.npz"
    core.write_deterministic_npz(prediction_path, prediction)
    paired_task = {name: value for name, value in task.items() if name != "arm"}
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "M34_EXPLORATORY_TRAIN_FACTORIZED_LAZY",
        "status": "PASS_TRAINED_VALID_ONLY_FACTORIZED_LAZY",
        "claim_level": "exploratory",
        "task": task,
        "paired_task_sha256_without_arm": canonical_sha256(paired_task),
        "model_spec": asdict(specification),
        "parameter_count": models.parameter_count(model),
        "boundary_loss": boundary,
        "fit_max_callable_an": dict(max_callable_an),
        "sample_shard_size": args.sample_shard_size,
        "marker_shard_size": args.marker_shard_size,
        "maximum_rows_per_batch": args.maximum_rows_per_batch,
        "maximum_tokens_per_batch": args.maximum_tokens_per_batch,
        "fit_factor_count": len(bundles["FIT"]),
        "valid_factor_count": len(bundles["VALID"]),
        "fit_sample_count": sum(int(bundle.dimensions["sample_count"])
                                for bundle in bundles["FIT"]),
        "valid_sample_count": sum(int(bundle.dimensions["sample_count"])
                                  for bundle in bundles["VALID"]),
        "fit_lazy_shard_count": len(fit_shards),
        "valid_lazy_shard_count": len(valid_shards),
        "updates_executed": update,
        "selected_update": best_update,
        "selected_valid_loss": best_loss,
        "device": str(device),
        "input_sha256": input_hashes,
        "contract_sha256": contract_hash,
        "manifest_sha256": manifest_hash,
        "task_sha256": task_hash,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "valid_prediction_sha256": sha256_file(prediction_path),
        "rd_re_pair_policy": "same_factors_axes_masks_geometry_F0_seed_and_task_except_arm",
        "rd_zero_channel_indices": list(
            materialize.rare_value_channel_indices(len(manifest["ancestry_names"])),
        ),
        "expanded_input_artifacts_written": False,
        "test_opened": False,
        "wall_seconds": time.monotonic() - started,
    }
    receipt["semantic_sha256"] = canonical_sha256({
        name: value for name, value in receipt.items() if name != "wall_seconds"
    })
    core.write_exclusive_json(args.outdir / "train.receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--sample-shard-size", type=int, required=True)
    parser.add_argument("--marker-shard-size", type=int, required=True)
    parser.add_argument("--maximum-rows-per-batch", type=int, required=True)
    parser.add_argument("--maximum-tokens-per-batch", type=int, required=True)
    parser.add_argument("--validation-every", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "selected_update": result["selected_update"],
        "valid_loss": result["selected_valid_loss"],
    }, sort_keys=True))
