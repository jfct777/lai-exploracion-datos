#!/usr/bin/env python3
"""Cache GQ pre-head vectors and fit finite weighted FIT readouts.

Extraction authenticates one completed GQ checkpoint, assembles each of the 126
physical GT segments once, and stores only the exact 117-value final-head input,
the frozen GQ logits and query audit fields.  Training reuses the GU weighted
loss, Adam declaration and recoverable checkpoint loop; upstream model weights
and genotype arrays are absent from every readout fit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from r02_context_checkpoint import CheckpointStore, _sha_file, identity_digest
from r02_lai_donor_context import (
    FINAL_HEAD_FACTORIZED_LINEAR, FINAL_HEAD_MLP, JOINT_ANCESTRY_STATES,
    STATE_NAMES,
)
from r02_lai_donor_experiment import (
    QueryBlock, _adam, _assembled, _model, build_region_ownership,
    load_config, require_training_catalogue_coverage,
)
from r02_lai_donor_inputs import CONDITIONS, DonorInputReader, sources_from_spec
from r02_lai_donor_train import (
    AdamSpecification, FIT_SCOPE, FROZEN_READOUT_MODEL_KIND,
    FROZEN_READOUT_OBJECTIVE, FrozenReadoutTrainingBatch,
    WeightedLogicalBatchPlan, WeightedMicrobatchPlan, build_adam,
    donor_training_identity, run_donor_training_call,
)
from r02_lai_weighted_train import (
    WEIGHTED_REFERENCE_SCHEMA, load_weighted_fit_plan,
    validate_weighted_plan_reference,
)


CACHE_SCHEMA = "r02_lai_frozen_readout_cache_v1"
SEGMENT_SCHEMA = "r02_lai_frozen_readout_segment_v1"
TRAIN_CONFIG_SCHEMA = "r02_lai_frozen_readout_train_v1"
RESULT_SCHEMA = "r02_lai_frozen_readout_result_v1"
PRIOR_CODE_SCHEMA = "r02_lai_frozen_readout_prior_code_v1"
MODES = (FINAL_HEAD_FACTORIZED_LINEAR, FINAL_HEAD_MLP, "bias_only")
EXTRACTION_CODE_FILES = (
    "r02_lai_frozen_readout.py", "r02_lai_donor_context.py",
    "r02_lai_donor_experiment.py", "r02_lai_donor_inputs.py",
    "r02_lai_donor_train.py", "r02_lai_weighted_train.py",
)
DOSAGE_BY_STATE = torch.tensor(
    ((2, 0, 0), (1, 1, 0), (1, 0, 1),
     (0, 2, 0), (0, 1, 1), (0, 0, 2)), dtype=torch.float64)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: object) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def sha256(path: Path) -> str:
    return _sha_file(Path(path))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: object) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def tensor_state_sha256(state: Mapping[str, torch.Tensor], *, exclude_head: bool) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if exclude_head and name.startswith("joint_state_head."):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b"\0" + str(array.dtype).encode() + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_files(checkpoint_dir: Path) -> tuple[dict, dict[str, dict[str, object]]]:
    current = checkpoint_dir / "CURRENT"
    require(current.is_file(), "source checkpoint CURRENT is absent")
    pointer = json.loads(current.read_text())
    manifest_path = checkpoint_dir / str(pointer.get("manifest"))
    require(manifest_path.is_file(), "source checkpoint manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    payload_path = checkpoint_dir / str(manifest.get("payload"))
    require(payload_path.is_file(), "source checkpoint payload is absent")
    files = {name: {"path": str(path.resolve()), "sha256": sha256(path)}
             for name, path in (("current", current), ("manifest", manifest_path),
                                ("payload", payload_path))}
    require(files["manifest"]["sha256"] == pointer.get("manifest_sha256")
            and files["payload"]["sha256"] == manifest.get("payload_sha256"),
            "source checkpoint pointer or payload hash differs")
    return manifest, files


def _weighted_reference(plan_dir: Path) -> dict[str, object]:
    plan = plan_dir / "PLAN.json"; occurrences = plan_dir / "OCCURRENCES.npz"
    require(plan.is_file() and occurrences.is_file(),
            "weighted plan directory requires PLAN.json and OCCURRENCES.npz")
    return {"schema": WEIGHTED_REFERENCE_SCHEMA,
            "plan_json": str(plan.resolve()), "plan_sha256": sha256(plan),
            "occurrences_npz": str(occurrences.resolve()),
            "occurrences_sha256": sha256(occurrences)}


def _source_model(config: Mapping[str, object], checkpoint_dir: Path,
                  source_spec: Path, source_config: Path, condition: str):
    random.seed(config["model_seed"]); np.random.seed(config["model_seed"])
    torch.manual_seed(config["model_seed"])
    model = _model(config); specification = _adam(config)
    optimizer = build_adam(model, specification)
    manifest, checkpoint_files = _checkpoint_files(checkpoint_dir)
    identity = manifest.get("identity")
    require(isinstance(identity, dict)
            and identity.get("model_configuration") == model.configuration
            and identity.get("optimizer") == specification.receipt()
            and identity.get("task_id") == f'{config["task_id"]}:{condition}'
            and identity.get("data_scope") == FIT_SCOPE,
            "source checkpoint task, scope, model or Adam identity differs")
    identity_sources = identity.get("sources")
    require(isinstance(identity_sources, Mapping)
            and identity_sources.get("input_spec", {}).get("sha256") == sha256(source_spec)
            and identity_sources.get("experiment_config", {}).get("sha256") == sha256(source_config),
            "source checkpoint input-spec or experiment-config identity differs")
    # GQ is a frozen read-only source.  CheckpointStore uses a local advisory
    # lock, so restore from an authenticated temporary copy rather than writing
    # even a lock file beside the source checkpoint.
    with tempfile.TemporaryDirectory(prefix="r02-frozen-source-") as temporary:
        copied = Path(temporary)
        require(len({Path(value['path']).name for value in checkpoint_files.values()})
                == len(checkpoint_files), "source checkpoint basename collision")
        for value in checkpoint_files.values():
            destination = copied / Path(value["path"]).name
            shutil.copy2(value["path"], destination)
            require(sha256(destination) == value['sha256'],
                    "temporary checkpoint copy differs from authenticated source")
        store = CheckpointStore(copied, identity)
        try:
            completed, history = store.restore(model, optimizer)
        finally:
            store.close()
    require(completed == 16 and len(history) == 16,
            "frozen GQ source must be the completed 16-update checkpoint")
    model.eval()
    return model, identity, checkpoint_files


def _segment_arrays(value, represented, source_logits) -> dict[str, np.ndarray]:
    arrays = {
        "head_input": represented.head_input.detach().cpu().numpy(),
        "frozen_source_logits": source_logits.detach().cpu().numpy(),
        "support": represented.support.detach().cpu().numpy(),
        "truth_state": value.targets.truth_state.detach().cpu().numpy(),
        "truth_observed": value.targets.truth_observed.detach().cpu().numpy(),
        "query_valid": value.batch.query_valid.detach().cpu().numpy(),
        "query_global_index": value.targets.query_global_index.detach().cpu().numpy(),
    }
    require(arrays["head_input"].ndim == 3 and arrays["head_input"].shape[0] == 1
            and arrays["head_input"].shape[2] == 117,
            "cached pre-head representation must have exact width 117")
    require(np.isfinite(arrays["head_input"]).all()
            and np.isfinite(arrays["frozen_source_logits"]).all(),
            "cached representation or source logits are nonfinite")
    return arrays


def _restore_segment(manifest_path: Path, expected_identity: dict) -> bool:
    if not manifest_path.exists():
        return False
    value = json.loads(manifest_path.read_text())
    payload_name = value.get("payload")
    require(isinstance(payload_name, str) and Path(payload_name).name == payload_name,
            "existing frozen-readout payload path is not a local filename")
    payload = manifest_path.parent / payload_name
    require(value.get("schema") == SEGMENT_SCHEMA
            and value.get("identity") == expected_identity
            and value.get("identity_sha256") == canonical_sha256(expected_identity)
            and payload.is_file() and sha256(payload) == value.get("payload_sha256"),
            "existing frozen-readout segment differs from requested identity")
    with np.load(payload, allow_pickle=False) as stored:
        require(set(stored.files) == {"head_input", "frozen_source_logits", "support",
                "truth_state", "truth_observed", "query_valid", "query_global_index"},
                "existing frozen-readout segment fields differ")
        require(np.isfinite(stored["head_input"]).all()
                and np.isfinite(stored["frozen_source_logits"]).all(),
                "existing frozen-readout segment contains nonfinite values")
    return True


def _load_prior_code_manifest(path: Path, current: Mapping[str, str]) -> dict[str, str]:
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict) and set(value) == {"schema", "code_sha256"}
            and value.get("schema") == PRIOR_CODE_SCHEMA,
            "prior code manifest schema or fields differ")
    hashes = value.get("code_sha256")
    require(isinstance(hashes, dict) and set(hashes) == set(EXTRACTION_CODE_FILES)
            and all(isinstance(digest, str) and len(digest) == 64
                    and digest == digest.lower()
                    and all(character in "0123456789abcdef" for character in digest)
                    for digest in hashes.values()),
            "prior code manifest must contain the exact six lowercase SHA256 values")
    unchanged = set(EXTRACTION_CODE_FILES) - {"r02_lai_frozen_readout.py"}
    require(all(hashes[name] == current[name] for name in unchanged),
            "prior code manifest changes a non-extractor source hash")
    return dict(hashes)


def _query_budget_receipt(model, blocks: Sequence[QueryBlock],
                          reference_people: int,
                          override: int | None) -> dict[str, object]:
    require(override is None or (type(override) is int and override > 0),
            "max query-state elements must be a positive integer")
    require(type(reference_people) is int and reference_people > 0 and blocks,
            "query-state preflight needs references and physical segments")
    original = model.max_query_state_elements
    effective = original if override is None else override
    requirements = {
        block.region_id: reference_people * (block.query_stop - block.query_start) * model.width
        for block in blocks
    }
    maximum = max(requirements.values())
    require(maximum <= effective,
            "B*D*Q*W query-state budget is too small for at least one physical segment")
    receipt = {
        "scope": "EXTRACTOR_ONLY_RUNTIME_ALLOCATION_GUARD",
        "formula": "B*D*Q*W",
        "batch_people": 1,
        "reference_people": reference_people,
        "representation_width": model.width,
        "physical_segments_checked": len(blocks),
        "maximum_queries_per_segment": max(
            block.query_stop - block.query_start for block in blocks),
        "maximum_required_elements": maximum,
        "source_configured_max_query_state_elements": original,
        "effective_max_query_state_elements": effective,
        "override_supplied": override is not None,
    }
    # This property is an allocation guard, not a parameter or persistent tensor.
    # The caller records the source configuration and state hashes before mutation.
    model.max_query_state_elements = effective
    return receipt


def _atomic_link_or_copy(source: Path, destination: Path) -> str:
    require(source.is_file() and not source.is_symlink(),
            "prior segment payload must be a regular non-symbolic file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor); os.unlink(temporary_name)
    temporary = Path(temporary_name)
    method = "hardlink"
    try:
        try:
            os.link(source, temporary)
        except OSError:
            method = "copy"
            shutil.copy2(source, temporary)
        require(sha256(temporary) == sha256(source),
                "imported prior segment payload bytes differ")
        os.replace(temporary, destination); _fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return method


def _import_prior_segment(prior_manifest_path: Path, destination_manifest_path: Path,
                          prior_identity: dict, new_identity: dict,
                          prior_code_receipt: dict[str, object]) -> bool:
    if not prior_manifest_path.exists():
        return False
    require(not prior_manifest_path.is_symlink(),
            "prior segment manifest must be a regular non-symbolic file")
    _restore_segment(prior_manifest_path, prior_identity)
    prior = json.loads(prior_manifest_path.read_text())
    prior_payload = prior_manifest_path.parent / prior["payload"]
    destination_payload = destination_manifest_path.parent / prior_payload.name
    method = _atomic_link_or_copy(prior_payload, destination_payload)
    imported = dict(prior)
    imported.update({
        "identity": new_identity,
        "identity_sha256": canonical_sha256(new_identity),
        "payload": destination_payload.name,
        "payload_sha256": sha256(destination_payload),
        "reused_from": {
            "prior_manifest_path": str(prior_manifest_path.resolve()),
            "prior_manifest_sha256": sha256(prior_manifest_path),
            "prior_payload_path": str(prior_payload.resolve()),
            "prior_payload_sha256": sha256(prior_payload),
            "prior_identity_sha256": prior["identity_sha256"],
            "prior_cache_identity_sha256": prior_identity["cache_identity_sha256"],
            "prior_code_manifest": prior_code_receipt,
            "payload_import_method": method,
            "byte_identical_payload": True,
        },
    })
    atomic_json(destination_manifest_path, imported)
    _restore_segment(destination_manifest_path, new_identity)
    require(sha256(destination_payload) == sha256(prior_payload),
            "migrated segment payload differs from prior bytes")
    return True


def extract(source_spec: Path, source_config: Path, checkpoint_dir: Path,
            weighted_plan_dir: Path, condition: str, output_dir: Path, *,
            max_query_state_elements: int | None = None,
            reuse_segments_from: Path | None = None,
            prior_code_manifest: Path | None = None) -> dict:
    require(condition in CONDITIONS, "unknown source-rare condition")
    require((reuse_segments_from is None) == (prior_code_manifest is None),
            "prior segment reuse requires both old cache root and prior code manifest")
    config = load_config(source_config)
    sources = sources_from_spec(source_spec)
    admission = require_training_catalogue_coverage(
        sources, policy=config.get("catalogue_policy"),
        input_spec_sha256=sha256(source_spec))
    torch.set_num_threads(config["cpu_threads"]); torch.use_deterministic_algorithms(True)
    model, source_identity, checkpoint_files = _source_model(
        config, checkpoint_dir, source_spec, source_config, condition)
    reference_value = _weighted_reference(weighted_plan_dir)
    weighted_reference = validate_weighted_plan_reference(reference_value, Path("/"))
    code_root = Path(__file__).resolve().parent
    code_files = {name: sha256(code_root / name) for name in EXTRACTION_CODE_FILES}
    prior_hashes = (_load_prior_code_manifest(prior_code_manifest, code_files)
                    if prior_code_manifest is not None else None)
    prior_code_receipt = ({"path": str(prior_code_manifest.resolve()),
                           "sha256": sha256(prior_code_manifest),
                           "schema": PRIOR_CODE_SCHEMA,
                           "code_sha256": prior_hashes}
                          if prior_code_manifest is not None else None)
    if reuse_segments_from is not None:
        require(reuse_segments_from.resolve() != output_dir.resolve(),
                "prior and new cache roots must differ")
    with DonorInputReader(sources) as reader:
        query_valid = np.asarray(reader.query["query_valid"], dtype=np.bool_)
        truth_observed = np.asarray(reader.query["truth_observed"], dtype=np.bool_)
        truth_state = np.asarray(reader.query["truth_state"], dtype=np.int64)
        person_ids = tuple(reader.query["person_id"].astype(str).tolist())
        conversion = load_weighted_fit_plan(
            weighted_reference, person_ids=person_ids, query_valid=query_valid,
            truth_observed=truth_observed, truth_state=truth_state,
            source_spec_path=source_spec,
            query_truth_overlay_path=reader.paths["query_overlay"])
        require(config["plan_seed"] == conversion.plan_metadata.get("seed"),
                "source config and weighted GT plan seeds differ")
        blocks = tuple(QueryBlock(x.region_id, x.person_index, x.query_start, x.query_stop)
                       for x in conversion.physical_segments)
        ownership = build_region_ownership(query_valid, blocks)
        state_counts = np.bincount(truth_state[query_valid & truth_observed],
                                   minlength=JOINT_ANCESTRY_STATES)
        require(int(state_counts.sum()) == int((query_valid & truth_observed).sum())
                and bool((state_counts > 0).all()),
                "complete FIT rectangle must contain all six truth states")
        source_state = model.state_dict()
        source_model_configuration = model.configuration
        source_model_state_sha = tensor_state_sha256(source_state, exclude_head=False)
        source_upstream_state_sha = tensor_state_sha256(source_state, exclude_head=True)
        query_budget = _query_budget_receipt(
            model, blocks, config["reference_policy"]["expected_reference_people"],
            max_query_state_elements)
        require(source_model_state_sha == tensor_state_sha256(model.state_dict(), exclude_head=False)
                and source_upstream_state_sha == tensor_state_sha256(
                    model.state_dict(), exclude_head=True),
                "extractor query-state limit changed source weights")
        common_identity = {
            "schema": CACHE_SCHEMA, "condition": condition,
            "source_spec": {"path": str(source_spec.resolve()), "sha256": sha256(source_spec)},
            "source_config": {"path": str(source_config.resolve()),
                              "sha256": sha256(source_config)},
            "weighted_plan": reference_value,
            "checkpoint_files": checkpoint_files,
            "source_training_identity_sha256": identity_digest(source_identity),
            "source_model_state_sha256": source_model_state_sha,
            "source_upstream_state_sha256": source_upstream_state_sha,
            "source_model_configuration": source_model_configuration,
            "catalogue_admission": admission,
            "physical_segment_receipts_sha256":
                conversion.source_receipt["physical_segment_receipts_sha256"],
            "code_sha256": code_files,
            "fit_truth_state_counts": dict(zip(STATE_NAMES, state_counts.tolist())),
            "fit_truth_total": int(state_counts.sum()),
        }
        if max_query_state_elements is not None:
            common_identity["execution_budget"] = query_budget
        cache_identity_sha = canonical_sha256(common_identity)
        prior_common_identity = None
        prior_cache_identity_sha = None
        if prior_hashes is not None:
            prior_common_identity = dict(common_identity)
            prior_common_identity.pop("execution_budget", None)
            prior_common_identity["code_sha256"] = prior_hashes
            prior_cache_identity_sha = canonical_sha256(prior_common_identity)
        segment_root = output_dir / "segments"; segment_root.mkdir(parents=True, exist_ok=True)
        segment_rows = []
        direct_forward_equivalence_checked = False
        for block in blocks:
            person_id = person_ids[block.person_index]
            segment_identity = {"cache_identity_sha256": cache_identity_sha,
                                "block": block.receipt(), "person_id": person_id}
            manifest_path = segment_root / f"{block.region_id}.json"
            if not _restore_segment(manifest_path, segment_identity):
                payload_path = segment_root / f"{block.region_id}.npz"
                if payload_path.exists():
                    # A crash can occur after durable payload replacement and before
                    # its manifest.  Preserve that incomplete object for diagnosis,
                    # then recompute only this segment under the complete identity.
                    orphan = payload_path.with_name(
                        f"{payload_path.stem}.orphan-{sha256(payload_path)[:16]}.npz")
                    serial = 0
                    while orphan.exists():
                        serial += 1
                        orphan = payload_path.with_name(
                            f"{payload_path.stem}.orphan-{sha256(payload_path)[:16]}-{serial}.npz")
                    payload_path.replace(orphan); _fsync_directory(payload_path.parent)
                migrated = False
                if reuse_segments_from is not None:
                    prior_identity = {
                        "cache_identity_sha256": prior_cache_identity_sha,
                        "block": block.receipt(), "person_id": person_id,
                    }
                    migrated = _import_prior_segment(
                        reuse_segments_from / "segments" / f"{block.region_id}.json",
                        manifest_path, prior_identity, segment_identity,
                        prior_code_receipt)
                if migrated:
                    stored_manifest = json.loads(manifest_path.read_text())
                    segment_rows.append({
                        "region_id": block.region_id, "person_id": person_id,
                        "manifest": str(manifest_path.relative_to(output_dir)),
                        "manifest_sha256": sha256(manifest_path),
                        "payload": str((segment_root / stored_manifest["payload"])
                                       .relative_to(output_dir)),
                        "payload_sha256": stored_manifest["payload_sha256"],
                    })
                    continue
                value = _assembled(reader, ownership, block, config, condition)
                require(value.targets.query_global_index.shape[0] == 1
                        and value.lineage["training_performed"] is False,
                        "segment assembly axes or lineage differ")
                with torch.no_grad():
                    represented = model.readout_representation(value.batch)
                    source_logits = model.joint_state_head(represented.head_input)
                    if not direct_forward_equivalence_checked:
                        old_logits = model(value.batch).logits
                        torch.testing.assert_close(source_logits, old_logits, atol=0, rtol=0)
                        direct_forward_equivalence_checked = True
                arrays = _segment_arrays(value, represented, source_logits)
                atomic_npz(payload_path, arrays)
                segment_manifest = {
                    "schema": SEGMENT_SCHEMA, "identity": segment_identity,
                    "identity_sha256": canonical_sha256(segment_identity),
                    "payload": payload_path.name, "payload_sha256": sha256(payload_path),
                    "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                               for name, array in arrays.items()},
                    "lineage": value.lineage,
                    "unsupported_valid_queries": int((arrays["query_valid"]
                                                       & ~arrays["support"]).sum()),
                }
                atomic_json(manifest_path, segment_manifest)
            stored_manifest = json.loads(manifest_path.read_text())
            segment_rows.append({"region_id": block.region_id, "person_id": person_id,
                                 "manifest": str(manifest_path.relative_to(output_dir)),
                                 "manifest_sha256": sha256(manifest_path),
                                 "payload": str((segment_root / stored_manifest["payload"])
                                                .relative_to(output_dir)),
                                 "payload_sha256": stored_manifest["payload_sha256"]})
        result = {
            "schema": CACHE_SCHEMA, "identity": common_identity,
            "identity_sha256": cache_identity_sha,
            "physical_segments": len(segment_rows), "segments": segment_rows,
            "logical_batches": [plan.receipt() for plan in conversion.logical_batches],
            "weighted_conversion_receipt": conversion.source_receipt,
            "cache_contains_genotype_arrays": False,
            "cache_contains_only_vectors_logits_truth_masks_indices": True,
            "direct_forward_equivalence_checked_on_one_new_segment":
                direct_forward_equivalence_checked,
            "extractor_query_state_budget": query_budget,
            "reused_prior_segment_count": sum(
                "reused_from" in json.loads((output_dir / row["manifest"]).read_text())
                for row in segment_rows),
            "scientific_scope": "CONDITIONAL_FIT_DEVELOPMENT_NO_SELECT_OR_EVALUATE",
        }
    atomic_json(output_dir / "CACHE.json", result)
    return result


@dataclass(frozen=True)
class ReadoutOutput:
    logits: torch.Tensor
    support: torch.Tensor


class FrozenReadoutModel(nn.Module):
    training_model_kind = FROZEN_READOUT_MODEL_KIND

    def __init__(self, *, mode: str, input_width: int, hidden_width: int | None,
                 fit_prior: Sequence[float], cache_identity_sha256: str):
        super().__init__()
        require(mode in MODES and type(input_width) is int and input_width > 0,
                "invalid frozen readout mode or input width")
        prior64 = torch.as_tensor(fit_prior, dtype=torch.float64)
        require(prior64.shape == (JOINT_ANCESTRY_STATES,) and bool((prior64 > 0).all())
                and math.isclose(float(prior64.sum()), 1.0, abs_tol=1e-12),
                "frozen readout requires six positive complete-FIT frequencies")
        prior = prior64.to(torch.float32)
        self.mode = mode; self.input_width = input_width
        self.hidden_width = hidden_width
        self.cache_identity_sha256 = cache_identity_sha256
        if mode == "bias_only":
            require(hidden_width is None, "bias-only readout has no hidden width")
            self.bias = nn.Parameter(torch.zeros(JOINT_ANCESTRY_STATES))
        else:
            require(hidden_width == input_width,
                    "Identity/GELU comparison fixes hidden width equal to input width")
            self.input_layer = nn.Linear(input_width, hidden_width)
            self.activation = nn.Identity() if mode == FINAL_HEAD_FACTORIZED_LINEAR else nn.GELU()
            self.output_layer = nn.Linear(hidden_width, JOINT_ANCESTRY_STATES)
            with torch.no_grad():
                self.input_layer.weight.copy_(torch.eye(input_width))
                self.input_layer.bias.zero_(); self.output_layer.weight.zero_()
                self.output_layer.bias.copy_(prior.log())
        self._fit_prior = tuple(float(value) for value in prior.double().tolist())

    @property
    def configuration(self) -> dict[str, object]:
        return {"schema": "r02_lai_frozen_readout_model_v1", "mode": self.mode,
                "input_width": self.input_width, "hidden_width": self.hidden_width,
                "output_width": JOINT_ANCESTRY_STATES,
                "activation": ("GELU" if self.mode == FINAL_HEAD_MLP else
                               "identity" if self.mode == FINAL_HEAD_FACTORIZED_LINEAR else
                               "frozen_source_logits_plus_bias"),
                "initialization": ("W1_IDENTITY_B1_ZERO_W2_ZERO_B2_LOG_COMPLETE_FIT_FREQUENCY"
                                   if self.mode != "bias_only" else
                                   "FROZEN_GQ_LOGITS_PLUS_ZERO_BIAS"),
                "complete_fit_prior": list(self._fit_prior),
                "cache_identity_sha256": self.cache_identity_sha256}

    def forward(self, batch: FrozenReadoutTrainingBatch) -> ReadoutOutput:
        require(type(batch) is FrozenReadoutTrainingBatch,
                "FrozenReadoutTrainingBatch required")
        if self.mode == "bias_only":
            logits = batch.frozen_source_logits + self.bias
        else:
            logits = self.output_layer(self.activation(self.input_layer(batch.head_input)))
        zero = logits.sum(-1, keepdim=True) * 0
        logits = torch.where(batch.query_valid[..., None], logits, zero.expand_as(logits))
        return ReadoutOutput(logits, batch.support & batch.query_valid)


def _microbatch(value: Mapping[str, object]) -> WeightedMicrobatchPlan:
    return WeightedMicrobatchPlan(
        str(value["identity_sha256"]), str(value["region_id"]),
        int(value["query_start"]), int(value["query_stop"]),
        tuple(str(x) for x in value["person_ids"]),
        tuple(tuple(int(x) for x in row) for row in value["scored_query_global_indices"]),
        tuple(tuple(float(x) for x in row) for row in value["loss_coefficients"]),
        tuple(tuple(int(x) for x in row) for row in value["occurrence_counts"]))


def _base_plans(cache: Mapping[str, object]) -> tuple[WeightedLogicalBatchPlan, ...]:
    plans = []
    for value in cache["logical_batches"]:
        microbatches = tuple(_microbatch(x) for x in value["microbatches"])
        plans.append(WeightedLogicalBatchPlan(
            int(value["step_index"]), str(value["logical_batch_id"]), microbatches,
            int(value["declared_occurrences"])))
    require(len(plans) == 16 and tuple(x.step_index for x in plans) == tuple(range(16)),
            "cache must preserve the exact 16-update GT plan")
    return tuple(plans)


def _repeated_plans(base: Sequence[WeightedLogicalBatchPlan], updates: int
                    ) -> tuple[WeightedLogicalBatchPlan, ...]:
    require(updates in (16, 64, 256), "updates must be one frozen endpoint")
    result = []
    for step in range(updates):
        source = base[step % len(base)]
        result.append(WeightedLogicalBatchPlan(
            step, f"pass{step // len(base):02d}-{source.logical_batch_id}",
            source.microbatches, source.declared_occurrences))
    return tuple(result)


def load_train_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    required = {"schema", "task_id", "mode", "hidden_width", "seed", "updates",
                "endpoints", "optimizer", "max_microbatches_per_call"}
    require(isinstance(value, dict) and set(value) == required
            and value.get("schema") == TRAIN_CONFIG_SCHEMA,
            "frozen-readout train config schema or fields differ")
    require(isinstance(value["task_id"], str) and value["task_id"]
            and value["mode"] in MODES and type(value["seed"]) is int and value["seed"] >= 0
            and value["updates"] == 256 and value["endpoints"] == [16, 64, 256],
            "frozen-readout task, mode, seed or finite endpoints differ")
    require((value["mode"] == "bias_only" and value["hidden_width"] is None)
            or (value["mode"] != "bias_only" and value["hidden_width"] == 117),
            "frozen-readout hidden width differs from frozen comparison")
    maximum = value["max_microbatches_per_call"]
    require(maximum is None or (type(maximum) is int and maximum > 0),
            "max_microbatches_per_call must be positive or null")
    optimizer = value["optimizer"]
    require(isinstance(optimizer, dict) and set(optimizer) == {
        "learning_rate", "betas", "epsilon", "weight_decay", "gradient_clip_norm",
        "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"},
        "frozen-readout optimizer fields differ")
    optimizer = dict(optimizer)
    require(optimizer["learning_rate"] in (0.0001, 0.001)
            and isinstance(optimizer["betas"], list) and len(optimizer["betas"]) == 2,
            "only the two frozen GQ learning rates and explicit Adam betas are allowed")
    optimizer["betas"] = tuple(optimizer["betas"])
    AdamSpecification(**optimizer)
    return value


class CachedBatchSource:
    def __init__(self, cache_dir: Path, cache: Mapping[str, object]):
        self.root = cache_dir
        self.rows = {value["region_id"]: value for value in cache["segments"]}
        require(len(self.rows) == len(cache['segments'])
                == cache.get('physical_segments', len(self.rows)),
                "cache segment count or unique region identity differs")
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        total = sum((self.root / str(value["payload"])).stat().st_size
                    for value in cache["segments"])
        # atomic_npz uses uncompressed np.savez. File bytes bound array bytes;
        # leave memory for Torch and copies inside the 1GiB task allocation.
        require(total < 128 * 1024 ** 2,
                "uncompressed frozen vectors exceed the 128 MiB cache budget")
        self.cache_payload_bytes = total

    def _arrays(self, region_id: str) -> dict[str, np.ndarray]:
        if region_id in self._cache:
            return self._cache[region_id]
        row = self.rows.get(region_id)
        require(isinstance(row, Mapping), "weighted plan references absent cache segment")
        manifest_path = self.root / str(row["manifest"])
        payload_path = self.root / str(row["payload"])
        require(manifest_path.is_file() and sha256(manifest_path) == row["manifest_sha256"]
                and payload_path.is_file() and sha256(payload_path) == row["payload_sha256"],
                "cached segment manifest or payload hash differs")
        with np.load(payload_path, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        self._cache[region_id] = arrays
        return arrays

    def __call__(self, _model, _step: int, index: int):
        require(self.plans[_step].step_index == _step,
                "cached batch step differs from the declared logical update")
        plan = self.plans[_step].microbatches[index]
        arrays = self._arrays(plan.region_id)
        global_index = torch.from_numpy(arrays["query_global_index"].copy()).long()
        query_valid = torch.from_numpy(arrays["query_valid"].copy()).bool()
        query_core = torch.zeros_like(query_valid)
        for row, indices in enumerate(plan.scored_query_global_indices):
            for query in indices:
                query_core[row] |= global_index[row] == query
        observed = torch.from_numpy(arrays["truth_observed"].copy()).bool()
        require(tuple(tuple(int(x) for x in global_index[row][query_core[row] & observed[row]].tolist())
                      for row in range(len(plan.person_ids))) ==
                plan.scored_query_global_indices,
                "cached scored indices differ from weighted plan")
        return FrozenReadoutTrainingBatch(
            plan.identity_sha256, plan.person_ids,
            torch.from_numpy(arrays["head_input"].copy()).float(),
            torch.from_numpy(arrays["frozen_source_logits"].copy()).float(),
            torch.from_numpy(arrays["support"].copy()).bool(),
            torch.from_numpy(arrays["truth_state"].copy()).long(), observed,
            query_core, query_valid, global_index)


def _evaluate(model: FrozenReadoutModel, plans: Sequence[WeightedLogicalBatchPlan],
              source: CachedBatchSource, state_counts: np.ndarray,
              *, trained_updates: int) -> dict[str, object]:
    loss_by_state = torch.zeros(JOINT_ANCESTRY_STATES, dtype=torch.float64)
    weight_by_state = torch.zeros_like(loss_by_state)
    brier_by_state = torch.zeros_like(loss_by_state)
    dosage_absolute_by_state = torch.zeros(
        (JOINT_ANCESTRY_STATES, 3), dtype=torch.float64)
    dosage_squared_by_state = torch.zeros_like(dosage_absolute_by_state)
    nn_probability_by_state = torch.zeros_like(loss_by_state)
    unsupported_weight = 0.0; occurrences = 0
    model.eval()
    with torch.no_grad():
        for plan in plans:
            for index, micro in enumerate(plan.microbatches):
                batch = source(model, plan.step_index, index)
                output = model(batch)
                scored = batch.truth_observed & batch.query_core & batch.query_valid
                truth = batch.truth_state[scored]
                nll = -F.log_softmax(output.logits, -1)[scored].gather(
                    -1, truth[:, None]).squeeze(-1).double()
                probability = F.softmax(output.logits, -1)[scored].double()
                one_hot = F.one_hot(truth, JOINT_ANCESTRY_STATES).double()
                brier = ((probability - one_hot) ** 2).sum(-1)
                predicted_dosage = probability @ DOSAGE_BY_STATE
                true_dosage = DOSAGE_BY_STATE[truth]
                dosage_absolute = (predicted_dosage - true_dosage).abs()
                weights = torch.cat([torch.tensor(row, dtype=torch.float64)
                                     for row in micro.loss_coefficients])
                for state in range(JOINT_ANCESTRY_STATES):
                    take = truth == state
                    loss_by_state[state] += (weights[take] * nll[take]).sum()
                    weight_by_state[state] += weights[take].sum()
                    brier_by_state[state] += (weights[take] * brier[take]).sum()
                    dosage_absolute_by_state[state] += (
                        weights[take, None] * dosage_absolute[take]).sum(0)
                    dosage_squared_by_state[state] += (
                        weights[take, None] * (dosage_absolute[take] ** 2)).sum(0)
                    nn_probability_by_state[state] += (
                        weights[take] * probability[take, 5]).sum()
                unsupported_weight += float(weights[(~output.support)[scored]].sum())
                occurrences += micro.occurrence_count
    passes = len(plans) / 16
    require(passes in (1, 4, 16), "metrics require complete frozen GT passes")
    prior = torch.from_numpy(state_counts.astype(np.float64) / state_counts.sum())
    # Each of the 16 updates in a pass has total coefficient one.  Report every
    # metric per update; dividing only by passes would leave an erroneous factor
    # of 16 in both the state mass and the fixed-mass conditional metric.
    mean_state_loss = loss_by_state / len(plans)
    mean_state_weight = weight_by_state / len(plans)
    mean_state_brier = brier_by_state / len(plans)
    mean_state_dosage = dosage_absolute_by_state / len(plans)
    mean_state_dosage_squared = dosage_squared_by_state / len(plans)
    mean_state_nn_probability = nn_probability_by_state / len(plans)
    conditional = mean_state_loss / prior
    conditional_brier = mean_state_brier / prior
    conditional_dosage = mean_state_dosage / prior[:, None]
    conditional_dosage_squared = mean_state_dosage_squared / prior[:, None]
    return {"schema": "r02_lai_frozen_readout_metrics_v1",
            "trained_updates": trained_updates,
            "evaluation_plan_updates": len(plans),
            "complete_gt_passes": int(passes),
            "model_configuration": model.configuration,
            "weighted_nll": float(mean_state_loss.sum()),
            "weighted_brier": float(mean_state_brier.sum()),
            "weighted_dosage_mae_by_ancestry":
                dict(zip(("AFR", "EUR", "NAM"), mean_state_dosage.sum(0).tolist())),
            "weighted_dosage_mse_by_ancestry":
                dict(zip(("AFR", "EUR", "NAM"),
                         mean_state_dosage_squared.sum(0).tolist())),
            "weighted_mean_predicted_probability_NN":
                float(mean_state_nn_probability.sum()),
            "state_order": list(STATE_NAMES),
            "complete_fit_state_counts": state_counts.tolist(),
            "complete_fit_state_frequency": prior.tolist(),
            "observed_weight_per_update_by_state": mean_state_weight.tolist(),
            "weighted_nll_numerator_per_update_by_state": mean_state_loss.tolist(),
            "conditional_nll_fixed_fit_mass_by_state": conditional.tolist(),
            "conditional_brier_fixed_fit_mass_by_state": conditional_brier.tolist(),
            "conditional_dosage_mae_fixed_fit_mass_by_state_and_ancestry":
                conditional_dosage.tolist(),
            "conditional_dosage_mse_fixed_fit_mass_by_state_and_ancestry":
                conditional_dosage_squared.tolist(),
            "weighted_mean_predicted_probability_NN_numerator_by_truth_state":
                mean_state_nn_probability.tolist(),
            "conditional_denominator_rule":
                "(SUM_COEFFICIENT_TIMES_STATE_NLL/UPDATES)/COMPLETE_FIT_STATE_FREQUENCY",
            "unsupported_weight_per_update": unsupported_weight / len(plans),
            "occurrences": occurrences,
            "selection_or_evaluate_read": False}


def _endpoint_snapshot(root: Path, identity: dict, endpoint: int,
                       model: nn.Module, optimizer, history: Sequence[float]) -> None:
    endpoint_identity = {"schema": "r02_lai_frozen_readout_endpoint_v1",
                         "training_identity_sha256": identity_digest(identity),
                         "endpoint_updates": endpoint}
    store = CheckpointStore(root / f"u{endpoint:06d}", endpoint_identity)
    try:
        restored_step, restored_history = store.restore(model, optimizer)
        if restored_step == 0:
            store.save(model, optimizer, endpoint, list(history))
        else:
            require(restored_step == endpoint and restored_history == list(history),
                    "existing endpoint snapshot differs")
    finally:
        store.close()


def train_head(cache_dir: Path, config_path: Path, output_dir: Path) -> dict:
    config = load_train_config(config_path)
    cache_path = cache_dir / "CACHE.json"
    require(cache_path.is_file(), "CACHE.json is absent")
    cache = json.loads(cache_path.read_text())
    require(cache.get("schema") == CACHE_SCHEMA
            and cache.get("identity_sha256") == canonical_sha256(cache["identity"])
            and cache.get("physical_segments") == 126,
            "frozen-readout cache identity or geometry differs")
    counts_map = cache["identity"]["fit_truth_state_counts"]
    state_counts = np.asarray([counts_map[name] for name in STATE_NAMES], dtype=np.int64)
    require(int(state_counts.sum()) == cache["identity"]["fit_truth_total"] == 930176
            and bool((state_counts > 0).all()),
            "frozen complete-FIT truth denominator differs")
    prior = state_counts.astype(np.float64) / state_counts.sum()
    torch.set_num_threads(1)
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    model = FrozenReadoutModel(
        mode=config["mode"], input_width=117, hidden_width=config["hidden_width"],
        fit_prior=prior, cache_identity_sha256=cache["identity_sha256"])
    optimizer_value = dict(config["optimizer"]); optimizer_value["betas"] = tuple(optimizer_value["betas"])
    specification = AdamSpecification(**optimizer_value)
    optimizer = build_adam(model, specification)
    base = _base_plans(cache); plans = _repeated_plans(base, config["updates"])
    source = CachedBatchSource(cache_dir, cache); source.plans = plans
    code_root = Path(__file__).resolve().parent
    identity = donor_training_identity(
        task_id=config["task_id"], data_scope=FIT_SCOPE, model=model, optimizer=optimizer,
        optimizer_specification=specification,
        source_paths={"cache_manifest": cache_path, "train_config": config_path,
                      "code:r02_lai_frozen_readout.py": code_root / "r02_lai_frozen_readout.py",
                      "code:r02_lai_donor_train.py": code_root / "r02_lai_donor_train.py"},
        policy_path=config_path, objective=FROZEN_READOUT_OBJECTIVE,
        seed=config["seed"], logical_batches=plans)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_path = output_dir / "METRICS-INITIAL.json"
    if initial_path.exists():
        existing_initial = json.loads(initial_path.read_text())
        require(existing_initial.get("model_configuration") == model.configuration
                and existing_initial.get("trained_updates") == 0,
                "existing initial metrics use another cache or readout configuration")
    else:
        source.plans = tuple(base)
        atomic_json(initial_path, _evaluate(
            model, base, source, state_counts, trained_updates=0))
        source.plans = plans
    endpoints = set(config["endpoints"])
    completed_store = CheckpointStore(output_dir / "checkpoints" / "completed", identity)
    try:
        completed, restored_history = completed_store.restore(model, optimizer)
    finally:
        completed_store.close()
    require(0 <= completed <= len(plans), "restored readout step is outside frozen horizon")

    def preserve_endpoint(step: int, history: Sequence[float]) -> None:
        _endpoint_snapshot(output_dir / "endpoints", identity, step,
                           model, optimizer, history)
        source.plans = tuple(plans[:step])
        try:
            metrics = _evaluate(model, plans[:step], source, state_counts,
                                trained_updates=step)
            metrics_path = output_dir / f"METRICS-u{step:06d}.json"
            if metrics_path.exists():
                require(json.loads(metrics_path.read_text()) == metrics,
                        "existing endpoint metrics differ; preserve for diagnosis")
            else:
                atomic_json(metrics_path, metrics)
        finally:
            source.plans = plans

    # A process may stop after the completed checkpoint becomes durable but
    # before its endpoint copy/metrics.  Restore and preserve that exact state
    # before allowing the next Adam update.
    if completed in endpoints:
        preserve_endpoint(completed, restored_history)
    for prior_endpoint in sorted(endpoint for endpoint in endpoints if endpoint < completed):
        require((output_dir / "endpoints" / f"u{prior_endpoint:06d}" / "CURRENT").is_file()
                and (output_dir / f"METRICS-u{prior_endpoint:06d}.json").is_file(),
                "a past endpoint is absent and cannot be reconstructed from a later state")
    result = None
    while completed < len(plans):
        result = run_donor_training_call(
            model, optimizer, specification, output_dir / "checkpoints", identity,
            plans, source, max_microbatches_this_call=config["max_microbatches_per_call"])
        completed = result.completed_steps
        atomic_json(output_dir / "PROGRESS.json", {
            "schema": "r02_lai_frozen_readout_progress_v1", "status": result.status,
            "completed_updates": completed, "total_updates": len(plans),
            "history_weighted_occurrence_nll": list(result.history),
            "cache_identity_sha256": cache["identity_sha256"],
            "training_identity_sha256": identity_digest(identity)})
        if completed in endpoints:
            preserve_endpoint(completed, result.history)
        require(completed <= len(plans), "readout trainer exceeded frozen horizon")
    require(completed == 256, "readout training did not reach finite horizon")
    final_history = tuple(result.history) if result is not None else tuple(restored_history)
    require(len(final_history) == 256, "completed readout history differs from finite horizon")
    receipt = {"schema": RESULT_SCHEMA, "status": "COMPLETE",
               "task_id": config["task_id"], "mode": config["mode"],
               "learning_rate": specification.learning_rate,
               "completed_updates": completed, "endpoints": config["endpoints"],
               "cache_identity_sha256": cache["identity_sha256"],
               "training_identity_sha256": identity_digest(identity),
               "history_weighted_occurrence_nll": list(final_history),
               "scientific_scope": "CONDITIONAL_FIT_DEVELOPMENT_NO_SELECT_OR_EVALUATE",
               "same_gt_occurrence_plan_repeated": True,
               "new_independent_observations_created_by_repetition": False}
    atomic_json(output_dir / "RESULT.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract")
    extraction.add_argument("--source-spec", type=Path, required=True)
    extraction.add_argument("--source-config", type=Path, required=True)
    extraction.add_argument("--checkpoint-dir", type=Path, required=True)
    extraction.add_argument("--weighted-plan-dir", type=Path, required=True)
    extraction.add_argument("--condition", choices=CONDITIONS, required=True)
    extraction.add_argument("--output-dir", type=Path, required=True)
    extraction.add_argument("--max-query-state-elements", type=int)
    extraction.add_argument("--reuse-segments-from", type=Path)
    extraction.add_argument("--prior-code-manifest", type=Path)
    training = commands.add_parser("train-head")
    training.add_argument("--cache-dir", type=Path, required=True)
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        result = extract(args.source_spec, args.source_config, args.checkpoint_dir,
                         args.weighted_plan_dir, args.condition, args.output_dir,
                         max_query_state_elements=args.max_query_state_elements,
                         reuse_segments_from=args.reuse_segments_from,
                         prior_code_manifest=args.prior_code_manifest)
    else:
        result = train_head(args.cache_dir, args.config, args.output_dir)
    print(json.dumps({"schema": result["schema"], "status": result.get("status", "COMPLETE")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
