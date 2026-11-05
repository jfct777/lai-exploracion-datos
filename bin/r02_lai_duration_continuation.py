#!/usr/bin/env python3
"""Continue the authenticated FR 1/128 LAI duration update.

The FR request and frozen computational package remain the scientific identity.
This file is a new executor and records its own hash in an amendment; it never
claims to execute the historical ``runner/measure.py``.  Technical destinations
and per-call chunk limits are deliberately outside the scientific identity.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Callable, Mapping


SCHEMA = "r02_lai_duration_continuation_v1"
AMENDMENT_SCHEMA = "r02_lai_duration_continuation_amendment_v1"
COPY_SCHEMA = "r02_lai_duration_partial_copy_v1"
SHA256_LENGTH = 64
FR_ROWS = 128
FR_OBSERVED_LABELS = 3_664_512
FR_PERSON_CHUNKS = 128
FR_START_CURSOR = 1
FR_TASK_ID = "FQ_LAI_COMPLETE_PERSON_GRADIENT"


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == SHA256_LENGTH
            and all(character in "0123456789abcdef" for character in value))


def _relative_path(value: str) -> Path:
    path = Path(value)
    _require(bool(value) and not path.is_absolute() and ".." not in path.parts,
             "checkpoint receipt path must be relative and confined")
    return path


def _read_authenticated_json(path: str | Path, expected_sha256: str,
                             label: str) -> dict[str, object]:
    _require(_valid_sha(expected_sha256), f"{label} SHA256 is invalid")
    source = Path(path)
    _require(source.is_file() and sha256_file(source) == expected_sha256,
             f"{label} SHA256 differs")
    value = json.loads(source.read_bytes())
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class PartialCheckpointReceipt:
    """Expected bytes for one durable partial-update generation."""

    current_path: str
    current_sha256: str
    manifest_path: str
    manifest_sha256: str
    payload_path: str
    payload_sha256: str

    def __post_init__(self) -> None:
        for value in (self.current_path, self.manifest_path, self.payload_path):
            _relative_path(value)
        for value in (self.current_sha256, self.manifest_sha256,
                      self.payload_sha256):
            _require(_valid_sha(value), "partial checkpoint SHA256 is invalid")
        current = _relative_path(self.current_path)
        manifest = _relative_path(self.manifest_path)
        payload = _relative_path(self.payload_path)
        _require(current.name == "CURRENT" and current.parent == manifest.parent
                 and manifest.parent == payload.parent,
                 "partial checkpoint files must share one update directory")

    def as_dict(self) -> dict[str, str]:
        return {
            "current_path": self.current_path,
            "current_sha256": self.current_sha256,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "payload_path": self.payload_path,
            "payload_sha256": self.payload_sha256,
        }


def partial_receipt_from_fr_verification(
        checkpoint_root: str | Path, verification_path: str | Path,
        verification_sha256: str, current_sha256: str) -> PartialCheckpointReceipt:
    """Translate the existing FR verification into a confined copy receipt."""
    root = Path(checkpoint_root).resolve()
    value = _read_authenticated_json(
        verification_path, verification_sha256, "FR checkpoint verification")
    files = value.get("files")
    _require(value.get("status") == "MANIFEST_PAYLOAD_HASH_VERIFIED"
             and isinstance(files, list) and len(files) == 1,
             "FR verification must identify one partial generation")
    row = files[0]
    _require(isinstance(row, dict), "FR verification row differs")
    manifest = Path(str(row.get("manifest", ""))).resolve()
    payload = Path(str(row.get("payload", ""))).resolve()
    try:
        manifest_relative = manifest.relative_to(root)
        payload_relative = payload.relative_to(root)
    except ValueError as error:
        raise ValueError("FR verification points outside its checkpoint root") from error
    current_relative = manifest_relative.parent / "CURRENT"
    return PartialCheckpointReceipt(
        str(current_relative), current_sha256, str(manifest_relative),
        str(row.get("manifest_sha256", "")), str(payload_relative),
        str(row.get("payload_sha256", "")))


def authenticate_partial_checkpoint(
        checkpoint_root: str | Path,
        receipt: PartialCheckpointReceipt) -> dict[str, object]:
    """Authenticate pointer, manifest, payload, identity, and cursor links."""
    root = Path(checkpoint_root)
    current_path = root / _relative_path(receipt.current_path)
    manifest_path = root / _relative_path(receipt.manifest_path)
    payload_path = root / _relative_path(receipt.payload_path)
    expected = ((current_path, receipt.current_sha256),
                (manifest_path, receipt.manifest_sha256),
                (payload_path, receipt.payload_sha256))
    for path, digest in expected:
        _require(path.is_file() and sha256_file(path) == digest,
                 f"partial checkpoint digest differs: {path.name}")

    pointer = json.loads(current_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    _require(isinstance(pointer, dict) and isinstance(manifest, dict)
             and pointer.get("schema") == "r02_context_partial_update_v1"
             and manifest.get("schema") == "r02_context_partial_update_v1",
             "partial checkpoint schema differs")
    _require(pointer.get("manifest") == manifest_path.name
             and pointer.get("manifest_sha256") == receipt.manifest_sha256
             and pointer.get("identity_sha256") == manifest.get("identity_sha256")
             and manifest.get("payload") == payload_path.name
             and manifest.get("payload_sha256") == receipt.payload_sha256,
             "partial checkpoint pointer/manifest links differ")
    identity = manifest.get("identity")
    progress = manifest.get("progress")
    _require(isinstance(identity, dict) and isinstance(progress, dict)
             and identity.get("schema") == "r02_lai_duration_partial_update_v1"
             and progress.get("phase") == "DURATION_PERSON_CHUNKS"
             and pointer.get("phase") == progress.get("phase")
             and pointer.get("next_person_batch_index") ==
                progress.get("next_person_batch_index")
             and pointer.get("next_tile_index") == progress.get("next_tile_index"),
             "partial duration identity or cursor differs")
    return manifest


def copy_verified_partial_checkpoint(
        source_checkpoint_root: str | Path,
        destination_checkpoint_root: str | Path,
        receipt: PartialCheckpointReceipt) -> dict[str, object]:
    """Copy one authenticated generation without modifying its source."""
    source = Path(source_checkpoint_root).resolve()
    destination = Path(destination_checkpoint_root).resolve()
    _require(source != destination and source.is_dir(),
             "source and destination checkpoint roots must differ")
    _require(not destination.exists() and destination.parent.is_dir(),
             "destination checkpoint root must be fresh with an existing parent")
    source_manifest = authenticate_partial_checkpoint(source, receipt)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.copy-", dir=destination.parent))
    try:
        for relative in (receipt.current_path, receipt.manifest_path,
                         receipt.payload_path):
            relative_path = _relative_path(relative)
            target = staging / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative_path, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
        authenticate_partial_checkpoint(staging, receipt)
        os.replace(staging, destination)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    destination_manifest = authenticate_partial_checkpoint(destination, receipt)
    # Re-read after the copy to prove this function did not mutate the source.
    authenticate_partial_checkpoint(source, receipt)
    _require(destination_manifest == source_manifest,
             "copied partial checkpoint content differs")
    return {
        "schema": COPY_SCHEMA,
        "status": "VERIFIED_PARTIAL_GENERATION_COPIED_SOURCE_UNCHANGED",
        "source_checkpoint_root": str(source),
        "destination_checkpoint_root": str(destination),
        "files": receipt.as_dict(),
        "training_identity_sha256":
            source_manifest["identity"]["training_identity_sha256"],
        "next_person_chunk": source_manifest["progress"]["next_person_chunk"],
    }


def verify_frozen_package(package_root: str | Path,
                          manifest_sha256: str) -> dict[str, str]:
    """Verify every member of the original FR package before importing it."""
    root = Path(package_root).resolve()
    manifest = _read_authenticated_json(
        root / "MANIFEST.json", manifest_sha256, "frozen package manifest")
    _require(manifest and all(isinstance(name, str) and _valid_sha(digest)
                              for name, digest in manifest.items()),
             "frozen package manifest inventory differs")
    for name, digest in manifest.items():
        relative = _relative_path(name)
        _require(sha256_file(root / relative) == digest,
                 f"frozen package member differs: {name}")
    required = {
        "bin/r02_context_checkpoint.py", "bin/r02_context_train.py",
        "bin/r02_lai_duration_fit.py", "bin/r02_lai_duration_input.py",
        "bin/r02_lai_duration_model.py", "bin/r02_lai_duration_neural.py",
        "bin/r02_lai_duration_training.py", "runner/measure.py",
    }
    _require(required <= set(manifest), "frozen FR package lacks required computation")
    return {str(name): str(digest) for name, digest in manifest.items()}


def _import_frozen_modules(package_root: Path) -> dict[str, ModuleType]:
    names = (
        "r02_context_checkpoint", "r02_context_train", "r02_lai_duration_input",
        "r02_lai_duration_model", "r02_lai_duration_neural",
        "r02_lai_duration_fit", "r02_lai_duration_training",
    )
    frozen_bin = (package_root / "bin").resolve()
    for name in names:
        loaded = sys.modules.get(name)
        if loaded is not None:
            source = Path(str(getattr(loaded, "__file__", ""))).resolve()
            _require(source.parent == frozen_bin,
                     f"effective module was imported outside frozen FR package: {name}")
    sys.path.insert(0, str(frozen_bin))
    return {name: importlib.import_module(name) for name in names}


def _verify_loaded_frozen_modules(package_root: Path,
                                  manifest: Mapping[str, str]) -> dict[str, str]:
    frozen_bin = (package_root / "bin").resolve()
    effective = {}
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("r02_") or name == Path(__file__).stem:
            continue
        source_value = getattr(module, "__file__", None)
        if not source_value:
            continue
        source = Path(source_value).resolve()
        _require(source.parent == frozen_bin,
                 f"effective computational module is not frozen: {name}")
        key = f"bin/{source.name}"
        _require(key in manifest and sha256_file(source) == manifest[key],
                 f"effective computational module hash differs: {name}")
        effective[key] = manifest[key]
    _require(effective, "no effective frozen computational modules were recorded")
    return effective


@dataclass
class FrozenFRRuntime:
    request: dict[str, object]
    request_sha256: str
    package_root: Path
    package_manifest_sha256: str
    original_code_sha256: dict[str, str]
    effective_code_sha256: dict[str, str]
    state: object
    optimizer: object
    optimizer_specification: object
    identity: dict[str, object]
    plan: object
    provider: Callable
    training_module: ModuleType
    source_model_receipt: dict[str, object]
    gate_receipt: dict[str, object]
    source_identities: dict[str, object]


def build_frozen_fr_runtime(
        request_path: str | Path, request_sha256: str,
        package_root: str | Path, package_manifest_sha256: str,
        expected_training_identity_sha256: str) -> FrozenFRRuntime:
    """Reconstruct FR state/provider/identity using only frozen computation."""
    _require(_valid_sha(expected_training_identity_sha256),
             "expected FR training identity SHA256 is required")
    package = Path(package_root).resolve()
    package_manifest = verify_frozen_package(package, package_manifest_sha256)
    request = _read_authenticated_json(request_path, request_sha256, "original FR request")
    modules = _import_frozen_modules(package)
    checkpoint = modules["r02_context_checkpoint"]
    training = modules["r02_lai_duration_training"]
    neural = modules["r02_lai_duration_neural"]
    fit = modules["r02_lai_duration_fit"]
    model_module = modules["r02_lai_duration_model"]
    input_module = modules["r02_lai_duration_input"]

    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(request["seed"])
    np.random.seed(request["seed"])
    contract = _read_authenticated_json(
        request["contract"]["path"], request["contract"]["sha256"], "FR contract")
    resolved = _read_authenticated_json(
        request["resolved"]["path"], request["resolved"]["sha256"], "FR resolved config")
    helper_spec = request["helper"]
    _require(sha256_file(helper_spec["path"]) == helper_spec["sha256"],
             "FR model helper SHA256 differs")
    # The helper is authenticated separately by the original request.  Its
    # neutral module name keeps the r02_* inventory reserved for package files.
    loader = importlib.util.spec_from_file_location(
        "fixed_checkpoint_source", helper_spec["path"])
    _require(loader is not None and loader.loader is not None,
             "FR model helper could not be loaded")
    helper = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helper)

    source_checkpoint = Path(request["checkpoint"]["root"])
    _require(sha256_file(source_checkpoint / "CURRENT") ==
             request["checkpoint"]["pointer_sha256"]
             and sha256_file(source_checkpoint / request["checkpoint"]["manifest"]) ==
             request["checkpoint"]["manifest_sha256"],
             "FR pretrained checkpoint source differs")
    original_manifest = json.loads(
        (source_checkpoint / request["checkpoint"]["manifest"]).read_bytes())
    old_identity = original_manifest["identity"]
    _require(old_identity["contract_sha256"] == request["contract"]["sha256"]
             and old_identity["resolved_sha256"] == request["resolved"]["sha256"],
             "FR pretrained model contract/resolved identity differs")
    model = modules["r02_context_train"].TaskModel(resolved, contract)
    source_model = helper.load_model_only(
        source_checkpoint, old_identity, checkpoint, model)
    gate = neural.freeze_legacy_probability_mixture_head(model)
    duration = neural.build_fixed_d1_parameters(
        path=request["duration_initial"]["path"],
        expected_sha256=request["duration_initial"]["sha256"],
        floating_dtype=torch.float64)
    state = training.DurationTrainingState(model, duration)
    inventory = input_module.load_duration_case_inventory(
        request["contract"]["path"], request["contract"]["sha256"],
        request["case_id"])
    declaration = model_module.SequenceDeclaration(
        "chromosome", inventory.case_id, inventory.chromosome,
        inventory.query_positions_cm, model_module.COMPLETE_SCIENTIFIC_UNIT,
        duration.specification.prior_provenance.core_declaration())
    fit_rows = fit.build_fit_rows(
        [inventory], people_per_chunk=1, max_optimizer_updates=1,
        design_sha256=request_sha256)
    reconstruction_source = _read_authenticated_json(
        request["reconstruction"]["path"], request["reconstruction"]["sha256"],
        "FR reconstruction limits")
    reconstruction_keys = (
        "max_factorized_source_bytes", "max_reconstructed_join_bytes",
        "max_source_plus_reconstructed_bytes")
    reconstruction = {key: reconstruction_source[key] for key in reconstruction_keys}
    case = neural.DurationNeuralCase(
        inventory, declaration, request["contract"]["path"],
        request["contract"]["sha256"], request["resolved"]["path"],
        request["resolved"]["sha256"], 1, contract["limits"]["max_targets"] * 6,
        len(inventory.query_global_indices) * 6, reconstruction,
        neural.BOUNDED_ONE_CASE_MEMORY_CACHE, request["limits"]["cache_bytes"],
        neural.RAW_EMISSION_SEMANTICS,
        cache_directory=request["regional_cache"]["directory"],
        cache_load_workers=request["regional_cache"]["workers"])
    provider = neural.DurationNeuralChunkProvider([case], fit_rows)
    _require(provider.plan.global_rows == FR_ROWS
             and provider.plan.global_observed_labels == FR_OBSERVED_LABELS
             and len(provider.plan.chunk_identity_sha256) == FR_PERSON_CHUNKS,
             "FR plan rows, labels, or chunks differ")
    optimizer_settings = dict(request["optimizer"])
    optimizer_settings["betas"] = tuple(optimizer_settings["betas"])
    optimizer_specification = training.OptimizerSpecification(**optimizer_settings)
    optimizer = training.build_duration_optimizer(state, optimizer_specification)
    original_code = {name: digest for name, digest in package_manifest.items()
                     if name.startswith(("bin/", "runner/"))}
    identity = training.duration_training_identity(
        task_id=FR_TASK_ID,
        data_declaration_sha256=checkpoint.identity_digest(fit_rows),
        data_scope=fit.FIT_SCOPE, data_source=request["contract"]["path"],
        randomness_declaration_sha256=checkpoint.identity_digest(
            {"seed": request["seed"]}),
        code_sha256=original_code, state=state,
        optimizer=optimizer_specification, plan=provider.plan, fit_rows=fit_rows)
    identity["pretrained_model_source"] = source_model
    identity["request_sha256"] = request_sha256
    identity["package_manifest_sha256"] = package_manifest_sha256
    _require(checkpoint.identity_digest(identity) == expected_training_identity_sha256,
             "reconstructed FR training identity differs")
    _require(not optimizer.state and not any(
        parameter.requires_grad for parameter in duration.parameters()),
        "FR Adam or fixed D1 initial state differs")
    effective = _verify_loaded_frozen_modules(package, package_manifest)
    sources = {
        "request": {"path": str(Path(request_path).resolve()),
                    "sha256": request_sha256},
        "package_manifest": {"path": str(package / "MANIFEST.json"),
                             "sha256": package_manifest_sha256},
        "contract": dict(request["contract"]),
        "resolved": dict(request["resolved"]),
        "duration_initial": dict(request["duration_initial"]),
        "reconstruction": dict(request["reconstruction"]),
        "model_helper": dict(helper_spec),
        "pretrained_checkpoint": dict(request["checkpoint"]),
    }
    return FrozenFRRuntime(
        request, request_sha256, package, package_manifest_sha256,
        original_code, effective, state, optimizer, optimizer_specification,
        identity, provider.plan, provider, training, source_model, gate, sources)


PreChunkCallback = Callable[[object, int, object], None]


def continue_verified_update(
        *, state: object, optimizer: object, optimizer_specification: object,
        checkpoint_root: str | Path, training_identity: dict[str, object],
        plan: object, provider: Callable, checkpoint_receipt: PartialCheckpointReceipt,
        expected_start_cursor: int, expected_rows: int,
        expected_observed_labels: int, max_chunks_this_call: int,
        training_module: ModuleType | None = None,
        pre_chunk_callback: PreChunkCallback | None = None) -> object:
    """Authenticate a partial cursor and advance the same single update."""
    _require(type(max_chunks_this_call) is int and max_chunks_this_call > 0,
             "max_chunks_this_call is required external execution configuration")
    _require(type(expected_start_cursor) is int and expected_start_cursor > 0,
             "positive expected starting cursor required")
    manifest = authenticate_partial_checkpoint(checkpoint_root, checkpoint_receipt)
    progress = manifest["progress"]
    partial_identity = manifest["identity"]
    if training_module is None:
        training_module = importlib.import_module("r02_lai_duration_training")
    checkpoint_module = importlib.import_module("r02_context_checkpoint")
    identity_sha256 = checkpoint_module.identity_digest(training_identity)
    _require(partial_identity.get("training_identity_sha256") == identity_sha256
             and partial_identity.get("update_index") == 0,
             "partial checkpoint belongs to a different training identity or update")
    _require(progress.get("next_person_chunk") == expected_start_cursor
             and progress.get("next_person_batch_index") == expected_start_cursor,
             "partial checkpoint starting cursor differs")
    _require(plan.global_rows == expected_rows
             and plan.global_observed_labels == expected_observed_labels
             and len(plan.chunk_identity_sha256) == progress.get("total_person_chunks"),
             "continuation rows, denominator, or chunk count differs")

    def observed_provider(live_state: object, index: int) -> object:
        chunk = provider(live_state, index)
        if pre_chunk_callback is not None:
            # The callback runs before the existing trainer evaluates the loss.
            # It must detach anything it retains and must not mutate the chunk.
            pre_chunk_callback(live_state, index, chunk)
        return chunk

    result = training_module.run_duration_training_call(
        state, optimizer, optimizer_specification, Path(checkpoint_root),
        training_identity, plan, observed_provider,
        max_chunks_this_call=max_chunks_this_call)
    _require(result.completed_updates in (0, 1)
             and not (result.status == "COMPLETE" and result.completed_updates != 1),
             "continuation exceeded its single declared optimizer update")
    return result


def _atomic_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    _require(destination.parent.is_dir(), "amendment parent directory is absent")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(_canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, destination)


def amendment(runtime: FrozenFRRuntime, copy_receipt: Mapping[str, object],
              max_chunks_this_call: int, result: object | None = None) -> dict[str, object]:
    """Describe the new executor separately from the frozen FR computation."""
    value = {
        "schema": AMENDMENT_SCHEMA,
        "status": "PREPARED" if result is None else "CALL_FINISHED",
        "scientific_training_identity_sha256":
            runtime.training_module.identity_digest(runtime.identity)
            if hasattr(runtime.training_module, "identity_digest") else
            importlib.import_module("r02_context_checkpoint").identity_digest(runtime.identity),
        "original_request_sha256": runtime.request_sha256,
        "original_package_manifest_sha256": runtime.package_manifest_sha256,
        "original_code_sha256": runtime.original_code_sha256,
        "effective_frozen_module_sha256": runtime.effective_code_sha256,
        "new_executor": {"path": str(Path(__file__).resolve()),
                         "sha256": sha256_file(__file__)},
        "historical_runner": {
            "path_in_frozen_package": "runner/measure.py",
            "sha256": runtime.original_code_sha256["runner/measure.py"],
            "executed_by_continuation": False,
            "role": "retained_only_in_original_training_identity"},
        "source_identities": runtime.source_identities,
        "partial_copy": dict(copy_receipt),
        "external_execution_configuration": {
            "max_chunks_this_call": max_chunks_this_call,
            "destination_checkpoint_root":
                copy_receipt["destination_checkpoint_root"],
            "excluded_from_scientific_identity": True},
        "limits": [
            "FIT development with shared donors; no SELECT or EVALUATE",
            "One declared Adam update maximum; no biological-superiority claim",
            "Optional pre-update chunk callback does not choose or change inputs"],
    }
    if result is not None:
        value["result"] = {
            "status": result.status,
            "completed_updates": result.completed_updates,
            "next_person_chunk": result.next_person_chunk,
            "history": list(result.history),
            "budgets": result.budgets,
        }
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--expected-training-identity-sha256", required=True)
    parser.add_argument("--source-checkpoint-root", required=True)
    parser.add_argument("--source-verification", required=True)
    parser.add_argument("--source-verification-sha256", required=True)
    parser.add_argument("--source-current-sha256", required=True)
    parser.add_argument("--destination-checkpoint-root", required=True)
    parser.add_argument("--max-chunks-this-call", required=True, type=int)
    parser.add_argument("--amendment", required=True)
    args = parser.parse_args(argv)

    runtime = build_frozen_fr_runtime(
        args.request, args.request_sha256, args.package,
        args.package_manifest_sha256, args.expected_training_identity_sha256)
    source_receipt = partial_receipt_from_fr_verification(
        args.source_checkpoint_root, args.source_verification,
        args.source_verification_sha256, args.source_current_sha256)
    copy_receipt = copy_verified_partial_checkpoint(
        args.source_checkpoint_root, args.destination_checkpoint_root, source_receipt)
    prepared = amendment(runtime, copy_receipt, args.max_chunks_this_call)
    _atomic_json(args.amendment, prepared)
    result = continue_verified_update(
        state=runtime.state, optimizer=runtime.optimizer,
        optimizer_specification=runtime.optimizer_specification,
        checkpoint_root=args.destination_checkpoint_root,
        training_identity=runtime.identity, plan=runtime.plan,
        provider=runtime.provider, checkpoint_receipt=source_receipt,
        expected_start_cursor=FR_START_CURSOR, expected_rows=FR_ROWS,
        expected_observed_labels=FR_OBSERVED_LABELS,
        max_chunks_this_call=args.max_chunks_this_call,
        training_module=runtime.training_module)
    final = amendment(runtime, copy_receipt, args.max_chunks_this_call, result)
    _atomic_json(args.amendment, final)
    print(_canonical(final).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
