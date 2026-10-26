"""One fragmented Adam update for the masked-genotype objective.

Owned loci contribute targets exactly once.  Halo loci provide context only.
The complete-update common and rare denominators are known before backward;
one Adam step occurs only after every declared fragment gradient is durable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import zipfile

import numpy as np
import torch
from torch import Tensor

from r02_context_checkpoint import (_atomic_bytes, _atomic_npz, _canonical, _decode,
    _encode, _fsync_directory, _require_finite_arrays, _sha_bytes, _sha_file,
    identity_digest, MAX_UNCOMPRESSED_BYTES)
from r02_context_update_checkpoint import (capture_parameter_gradients,
                                             optimizer_start_digest, _state_digest)
from r02_masked_genotype_loss import (masked_genotype_loss_terms,
                                      scaled_fragment_loss)
from r02_masked_genotype_model import (INFORMATION_CONDITIONS, MaskedGenotypeModel,
                                       MaskedTensorChannel)


SCHEMA = "r02_masked_genotype_partial_update_v2"


def masked_update_code_hashes():
    """Share the complete numerical dependency identity with outer campaigns."""
    return {name: _sha_file(Path(__file__).parent/name) for name in (
        'r02_masked_genotype_update.py', 'r02_masked_genotype_model.py',
        'r02_masked_genotype_loss.py', 'r02_masked_genotypes.py',
        'r02_masked_store.py', 'r02_masked_training_batch.py',
        'genomic_context_runtime.py', 'r02_local_attention_blocks.py',
        'r02_context_checkpoint.py', 'r02_context_update_checkpoint.py',
        'r02_adam_fp64_state.py')}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class MaskedTargets:
    dosage: Tensor
    target_scored: Tensor
    owned: Tensor

    def check(self, channel: MaskedTensorChannel) -> None:
        shape = channel.dosage.shape
        _require(self.dosage.shape == shape and self.dosage.dtype == torch.int64,
                 "target dosage must be int64 person by locus")
        _require(self.target_scored.shape == shape and self.target_scored.dtype == torch.bool,
                 "target_scored must be boolean person by locus")
        _require(self.owned.shape == (shape[1],) and self.owned.dtype == torch.bool,
                 "owned must be one boolean per locus")
        _require(all(value.device == channel.dosage.device
                     for value in (self.dosage, self.target_scored, self.owned)),
                 "targets and channel must share a device")
        score = self.score_mask
        _require(bool((channel.locus_valid[self.owned]).all()),
                 "owned loci must be structurally valid")
        _require(bool(((self.dosage[score] >= 0) & (self.dosage[score] <= 2)).all()),
                 "scored target dosage must be 0, 1 or 2")
        _require(not bool((score & channel.genotype_visible).any()),
                 "a scored target genotype remains visible to the model")

    @property
    def score_mask(self) -> Tensor:
        return self.target_scored & self.owned.unsqueeze(0)

    @classmethod
    def from_masked_channel(cls, masked, *, owned: np.ndarray,
                            device: torch.device | str) -> "MaskedTargets":
        targets = masked.targets
        return cls(torch.as_tensor(np.asarray(targets["dosage"]), device=device,
                                   dtype=torch.int64),
                   torch.as_tensor(np.asarray(targets["target_scored"]), device=device,
                                   dtype=torch.bool),
                   torch.as_tensor(np.asarray(owned), device=device, dtype=torch.bool))


@dataclass(frozen=True)
class MaskedFragment:
    common: MaskedTensorChannel
    rare: MaskedTensorChannel
    common_targets: MaskedTargets
    rare_targets: MaskedTargets
    lineage: str

    def check(self, max_loci: int) -> None:
        self.common.check(max_loci); self.rare.check(max_loci)
        self.common_targets.check(self.common); self.rare_targets.check(self.rare)
        _require(self.common.person_ids == self.rare.person_ids
                 and self.common.plan_identity == self.rare.plan_identity,
                 "fragment channels use different people or masks")
        _require(isinstance(self.lineage, str) and self.lineage,
                 "fragment lineage required")


@dataclass(frozen=True)
class MaskedUpdateResult:
    loss: float
    common_cross_entropy_sum: float
    rare_cross_entropy_sum: float
    common_objective_count: int
    rare_objective_count: int
    fragments: int


def _owned_keys(fragment: MaskedFragment, channel_name: str):
    channel = getattr(fragment, channel_name)
    targets = getattr(fragment, channel_name + "_targets")
    score = targets.score_mask.detach().cpu()
    chromosome = channel.chromosome.detach().cpu()
    position_bp = channel.position_bp.detach().cpu()
    ref = channel.ref.detach().cpu(); alt = channel.alt.detach().cpu()
    counted = channel.counted_allele.detach().cpu()
    indices = torch.where(score)
    for person, locus in zip(indices[0].tolist(), indices[1].tolist()):
        yield (channel_name, channel.person_ids[person], int(chromosome[locus]),
               int(position_bp[locus]), int(ref[locus]), int(alt[locus]),
               int(counted[locus]))


def validate_owned_targets(fragments: list[MaskedFragment], max_loci: int) -> tuple[int, int]:
    """Reject duplicated scored person/locus targets across overlapping fragments."""
    _require(isinstance(fragments, list) and fragments, "nonempty fragment list required")
    _require(len({item.lineage for item in fragments}) == len(fragments),
             "fragment lineage must be unique within one update")
    seen = set(); counts = {"common": 0, "rare": 0}
    for fragment in fragments:
        fragment.check(max_loci)
        for channel in ("common", "rare"):
            for key in _owned_keys(fragment, channel):
                _require(key not in seen, "owned target appears in more than one fragment")
                seen.add(key); counts[channel] += 1
    _require(counts["common"] > 0 and counts["rare"] > 0,
             "complete update requires positive common and rare objective counts")
    return counts["common"], counts["rare"]


def masked_update_identity(model: MaskedGenotypeModel, optimizer, fragments: list[MaskedFragment],
                           *, information_condition: str) -> dict:
    _require(information_condition in INFORMATION_CONDITIONS,
             "unknown rare-information condition")
    common_count, rare_count = validate_owned_targets(
        fragments, model.config.max_loci_per_channel)
    return {"schema": SCHEMA,
            "model_config": asdict(model.config),
            "fragment_content_sha256": [_state_digest({
                'common': vars(item.common), 'rare': vars(item.rare),
                'common_targets': vars(item.common_targets),
                'rare_targets': vars(item.rare_targets)}) for item in fragments],
            "code_sha256": masked_update_code_hashes(),
            "parameter_schema": {name: [list(value.shape), str(value.dtype)]
                                 for name, value in model.state_dict().items()},
            "optimizer_algorithm": type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
            "optimizer_options_sha256": _state_digest([
                {k: v for k, v in group.items() if k != 'params'}
                for group in optimizer.param_groups]),
            "information_condition": information_condition,
            "fragment_lineage": [item.lineage for item in fragments],
            "channel_lineage": [[item.common.lineage, item.rare.lineage] for item in fragments],
            "mask_plan_identity": [item.common.plan_identity for item in fragments],
            "person_ids": [list(item.common.person_ids) for item in fragments],
            "global_common_count": common_count,
            "global_rare_count": rare_count,
            "torch_version": torch.__version__,
            "parameter_device": str(next(model.parameters()).device),
            "model_optimizer_start_sha256": optimizer_start_digest(model, optimizer)}


def masked_fragment_record(fragment: MaskedFragment, max_loci: int):
    """Return a compact authenticated record and owned keys for one fragment."""
    fragment.check(max_loci)
    keys = tuple(key for channel in ("common", "rare")
                 for key in _owned_keys(fragment, channel))
    common_count = int(fragment.common_targets.score_mask.sum().item())
    rare_count = int(fragment.rare_targets.score_mask.sum().item())
    record = {"content_sha256": _state_digest({
                  "common": vars(fragment.common), "rare": vars(fragment.rare),
                  "common_targets": vars(fragment.common_targets),
                  "rare_targets": vars(fragment.rare_targets)}),
              "owned_targets_sha256": _state_digest(keys),
              "fragment_lineage": fragment.lineage,
              "channel_lineage": [fragment.common.lineage, fragment.rare.lineage],
              "mask_plan_identity": fragment.common.plan_identity,
              "person_ids": list(fragment.common.person_ids),
              "common_count": common_count, "rare_count": rare_count}
    return record, keys


def masked_streaming_update_identity(model, optimizer, records, *, information_condition):
    """Build the update identity from preflight records without retaining tensors."""
    _require(information_condition in INFORMATION_CONDITIONS,
             "unknown rare-information condition")
    values = tuple(records)
    _require(values and len({value["fragment_lineage"] for value in values}) == len(values),
             "unique nonempty streaming fragment records required")
    common_count = sum(value["common_count"] for value in values)
    rare_count = sum(value["rare_count"] for value in values)
    _require(common_count > 0 and rare_count > 0,
             "complete streaming update requires positive channel counts")
    return {"schema": SCHEMA, "model_config": asdict(model.config),
            "fragment_content_sha256": [value["content_sha256"] for value in values],
            "fragment_owned_targets_sha256": [value["owned_targets_sha256"] for value in values],
            "code_sha256": masked_update_code_hashes(),
            "parameter_schema": {name: [list(value.shape), str(value.dtype)]
                                 for name, value in model.state_dict().items()},
            "optimizer_algorithm": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
            "optimizer_options_sha256": _state_digest([
                {k: v for k, v in group.items() if k != "params"}
                for group in optimizer.param_groups]),
            "information_condition": information_condition,
            "fragment_lineage": [value["fragment_lineage"] for value in values],
            "channel_lineage": [value["channel_lineage"] for value in values],
            "mask_plan_identity": [value["mask_plan_identity"] for value in values],
            "person_ids": [value["person_ids"] for value in values],
            "global_common_count": common_count, "global_rare_count": rare_count,
            "torch_version": torch.__version__,
            "parameter_device": str(next(model.parameters()).device),
            "model_optimizer_start_sha256": optimizer_start_digest(model, optimizer)}


def _restore_parameter_gradients_on_model_device(model, gradients) -> None:
    expected = dict(model.named_parameters())
    _require(isinstance(gradients, dict) and set(gradients) == set(expected),
             "partial gradient parameter inventory differs")
    for name, parameter in expected.items():
        value = gradients[name]
        if value is None:
            parameter.grad = None
            continue
        _require(torch.is_tensor(value) and value.shape == parameter.shape
                 and value.dtype == parameter.dtype and bool(torch.isfinite(value).all()),
                 f"partial gradient shape/dtype/value differs: {name}")
        parameter.grad = value.detach().to(parameter.device).clone()


class MaskedUpdateStore:
    """Atomic data-only checkpoints for gradients between fragment backwards."""

    def __init__(self, root: str | Path, identity: dict):
        from r02_context_checkpoint import CheckpointStore
        _require(identity.get("schema") == SCHEMA, "masked-update identity required")
        self._lock = CheckpointStore(root, identity)
        self.root = self._lock.root
        self.identity = identity
        self.identity_sha256 = identity_digest(identity)
        self._generation = 0

    def close(self) -> None:
        self._lock.close()

    def save(self, model, optimizer, *, next_fragment: int,
             common_sum: float, rare_sum: float,
             common_count: int, rare_count: int,
             optimizer_step_completed: bool = False) -> dict:
        _require(type(next_fragment) is int and 0 <= next_fragment <=
                 len(self.identity["fragment_lineage"]), "invalid fragment cursor")
        _require(all(np.isfinite(value) and value >= 0 for value in (common_sum, rare_sum)),
                 "finite nonnegative accumulated loss sums required")
        _require(type(common_count) is int and type(rare_count) is int
                 and 0 <= common_count <= self.identity["global_common_count"]
                 and 0 <= rare_count <= self.identity["global_rare_count"],
                 "invalid accumulated objective counts")
        _require(type(optimizer_step_completed) is bool,
                 "optimizer_step_completed must be boolean")
        if optimizer_step_completed:
            _require(next_fragment == len(self.identity["fragment_lineage"])
                     and common_count == self.identity["global_common_count"]
                     and rare_count == self.identity["global_rare_count"],
                     "completed Adam state requires every declared fragment and objective")
        arrays = {}
        state = {"model": _encode(model.state_dict(), arrays, "model"),
                 "optimizer": _encode(optimizer.state_dict(), arrays, "optimizer"),
                 "gradients": _encode(capture_parameter_gradients(model), arrays, "gradients"),
                 "python_rng": _encode(random.getstate(), arrays, "python_rng"),
                 "numpy_rng": _encode(np.random.get_state(), arrays, "numpy_rng"),
                 "torch_rng": _encode(torch.get_rng_state(), arrays, "torch_rng"),
                 "cuda_rng": _encode(torch.cuda.get_rng_state_all()
                                     if torch.cuda.is_available() else [], arrays, "cuda_rng")}
        _require_finite_arrays(arrays)
        self._generation += 1
        generation = f"masked-update-{self._generation:012d}"
        payload = self.root / f"{generation}.npz"
        manifest_path = self.root / f"{generation}.json"
        _atomic_npz(payload, arrays)
        progress = {"next_fragment": next_fragment, "common_sum": float(common_sum),
                    "rare_sum": float(rare_sum), "common_count": common_count,
                    "rare_count": rare_count,
                    "optimizer_step_completed": optimizer_step_completed}
        manifest = {"schema": SCHEMA, "generation": generation,
                    "identity": self.identity, "identity_sha256": self.identity_sha256,
                    "progress": progress, "payload": payload.name,
                    "payload_sha256": _sha_file(payload), "array_keys": sorted(arrays),
                    "state": state}
        manifest_bytes = _canonical(manifest)
        _atomic_bytes(manifest_path, manifest_bytes)
        _atomic_bytes(self.root / "CURRENT", _canonical({
            "schema": SCHEMA, "manifest": manifest_path.name,
            "manifest_sha256": _sha_bytes(manifest_bytes),
            "identity_sha256": self.identity_sha256}))
        for path in self.root.glob("masked-update-*"):
            if path.stem != generation:
                path.unlink()
        _fsync_directory(self.root)
        return manifest

    def restore(self, model, optimizer):
        current = self.root / "CURRENT"
        if not current.exists():
            return None
        try:
            pointer = json.loads(current.read_bytes())
            _require(pointer.get("schema") == SCHEMA
                     and pointer.get("identity_sha256") == self.identity_sha256,
                     "masked-update identity is incompatible")
            name = pointer.get("manifest")
            _require(isinstance(name, str) and Path(name).name == name,
                     "invalid masked-update manifest name")
            manifest_bytes = (self.root / name).read_bytes()
            _require(_sha_bytes(manifest_bytes) == pointer.get("manifest_sha256"),
                     "masked-update manifest integrity failure")
            manifest = json.loads(manifest_bytes)
            _require(manifest.get("schema") == SCHEMA and manifest.get("identity") == self.identity,
                     "masked-update manifest identity is incompatible")
            payload_name = manifest.get("payload")
            _require(isinstance(payload_name, str) and Path(payload_name).name == payload_name,
                     "invalid masked-update payload name")
            payload = self.root / payload_name
            _require(_sha_file(payload) == manifest.get("payload_sha256"),
                     "masked-update payload integrity failure")
            with zipfile.ZipFile(payload) as archive:
                _require(sum(item.file_size for item in archive.infolist()) <= MAX_UNCOMPRESSED_BYTES,
                         "masked-update payload exceeds safety bound")
            with np.load(payload, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            _require(sorted(arrays) == manifest.get("array_keys"),
                     "masked-update array inventory differs")
            used = set(); state = manifest["state"]
            model_state = _decode(state["model"], arrays, used)
            optimizer_state = _decode(state["optimizer"], arrays, used)
            gradients = _decode(state["gradients"], arrays, used)
            python_rng = _decode(state["python_rng"], arrays, used)
            numpy_rng = _decode(state["numpy_rng"], arrays, used)
            torch_rng = _decode(state["torch_rng"], arrays, used)
            cuda_rng = _decode(state["cuda_rng"], arrays, used)
            _require(used == set(arrays), "unreferenced masked-update arrays")
            model.load_state_dict(model_state, strict=True)
            optimizer.load_state_dict(optimizer_state)
            _restore_parameter_gradients_on_model_device(model, gradients)
            random.setstate(python_rng); np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng:
                _require(torch.cuda.is_available() and len(cuda_rng) == torch.cuda.device_count(),
                         "CUDA RNG device inventory differs")
                torch.cuda.set_rng_state_all(cuda_rng)
            self._generation = int(manifest["generation"].rsplit("-", 1)[1])
            return manifest["progress"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
                zipfile.BadZipFile) as error:
            raise ValueError(f"invalid masked-update checkpoint: {error}") from error

    def clear(self) -> None:
        for path in tuple(self.root.iterdir()):
            if path.name == "CURRENT" or path.name.startswith("masked-update-"):
                path.unlink()
        _fsync_directory(self.root)


def run_masked_update(model: MaskedGenotypeModel, optimizer, fragments: list[MaskedFragment],
                      *, information_condition: str,
                      checkpoint: MaskedUpdateStore | None = None) -> MaskedUpdateResult:
    """Accumulate exact complete-update loss gradients and call Adam once."""
    _require(information_condition in INFORMATION_CONDITIONS,
             "unknown rare-information condition")
    global_common, global_rare = validate_owned_targets(
        fragments, model.config.max_loci_per_channel)
    start = 0; common_sum = rare_sum = 0.; common_count = rare_count = 0
    if checkpoint is not None:
        current = masked_update_identity(model, optimizer, fragments,
                                          information_condition=information_condition)
        expected = dict(checkpoint.identity)
        # Restoring deliberately replaces current weights/Adam with the saved
        # state. Everything defining the problem must still match. For a new
        # checkpoint even the initial numerical state must match exactly.
        if (checkpoint.root / 'CURRENT').exists():
            current.pop('model_optimizer_start_sha256')
            expected.pop('model_optimizer_start_sha256', None)
        _require(current == expected,
                 'checkpoint problem, inputs, targets, ownership, code or optimizer differs')
    restored = checkpoint.restore(model, optimizer) if checkpoint is not None else None
    if restored is None:
        optimizer.zero_grad(set_to_none=True)
    else:
        start = restored["next_fragment"]
        common_sum = restored["common_sum"]; rare_sum = restored["rare_sum"]
        common_count = restored["common_count"]; rare_count = restored["rare_count"]
        if restored.get("optimizer_step_completed"):
            _require((common_count, rare_count) == (global_common, global_rare),
                     "completed checkpoint objective counts differ")
            return MaskedUpdateResult(
                .5 * common_sum / global_common + .5 * rare_sum / global_rare,
                common_sum, rare_sum, global_common, global_rare, len(fragments))
    model.train()
    for index in range(start, len(fragments)):
        fragment = fragments[index]
        output = model(fragment.common, fragment.rare,
                       information_condition=information_condition)
        terms = masked_genotype_loss_terms(
            output.common_logits, fragment.common_targets.dosage,
            fragment.common_targets.score_mask,
            output.rare_logits, fragment.rare_targets.dosage,
            fragment.rare_targets.score_mask)
        value = scaled_fragment_loss(terms, global_common_count=global_common,
                                     global_rare_count=global_rare)
        _require(bool(torch.isfinite(value)), "nonfinite masked-genotype fragment loss")
        value.backward()
        _require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                     for parameter in model.parameters()), "nonfinite masked-genotype gradient")
        common_sum += float(terms.common.cross_entropy_sum.detach())
        rare_sum += float(terms.rare.cross_entropy_sum.detach())
        common_count += terms.common.objective_count
        rare_count += terms.rare.objective_count
        if checkpoint is not None:
            checkpoint.save(model, optimizer, next_fragment=index + 1,
                            common_sum=common_sum, rare_sum=rare_sum,
                            common_count=common_count, rare_count=rare_count)
    _require((common_count, rare_count) == (global_common, global_rare),
             "fragment cursor/counts do not cover the declared complete update")
    optimizer.step()
    loss = .5 * common_sum / global_common + .5 * rare_sum / global_rare
    if checkpoint is not None:
        # Promote the post-Adam weights/state before any partial generation can
        # be removed. A restart sees this flag and never applies Adam twice.
        checkpoint.save(model, optimizer, next_fragment=len(fragments),
                        common_sum=common_sum, rare_sum=rare_sum,
                        common_count=common_count, rare_count=rare_count,
                        optimizer_step_completed=True)
    return MaskedUpdateResult(loss, common_sum, rare_sum, global_common, global_rare,
                              len(fragments))


def run_masked_update_streaming(model, optimizer, load_fragment, records,
                                *, information_condition: str,
                                checkpoint: MaskedUpdateStore | None = None):
    """Load, authenticate, backpropagate and release one fragment at a time."""
    records = tuple(records)
    identity = masked_streaming_update_identity(
        model, optimizer, records, information_condition=information_condition)
    global_common = identity["global_common_count"]
    global_rare = identity["global_rare_count"]
    if checkpoint is not None:
        current = dict(identity); expected = dict(checkpoint.identity)
        if (checkpoint.root / "CURRENT").exists():
            current.pop("model_optimizer_start_sha256")
            expected.pop("model_optimizer_start_sha256", None)
        _require(current == expected,
                 "streaming checkpoint problem or initial state differs")
    restored = checkpoint.restore(model, optimizer) if checkpoint is not None else None
    start = 0; common_sum = rare_sum = 0.; common_count = rare_count = 0
    if restored is None:
        optimizer.zero_grad(set_to_none=True)
    else:
        start = restored["next_fragment"]
        common_sum = restored["common_sum"]; rare_sum = restored["rare_sum"]
        common_count = restored["common_count"]; rare_count = restored["rare_count"]
        if restored.get("optimizer_step_completed"):
            return MaskedUpdateResult(
                .5 * common_sum / global_common + .5 * rare_sum / global_rare,
                common_sum, rare_sum, global_common, global_rare, len(records))
    model.train()
    for index in range(start, len(records)):
        fragment = load_fragment(index)
        actual, _ = masked_fragment_record(fragment, model.config.max_loci_per_channel)
        _require(actual == records[index],
                 "streamed fragment differs from its preflight identity")
        output = model(fragment.common, fragment.rare,
                       information_condition=information_condition)
        terms = masked_genotype_loss_terms(
            output.common_logits, fragment.common_targets.dosage,
            fragment.common_targets.score_mask,
            output.rare_logits, fragment.rare_targets.dosage,
            fragment.rare_targets.score_mask)
        value = scaled_fragment_loss(terms, global_common_count=global_common,
                                     global_rare_count=global_rare)
        _require(bool(torch.isfinite(value)), "nonfinite streamed fragment loss")
        value.backward()
        common_sum += float(terms.common.cross_entropy_sum.detach())
        rare_sum += float(terms.rare.cross_entropy_sum.detach())
        common_count += terms.common.objective_count
        rare_count += terms.rare.objective_count
        del value, terms, output, fragment
        if checkpoint is not None:
            checkpoint.save(model, optimizer, next_fragment=index + 1,
                common_sum=common_sum, rare_sum=rare_sum,
                common_count=common_count, rare_count=rare_count)
    _require((common_count, rare_count) == (global_common, global_rare),
             "streamed fragment cursor/counts differ")
    optimizer.step()
    loss = .5 * common_sum / global_common + .5 * rare_sum / global_rare
    if checkpoint is not None:
        checkpoint.save(model, optimizer, next_fragment=len(records),
            common_sum=common_sum, rare_sum=rare_sum,
            common_count=common_count, rare_count=rare_count,
            optimizer_step_completed=True)
    return MaskedUpdateResult(loss, common_sum, rare_sum, global_common, global_rare,
                              len(records))
