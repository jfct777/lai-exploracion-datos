"""Streaming update adapter for masked-genotype carrier integration.

The numerical update, loss scaling, gradient checkpoint and restart logic stay
in ``run_masked_update_streaming``.  This module only authenticates the extra
reference/pair/COMMON inputs and supplies a fragment-forward callback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from r02_context_checkpoint import _sha_file
from r02_context_update_checkpoint import _state_digest
from r02_masked_carrier_batch import CarrierFragmentBatch
from r02_masked_carrier_integration import MaskedCarrierGenotypeModel
from r02_masked_genotype_update import (
    MaskedFragment, MaskedUpdateStore, masked_fragment_record,
    masked_streaming_update_identity, run_masked_update_streaming,
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def carrier_update_code_hashes() -> dict:
    names = ("r02_masked_carrier_batch.py", "r02_masked_carrier_integration.py",
             "r02_masked_carrier_update.py", "r02_carrier_interaction_model.py",
             "r02_carrier_interactions.py", "r02_carrier_common_context.py")
    return {name: _sha_file(Path(__file__).parent / name) for name in names}


@dataclass(frozen=True)
class MaskedCarrierFragment:
    fragment: MaskedFragment
    carrier_batch: CarrierFragmentBatch

    def check(self, model: MaskedCarrierGenotypeModel) -> None:
        _require(isinstance(self.fragment, MaskedFragment),
                 "MaskedFragment required")
        _require(isinstance(self.carrier_batch, CarrierFragmentBatch),
                 "CarrierFragmentBatch required")
        self.fragment.check(model.config.max_loci_per_channel)
        self.carrier_batch.check(
            self.fragment,
            common_context_width=model.interaction_config.common_context_width)


def masked_carrier_fragment_record(value: MaskedCarrierFragment,
                                   model: MaskedCarrierGenotypeModel) -> dict:
    value.check(model)
    base, owned = masked_fragment_record(
        value.fragment, model.config.max_loci_per_channel)
    carrier = value.carrier_batch.identities()
    return {"base": base, "carrier": carrier,
            "binding_sha256": _state_digest({
                "base_content_sha256": base["content_sha256"],
                "base_owned_targets_sha256": base["owned_targets_sha256"],
                "fragment_lineage": base["fragment_lineage"],
                "carrier": carrier})}, owned


def _identity_extension(model: MaskedCarrierGenotypeModel, records) -> dict:
    records = tuple(records)
    _require(records and all(set(record) == {"base", "carrier", "binding_sha256"}
                             for record in records),
             "complete masked-carrier records required")
    _require(all(record["carrier"].get("source_kind") == "SYNTHETIC_TEST"
                 and record["carrier"].get("human_training_authorized") is False
                 for record in records),
             "Adam update requires SYNTHETIC_TEST inputs; mapped human sources are preflight-only")
    return {"schema": "r02_masked_carrier_update_extension_v1",
            "integration_config": asdict(model.integration_config),
            "interaction_config": asdict(model.interaction_config),
            "carrier_binding_sha256": [record["binding_sha256"] for record in records],
            "carrier_panel_sha256": [record["carrier"]["panel_sha256"]
                                     for record in records],
            "carrier_pairs_sha256": [record["carrier"]["pairs_sha256"]
                                      for record in records],
            "carrier_context_sha256": [record["carrier"]["context_sha256"]
                                        for record in records],
            "carrier_feature_inputs_sha256": [
                record["carrier"]["feature_inputs_sha256"] for record in records],
            "per_target_policy_sha256": [
                record["carrier"]["per_target_policy_sha256"] for record in records],
            "source_kind": [record["carrier"]["source_kind"] for record in records],
            "human_training_authorized": False,
            "code_sha256": carrier_update_code_hashes()}


def masked_carrier_streaming_update_identity(model, optimizer, records, *,
                                             information_condition: str):
    records = tuple(records)
    extension = _identity_extension(model, records)
    return masked_streaming_update_identity(
        model, optimizer, [record["base"] for record in records],
        information_condition=information_condition,
        identity_extension=extension)


def masked_carrier_update_store(root, model, optimizer, records, *,
                                information_condition: str) -> MaskedUpdateStore:
    return MaskedUpdateStore(root, masked_carrier_streaming_update_identity(
        model, optimizer, records,
        information_condition=information_condition))


def run_masked_carrier_update_streaming(
        model: MaskedCarrierGenotypeModel, optimizer, load_fragment, records, *,
        information_condition: str, checkpoint: MaskedUpdateStore | None = None):
    """Authenticate carrier bindings, then delegate the complete update."""
    _require(isinstance(model, MaskedCarrierGenotypeModel),
             "MaskedCarrierGenotypeModel required")
    records = tuple(records)
    extension = _identity_extension(model, records)
    active: dict[str, CarrierFragmentBatch] = {}

    def load_base(index: int) -> MaskedFragment:
        bound = load_fragment(index)
        actual, _ = masked_carrier_fragment_record(bound, model)
        _require(actual == records[index],
                 "streamed carrier fragment differs from preflight identity")
        active[bound.fragment.lineage] = bound.carrier_batch
        return bound.fragment

    def forward_bound(current_model, fragment, condition):
        _require(fragment.lineage in active,
                 "carrier batch was not bound to streamed fragment")
        batch = active.pop(fragment.lineage)
        return current_model(fragment.common, fragment.rare,
                             information_condition=condition,
                             carrier_batch=batch)

    return run_masked_update_streaming(
        model, optimizer, load_base, [record["base"] for record in records],
        information_condition=information_condition, checkpoint=checkpoint,
        forward_fragment=forward_bound, identity_extension=extension)
