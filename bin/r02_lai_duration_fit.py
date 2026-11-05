"""Bind complete FIT chromosome rows to the existing resumable duration trainer.

This records development on simulated mosaics with shared donors. It neither
certifies biological validity nor admits SELECT/EVALUATE or a new donor cohort.
Inventories must come from ``load_duration_case_inventory``; the neural provider
still authenticates the reconstructed arrays when reading each region.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from r02_context_checkpoint import identity_digest
from r02_lai_duration_input import DurationCaseInventory
from r02_lai_duration_model import COMPLETE_SCIENTIFIC_UNIT, DurationSequenceBatch


FIT_SCOPE = "AUTHENTICATED_FIT_MOSAIC_DEVELOPMENT"
FIT_SCHEMA = "r02_lai_duration_fit_rows_v1"


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def build_fit_rows(inventories: Sequence[DurationCaseInventory], *,
                   people_per_chunk: int, max_optimizer_updates: int,
                   design_sha256: str) -> dict:
    """Declare all rows of the explicitly chosen cases, once, in source order."""
    _require(type(people_per_chunk) is int and people_per_chunk > 0,
             "positive people_per_chunk required")
    cases, chunks = {}, []
    for inventory in inventories:
        _require(isinstance(inventory, DurationCaseInventory)
                 and inventory.role == "FIT" and inventory.condition == "NONE"
                 and inventory.case_id not in cases,
                 "unique authenticated NONE FIT inventories required")
        case = {
            "contract": {"path": inventory.contract_path,
                         "sha256": inventory.contract_sha256},
            "inventory_sha256": inventory.inventory_sha256,
            "chromosome": inventory.chromosome,
            "person_ids": list(inventory.person_ids),
            "component_ids": list(inventory.component_ids),
            "query_positions_cm": list(inventory.query_positions_cm),
            "query_global_indices": list(inventory.query_global_indices),
            "observed_labels_by_person": list(inventory.observed_labels_by_person),
        }
        cases[inventory.case_id] = case
        for start in range(0, len(inventory.person_ids), people_per_chunk):
            chunks.append({"case_id": inventory.case_id, "person_start": start,
                           "person_end": min(start + people_per_chunk,
                                             len(inventory.person_ids))})
    manifest = {"schema": FIT_SCHEMA, "scope": FIT_SCOPE,
                "design_sha256": design_sha256,
                "max_optimizer_updates": max_optimizer_updates,
                "cases": cases, "chunks": chunks,
                "interpretation": "FIT_DEVELOPMENT_SHARED_DONORS_NO_INDEPENDENT_VALIDATION"}
    validate_fit_rows(manifest)
    return manifest


def validate_fit_rows(manifest: dict) -> None:
    _require(isinstance(manifest, dict) and manifest.get("schema") == FIT_SCHEMA
             and manifest.get("scope") == FIT_SCOPE,
             "authenticated FIT row declaration required")
    import re
    sha = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    _require(sha(manifest.get("design_sha256")), "FIT design SHA256 required")
    maximum = manifest.get("max_optimizer_updates")
    _require(type(maximum) is int and maximum > 0, "finite positive update budget required")
    cases, chunks = manifest.get("cases"), manifest.get("chunks")
    _require(isinstance(cases, dict) and cases and isinstance(chunks, list) and chunks,
             "FIT cases and chunks must be nonempty")
    cursors = {}
    for key, case in cases.items():
        people, components = case.get("person_ids"), case.get("component_ids")
        positions, counts = case.get("query_positions_cm"), case.get("observed_labels_by_person")
        indices = case.get("query_global_indices")
        _require(isinstance(key, str) and key and isinstance(people, list) and people
                 and all(isinstance(p, str) and p for p in people)
                 and len(set(people)) == len(people)
                 and isinstance(components, list) and len(components) == len(people)
                 and all(isinstance(c, str) and c for c in components),
                 "FIT identities or components differ")
        _require(type(case.get("chromosome")) is int and 1 <= case["chromosome"] <= 22
                 and isinstance(positions, list) and positions
                 and all(type(p) in (int, float) and math.isfinite(p) for p in positions)
                 and all(a < b for a, b in zip(positions, positions[1:])),
                 "complete increasing autosomal axis required")
        _require(isinstance(indices, list) and len(indices) == len(positions)
                 and all(type(i) is int and i >= 0 for i in indices)
                 and all(a < b for a, b in zip(indices, indices[1:])),
                 "complete global query identities required")
        _require(isinstance(counts, list) and len(counts) == len(people)
                 and all(type(n) is int and 0 <= n <= len(positions) for n in counts),
                 "observed label counts differ from source rows")
        contract = case.get("contract", {})
        _require(isinstance(contract.get("path"), str) and contract["path"]
                 and sha(contract.get("sha256")) and sha(case.get("inventory_sha256")),
                 "source identities required")
        cursors[key] = 0
    for chunk in chunks:
        key, start, end = (chunk.get(k) for k in ("case_id", "person_start", "person_end"))
        _require(key in cases and type(start) is int and type(end) is int
                 and start == cursors[key] and start < end <= len(cases[key]["person_ids"]),
                 "FIT chunks omit, repeat or reorder source rows")
        cursors[key] = end
    _require(all(cursors[key] == len(case["person_ids"]) for key, case in cases.items()),
             "FIT plan must cover every declared source row")
    _require(sum(sum(c["observed_labels_by_person"]) for c in cases.values()) > 0,
             "FIT objective has no observed labels")


def fit_chunk_identity(manifest: dict, index: int) -> str:
    chunk = manifest["chunks"][index]
    return identity_digest({"fit_rows_sha256": identity_digest(manifest),
                            "chunk_index": index, "chunk": chunk})


def validate_fit_plan(manifest: dict, plan) -> None:
    validate_fit_rows(manifest)
    _require(plan.chunk_identity_sha256 == tuple(
        fit_chunk_identity(manifest, i) for i in range(len(manifest["chunks"])))
        and plan.global_rows == sum(len(c["person_ids"]) for c in manifest["cases"].values())
        and plan.global_observed_labels == sum(sum(c["observed_labels_by_person"])
                                              for c in manifest["cases"].values()),
        "FIT plan identities or global denominators differ")


def validate_fit_batch(manifest: dict, index: int, batch: DurationSequenceBatch) -> None:
    chunk = manifest["chunks"][index]
    case = manifest["cases"][chunk["case_id"]]
    start, end = chunk["person_start"], chunk["person_end"]
    _require(batch.person_ids == tuple(case["person_ids"][start:end])
             and batch.component_ids == tuple(case["component_ids"][start:end])
             and batch.roles == ("FIT",) * (end - start),
             "actual FIT people, components or roles differ from declared chunk")
    positions = torch.tensor(case["query_positions_cm"], device=batch.query_cm.device,
                             dtype=torch.float64).expand(end - start, -1)
    _require(batch.query_cm.dtype == torch.float64 and torch.equal(batch.query_cm, positions)
             and batch.query_chromosome.shape == positions.shape
             and bool((batch.query_chromosome == case["chromosome"]).all())
             and batch.query_valid.shape == positions.shape and bool(batch.query_valid.all()),
             "actual FIT chromosome axis or coverage differs")
    _require(torch.equal(batch.query_global_index, torch.tensor(case["query_global_indices"],
             device=batch.query_global_index.device, dtype=torch.long)),
             "actual FIT global query identities differ")
    _require(tuple(int(n) for n in batch.label_mask.sum(dim=1).tolist())
             == tuple(case["observed_labels_by_person"][start:end]),
             "actual FIT observed-label counts differ")
    _require(len(batch.sequence_declarations) == end - start and all(
        d.unit_kind == "chromosome" and d.coverage == COMPLETE_SCIENTIFIC_UNIT
        and d.chromosome == case["chromosome"]
        and d.expected_query_positions_cm == tuple(case["query_positions_cm"])
        for d in batch.sequence_declarations), "FIT sequence declaration differs")
