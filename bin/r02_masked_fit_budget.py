#!/usr/bin/env python3
"""Plan and measure one recoverable FIT masked-genotype Adam update.

The plan selects two cells from genetic-map geometry and catalogue chunk
density only.  It never reads genotype dosage to choose coordinates.  The run
uses the existing masked store, batch loader, model, update, and campaign
implementation.  It cannot read SELECT/EVALUATE and requires a PRE acceptance
receipt bound to the exact request, plan, and this executable.

This is a resource measurement, not model selection, biological evaluation,
or a scientific update-budget choice.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

REQUEST_SCHEMA = "r02_masked_fit_budget_request_v1"
REQUEST_STATUS = "PREPARED_FOR_PRE_REVIEW"
PLAN_SCHEMA = "r02_masked_fit_budget_plan_v1"
PRE_SCHEMA = "r02_masked_fit_budget_pre_acceptance_v1"
RECEIPT_SCHEMA = "r02_masked_fit_budget_receipt_v1"
COORDINATE_RULE = "CATALOG_CHUNK_DENSITY_CENTRAL_PAIR_V1"
PRE_STATUS = "PRE_ACCEPTED_FOR_ONE_FIT_RESOURCE_MEASUREMENT"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False).encode() + b"\n")
    os.replace(temporary, path)


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_request(request: dict) -> None:
    require(request.get("schema") == REQUEST_SCHEMA, "request schema differs")
    require(request.get("status") == REQUEST_STATUS,
            "request is still a template or has another status")
    require(request.get("role") == "FIT", "only FIT is permitted")
    require(request.get("coordinate_rule") == COORDINATE_RULE,
            "coordinate rule differs")
    require(request.get("measurement_updates") == 1,
            "exactly one measurement update is required")
    require(type(request.get("seed")) is int
            and 0 <= request["seed"] < 2**32,
            "explicit NumPy-compatible integer seed required")
    require(request.get("select_opened") is False
            and request.get("evaluate_opened") is False,
            "SELECT and EVALUATE must remain closed")
    people = request.get("person_ids")
    require(isinstance(people, list) and people
            and len(people) == len(set(people))
            and all(isinstance(person, str) and person for person in people),
            "unique explicit FIT person IDs required")
    sources = request.get("sources")
    require(isinstance(sources, list) and sources
            and len({source.get("chromosome") for source in sources}) == len(sources),
            "unique chromosome sources required")
    required_source = {"chromosome", "store", "manifest_sha256",
                       "axis_directory", "axis_receipt_sha256"}
    for source in sources:
        require(set(source) == required_source, "source fields differ")
        require(type(source["chromosome"]) is int
                and 1 <= source["chromosome"] <= 22,
                "source chromosome must be an autosome")
        require(is_sha256(source["manifest_sha256"])
                and is_sha256(source["axis_receipt_sha256"]),
                "source hashes must be SHA-256")
        require(all(isinstance(source[name], str) and source[name]
                    for name in ("store", "axis_directory")),
                "explicit source paths required")
    require(is_sha256(request.get("protocol_sha256")),
            "approved protocol SHA-256 required")
    model = request.get("model")
    model_fields = {"family", "width", "projection_depth", "attention_depth",
                    "dropout", "heads", "common_radius_cm", "rare_radius_cm",
                    "position_bp_scale", "position_cm_scale",
                    "max_loci_per_channel", "query_block_size",
                    "max_mask_elements", "checkpoint_blocks"}
    require(isinstance(model, dict) and set(model) == model_fields,
            "complete model configuration required")
    require(model["family"] in ("local_deepsets", "local_attention"),
            "unknown masked-genotype model family")
    for name in ("width", "projection_depth", "max_loci_per_channel",
                 "query_block_size", "max_mask_elements"):
        require(type(model[name]) is int and model[name] > 0,
                "positive integer model field required: " + name)
    for name in ("common_radius_cm", "rare_radius_cm", "position_bp_scale",
                 "position_cm_scale"):
        require(type(model[name]) in (int, float) and math.isfinite(model[name])
                and model[name] > 0,
                "positive finite model field required: " + name)
    require(type(model["dropout"]) in (int, float)
            and math.isfinite(model["dropout"])
            and 0 <= model["dropout"] < 1,
            "model dropout must be in [0,1)")
    require(type(model["checkpoint_blocks"]) is bool,
            "checkpoint_blocks must be boolean")
    if model["family"] == "local_deepsets":
        require(model["attention_depth"] == 0 and model["heads"] is None,
                "Deep Sets requires attention_depth=0 and heads=None")
    else:
        require(type(model["attention_depth"]) is int
                and model["attention_depth"] > 0
                and type(model["heads"]) is int and model["heads"] > 0
                and model["width"] % model["heads"] == 0
                and model["dropout"] == 0,
                "bounded local-attention configuration differs")
    optimizer = request.get("optimizer")
    require(isinstance(optimizer, dict)
            and set(optimizer) == {"learning_rate", "adam_state_precision"},
            "exact Adam configuration required")
    require(type(optimizer["learning_rate"]) in (int, float)
            and math.isfinite(optimizer["learning_rate"])
            and optimizer["learning_rate"] > 0,
            "positive finite Adam learning rate required")
    require(optimizer["adam_state_precision"] in ("fp32", "fp64"),
            "Adam state precision must be fp32 or fp64")
    require(request.get("information_condition") in
            ("full", "count_only", "no_rare"),
            "information condition differs")
    exclusion = request.get("context_exclusion_cm")
    require(type(exclusion) in (int, float) and math.isfinite(exclusion)
            and exclusion >= 0, "finite nonnegative context exclusion required")
    resources = request.get("resources")
    required_resources = {"max_people_per_fragment", "max_window_sites",
                          "max_cache_entries", "torch_threads", "max_seconds",
                          "external_memory_limit_bytes"}
    require(isinstance(resources, dict) and set(resources) == required_resources,
            "resource fields differ")
    for name in ("max_people_per_fragment", "max_window_sites",
                 "max_cache_entries", "torch_threads",
                 "external_memory_limit_bytes"):
        require(type(resources[name]) is int and resources[name] > 0,
                "positive integer resource required: " + name)
    require(resources["max_people_per_fragment"] == len(people),
            "one fragment must contain every explicitly requested person")
    require(type(resources["max_seconds"]) in (int, float)
            and math.isfinite(resources["max_seconds"])
            and resources["max_seconds"] > 0,
            "positive finite max_seconds required")


def open_reader(source: dict, request: dict):
    from r02_masked_store import MaskedStoreReader, role_person_ids

    resources = request["resources"]
    reader = MaskedStoreReader(
        source["store"], source["manifest_sha256"], source["axis_directory"],
        source["axis_receipt_sha256"], role="FIT",
        max_people=resources["max_people_per_fragment"],
        max_window_sites=resources["max_window_sites"],
        max_cache_entries=resources["max_cache_entries"])
    require(reader.chromosome == source["chromosome"],
            "source chromosome differs from authenticated store")
    require(set(request["person_ids"]) <= role_person_ids(reader.reader, "FIT"),
            "requested person is outside authenticated FIT")
    return reader


def _cell_bp_bounds(reader, cell: int) -> tuple[int, int]:
    from r02_masked_store import maps

    left_cm, right_cm = reader.edges[cell]
    left_bp = math.floor(maps.inverse_cm(reader.reader.points, left_cm, "left"))
    right_bp = math.ceil(maps.inverse_cm(reader.reader.points, right_cm, "right"))
    return left_bp, right_bp


def _channel_estimate(chunks: list[dict], left_bp: int, right_bp: int) -> float:
    """Estimate catalogue loci from chunk count and physical overlap only."""
    estimate = 0.0
    for chunk in chunks:
        overlap = max(0, min(right_bp, int(chunk["end_bp"]))
                      - max(left_bp, int(chunk["start_bp"])) + 1)
        if overlap:
            span = int(chunk["end_bp"]) - int(chunk["start_bp"]) + 1
            estimate += int(chunk["n_sites"]) * overlap / span
    return estimate


def catalogue_density_cells(reader) -> list[dict]:
    rows = []
    channels = reader.reader.manifest["channels"]
    for cell, edges in enumerate(reader.edges):
        left_bp, right_bp = _cell_bp_bounds(reader, cell)
        common = _channel_estimate(channels["common"], left_bp, right_bp)
        rare = _channel_estimate(channels["rare"], left_bp, right_bp)
        width_cm = float(edges[1] - edges[0])
        require(width_cm > 0, "cell has nonpositive genetic-map width")
        total = common + rare
        rows.append({"chromosome": reader.chromosome, "cell": cell,
                     "left_edge_cm": float(edges[0]),
                     "right_edge_cm": float(edges[1]),
                     "left_edge_bp": left_bp, "right_edge_bp": right_bp,
                     "common_catalog_loci_estimate": common,
                     "rare_catalog_loci_estimate": rare,
                     "catalog_loci_estimate": total,
                     "catalog_loci_per_cm_estimate": total / width_cm})
    return rows


def central_density_pair(cells: list[dict]) -> tuple[dict, dict]:
    """Return the two cells bracketing catalogue-density median.

    The rule has no genotype-, carrier-, loss-, or person-dependent input.
    """
    eligible = [row for row in cells if row["common_catalog_loci_estimate"] > 0
                and row["rare_catalog_loci_estimate"] > 0]
    require(len(eligible) >= 2,
            "fewer than two cells have both catalogue channels")
    ordered = sorted(eligible, key=lambda row: (
        row["catalog_loci_per_cm_estimate"], row["chromosome"], row["cell"]))
    size = len(ordered)
    lower = size // 2 - 1 if size % 2 == 0 else size // 2
    upper = size // 2 if size % 2 == 0 else size // 2 + 1
    return ordered[lower], ordered[upper]


def make_plan(request: dict) -> dict:
    validate_request(request)
    all_cells = []
    source_receipts = []
    for source in sorted(request["sources"], key=lambda item: item["chromosome"]):
        reader = open_reader(source, request)
        all_cells.extend(catalogue_density_cells(reader))
        source_receipts.append({
            "chromosome": reader.chromosome,
            "manifest_sha256": reader.reader.manifest_sha256,
            "axis_receipt_sha256": reader.axis_receipt_sha256,
            "cells_sha256": reader.cells_sha256,
            "cell_count": len(reader.edges)})
    selected = central_density_pair(all_cells)
    request_sha = digest(request)
    return {"schema": PLAN_SCHEMA, "status": "PLANNED_NOT_EXECUTED",
            "request_sha256": request_sha, "code_sha256": sha256(__file__),
            "coordinate_rule": COORDINATE_RULE,
            "coordinate_inputs": "genetic map plus manifest chunk start_bp/end_bp/n_sites only",
            "coordinate_inputs_exclude": ["dosage", "carrier status", "person genotype",
                                          "loss", "SELECT", "EVALUATE"],
            "eligible_cells": sum(row["common_catalog_loci_estimate"] > 0
                                  and row["rare_catalog_loci_estimate"] > 0
                                  for row in all_cells),
            "all_cells": len(all_cells), "selected_coordinates": list(selected),
            "people": len(request["person_ids"]), "cells": 2,
            "person_cell_units": 2 * len(request["person_ids"]),
            "optimizer_updates": 1, "seed": request["seed"],
            "source_receipts": source_receipts,
            "source_identity_sha256": digest(source_receipts),
            "scientific_sampling_claim": False,
            "select_opened": False, "evaluate_opened": False}


@dataclass(frozen=True)
class MeasurementCellExample:
    epoch: int
    chromosome: int
    cell: int
    person_ids: tuple[str, ...]


@dataclass(frozen=True)
class MeasurementSchedule:
    """One explicit update; coordinate choice is recorded in the external plan."""

    person_ids: tuple[str, ...]
    coordinates: tuple[tuple[int, int], ...]

    def __post_init__(self):
        require(len(self.coordinates) == 2 and len(set(self.coordinates)) == 2,
                "exactly two distinct measurement coordinates required")

    def updates(self):
        yield tuple(MeasurementCellExample(0, chromosome, cell, self.person_ids)
                    for chromosome, cell in self.coordinates)

    @property
    def update_count(self) -> int:
        return 1

    @property
    def minimum_updates_for_all_people(self) -> int:
        return 1


def validate_plan(request: dict, plan: dict) -> None:
    require(plan.get("schema") == PLAN_SCHEMA
            and plan.get("status") == "PLANNED_NOT_EXECUTED",
            "measurement plan differs")
    require(plan.get("request_sha256") == digest(request),
            "plan belongs to another request")
    require(plan.get("code_sha256") == sha256(__file__),
            "plan was made by another executable")
    require(is_sha256(plan.get("source_identity_sha256")),
            "plan source identity SHA-256 required")
    coordinates = plan.get("selected_coordinates")
    require(isinstance(coordinates, list) and len(coordinates) == 2
            and all(type(row.get("chromosome")) is int
                    and type(row.get("cell")) is int for row in coordinates),
            "plan must contain two coordinates")
    require(plan.get("people") == len(request["person_ids"])
            and plan.get("cells") == 2
            and plan.get("person_cell_units") == 2 * len(request["person_ids"])
            and plan.get("optimizer_updates") == 1
            and plan.get("seed") == request["seed"],
            "plan measurement units differ")


def campaign_external_identity(request: dict, plan: dict) -> str:
    """Bind campaign checkpoints to every reviewed external input.

    ``run_masked_campaign`` already binds model/update code, model/Adam state,
    schedule and its own limits.  This digest additionally binds the complete
    request (including context exclusion and seed), selected-coordinate plan,
    approved protocol and authenticated source inventory.
    """
    return digest({"schema": "r02_masked_fit_budget_external_identity_v1",
                   "request_sha256": digest(request),
                   "plan_sha256": digest(plan),
                   "protocol_sha256": request["protocol_sha256"],
                   "authenticated_sources_sha256":
                       plan["source_identity_sha256"]})


def validate_pre(request: dict, plan: dict, pre: dict) -> None:
    require(pre.get("schema") == PRE_SCHEMA and pre.get("status") == PRE_STATUS,
            "accepted PRE receipt required before genotype access")
    require(pre.get("request_sha256") == digest(request)
            and pre.get("plan_sha256") == digest(plan)
            and pre.get("code_sha256") == sha256(__file__)
            and pre.get("protocol_sha256") == request["protocol_sha256"],
            "PRE receipt is not bound to this request, plan, code, and protocol")
    require(pre.get("authorized_updates") == 1
            and pre.get("authorized_role") == "FIT"
            and pre.get("select_opened") is False
            and pre.get("evaluate_opened") is False,
            "PRE scope differs from one FIT resource measurement")


def peak_rss_bytes() -> int:
    # Linux reports KiB; this project executes on GCP Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def seed_random_generators(seed: int) -> None:
    """Seed every random generator used by the exact CPU measurement path."""
    import random
    import numpy as np
    import torch

    require(type(seed) is int and 0 <= seed < 2**32,
            "explicit NumPy-compatible integer seed required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_measurement(request: dict, plan: dict, pre: dict,
                    output: str | Path) -> dict:
    validate_request(request)
    validate_plan(request, plan)
    validate_pre(request, plan, pre)
    import torch
    from r02_masked_genotype_model import (MaskedGenotypeModel, MaskedModelConfig,
                                           build_masked_adam)
    from r02_masked_training_campaign import (fit_cell_loader,
                                              run_masked_campaign)

    output = Path(output).resolve()
    source_paths = [Path(source["store"]).resolve() for source in request["sources"]]
    axis_paths = [Path(source["axis_directory"]).resolve()
                  for source in request["sources"]]
    require(not any(output == path or output.is_relative_to(path)
                    for path in source_paths + axis_paths),
            "measurement output must not be inside a source")
    output.mkdir(parents=True, exist_ok=True)
    coordinates = tuple((row["chromosome"], row["cell"])
                        for row in plan["selected_coordinates"])
    selected_chromosomes = {chromosome for chromosome, _ in coordinates}
    readers = {source["chromosome"]: open_reader(source, request)
               for source in request["sources"]
               if source["chromosome"] in selected_chromosomes}
    require(set(readers) == selected_chromosomes,
            "selected coordinate lacks an authenticated source")
    config = MaskedModelConfig(**request["model"])
    torch.set_num_threads(request["resources"]["torch_threads"])
    seed_random_generators(request["seed"])
    model = MaskedGenotypeModel(config)
    optimizer = build_masked_adam(model, **request["optimizer"])
    schedule = MeasurementSchedule(tuple(request["person_ids"]), coordinates)
    loader = fit_cell_loader(readers, config,
        context_exclusion_cm=request["context_exclusion_cm"], device="cpu")
    started = time.monotonic()
    result = run_masked_campaign(
        model, optimizer, schedule, loader, output / "checkpoint",
        source_identity_sha256=campaign_external_identity(request, plan),
        protocol_sha256=request["protocol_sha256"],
        information_condition=request["information_condition"], max_updates=1,
        max_seconds=request["resources"]["max_seconds"])
    elapsed = time.monotonic() - started
    update_path = output / "checkpoint" / "update-00000001.json"
    update = load_json(update_path) if update_path.exists() else None
    rss = peak_rss_bytes()
    receipt = {"schema": RECEIPT_SCHEMA, "status": result["status"],
        "request_sha256": digest(request), "plan_sha256": digest(plan),
        "pre_sha256": digest(pre), "code_sha256": sha256(__file__),
        "campaign_external_identity_sha256":
            campaign_external_identity(request, plan),
        "campaign_result": result, "update_receipt": str(update_path),
        "update_result": None if update is None else update["result"],
        "units": {"people": len(request["person_ids"]), "cells": 2,
                  "person_cell_units": 2 * len(request["person_ids"]),
                  "optimizer_updates": result["updates_completed"]},
        "resources": {"elapsed_this_invocation_seconds": elapsed,
                      "process_peak_rss_bytes": rss,
                      "external_memory_limit_bytes":
                          request["resources"]["external_memory_limit_bytes"],
                      "torch_threads": request["resources"]["torch_threads"],
                      "seed": request["seed"]},
        "checkpoint_root": str(output / "checkpoint"),
        "checkpoint_reusable": True, "scientific_training_campaign": False,
        "scientific_sampling_claim": False, "select_opened": False,
        "evaluate_opened": False}
    if rss > request["resources"]["external_memory_limit_bytes"]:
        receipt["resource_limit_observation"] = "PEAK_RSS_EXCEEDED_DECLARED_EXTERNAL_LIMIT"
    else:
        receipt["resource_limit_observation"] = "PEAK_RSS_WITHIN_DECLARED_EXTERNAL_LIMIT"
    atomic_json(output / "MEASUREMENT_RECEIPT.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="authenticate metadata and select two cells")
    plan.add_argument("--request", required=True)
    plan.add_argument("--output", required=True)
    run = commands.add_parser("run", help="run or resume one accepted FIT update")
    run.add_argument("--request", required=True)
    run.add_argument("--plan", required=True)
    run.add_argument("--pre-receipt", required=True)
    run.add_argument("--output", required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    request = load_json(args.request)
    if args.command == "plan":
        plan = make_plan(request)
        atomic_json(args.output, plan)
        print(json.dumps({"status": plan["status"],
                          "people": plan["people"], "cells": plan["cells"],
                          "person_cell_units": plan["person_cell_units"],
                          "optimizer_updates": 1, "output": str(Path(args.output))}))
        return 0
    plan = load_json(args.plan)
    pre = load_json(args.pre_receipt)
    receipt = run_measurement(request, plan, pre, args.output)
    print(json.dumps({"status": receipt["status"],
                      "units": receipt["units"],
                      "elapsed_seconds": receipt["resources"]["elapsed_this_invocation_seconds"],
                      "process_peak_rss_bytes": receipt["resources"]["process_peak_rss_bytes"],
                      "receipt": str(Path(args.output) / "MEASUREMENT_RECEIPT.json")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
