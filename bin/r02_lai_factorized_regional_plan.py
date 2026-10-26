#!/usr/bin/env python3
"""Plan each regional LAI predictor once and five hash-bound condition links.

The plan contains small JSON requests.  It never serializes predictor arrays,
truth arrays, control arrays, or burden prefixes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r02_lai_query_burden import BOUNDARY, COUNT_NAMES, SCOPE
from r02_lai_regional_inputs import RegionalReader

PLAN_SCHEMA = "r02_lai_factorized_regional_request_plan_v1"
TEMPLATE_SCHEMA = "r02_lai_factorized_regional_link_template_v1"
CONDITIONS = ("NONE", "NO_RARE", "COUNT_MATCHED", "DENSITY_MATCHED", "IDENTITY_SHAM")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def descriptor(path):
    path = Path(path).resolve()
    require(path.is_file(), "input absent: " + str(path))
    return {"path": str(path), "sha256": sha256(path)}


def write_fresh(path, value):
    path = Path(path)
    require(not path.exists() and path.parent.is_dir(), "fresh JSON output required: " + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return descriptor(path)


def radius_label(value):
    return format(float(value), "g")


def _source_products(task_id, radius, condition, roots, density_requests):
    four = Path(roots["four_control_root"]) / task_id / "prepared"
    if condition == "DENSITY_MATCHED":
        density_id = task_id + ".r" + radius_label(radius)
        control = Path(roots["density_root"]) / density_id / "prepared"
        prefix = Path(roots["density_prefix_root"]) / density_id / "prepared"
        control_request = Path(density_requests[density_id]["request"]["path"])
    else:
        control = four
        prefix = Path(roots["four_prefix_root"]) / task_id / condition
        control_request = Path(roots["four_control_request_root"]) / (task_id + ".json")
    return {"control_request": descriptor(control_request),
            "control_receipt": descriptor(control / "receipt.json"),
            "control_batch": descriptor(control / "control-views.npz"),
            "event_burden_prefix": descriptor(prefix / "event-burden-prefix.npz"),
            "event_burden_prefix_receipt": descriptor(prefix / "receipt.json")}


def build(args):
    import numpy as np
    output = Path(args.output_dir)
    require(not output.exists() and output.parent.is_dir(), "fresh plan directory required")
    output.mkdir(mode=0o700)
    predictor_dir = output / "predictors"; predictor_dir.mkdir()
    link_dir = output / "links"; link_dir.mkdir()
    density_plan_path = Path(args.density_plan)
    density_plan = json.loads(density_plan_path.read_text())
    require(density_plan.get("schema") == "r02_lai_density_plan_v1"
            and len(density_plan.get("tasks", [])) == 36, "density plan differs")
    density_requests = {row["density_task_id"]: row for row in density_plan["tasks"]}
    roots = {name: getattr(args, name) for name in ("four_control_root", "four_control_request_root",
        "density_root", "four_prefix_root", "density_prefix_root")}
    layout_root = Path(args.layout_root)
    cases = sorted(path.parent.parent for path in layout_root.glob("*/prepared/receipt.json"))
    require(len(cases) == 12, "exact twelve regional layout cases required")
    predictors, links = [], []
    for case_dir in cases:
        prepared = case_dir / "prepared"; layout_path = prepared / "layout.json"
        layout_receipt_path = prepared / "receipt.json"
        layout_receipt = json.loads(layout_receipt_path.read_text())
        require(layout_receipt.get("schema") == "r02_lai_regional_layout_receipt_v2"
                and layout_receipt.get("layout_sha256") == sha256(layout_path),
                "regional layout receipt differs")
        reader = RegionalReader(layout_path, layout_receipt["layout_sha256"], authenticate=True)
        layout = reader.layout; task_id = layout["task_id"]
        sources = reader._sources(); query = sources["query"]
        people = list(range(layout["people"]))
        expected_people = sources["index"]["person_id"].astype(str).tolist()
        expected_components = query["component_id"].astype(str).tolist()
        require(query["person_id"].astype(str).tolist() == expected_people,
                "factorized/query person axes differ")
        overlay = layout["sources"]["query_overlay"]
        overlay_receipt_path = Path(overlay["path"]).with_name("receipt.json")
        overlay_receipt = json.loads(overlay_receipt_path.read_text())
        require(overlay_receipt.get("schema") == "r02_lai_truthblind_context_receipt_v2"
                and overlay_receipt.get("output_sha256") == overlay["sha256"]
                and overlay_receipt.get("truth_read_for_query_geometry") is False,
                "truth-blind overlay receipt differs")
        for recipe_row in layout["layouts"]:
            recipe = recipe_row["recipe"]; radius = float(recipe["rare_radius_cm"])
            require(float(recipe["common_radius_cm"]) == radius, "COMMON/EVENT radii differ")
            resolved = dict(recipe)
            for region in recipe_row["regions"]:
                predictor_id = ".".join((task_id, recipe["family"], "r" + radius_label(radius), region["region_id"]))
                predictor = {"schema": "r02_lai_regional_predictor_request_v1", "operation": "PREPARE",
                    "layout": descriptor(layout_path), "task_id": task_id, "case_id": layout["case_id"],
                    "direction": layout["direction"], "role": layout["role"],
                    "chromosome": str(layout["chromosome"]), "family": recipe["family"],
                    "scale_cm": radius, "region_id": region["region_id"], "person_indices": people,
                    "expected_person_ids": expected_people, "expected_component_ids": expected_components,
                    "expected_layout_recipe": recipe, "resolved_model_config": resolved,
                    "reference_identity": json.loads(Path(layout["sources"]["factorized_receipt"]["path"]).read_text())["reference_identity"],
                    "training_allowed": False, "controls_materialized": False, "truth_loaded": False,
                    "selection_performed": False, "evaluate_opened": False}
                predictor_path = predictor_dir / (predictor_id + ".json")
                predictor_desc = write_fresh(predictor_path, predictor)
                link_ids = []
                for condition in CONDITIONS:
                    link_id = predictor_id + "." + condition
                    template = {"schema": TEMPLATE_SCHEMA, "operation": "PREPARE",
                        "link_id": link_id, "predictor_id": predictor_id, "condition": condition,
                        "task_id": task_id, "case_id": layout["case_id"], "direction": layout["direction"],
                        "role": layout["role"], "chromosome": str(layout["chromosome"]),
                        "rare_radius_cm": radius, "overlay": overlay,
                        "overlay_receipt": descriptor(overlay_receipt_path),
                        **_source_products(task_id, radius, condition, roots, density_requests),
                        "lai_burden_scope": SCOPE, "lai_burden_boundary": BOUNDARY,
                        "lai_burden_count_names": list(COUNT_NAMES),
                        "training_authorized": False, "selection_authorized": False,
                        "evaluate_opened": False, "controls_materialized": False}
                    template_path = link_dir / (link_id + ".json")
                    template_desc = write_fresh(template_path, template)
                    links.append({"link_id": link_id, "predictor_id": predictor_id,
                                  "condition": condition, "chromosome": str(layout["chromosome"]),
                                  "template": template_desc})
                    link_ids.append(link_id)
                predictors.append({"predictor_id": predictor_id, "request": predictor_desc,
                                   "chromosome": str(layout["chromosome"]), "link_ids": link_ids})
    require(len(predictors) == 12 * 2 * 3 * 16 and len(links) == len(predictors) * 5,
            "factorized plan cardinality differs")
    plan = {"schema": PLAN_SCHEMA,
        "status": "ONE_PREDICTOR_PER_REGION_RECIPE_WITH_FIVE_METADATA_LINKS_PREPARE_ONLY",
        "predictors": predictors, "links": links, "predictor_count": len(predictors),
        "link_count": len(links), "conditions": list(CONDITIONS),
        "source_density_plan_sha256": sha256(density_plan_path),
        "large_predictor_array_copies_per_condition": 0,
        "training_authorized": False, "selection_authorized": False, "evaluate_opened": False,
        "scientific_limit": "Request plan only; no model fit, selection, evaluation, or LAI result."}
    plan_path = output / "PLAN.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def finalize_link(args):
    import r02_lai_neural_input as neural
    template_path = Path(args.template)
    require(sha256(template_path) == args.template_sha256, "link template hash differs")
    template = json.loads(template_path.read_text())
    require(template.get("schema") == TEMPLATE_SCHEMA and template.get("operation") == "PREPARE",
            "link template differs")
    full = {key: template[key] for key in ("task_id", "case_id", "direction", "role", "chromosome",
        "condition", "overlay", "overlay_receipt", "control_request", "control_receipt", "control_batch",
        "event_burden_prefix", "event_burden_prefix_receipt", "rare_radius_cm", "lai_burden_scope",
        "lai_burden_boundary", "lai_burden_count_names", "training_authorized", "selection_authorized",
        "evaluate_opened", "controls_materialized")}
    full.update(schema=neural.REGIONAL_TRUTH_REQUEST_SCHEMA_V2, operation="PREPARE",
        regional_request=descriptor(args.regional_request),
        regional_receipt=descriptor(args.regional_receipt),
        regional_batch=descriptor(args.regional_batch))
    request_output = Path(args.request_output)
    write_fresh(request_output, full)
    receipt = neural.join_regional_truth(request_output, sha256(request_output), args.output_dir,
                                         factorized_link=True)
    require(receipt["regional_batch_reused_without_copy"] is True
            and receipt["batch_sha256"] is None, "factorized link wrote a combined batch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("build")
    for name in ("layout_root", "four_control_root", "four_control_request_root", "density_root",
                 "four_prefix_root", "density_prefix_root", "density_plan", "output_dir"):
        plan.add_argument("--" + name.replace("_", "-"), required=True)
    link = sub.add_parser("finalize-link")
    for name in ("template", "template_sha256", "regional_request", "regional_receipt",
                 "regional_batch", "request_output", "output_dir"):
        link.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args(); os.umask(0o077)
    result = build(args) if args.command == "build" else finalize_link(args)
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "predictors": result.get("predictor_count"), "links": result.get("link_count")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
