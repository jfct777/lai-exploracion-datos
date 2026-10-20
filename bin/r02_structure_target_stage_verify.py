#!/usr/bin/env python3
"""Verify locally staged K03 FIT/SELECT target bundles and the 88-view campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re


CATALOGUE_SCHEMA = "r02_structure_88_target_catalogue_v1"
BUNDLE_SCHEMA = "r02_structure_target_stage_receipt_v1"
CAMPAIGN_SCHEMA = "r02_structure_target_stage_campaign_v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
FILES = ("manifest.json", "FIT.target.private.npz", "FIT.target.receipt.json",
         "SELECT.target.private.npz", "SELECT.target.receipt.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path, expected_sha256):
    path = Path(path).resolve()
    require(path.is_file() and HEX64.fullmatch(expected_sha256 or "") and
            sha256(path) == expected_sha256, "pinned input differs: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def catalogue_target(catalogue, view_id):
    require(catalogue.get("schema") == CATALOGUE_SCHEMA and
            catalogue.get("status") == "AUTHENTICATED_88_FIT_SELECT_TARGETS_NO_TRAINING" and
            catalogue.get("views") == 88 and catalogue.get("training_allowed") is False and
            catalogue.get("evaluate_targets_present") is False,
            "target catalogue state differs")
    matches = [row for row in catalogue.get("targets", []) if row.get("view_id") == view_id]
    require(len(matches) == 1, "unknown or duplicate target view: " + view_id)
    return matches[0]


def load_stager(path, expected_sha256):
    path = Path(path).resolve()
    require(path.is_file() and sha256(path) == expected_sha256,
            "frozen r02_neural_target_stage.py differs")
    spec = importlib.util.spec_from_file_location("frozen_neural_target_stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "validate_bundle", None)),
            "frozen stager lacks validate_bundle")
    return module


def verify_bundle(catalogue_path, catalogue_sha256, stager_path, stager_sha256,
                  view_id, target_dir, output):
    catalogue = load_pinned(catalogue_path, catalogue_sha256)
    target = catalogue_target(catalogue, view_id)
    target_dir = Path(target_dir).resolve()
    require(target_dir.is_dir() and target_dir.name == view_id,
            "target directory missing or not named by exact view")
    stager = load_stager(stager_path, stager_sha256)
    handoff = stager.validate_bundle(target_dir, view_id)
    require(handoff.get("view_id") == view_id and handoff.get("roles") == ["FIT", "SELECT"] and
            handoff.get("evaluate") is False and handoff.get("training_allowed") is False,
            "trainer target handoff differs")

    files = {}
    for name in FILES:
        path = target_dir / name
        require(path.is_file() and not path.is_symlink(), "staged target file missing: " + name)
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if target.get("source") == "common88_direct_objects":
        require(set(target.get("objects", {})) == set(FILES), "catalogue file inventory differs")
        for name in FILES:
            expected = target["objects"][name]
            require(files[name] == {"bytes": expected["bytes"], "sha256": expected["sha256"]},
                    "staged object differs from pinned GCS object: " + name)
    else:
        require(target.get("source") == "verified_archive_member" and
                files["manifest.json"] == {"bytes": target["manifest"]["bytes"],
                                           "sha256": target["manifest"]["sha256"]},
                "verified archive target manifest differs")
    receipt = {
        "schema": BUNDLE_SCHEMA,
        "status": "COMPLETE_AUTHENTICATED_FIT_SELECT_TARGET_STAGE",
        "view_id": view_id,
        "source": target["source"],
        "target_dir": str(target_dir),
        "catalogue_sha256": catalogue_sha256,
        "stager_sha256": stager_sha256,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "roles": ["FIT", "SELECT"],
        "evaluate_opened": False,
        "training_allowed": False,
    }
    output = Path(output)
    require(not output.exists(), "fresh bundle receipt output required")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_campaign(catalogue_path, catalogue_sha256, receipts, output):
    catalogue = load_pinned(catalogue_path, catalogue_sha256)
    expected = {row["view_id"] for row in catalogue.get("targets", [])}
    require(len(expected) == 88, "catalogue does not contain 88 unique views")
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in receipts]
    observed = {row.get("view_id") for row in rows}
    require(len(rows) == 88 and observed == expected, "staged receipt view universe differs")
    require(all(row.get("schema") == BUNDLE_SCHEMA and
                row.get("status") == "COMPLETE_AUTHENTICATED_FIT_SELECT_TARGET_STAGE" and
                row.get("catalogue_sha256") == catalogue_sha256 and
                row.get("roles") == ["FIT", "SELECT"] and
                row.get("evaluate_opened") is False and row.get("training_allowed") is False
                for row in rows), "staged bundle receipt state differs")
    stager_hashes = {row["stager_sha256"] for row in rows}
    require(len(stager_hashes) == 1, "more than one frozen stager built the campaign")
    result = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "ALL_88_FIT_SELECT_TARGETS_STAGED_AND_AUTHENTICATED_NO_TRAINING",
        "views": 88,
        "direct_views": sum(row["source"] == "common88_direct_objects" for row in rows),
        "archive_views": sum(row["source"] == "verified_archive_member" for row in rows),
        "total_staged_bytes": sum(row["total_bytes"] for row in rows),
        "catalogue_sha256": catalogue_sha256,
        "stager_sha256": next(iter(stager_hashes)),
        "view_ids": sorted(observed),
        "evaluate_opened": False,
        "training_allowed": False,
    }
    require(result["direct_views"] == 87 and result["archive_views"] == 1,
            "87-direct/one-archive staged provenance differs")
    output = Path(output)
    require(not output.exists(), "fresh campaign receipt output required")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--catalogue", required=True)
    bundle.add_argument("--catalogue-sha256", required=True)
    bundle.add_argument("--stager", required=True)
    bundle.add_argument("--stager-sha256", required=True)
    bundle.add_argument("--view-id", required=True)
    bundle.add_argument("--target-dir", required=True)
    bundle.add_argument("--output", required=True)
    campaign = sub.add_parser("campaign")
    campaign.add_argument("--catalogue", required=True)
    campaign.add_argument("--catalogue-sha256", required=True)
    campaign.add_argument("--receipts", nargs="+", required=True)
    campaign.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "bundle":
        result = verify_bundle(args.catalogue, args.catalogue_sha256, args.stager,
                               args.stager_sha256, args.view_id, args.target_dir, args.output)
    else:
        result = verify_campaign(args.catalogue, args.catalogue_sha256, args.receipts, args.output)
    print(json.dumps({"status": result["status"], "views": result.get("views", 1)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
