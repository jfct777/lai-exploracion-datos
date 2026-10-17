#!/usr/bin/env python3
"""Stage one generation-pinned prepared-shard manifest for K02/K07 assembly."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ACCOUNT = "653458115080-compute@developer.gserviceaccount.com"
PROJECT = "uspbr-242713"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def resolve(plan_path, inventory_path, view_id):
    plan = load(plan_path)
    inventory = load(inventory_path)
    require(len(plan.get("tasks", [])) == 88 and plan.get("evaluate_allowed") is False,
            "Exact 88-view EVALUATE-closed plan required")
    require(inventory.get("schema") == "r02_pair_shard_manifest_inventory_v1" and
            inventory.get("status") == "COMPLETE_GENERATION_PINNED" and
            inventory.get("view_count") == 88 and len(inventory.get("objects", {})) == 88,
            "Exact 88-view generation-pinned manifest inventory required")
    matches = [task for task in plan["tasks"] if task["view_id"] == view_id]
    require(len(matches) == 1, "Exact view required")
    task = matches[0]
    obj = inventory["objects"].get(view_id)
    require(isinstance(obj, dict), "Shard manifest object absent")
    expected_uri = next(iter(task["prepared_shard_objects"].values()))["uri"].rsplit("/", 1)[0] + "/manifest.json"
    require(obj.get("uri") == expected_uri and obj.get("generation_uri") ==
            expected_uri + "#" + str(obj.get("generation")) and
            str(obj.get("generation", "")).isdigit() and type(obj.get("bytes")) is int and
            obj["bytes"] > 0 and obj.get("sha256") == task["prepared_shard_manifest_sha256"],
            "Shard manifest identity differs from assembly plan")
    return obj


def stage(plan_path, inventory_path, view_id, output):
    obj = resolve(plan_path, inventory_path, view_id)
    output = Path(output)
    require(not output.exists(), "Output exists")
    subprocess.run(["gcloud", "--account=" + ACCOUNT, "--project=" + PROJECT,
                    "storage", "cp", obj["generation_uri"], str(output)], check=True)
    require(output.stat().st_size == obj["bytes"] and sha(output) == obj["sha256"],
            "Staged shard manifest differs")
    return {"view_id": view_id, "generation_uri": obj["generation_uri"],
            "bytes": obj["bytes"], "sha256": obj["sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.plan, args.inventory, args.view_id, args.output), sort_keys=True))
