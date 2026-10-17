#!/usr/bin/env python3
"""Verify or add the project team label to one identified VM and its disks.

Only ``team=frank`` may be added. Identity, attachments, existing labels and
configuration are checked before and after the update. This helper never starts,
stops, resizes or deletes a resource. A partial failure is reported, not rolled
back by removing labels. Reusing the captured plan makes retries idempotent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

SCHEMA = "r02_resource_labels_v1"
TEAM = "frank"
NAME = re.compile(r"[a-z][a-z0-9-]{0,61}[a-z0-9]|[a-z]")
LINK = re.compile(r"https://(?:www\.googleapis\.com|compute\.googleapis\.com)/compute/v1/"
                  r"projects/([^/]+)/zones/([^/]+)/(instances|disks)/([^/]+)")
# Output-only observations may change during a running job. Every other field,
# including metadata, scheduling, disk attachments and networking, is protected.
OBSERVATIONS = {"labels", "labelFingerprint", "fingerprint", "status", "statusMessage",
                "lastStartTimestamp", "lastStopTimestamp", "lastSuspendedTimestamp",
                "resourceStatus", "satisfiesPzi", "satisfiesPzs"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def settings_sha256(value):
    settings = {k:v for k,v in value.items() if k not in OBSERVATIONS}
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resource_identity(value, project, zone, kind, name, resource_id=None):
    match = LINK.fullmatch(value.get("selfLink", ""))
    require(match and match.groups() == (project, zone, kind, name), "Resource project/zone/name differs")
    require(value.get("name") == name and str(value.get("id", "")).isdigit(), "Resource identity is incomplete")
    require(value.get("zone", "").endswith("/projects/"+project+"/zones/"+zone), "Resource zone differs")
    if resource_id is not None:
        require(str(value["id"]) == str(resource_id), "Resource ID changed: "+name)
    labels = value.get("labels", {})
    require(isinstance(labels, dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in labels.items()),
            "Invalid labels: "+name)
    require("team" not in labels or labels["team"] == TEAM, "Conflicting team label: "+name)
    return dict(kind=kind, name=name, id=str(value["id"]), self_link=value["selfLink"],
                settings_sha256=settings_sha256(value))


def gcloud(args, project, zone, account, runner=None):
    """Restrict the command surface to describe and one additive label update."""
    require(len(args) >= 4 and args[:1] == ["compute"] and args[1] in {"instances", "disks"}
            and args[2] in {"describe", "add-labels"}, "Only describe/add-labels are supported")
    require(NAME.fullmatch(args[3]) is not None, "Invalid resource name")
    require(args[4:] == ([] if args[2] == "describe" else ["--labels=team="+TEAM]), "Unexpected mutation flags")
    command = ["gcloud", *args, "--project="+project, "--zone="+zone, "--account="+account,
               "--format=json", "--quiet"]
    result = (runner or subprocess.run)(command, capture_output=True, text=True, timeout=180)
    require(result.returncode == 0, "Label operation failed; resources left running: "+result.stderr[-1000:])
    # The update response is not our verification source and some CLI versions
    # print nothing. The caller always performs a fresh identity-checked describe.
    if args[2] == "add-labels":
        return None
    require(bool(result.stdout.strip()), "Describe returned no resource")
    return json.loads(result.stdout)


def validate_plan(plan):
    require(plan.get("schema") == SCHEMA and plan.get("team") == TEAM, "Unsupported label plan")
    for key in ("project", "zone"):
        require(isinstance(plan.get(key),str) and NAME.fullmatch(plan[key]), "Explicit safe scope required: "+key)
    require(isinstance(plan.get("account"),str) and re.fullmatch(r"[A-Za-z0-9._+@-]+",plan["account"])
            and "@" in plan["account"], "Explicit account required")
    resources = plan.get("resources")
    require(isinstance(resources,list) and resources and resources[0]["kind"] == "instances"
            and all(r["kind"] == "disks" for r in resources[1:]), "One VM followed by its exact disks required")
    require(len({r["self_link"] for r in resources}) == len(resources), "Repeated resource")
    for resource in resources:
        require(NAME.fullmatch(resource["name"]) and re.fullmatch(r"[1-9][0-9]*",resource["id"])
                and re.fullmatch(r"[a-f0-9]{64}",resource["settings_sha256"]), "Invalid expected resource")
        match = LINK.fullmatch(resource["self_link"])
        require(match and match.groups() == (plan["project"],plan["zone"],resource["kind"],resource["name"]),
                "Expected resource escaped project/zone")


def capture_targets(created_instance, *, project, zone, account, expected_disk_ids=None, runner=None):
    """Capture disk IDs from the exact just-created VM, without changing labels."""
    name = created_instance["name"]
    resource_identity(created_instance, project, zone, "instances", name, created_instance["id"])
    current = gcloud(["compute","instances","describe",name], project, zone, account, runner)
    vm = resource_identity(current, project, zone, "instances", name, created_instance["id"])
    require(current.get("disks") == created_instance.get("disks"), "VM attachments changed after creation")
    resources = [vm]
    for attachment in current.get("disks", []):
        # Local scratch disks have no independent Compute Engine disk resource.
        if attachment.get("type") == "SCRATCH":
            require(not attachment.get("source"), "Unexpected scratch disk source")
            continue
        match = LINK.fullmatch(attachment.get("source", ""))
        require(match and match.groups()[:3] == (project,zone,"disks"), "Attached disk escaped project/zone")
        disk_name = match.group(4)
        disk = gcloud(["compute","disks","describe",disk_name], project, zone, account, runner)
        expected_id = None if expected_disk_ids is None else expected_disk_ids.get(attachment["source"])
        require(expected_disk_ids is None or expected_id is not None, "Pinned attached disk ID is missing")
        entry = resource_identity(disk, project, zone, "disks", disk_name, expected_id)
        require(entry["self_link"] == attachment["source"] and current["selfLink"] in disk.get("users",[]),
                "Disk attachment not confirmed in both directions")
        resources.append(entry)
    require(expected_disk_ids is None or set(expected_disk_ids) == {r["self_link"] for r in resources[1:]},
            "Pinned disk set differs from VM attachments")
    plan = dict(schema=SCHEMA,team=TEAM,project=project,zone=zone,account=account,resources=resources)
    validate_plan(plan)
    return plan


def ensure_team_labels(plan, *, apply=False, runner=None):
    """Fail closed on conflicting labels or changed IDs; never interrupt jobs.

    gcloud's additive label command preserves unspecified labels using its
    fingerprint-based update. Checks bracket each command; this is not an atomic
    multi-resource transaction. Do not delete/recreate these resources concurrently.
    """
    validate_plan(plan)
    project, zone, account = (plan[k] for k in ("project","zone","account"))

    def inspect(resource):
        value = gcloud(["compute",resource["kind"],"describe",resource["name"]], project,zone,account,runner)
        actual = resource_identity(value,project,zone,resource["kind"],resource["name"],resource["id"])
        require(actual == resource, "Resource configuration changed: "+resource["name"])
        return value

    # Inspect every target before making even the first update. An existing
    # conflicting label on a disk must not leave the VM needlessly relabelled.
    initial = {r["self_link"]:inspect(r) for r in plan["resources"]}
    expected_disks = {r["self_link"] for r in plan["resources"][1:]}
    vm = initial[plan["resources"][0]["self_link"]]
    require({d["source"] for d in vm.get("disks",[]) if d.get("type") != "SCRATCH"} == expected_disks,
            "Label plan omits or adds attached disks")
    rows = []
    for resource in plan["resources"]:
        before = inspect(resource)
        labels = dict(before.get("labels",{}))
        require(labels == initial[resource["self_link"]].get("labels",{}), "Labels changed during preflight")
        state = "ALREADY_PRESENT" if labels.get("team") == TEAM else "REQUIRED"
        if state == "REQUIRED" and apply:
            gcloud(["compute",resource["kind"],"add-labels",resource["name"],"--labels=team="+TEAM],
                   project,zone,account,runner)
            after = inspect(resource)
            require(after.get("labels",{}) == {**labels,"team":TEAM}, "Labels changed beyond team: "+resource["name"])
            state = "ADDED_VERIFIED"
        rows.append(dict(**resource,status=state,labels_before=labels,
                         labels_after={**labels,"team":TEAM} if state == "ADDED_VERIFIED" else labels))
    # A final read detects changes in an earlier resource while another was updated.
    for resource,row in zip(plan["resources"],rows):
        require(inspect(resource).get("labels",{}) == row["labels_after"], "Post-verification labels differ")
    return dict(schema=SCHEMA,team=TEAM,project=project,zone=zone,
                status="TEAM_LABELS_VERIFIED" if all(r["status"] != "REQUIRED" for r in rows) else "LABEL_ADDITION_REQUIRED",
                resources=rows,processes_interrupted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True,help="Previously captured exact VM/disk IDs")
    parser.add_argument("--apply",action="store_true",help="Only add missing team=frank labels")
    args = parser.parse_args()
    print(json.dumps(ensure_team_labels(json.loads(args.plan.read_text()),apply=args.apply),sort_keys=True,indent=2))


if __name__ == "__main__":
    main()
