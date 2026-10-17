#!/usr/bin/env python3
"""Publish authenticated K02/K07 SCORE/SELECT checkpoints from Nextflow v7 layout."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile

ARMS = ("B0", "B1", "B2")
SCOPES = ("BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME")
SCORE_SCHEMA = "r02_structure_pair_score_checkpoint_v1"
SCORE_STATUS = "FIT_SELECT_SCORES_COMPLETE"
SELECT_SCHEMA = "r02_structure_distributed_select_checkpoint_v1"
SELECT_STATUS = "GLOBAL_SELECT_COMPLETE_AWAITING_REVIEW"
RECEIPT_SCHEMA = "r02_pair_published_checkpoint_v2"
DEFAULT_PLAN = Path("/opt/r02-pair/package/config/assembly-tasks.json")
DEFAULT_CLOUD_HELPER = Path("/opt/r02-pair/controller/publish_completed_v1.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def load_json(path):
    def unique(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, "Nonfinite JSON: " + value))


def load_cloud_helper(path=DEFAULT_CLOUD_HELPER):
    path = Path(os.environ.get("R02_PAIR_PUBLISH_V1", path))
    spec = importlib.util.spec_from_file_location("r02_pair_publish_v1_cloud", path)
    require(spec is not None and spec.loader is not None, "Cannot load v1 cloud helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "sha", None)) and callable(getattr(module, "publish", None)),
            "v1 cloud helper lacks sha/publish")
    def verify_record(record):
        metadata = json.loads(module.gc('storage', 'objects', 'describe',
                                        record['uri'], '--format=json').stdout)
        require(str(metadata.get('generation')) == record['generation'] and
                int(metadata.get('size', -1)) == record['bytes'],
                'Previously authenticated immutable object changed')
    module.verify_record = verify_record
    return module


def canonical_dir(path):
    path = Path(path).absolute()
    require(path.is_dir() and not path.is_symlink(), "Missing or symlink directory")
    require(path.resolve() == path, "Directory path traverses a symlink")
    return path


def files_without_symlinks(directory):
    found = []
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Symlink in checkpoint: " + str(path))
        if path.is_file():
            require(path.resolve().is_relative_to(directory), "Checkpoint file escaped directory")
            found.append(path)
    return found


def validate_plan(path):
    plan = load_json(path)
    tasks = plan.get("tasks")
    require(plan.get("evaluate_allowed") is False and isinstance(tasks, list) and len(tasks) == 88,
            "Exact 88-view plan with EVALUATE closed required")
    by_view = {}
    for task in tasks:
        require(isinstance(task, dict) and isinstance(task.get("view_id"), str) and
                isinstance(task.get("block_id"), str) and task.get("exclusion_scope") in SCOPES,
                "Invalid plan task identity")
        require(task["view_id"] not in by_view, "Duplicate plan view_id")
        by_view[task["view_id"]] = task
    require(all(sum(t["exclusion_scope"] == scope for t in tasks) == 44 for scope in SCOPES),
            "Plan must contain 44 views per scope")
    return by_view


def validate_common(directory, expected_status, sha):
    files = files_without_symlinks(directory)
    names = {p.relative_to(directory).as_posix() for p in files}
    require("manifest.json" in names and "complete.json" in names, "Checkpoint copy is incomplete")
    manifest_path, complete_path = directory / "manifest.json", directory / "complete.json"
    manifest, complete = load_json(manifest_path), load_json(complete_path)
    manifest_sha = sha(manifest_path)
    require(complete == {"status": expected_status, "manifest_sha256": manifest_sha},
            "Completion marker differs from manifest")
    return manifest, complete, files, names, manifest_sha


def validate_score(directory, view, arm, task, sha):
    require(arm in ARMS and directory.name == f"score-{view}-{arm}", "SCORE path identity differs")
    manifest, _, files, names, manifest_sha = validate_common(directory, SCORE_STATUS, sha)
    require(manifest.get("schema") == SCORE_SCHEMA and manifest.get("status") == SCORE_STATUS and
            manifest.get("block_id") == task["block_id"] and manifest.get("arm") == arm and
            manifest.get("exclusion_scope") == task["exclusion_scope"] and
            manifest.get("evaluate_targets_computed") is False and
            manifest.get("evaluate_predictions_computed") is False and
            manifest.get("alpha_selected") is False,
            "SCORE manifest identity/status differs")
    outputs = manifest.get("outputs_sha256")
    require(isinstance(outputs, dict) and outputs, "SCORE output hash inventory missing")
    require(set(outputs) | {"manifest.json", "complete.json"} == names,
            "SCORE file inventory differs")
    for relative, digest in outputs.items():
        require(isinstance(relative, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", relative) and
                ".." not in Path(relative).parts and re.fullmatch(r"[0-9a-f]{64}", str(digest)),
                "Unsafe SCORE output inventory")
        require(sha(directory / relative) == digest, "SCORE output SHA256 mismatch: " + relative)
    candidate = load_json(directory / "candidate_scores.json")
    alphas = candidate.get("ridge_alphas")
    trials = candidate.get("trials")
    require(candidate.get("schema") == SCORE_SCHEMA and candidate.get("status") == SCORE_STATUS and
            candidate.get("source") == "SELECT_ONLY_NO_ALPHA_CHOSEN" and
            candidate.get("block_id") == task["block_id"] and candidate.get("arm") == arm and
            candidate.get("exclusion_scope") == task["exclusion_scope"] and
            candidate.get("evaluate_targets_computed") is False and
            candidate.get("alpha_selected") is False and isinstance(alphas, list) and
            len(alphas) == 9 and len(set(alphas)) == 9 and
            all(type(a) in (int, float) and math.isfinite(a) and a > 0 for a in alphas) and
            isinstance(trials, list) and len(trials) == 9 and
            {(r.get("block_id"), r.get("arm"), r.get("alpha")) for r in trials} ==
            {(task["block_id"], arm, alpha) for alpha in alphas} and
            all(type(r.get("loss")) in (int, float) and math.isfinite(r["loss"]) and r["loss"] >= 0
                for r in trials), "SCORE candidate payload differs")
    return files, manifest_sha


def validate_selection(directory, scope, tasks, sha):
    require(scope in SCOPES and directory.name == f"selection-{scope}", "SELECT path identity differs")
    manifest, _, files, names, manifest_sha = validate_common(directory, SELECT_STATUS, sha)
    require(manifest.get("schema") == SELECT_SCHEMA and manifest.get("status") == SELECT_STATUS and
            manifest.get("exclusion_scope") == scope and manifest.get("n_blocks") == 44 and
            manifest.get("score_checkpoint_count") == 132 and
            manifest.get("evaluate_targets_computed") is False and
            manifest.get("biological_result") is False,
            "SELECT manifest identity/status differs")
    expected_blocks = sorted({t["block_id"] for t in tasks.values() if t["exclusion_scope"] == scope})
    require(len(expected_blocks) == 44 and manifest.get("block_ids") == expected_blocks,
            "SELECT block coverage differs from plan")
    require(names == {"frozen_selection.json", "manifest.json", "complete.json"} and
            sha(directory / "frozen_selection.json") == manifest.get("frozen_selection_sha256"),
            "SELECT files/hash differ")
    selection = load_json(directory / "frozen_selection.json")
    alphas, trials = selection.get("ridge_alphas"), selection.get("trials")
    require(selection.get("schema") == "r02_structure_distributed_ridge_selection_v1" and
            selection.get("source") == "SELECT_ONLY" and
            selection.get("exclusion_scope") == scope and selection.get("n_blocks") == 44 and
            selection.get("frozen_before_evaluation") is True and
            selection.get("evaluate_targets_computed") is False and
            set(selection.get("chosen", {})) == set(ARMS) and isinstance(alphas, list) and
            len(alphas) == 9 and len(set(alphas)) == 9 and
            all(type(a) in (int, float) and math.isfinite(a) and a > 0 for a in alphas) and
            all(selection["chosen"][arm] in alphas for arm in ARMS) and
            isinstance(trials, list) and len(trials) == 44 * 3 * 9 and
            isinstance(selection.get("score_sources"), dict) and len(selection["score_sources"]) == 132,
            "SELECT payload differs")
    require({(r.get('block_id'), r.get('arm'), r.get('alpha')) for r in trials} ==
            {(block, arm, alpha) for block in expected_blocks for arm in ARMS for alpha in alphas} and
            all(type(r.get('loss')) in (int, float) and math.isfinite(r['loss']) and r['loss'] >= 0
                for r in trials) and
            set(selection['score_sources']) == {block + ':' + arm for block in expected_blocks for arm in ARMS},
            'SELECT trial/source coverage differs')
    return files, manifest_sha


def candidates(root, tasks):
    for view_dir in sorted((root / "scores").glob("*")) if (root / "scores").is_dir() else []:
        view = view_dir.name
        for arm_dir in sorted(view_dir.glob("*")):
            arm = arm_dir.name
            for directory in sorted(arm_dir.glob("score-*")):
                yield "score", directory, view, arm, tasks.get(view)
    for scope_dir in sorted((root / "selection").glob("*")) if (root / "selection").is_dir() else []:
        scope = scope_dir.name
        for directory in sorted(scope_dir.glob("selection-*")):
            yield "selection", directory, scope, None, None


def atomic_write_new(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".pair-receipt-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_checkpoint(root, directory, kind, identity, files, manifest_sha, prefix, receipts, cloud):
    relative_dir = directory.relative_to(root).as_posix()
    receipt = receipts / (relative_dir.replace("/", "__") + ".json")
    ordered = sorted((p for p in files if p.name not in {"manifest.json", "complete.json"}),
                     key=lambda p: p.relative_to(directory).as_posix())
    ordered += [directory / "manifest.json", directory / "complete.json"]
    initial_sha = {path: cloud.sha(path) for path in ordered}
    if receipt.exists():
        require(not receipt.is_symlink(), 'Receipt cannot be a symlink')
        saved = load_json(receipt)
        require(saved.get('schema') == RECEIPT_SCHEMA and saved.get('kind') == kind and
                saved.get('relative') == relative_dir and saved.get('identity') == identity and
                saved.get('manifest_sha256') == manifest_sha and
                saved.get('complete_sha256') == initial_sha[ordered[-1]] and
                saved.get('complete_published_last') is True and
                len(saved.get('files', [])) == len(ordered), 'Existing receipt identity differs')
        for path, record in zip(ordered, saved['files']):
            require(record.get('uri') == prefix + '/checkpoints/' + path.relative_to(root).as_posix() and
                    record.get('bytes') == path.stat().st_size and
                    record.get('sha256') == initial_sha[path] and
                    re.fullmatch(r'[1-9][0-9]*', str(record.get('generation', ''))),
                    'Existing receipt file identity differs')
            # Its first publication was downloaded and SHA256-verified. The same
            # immutable generation preserves that evidence without downloading
            # every completed matrix again on every polling cycle.
            cloud.verify_record(record)
        cloud.publish(receipt, prefix + '/provenance/checkpoint-receipts/' + receipt.name)
        return saved, 'REUSED_VERIFIED'
    records = []
    for path in ordered:
        if path.name == "complete.json":
            require(all(cloud.sha(local) == digest for local, digest in initial_sha.items()),
                    "Checkpoint changed before completion publication")
        relative = path.relative_to(root).as_posix()
        records.append(cloud.publish(path, prefix + "/checkpoints/" + relative))
        if path.name == "manifest.json":
            require(cloud.sha(path) == manifest_sha, "Manifest changed during publication")
    # Completion was uploaded last and bound to a freshly rechecked manifest.
    require(ordered[-1].name == "complete.json" and
            load_json(ordered[-1]) == {"status": identity["status"], "manifest_sha256": manifest_sha},
            "Completion marker changed during publication")
    value = {"schema": RECEIPT_SCHEMA, "kind": kind, "relative": relative_dir,
             "identity": identity, "files": records, "complete_sha256": cloud.sha(ordered[-1]),
             "manifest_sha256": manifest_sha, "complete_published_last": True}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=receipts,
                                     prefix=".remote-receipt-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    try:
        cloud.publish(temporary, prefix + "/provenance/checkpoint-receipts/" + receipt.name)
        atomic_write_new(receipt, value)
    finally:
        temporary.unlink(missing_ok=True)
    return value, "CREATED_VERIFIED"


def run(root, prefix, receipts, plan=DEFAULT_PLAN, cloud=None):
    root, receipts = canonical_dir(root), Path(receipts).absolute()
    require(prefix.startswith("gs://") and not any(c in prefix for c in "\n\r#?"), "Unsafe GCS prefix")
    receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipts = canonical_dir(receipts)
    tasks = validate_plan(plan)
    cloud = cloud or load_cloud_helper()
    published, pending = [], []
    lock_path = receipts / ".publish.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another checkpoint publisher holds the lock") from exc
        for kind, directory, first, second, task in candidates(root, tasks):
            try:
                directory = canonical_dir(directory)
                if kind == "score":
                    require(task is not None, "SCORE view is absent from exact 88-view plan")
                    files, manifest_sha = validate_score(directory, first, second, task, cloud.sha)
                    identity = {"view_id": first, "block_id": task["block_id"], "arm": second,
                                "exclusion_scope": task["exclusion_scope"], "status": SCORE_STATUS}
                else:
                    files, manifest_sha = validate_selection(directory, first, tasks, cloud.sha)
                    identity = {"exclusion_scope": first, "status": SELECT_STATUS}
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                pending.append({"relative": str(directory.relative_to(root)),
                                "status": "PENDING_UNVERIFIED", "reason": str(exc)})
                continue
            value, operation = publish_checkpoint(root, directory, kind, identity, files,
                                                   manifest_sha, prefix.rstrip("/"), receipts, cloud)
            published.append({"relative": value["relative"], "operation": operation})
    return {"published_completed": len(published), "published": published,
            "pending_unverified": pending}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--receipts", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.prefix, args.receipts), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
