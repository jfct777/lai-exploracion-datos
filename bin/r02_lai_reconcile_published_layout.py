#!/usr/bin/env python3
"""Authenticate nested Nextflow publications and expose their canonical layout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

REQUIRED_OUTPUTS = {
    "common.revised.tsv.gz",
    "event.alt_ref.tsv.gz",
    "BASE/reference.vcf.gz",
    "BASE/reference.vcf.gz.csi",
    "BASE/target.vcf.gz",
    "BASE/target.vcf.gz.csi",
    "BASE/marker_inventory.tsv",
    "BASE/sample_map.tsv",
    "BASE/roles.tsv",
}


class PublicationPending(RuntimeError):
    """A receipt or one of its files is still being copied by publishDir."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_lai_expanded", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen producer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_relative_link(case_dir: Path, name: str, target: str) -> None:
    link = case_dir / name
    if link.is_symlink():
        if os.readlink(link) != target:
            raise RuntimeError(f"Existing canonical link differs: {link}")
        return
    if link.exists():
        raise RuntimeError(f"Refusing to overwrite canonical entry: {link}")
    temporary = case_dir / (name + ".layout-amendment-tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"Unexpected temporary link exists: {temporary}")
    os.symlink(target, temporary)
    os.replace(temporary, link)


def expose_canonical_layout(case_dir: Path, receipt: dict) -> list[dict]:
    outputs = receipt.get("outputs_sha256", {})
    if set(outputs) != REQUIRED_OUTPUTS:
        raise RuntimeError("Nested receipt does not contain exactly the nine required outputs")
    links = {
        "receipt.json": "case-output/receipt.json",
        "common.revised.tsv.gz": "case-output/common.revised.tsv.gz",
        "event.alt_ref.tsv.gz": "case-output/event.alt_ref.tsv.gz",
        "BASE": "case-output/BASE",
    }
    for name, target in links.items():
        ensure_relative_link(case_dir, name, target)
    return [{"path": name, "relative_target": target} for name, target in links.items()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_stable_receipt(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        receipt = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationPending(f"receipt.json is still incomplete: {type(error).__name__}") from error
    return receipt, hashlib.sha256(raw).hexdigest()


def validate_receipt_identity(module, case: dict, receipt: dict, plan_sha: str,
                              legacy_first_sha: str | None) -> None:
    required_fields = {"schema", "status", "task_id", "role", "chromosome", "outputs_sha256"}
    if not required_fields <= set(receipt):
        raise PublicationPending("receipt.json does not yet contain all identity/output fields")
    require(receipt["schema"] == "r02_lai_expanded_alt_ref_case_v1" and
            receipt["status"] == "PREPARED_NO_INFERENCE" and
            receipt["task_id"] == case["task_id"] and receipt["role"] == case["role"] and
            receipt["chromosome"] == case["chromosome"], "Published receipt identity/role differs")
    require(receipt.get("source_minor_definition") == "DNABR2723_FIXED" and
            receipt.get("source_minor_recalculated") is False and receipt.get("truth_read") is False and
            receipt.get("evaluate_opened") is False and receipt.get("inference_executed") is False and
            receipt.get("training_executed") is False and receipt.get("genotypes_modified") is False and
            receipt.get("phase_modified") is False and receipt.get("missingness_modified") is False,
            "Published receipt biological/scope invariants differ")
    require(receipt.get("input_descriptors") == case["inputs"] and
            receipt.get("catalogue_sources", {}).get("old_event") == case["old_event"] and
            receipt.get("catalogue_sources", {}).get("old_common") == case["old_common"] and
            receipt.get("catalogue_sources", {}).get("metadata") == case["metadata_sources"],
            "Published receipt source descriptors differ")
    producer = receipt.get("producer_code_sha256") or legacy_first_sha
    require(producer == sha256(Path(module.__file__)), "Published producer code hash differs")
    require(receipt.get("plan_sha256") == plan_sha, "Published receipt plan hash differs")
    outputs = receipt["outputs_sha256"]
    if not isinstance(outputs, dict) or not set(outputs) <= REQUIRED_OUTPUTS:
        raise RuntimeError("Published receipt output axis differs")
    if set(outputs) != REQUIRED_OUTPUTS:
        raise PublicationPending("receipt.json does not yet list exactly nine outputs")


def check_published_output_files(case_dir: Path, receipt: dict) -> None:
    for relative, expected in receipt["outputs_sha256"].items():
        path = (case_dir / relative).resolve()
        require(path.is_relative_to(case_dir.resolve()), "Published output path escapes case directory")
        if not path.is_file():
            raise PublicationPending(f"output is not published yet: {relative}")
        if sha256(path) != expected:
            raise PublicationPending(f"output is still partial or differs: {relative}")


def reconcile_case(module, plan_path: Path, plan_sha: str, case: dict,
                   case_dir: Path, legacy_first_sha: str,
                   checked_receipts: dict[str, str]) -> dict | None:
    top_receipt = case_dir / "receipt.json"
    nested = case_dir / "case-output"
    legacy = legacy_first_sha if case["case_id"].endswith("chr1.FOLD_A.r1") else None
    if top_receipt.is_file():
        verified = module.validate_case(plan_path, plan_sha, case, case_dir, legacy)
        return {"task_id": case["task_id"], "layout": "CANONICAL", "verification": verified}
    nested_receipt = nested / "receipt.json"
    if not nested_receipt.is_file():
        return None

    receipt, receipt_sha = load_stable_receipt(nested_receipt)
    prior_sha = checked_receipts.get(case["task_id"])
    if prior_sha is None:
        validate_receipt_identity(module, case, receipt, plan_sha, legacy)
        # Source incompatibility is never a publication wait.
        module.validate_case_sources(case)
        checked_receipts[case["task_id"]] = receipt_sha
    else:
        require(receipt_sha == prior_sha, "Published receipt changed after identity/source validation")
    check_published_output_files(nested, receipt)
    # Frozen validation repeats source and output authentication before links.
    nested_verified = module.validate_case(plan_path, plan_sha, case, nested, legacy)
    links = expose_canonical_layout(case_dir, receipt)
    canonical_verified = module.validate_case(plan_path, plan_sha, case, case_dir, legacy)
    amendment = {
        "schema": "r02_lai_published_layout_amendment_v1",
        "status": "NESTED_OUTPUTS_AUTHENTICATED_AND_CANONICAL_LINKS_EXPOSED",
        "task_id": case["task_id"],
        "plan_sha256": plan_sha,
        "nested_receipt": {"path": str(nested_receipt.resolve()), "sha256": sha256(nested_receipt)},
        "nested_verification": nested_verified,
        "canonical_verification": canonical_verified,
        "canonical_links": links,
        "verified_output_files_before_links": len(receipt["outputs_sha256"]),
        "source_minor_definition": "DNABR2723_FIXED",
        "truth_read": False,
        "evaluate_opened": False,
        "inference_executed": False,
        "training_executed": False,
    }
    amendment_path = case_dir / "LAYOUT_AMENDMENT.json"
    if amendment_path.exists():
        if json.loads(amendment_path.read_text()) != amendment:
            raise RuntimeError(f"Existing layout amendment differs: {amendment_path}")
    else:
        write_json(amendment_path, amendment)
    return {"task_id": case["task_id"], "layout": "AUTHENTICATED_NESTED_WITH_CANONICAL_LINKS",
            "verification": canonical_verified, "amendment_sha256": sha256(amendment_path)}


def scan(args, module, plan: dict, known: dict[str, dict] | None = None,
         checked_receipts: dict[str, str] | None = None,
         publication_waits: dict[str, dict] | None = None) -> tuple[list[dict], list[str], list[dict]]:
    known = known if known is not None else {}
    checked_receipts = checked_receipts if checked_receipts is not None else {}
    publication_waits = publication_waits if publication_waits is not None else {}
    verified, pending, publishing = [], [], []
    for case in plan["cases"]:
        if case["task_id"] in known:
            verified.append(known[case["task_id"]])
            continue
        case_dir = args.results / case["task_id"]
        if not case_dir.exists():
            pending.append(case["task_id"])
            continue
        try:
            result = reconcile_case(module, args.plan, args.plan_sha256, case, case_dir,
                                    args.legacy_first_code_sha, checked_receipts)
        except PublicationPending as error:
            now = time.monotonic()
            wait = publication_waits.setdefault(case["task_id"], {"first_seen": now})
            wait["reason"] = str(error); wait["elapsed_seconds"] = now - wait["first_seen"]
            if wait["elapsed_seconds"] >= args.publication_wait_seconds:
                raise RuntimeError(f"Publication wait expired for {case['task_id']}: {error}") from error
            pending.append(case["task_id"])
            publishing.append({"task_id": case["task_id"], "reason": str(error),
                               "elapsed_seconds": wait["elapsed_seconds"],
                               "limit_seconds": args.publication_wait_seconds})
            continue
        if result is None:
            pending.append(case["task_id"])
        else:
            verified.append(result)
            known[case["task_id"]] = result
            publication_waits.pop(case["task_id"], None)
    return verified, pending, publishing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--legacy-first-code-sha", required=True)
    parser.add_argument("--watch-seconds", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--publication-wait-seconds", type=int, default=300)
    args = parser.parse_args()
    args.producer = args.producer.resolve(); args.plan = args.plan.resolve(); args.results = args.results.resolve()
    started = time.monotonic()
    try:
        if sha256(args.producer) != args.producer_sha256:
            raise RuntimeError("Frozen producer hash differs")
        if sha256(args.plan) != args.plan_sha256:
            raise RuntimeError("Frozen plan hash differs")
        module = load_module(args.producer)
        plan = json.loads(args.plan.read_text())
        known: dict[str, dict] = {}
        checked_receipts: dict[str, str] = {}
        publication_waits: dict[str, dict] = {}
        while True:
            # Hash each completed publication once while waiting.  The final
            # campaign validation reopens all 12 sources and outputs together.
            verified, pending, publishing = scan(
                args, module, plan, known, checked_receipts, publication_waits)
            state = {
                "schema": "r02_lai_layout_reconciliation_service_v1",
                "status": "WAITING_FOR_PUBLISHED_CASES" if pending else "TWELVE_CANONICAL_CASES_VERIFIED",
                "plan_sha256": args.plan_sha256,
                "verified_cases": len(verified),
                "pending_cases": len(pending),
                "pending_task_ids": pending,
                "publishing_cases": publishing,
                "case_layouts": verified,
                "truth_read": False,
                "evaluate_opened": False,
                "inference_executed": False,
                "training_executed": False,
            }
            write_json(args.state, state)
            if not pending:
                candidate = args.final.with_name(args.final.name + f".candidate.{os.getpid()}")
                module.campaign_status(SimpleNamespace(
                    plan=args.plan, plan_sha256=args.plan_sha256, results=args.results,
                    output=candidate, legacy_first_code_sha=args.legacy_first_code_sha,
                    require_complete=True))
                if args.final.exists():
                    require(json.loads(args.final.read_text()) == json.loads(candidate.read_text()),
                            f"Pre-existing final verification differs; candidate preserved at {candidate}")
                else:
                    try:
                        os.link(candidate, args.final)
                    except FileExistsError:
                        require(json.loads(args.final.read_text()) == json.loads(candidate.read_text()),
                                f"Concurrent final verification differs; candidate preserved at {candidate}")
                state["final_verification"] = {"path": str(args.final), "sha256": sha256(args.final)}
                state["candidate_verification"] = {"path": str(candidate), "sha256": sha256(candidate)}
                write_json(args.state, state)
                return
            if args.watch_seconds == 0 or time.monotonic() - started >= args.watch_seconds:
                if args.watch_seconds:
                    state["status"] = "TIMED_OUT_WITH_PENDING_CASES"
                    write_json(args.state, state)
                    raise RuntimeError(f"Timed out with {len(pending)} pending cases")
                print(json.dumps(state, sort_keys=True))
                return
            time.sleep(args.poll_seconds)
    except Exception as error:
        failure = {
            "schema": "r02_lai_layout_reconciliation_service_v1",
            "status": "FAILED",
            "error": f"{type(error).__name__}: {error}",
            "plan_sha256": args.plan_sha256,
            "truth_read": False,
            "evaluate_opened": False,
            "inference_executed": False,
            "training_executed": False,
        }
        write_json(args.state, failure)
        raise


if __name__ == "__main__":
    main()
