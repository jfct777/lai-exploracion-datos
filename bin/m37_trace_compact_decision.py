#!/usr/bin/env python3
"""Apply the prespecified M37 R0 gate plus non-worsening safety guardrails."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from m37_trace_collect_metrics import ARMS
from m37_trace_core import require
from m37_trace_successive_halving import COMPARATORS, promote


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _metric(metric: dict[str, Any], name: str) -> float:
    if name in {"brier", "macro_ancestry_dose_mae"}:
        return float(metric[name])
    if name == "NAM_ancestry_dose_mae":
        return float(metric["ancestry_dose_mae"]["NAM"])
    require(name == "false_transitions_per_morgan",
            "M37 compact guardrail name differs")
    return float(metric[name])


def _validate_person_rows(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference_keys: list[str] | None = None
    for arm in ARMS:
        rows = arms[arm].get("per_individual")
        require(isinstance(rows, list) and rows,
                "M37 compact metrics lack per-individual sufficient statistics")
        keys = [str(row.get("sample_key_sha256", "")) for row in rows]
        require(all(keys) and len(keys) == len(set(keys)),
                "M37 compact per-individual keys differ")
        for row in rows:
            counts = row.get("boundary_counts")
            require(isinstance(counts, dict) and set(counts) == {"0.05", "0.1", "0.2", "0.5"} and
                    all(set(counts[tolerance]) == {"TP", "FP", "FN"}
                        for tolerance in counts),
                    "M37 compact per-individual boundary sufficient statistics differ")
        if reference_keys is None:
            reference_keys = keys
        else:
            require(keys == reference_keys,
                    "M37 compact TUNE people/order changed between paired arms")
    return {
        "person_count": len(reference_keys or []),
        "sample_axis_sha256": hashlib.sha256(
            "\n".join(reference_keys or []).encode("ascii")).hexdigest(),
    }


def decide(rows: list[dict[str, Any]], root: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(root == "R0" and rows, "M37 compact decision is restricted to R0")
    primary, primary_criteria = promote(
        rows, keep_fraction=1.0, boundary_tolerance_cm=0.2,
        minimum_f1_gain=0.0, maximum_log_loss_increase=0.0,
        bootstrap_seed=1103, bootstrap_draws=2000,
        minimum_replication_roots=3,
    )
    primary_by_id = {str(row["candidate_id"]): row for row in primary}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    family_by_candidate: dict[str, str] = {}
    run_by_candidate: dict[str, str] = {}
    for row in rows:
        candidate, arm = str(row.get("candidate_id", "")), str(row.get("arm", ""))
        metric = row.get("metrics")
        require(candidate and arm in ARMS and isinstance(metric, dict) and
                str(row.get("root")) == root and metric.get("evaluation_split") == "FIT_TUNE",
                "M37 compact decision row differs")
        require(metric.get("root") == root and metric.get("candidate_id") == candidate and
                metric.get("arm") == arm and metric.get("run_id"),
                "M37 compact metric identity differs")
        grouped[candidate][arm] = metric
        family_by_candidate.setdefault(candidate, str(row.get("family", "")))
        run_by_candidate.setdefault(candidate, str(metric["run_id"]))
        require(family_by_candidate[candidate] == str(row.get("family", "")) and
                run_by_candidate[candidate] == str(metric["run_id"]),
                "M37 compact candidate family/run changed between arms")

    decisions: list[dict[str, Any]] = []
    guardrail_names = (
        "brier", "macro_ancestry_dose_mae", "NAM_ancestry_dose_mae",
        "false_transitions_per_morgan",
    )
    for candidate in sorted(grouped):
        arms = grouped[candidate]
        require(set(arms) == set(ARMS), "M37 compact decision needs all five paired arms")
        person_audit = _validate_person_rows(arms)
        re_metric = arms["RE"]
        comparators = {"F0": re_metric["baseline"], **{
            arm: arms[arm] for arm in ARMS if arm != "RE"
        }}
        guardrail_deltas: dict[str, dict[str, float]] = {}
        guardrail_pass: dict[str, bool] = {}
        for metric_name in guardrail_names:
            deltas = {
                name: _metric(re_metric, metric_name) - _metric(metric, metric_name)
                for name, metric in comparators.items()
            }
            guardrail_deltas[metric_name] = deltas
            guardrail_pass[metric_name] = all(value <= 0.0 for value in deltas.values())
        primary_row = primary_by_id[candidate]
        primary_pass = bool(primary_row.get("promote"))
        passed = primary_pass and all(guardrail_pass.values())
        decisions.append({
            "candidate_id": candidate,
            "family": family_by_candidate[candidate],
            "run_id": run_by_candidate[candidate],
            "root": root,
            "status": "ADVANCE_EXPLORATORY" if passed else "STOP_EXPLORATORY",
            "scientific_closure": False,
            "primary_gate_pass": primary_pass,
            "primary_gate": primary_row,
            "guardrail_pass": guardrail_pass,
            "RE_minus_comparator_guardrails": guardrail_deltas,
            "per_individual_audit": person_audit,
            "pareto_only_promotion": "FORBIDDEN",
        })
    criteria = {
        "root_scope": "R0_FIT_TUNE_ONLY",
        "allowed_status": ["ADVANCE_EXPLORATORY", "STOP_EXPLORATORY"],
        "primary": primary_criteria,
        "guardrails": {
            "direction": "RE minus comparator must be <= 0 for every comparator",
            "metrics": list(guardrail_names),
        },
        "scientific_closure": "FORBIDDEN",
        "pareto_only_promotion": "FORBIDDEN",
    }
    return decisions, criteria


def apply_positive_control_gate(
    decisions: list[dict[str, Any]], criteria: dict[str, Any],
    control_by_family: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Allow scoring with an underpowered control, but never advancement."""
    require(control_by_family.get("hmm", {}).get("status") ==
            "PASS_ADDITIVE_DETECTABILITY",
            "M37 compact HMM detectability control did not pass")
    tcn_control = control_by_family.get("tcn", {})
    tcn_additive = tcn_control.get("additive", {}).get("status")
    tcn_xor = tcn_control.get("xor_interaction", {}).get("status")
    criteria["positive_control_precondition"] = {
        "hmm": "PASS_ADDITIVE_DETECTABILITY",
        "tcn": "additive PASS and nearby-XOR PASS",
        "incomplete_action": "STOP_EXPLORATORY",
    }
    for row in decisions:
        row["same_budget_control"] = control_by_family[row["family"]]
        row["same_budget_control_pass"] = (
            row["family"] == "hmm" or (tcn_additive == "PASS" and tcn_xor == "PASS")
        )
        if row["family"] == "tcn" and (tcn_additive != "PASS" or tcn_xor != "PASS"):
            row["status"] = "STOP_EXPLORATORY"
            row["budget_assessment"] = "BUDGET_INSUFFICIENT_FOR_REQUIRED_CONTROLS"
            row["interpretation"] = (
                "At least one 200-update held-out control did not pass; this candidate "
                "cannot advance or support a biological absence claim."
            )
        if row["family"] == "tcn" and tcn_xor != "PASS":
            row["interaction_assessment"] = "BUDGET_INSUFFICIENT_FOR_INTERACTION"
            row["interaction_absence_claim"] = "FORBIDDEN"
    return decisions, criteria


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--metrics-receipt", type=Path, required=True)
    parser.add_argument("--family-audit", action="append", type=Path, required=True)
    parser.add_argument("--family-audit-receipt", action="append", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--auth-file", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite M37 compact triage decision")
    collection, receipt = _json(args.metrics_json), _json(args.metrics_receipt)
    rows = collection.get("rows")
    require(collection.get("stage") == "M37_TRACE_COLLECT_METRICS" and
            collection.get("root") == args.root and collection.get("evaluation_split") == "FIT_TUNE" and
            isinstance(rows, list) and rows and receipt.get("stage") == "M37_TRACE_COLLECT_METRICS" and
            receipt.get("root") == args.root and receipt.get("row_count") == len(rows) and
            receipt.get("output_sha256") == sha256(args.metrics_json),
            "M37 compact collection/receipt differs")
    family_audits = [_json(path) for path in args.family_audit]
    family_audit_receipts = [_json(path) for path in args.family_audit_receipt]
    require({row.get("family") for row in family_audits} == {"hmm", "tcn"} and
            len(family_audit_receipts) == len(family_audits) and
            all(row.get("stage") == "M37_TRACE_COMPACT_SWEEP" and
                row.get("status") == "PASS_FIT_TUNE_ONLY" and
                row.get("root") == args.root and row.get("run_id") == args.run_id
                for row in family_audits),
            "M37 compact family audits differ")
    container_digests = {str(row.get("container_digest", "")) for row in family_audits}
    require(len(container_digests) == 1 and "@sha256:" in next(iter(container_digests)),
            "M37 compact family container digest differs")
    # Prevent mixing two technically valid family bundles produced from
    # different PRE contracts, inputs, controls, or run overlays.
    shared_evidence_fields = (
        "manifest_sha256", "parent_contract_sha256", "contract_amendment_sha256",
        "positive_control_sha256", "positive_control_receipt_sha256",
        "canonical_metrics_sha256", "canonical_metrics_receipt_sha256",
        "truth_sha256", "baseline_provenance", "run_overlay", "feature_evidence",
        "container_digest",
    )
    for field in shared_evidence_fields:
        encoded = {
            json.dumps(row.get(field), sort_keys=True, separators=(",", ":"))
            for row in family_audits
        }
        require(len(encoded) == 1 and next(iter(encoded)) not in ("null", '""'),
                f"M37 compact family {field} differs")
    baseline_provenance = family_audits[0]["baseline_provenance"]
    require(isinstance(baseline_provenance, dict) and
            baseline_provenance.get("method") == "FLARE" and
            baseline_provenance.get("upstream_stage") == "M34_PARSE_FLARE_F0" and
            baseline_provenance.get("truth_blind") is True,
            "M37 compact baseline provenance is not truth-blind FLARE")
    for row in rows:
        metadata = row.get("metrics", {}).get("baseline_metadata")
        require(isinstance(metadata, dict) and
                metadata.get("method") == baseline_provenance.get("method") and
                metadata.get("source_sha256") == baseline_provenance.get("source_sha256") and
                metadata.get("upstream_stage") == baseline_provenance.get("upstream_stage") and
                metadata.get("upstream_receipt_sha256") ==
                baseline_provenance.get("upstream_receipt_sha256"),
                "M37 compact metric/F0 provenance binding differs")
    receipt_by_family = {str(row.get("family", "")): row for row in family_audit_receipts}
    require(set(receipt_by_family) == {"hmm", "tcn"},
            "M37 compact family audit receipt identities differ")
    for path, audit in zip(args.family_audit, family_audits):
        audit_receipt = receipt_by_family[str(audit["family"])]
        require(audit_receipt.get("stage") == "M37_TRACE_COMPACT_SWEEP" and
                audit_receipt.get("run_id") == args.run_id and
                audit_receipt.get("root") == args.root and
                audit_receipt.get("container_digest") in container_digests and
                audit_receipt.get("output_sha256") == sha256(path),
                "M37 compact family audit/receipt hash differs")
    collection_evidence = collection.get("input_evidence")
    require(isinstance(collection_evidence, dict) and
            receipt.get("input_evidence") == collection_evidence,
            "M37 compact collection evidence differs from its receipt")
    audit_metric_sha: dict[str, str] = {}
    audit_metric_receipt_sha: dict[str, str] = {}
    for audit in family_audits:
        for observed, source in ((audit_metric_sha, audit.get("metric_sha256")),
                                 (audit_metric_receipt_sha, audit.get("metric_receipt_sha256"))):
            require(isinstance(source, dict) and not (set(observed) & set(source)),
                    "M37 compact family audit evidence overlaps")
            observed.update({str(key): str(value) for key, value in source.items()})
    require(audit_metric_sha == collection_evidence.get("metric_sha256") and
            audit_metric_receipt_sha == collection_evidence.get("score_receipt_sha256"),
            "M37 compact collection is not the exact union of family audit evidence")
    decisions, criteria = decide(rows, args.root)
    require(all(row["run_id"] == args.run_id for row in decisions),
            "M37 compact decision run_id differs")
    control_by_family = {
        str(row["family"]): row.get("positive_control_status") for row in family_audits
    }
    decisions, criteria = apply_positive_control_gate(
        decisions, criteria, control_by_family,
    )
    authenticated_sources = {path.name: sha256(path) for path in args.auth_file}
    require(len(authenticated_sources) == len(args.auth_file) and
            {"m37_trace_compact_decision.py", "m37_trace_collect_metrics.py",
             "m37_trace_successive_halving.py"} <= set(authenticated_sources),
            "M37 compact decision authenticated source set differs")
    payload = {
        "schema_version": "1.0.0",
        "stage": "M37_TRACE_COMPACT_TRIAGE",
        "status": ("ADVANCE_EXPLORATORY" if any(row["status"] == "ADVANCE_EXPLORATORY"
                                                   for row in decisions)
                   else "STOP_EXPLORATORY"),
        "run_id": args.run_id,
        "root": args.root,
        "criteria": criteria,
        "container_digest": next(iter(container_digests)),
        "decisions": decisions,
        "positive_control_status": control_by_family,
        "metrics_collection_sha256": sha256(args.metrics_json),
        "metrics_collection_receipt_sha256": sha256(args.metrics_receipt),
        "family_audit_sha256": {path.name: sha256(path) for path in args.family_audit},
        "family_audit_receipt_sha256": {
            path.name: sha256(path) for path in args.family_audit_receipt
        },
        "authenticated_source_sha256": authenticated_sources,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps({
        "schema_version": "1.0.0", "stage": "M37_TRACE_COMPACT_TRIAGE",
        "run_id": args.run_id, "root": args.root,
        "container_digest": next(iter(container_digests)),
        "authenticated_source_sha256": authenticated_sources,
        "status": payload["status"], "candidate_count": len(decisions),
        "output_sha256": sha256(args.output),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
