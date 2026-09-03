#!/usr/bin/env python3
"""Combine prespecified M38B gates without selecting a model family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


class M38BDecisionError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BDecisionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_family(path: Path, receipt_path: Path, family: str) -> tuple[dict, tuple[str, ...]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        result.get("stage") == "M38B_OOF_SCORE" and result.get("status") == "PASS_SCORED"
        and receipt.get("stage") == "M38B_OOF_SCORE" and receipt.get("status") == "PASS_SCORED"
        and result.get("family") == family and receipt.get("family") == family
        and receipt.get("output_sha256") == sha256(path)
        and receipt.get("arms") == ["RD", "RE", "SHAM", "full", "minus"],
        f"M38B {family} score receipt differs",
    )
    provenance = tuple(str(receipt.get(name, "")) for name in (
        "model_contract_receipt_sha256", "base_contract_sha256", "amendment_sha256",
        "amendment_2_sha256", "folds_sha256",
        "folds_receipt_sha256",
    ))
    return result, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for family in ("analytic", "tcn"):
        parser.add_argument(f"--{family}", type=Path, required=True)
        parser.add_argument(f"--{family}-receipt", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--positive-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analytic, analytic_provenance = load_family(args.analytic, args.analytic_receipt, "analytic")
    tcn, tcn_provenance = load_family(args.tcn, args.tcn_receipt, "tcn")
    positive = json.loads(args.positive.read_text(encoding="utf-8"))
    positive_receipt = json.loads(args.positive_receipt.read_text(encoding="utf-8"))
    require(
        positive.get("stage") == "M38B_SCORE_DIAGNOSTIC_POSITIVE_CONTROL"
        and positive.get("family") == "tcn"
        and positive.get("logical_ids") == ["POS_d0", "POS_d0p25", "POS_d0p5", "POS_d1", "POS_d2"]
        and positive_receipt.get("stage") == positive.get("stage")
        and positive_receipt.get("status") == "PASS_DIAGNOSTIC_GRID_SCORED"
        and positive_receipt.get("diagnostic_only") is True
        and positive_receipt.get("family") == "tcn"
        and positive_receipt.get("logical_ids") == positive.get("logical_ids")
        and positive_receipt.get("output_sha256") == sha256(args.positive),
        "M38B positive-control score receipt differs",
    )
    positive_provenance = tuple(str(positive_receipt.get(name, "")) for name in (
        "model_contract_receipt_sha256", "base_contract_sha256", "amendment_sha256",
        "amendment_2_sha256", "folds_sha256",
        "folds_receipt_sha256",
    ))
    require(len(set((analytic_provenance, tcn_provenance, positive_provenance))) == 1
            and all(len(value) == 64 for value in analytic_provenance),
            "M38B final gates do not share provenance/folds")
    decisions = {}
    for family, result in (("analytic", analytic), ("tcn", tcn)):
        incremental = bool(result["candidate_incremental_gate"]["pass"])
        secondary = result["secondary_gates"]
        capacity = True if family == "analytic" else bool(positive["capacity_gate"]["pass"])
        supported = bool(
            incremental and capacity
            and secondary["weighted_uniform_no_sign_reversal"]["pass"]
            and secondary["no_statistically_clear_harm"]["pass"]
        )
        deploy = bool(
            supported and secondary["deploy_improvement_over_full_flare"]["pass"]
            and secondary["no_statistically_clear_harm_vs_full"]["pass"]
        )
        decisions[family] = {
            "incremental_information_supported": supported,
            "improvement_over_full_flare_supported": deploy,
            "capacity_gate": "NOT_APPLICABLE_EXPLICIT_ANALYTIC_TRANSFORM" if family == "analytic" else capacity,
            "family_selected": False,
            "status": ("SUPPORTED" if supported else
                       "CAPACITY_INCONCLUSIVE" if family == "tcn" and not capacity else "NOT_SUPPORTED"),
        }
    document = {
        "schema_version": "1.0.0", "stage": "M38B_FINAL_PRESPECIFIED_DECISION",
        "status": "PASS_GATES_EVALUATED_NO_FAMILY_SELECTION",
        "families": decisions,
        "any_incremental_candidate_supported": any(
            row["incremental_information_supported"] for row in decisions.values()
        ),
        "any_improvement_over_full_flare_supported": any(
            row["improvement_over_full_flare_supported"] for row in decisions.values()
        ),
        "full_minus_scope": "ALL_S660_LOCI",
        "trace_tcn_scope": "S_STAR_123_LOO_STABLE_LOCI",
        "claim_scope": "EXPLORATORY_CHR22_R0_FIT_DONOR_CONDITIONAL",
        "provenance": dict(zip(("model_contract_receipt_sha256", "base_contract_sha256",
                                "amendment_sha256", "amendment_2_sha256", "folds_sha256",
                                "folds_receipt_sha256"),
                               analytic_provenance, strict=True)),
    }
    require(not args.output.exists(), "refusing to overwrite M38B final decision")
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = args.output.with_suffix(".receipt.json")
    receipt.write_text(json.dumps({
        "schema_version": "1.0.0", "stage": document["stage"], "status": document["status"],
        "output_sha256": sha256(args.output), **document["provenance"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": document["status"],
                      "any_supported": document["any_incremental_candidate_supported"]}, sort_keys=True))


if __name__ == "__main__":
    main()
