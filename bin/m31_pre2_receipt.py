#!/usr/bin/env python3
"""Hash-bound root17 gate receipt for the M31 PRE2 one-way evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import run_m31_ordered_linear as runner


BINDING_KEYS = {
    "contract_sha256",
    "git_commit",
    "runner_sha256",
    "core_sha256",
    "container_digest",
    "prediction_manifest_sha256",
    "context_sha256",
    "root17_metrics_sha256",
    "technical_evidence_sha256",
    "receipt_code_sha256",
}
RECEIPT_KEYS = {
    "schema_version",
    "experiment_id",
    "stage",
    "binding",
    "checkpoint_guarded",
    "technical_requirements",
    "root17_metrics",
    "decision",
    "claims_excluded",
    "receipt_semantic_sha256",
}
EXPECTED_CLAIMS_EXCLUDED = (
    "confirmatory_validation",
    "independent_replication",
    "rare_support_mechanism_without_DSHAM",
    "haplotype_specific_rare_effect",
    "phase_resolved_boundary_mechanism",
    "ancestry_label_mapping_effect_without_DSHAM",
    "capacity_controlled_increment_without_capacity_matched_control",
    "ancestry_specific_boundary_improvement",
    "multiplicity_controlled_significance",
    "family_wise_error_control",
    "p_value_claims",
    "DNABR_generalization",
    "Native_American_LAI",
    "Brazil_novel_variant_effect",
    "deep_learning_benefit",
)


class ReceiptError(ValueError):
    """Raised when a PRE2 gate receipt is incomplete, altered or closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        runner.json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str) and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalize_binding(binding: Mapping[str, Any]) -> dict[str, str]:
    require(set(binding) == BINDING_KEYS, "PRE2 receipt binding fields differ")
    normalized = {key: str(value) for key, value in binding.items()}
    for key in BINDING_KEYS - {"git_commit", "container_digest"}:
        require(_valid_hex(normalized[key], 64), f"PRE2 receipt {key} is not SHA-256")
    require(_valid_hex(normalized["git_commit"], 40), "PRE2 receipt git commit is invalid")
    require(
        normalized["container_digest"].startswith("sha256:")
        and _valid_hex(normalized["container_digest"][7:], 64),
        "PRE2 receipt container digest is invalid",
    )
    return dict(sorted(normalized.items()))


def _derive_checkpoint_guarded(checkpoint_fits: Mapping[str, Any]) -> dict[str, bool]:
    require(set(checkpoint_fits) == {"L", "D"}, "PRE2 receipt requires L and D checkpoint fits")
    guarded = {}
    for arm in ("L", "D"):
        fit = checkpoint_fits[arm]
        runner._validate_fit_checkpoint(arm, fit)
        require(isinstance(fit["guarded"], bool), f"PRE2 checkpoint guarded flag invalid for {arm}")
        guarded[arm] = fit["guarded"]
    return guarded


def _normalize_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    require(set(metrics) == {"F0", "L", "D"}, "PRE2 receipt metrics require exactly F0, L and D")
    normalized = {}
    for arm in ("F0", "L", "D"):
        require(set(metrics[arm]) == set(runner.PRE2_GATE_METRIC_NAMES),
                f"PRE2 receipt metric fields differ for {arm}")
        normalized[arm] = {
            name: runner._finite_metric(metrics[arm], name, arm)
            for name in runner.PRE2_GATE_METRIC_NAMES
        }
    return normalized


def _reconcile_checkpoint_metrics(
    metrics: Mapping[str, Mapping[str, float]], checkpoint_fits: Mapping[str, Any],
) -> None:
    """Bind the gate's global OOF metrics to the authenticated selected checkpoints."""
    f0_expected = None
    for arm in ("L", "D"):
        fit = checkpoint_fits[arm]
        observed = {
            "boundary_f1_0.2cM": float(fit["cv_boundary_f1_0.2cM"]),
            "false_transitions_per_cM_0.2cM": float(
                fit["cv_false_transitions_per_cM_0.2cM"]
            ),
            "macro_ancestry_dose_mae": float(fit["cv_macro_ancestry_dose_mae"]),
        }
        require(
            all(metrics[arm][name] == value for name, value in observed.items()),
            f"PRE2 root17 metrics do not match authenticated {arm} checkpoint",
        )
        current_f0 = {
            name: float(fit["f0_cv_metrics"][name])
            for name in (
                "boundary_f1_0.2cM",
                "false_transitions_per_cM_0.2cM",
                "macro_ancestry_dose_mae",
            )
        }
        if f0_expected is None:
            f0_expected = current_f0
        require(current_f0 == f0_expected, "PRE2 L/D checkpoints disagree on F0 OOF metrics")
    require(
        all(metrics["F0"][name] == value for name, value in f0_expected.items()),
        "PRE2 root17 F0 metrics do not match authenticated checkpoints",
    )


def build_root17_gate_receipt(
    *,
    metrics: Mapping[str, Mapping[str, Any]],
    checkpoint_fits: Mapping[str, Any],
    technical_requirements: Mapping[str, bool],
    binding: Mapping[str, Any],
    claims_excluded: Sequence[str],
) -> dict[str, Any]:
    """Build a deterministic receipt from authenticated checkpoints and OOF metrics."""
    normalized_metrics = _normalize_metrics(metrics)
    guarded = _derive_checkpoint_guarded(checkpoint_fits)
    _reconcile_checkpoint_metrics(normalized_metrics, checkpoint_fits)
    require(
        set(technical_requirements) == set(runner.PRE2_TECHNICAL_REQUIREMENTS)
        and all(type(value) is bool for value in technical_requirements.values()),
        "PRE2 receipt technical requirements differ",
    )
    require(
        isinstance(claims_excluded, Sequence) and not isinstance(claims_excluded, (str, bytes))
        and bool(claims_excluded) and len(set(claims_excluded)) == len(claims_excluded)
        and all(isinstance(value, str) and value for value in claims_excluded),
        "PRE2 receipt claims_excluded are invalid",
    )
    require(tuple(claims_excluded) == EXPECTED_CLAIMS_EXCLUDED,
            "PRE2 receipt claims_excluded differ from frozen contract")
    decision = runner.evaluate_pre2_root17_gate(
        normalized_metrics,
        d_guarded=guarded["D"],
        l_guarded=guarded["L"],
        technical_requirements=technical_requirements,
    )
    body = {
        "schema_version": "2.0.0",
        "experiment_id": "M31_ORDERED_LINEAR_DEV_PRE2",
        "stage": "ROOT17_HASH_BOUND_OPEN_GATE",
        "binding": _normalize_binding(binding),
        "checkpoint_guarded": guarded,
        "technical_requirements": dict(sorted(technical_requirements.items())),
        "root17_metrics": normalized_metrics,
        "decision": decision,
        "claims_excluded": list(claims_excluded),
    }
    return {**body, "receipt_semantic_sha256": _sha256_payload(body)}


def validate_root17_gate_receipt(
    receipt: Mapping[str, Any], *, expected_binding: Mapping[str, Any],
    checkpoint_fits: Mapping[str, Any], expected_metrics: Mapping[str, Mapping[str, Any]],
    expected_technical_requirements: Mapping[str, bool],
    expected_claims_excluded: Sequence[str],
) -> dict[str, Any]:
    """Reconstruct the receipt and reject any drift before root18 truth is mounted."""
    require(isinstance(receipt, Mapping) and set(receipt) == RECEIPT_KEYS,
            "PRE2 root17 receipt fields differ")
    semantic = receipt.get("receipt_semantic_sha256")
    require(_valid_hex(semantic, 64), "PRE2 receipt semantic SHA-256 is invalid")
    body = {key: receipt[key] for key in RECEIPT_KEYS - {"receipt_semantic_sha256"}}
    require(_sha256_payload(body) == semantic, "PRE2 root17 receipt semantic SHA-256 mismatch")
    rebuilt = build_root17_gate_receipt(
        metrics=expected_metrics,
        checkpoint_fits=checkpoint_fits,
        technical_requirements=expected_technical_requirements,
        binding=expected_binding,
        claims_excluded=expected_claims_excluded,
    )
    require(dict(receipt) == rebuilt, "PRE2 root17 receipt does not reconstruct exactly")
    require(receipt["decision"]["status"] == "OPEN_ROOT18",
            "PRE2 root17 receipt does not authorize opening root18")
    return rebuilt
