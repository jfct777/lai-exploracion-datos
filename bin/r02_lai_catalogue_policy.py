"""Explicit locus coverage scope; never upgrades it to biological admission."""
from __future__ import annotations

from collections.abc import Mapping
import re


AVAILABLE_SCOPE = "FIXED_PANEL_AVAILABLE_AXIS_DEVELOPMENT"
COUNT_FIELDS = ("source_rare_total", "material_loci_total",
                "source_rare_intersect_material")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_catalogue_policy(policy: Mapping[str, object]) -> None:
    """Require an explicit, pinned alternative to legacy full-locus coverage."""
    fields = {"scope", "input_spec_sha256", "material_axis_sha256", *COUNT_FIELDS}
    _require(isinstance(policy, Mapping) and set(policy) == fields,
             "catalogue policy fields differ")
    _require(policy["scope"] == AVAILABLE_SCOPE, "unsupported catalogue scope")
    for field in ("input_spec_sha256", "material_axis_sha256"):
        _require(isinstance(policy[field], str)
                 and re.fullmatch(r"[0-9a-f]{64}", policy[field]) is not None,
                 f"catalogue policy requires a valid {field}")
    for field in COUNT_FIELDS:
        _require(type(policy[field]) is int and policy[field] > 0,
                 f"catalogue policy requires positive integer {field}")
    _require(policy["source_rare_intersect_material"] <=
             min(policy["source_rare_total"], policy["material_loci_total"]),
             "catalogue intersection exceeds its source or material")


def admit_catalogue_coverage(report: Mapping[str, object], *,
                            policy: Mapping[str, object] | None = None,
                            input_spec_sha256: str | None = None,
                            material_axis_sha256: str | None = None) -> dict:
    """Check authenticated reader output; preserve all coverage/quality flags.

    This function cannot authenticate a fabricated report. Its caller must use
    DonorInputReader and retain source, role, missingness and exclusion checks.
    Passing it establishes only the declared locus scope, not an independent
    evaluation, an observed rare allele, or scientific training authorization.
    """
    result = dict(report)
    if policy is None:
        _require(result.get("complete_catalogue_covered") is True,
                 "complete source-rare catalogue coverage is required for training")
        return result
    validate_catalogue_policy(policy)
    _require(input_spec_sha256 == policy["input_spec_sha256"],
             "catalogue policy input-spec identity differs")
    _require(material_axis_sha256 == policy["material_axis_sha256"],
             "catalogue policy material axis differs")
    _require(result.get("coverage_scope") == "LOCUS_AXIS_MEMBERSHIP_ONLY",
             "catalogue coverage scope differs")
    for field in COUNT_FIELDS:
        _require(type(result.get(field)) is int and result[field] == policy[field],
                 f"catalogue coverage count differs: {field}")
    missing = result["source_rare_total"] - result["source_rare_intersect_material"]
    _require(type(result.get("source_rare_missing_from_material")) is int
             and result["source_rare_missing_from_material"] == missing,
             "catalogue missing-locus denominator differs")
    _require(result.get("complete_catalogue_covered") is (missing == 0),
             "catalogue complete-coverage flag contradicts counts")
    result.update(training_locus_scope=AVAILABLE_SCOPE,
                  input_spec_sha256=input_spec_sha256,
                  classified_material_axis_sha256=material_axis_sha256,
                  locus_scope_check_passed=True)
    return result
