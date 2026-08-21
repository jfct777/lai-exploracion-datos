#!/usr/bin/env python3
"""Frozen-contract validation for the synthetic M32 locus-sequence smoke."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


EXPECTED_RADII = (0.05, 0.10, 0.20, 0.50)
EXPECTED_STATUS = "CONTRACT_AND_SYNTHETIC_SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_git_commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("git_commit must be an exact lowercase 40-character hexadecimal commit")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_all_true(mapping: dict[str, Any], keys: set[str], label: str) -> None:
    _require(keys.issubset(mapping), f"{label} lacks required fields")
    _require(all(mapping[key] is True for key in keys), f"{label} must freeze all capacity invariants")


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    _require(contract.get("stage") == "M32_LOCUS_SEQUENCE_SMOKE", "unsupported M32 stage")
    _require(contract.get("version") == 1, "unsupported M32 contract version")
    _require(contract.get("status") == EXPECTED_STATUS, "contract is not smoke-only")
    _require(contract.get("chromosome") == "chr22", "M32 smoke is restricted to chr22")
    _require(contract.get("allele_orientation") == "FREQ_defined_minor_allele", "minor-allele orientation is not frozen")

    representation = contract["representation"]
    _require(representation.get("preserve_each_locus") is True, "individual loci must be preserved")
    _require(representation.get("preserve_genetic_order") is True, "locus order must be preserved")
    _require(representation.get("full_flare_grid") is True, "the full FLARE grid must be retained")
    primary = representation["primary"]
    _require(primary.get("phase_invariant") is True, "primary representation must be phase-invariant")
    _require(primary.get("missing_value_policy") == "explicit_mask_never_encode_as_zero", "missingness policy is unsafe")
    upper = representation["upper_bound"]
    _require(upper.get("simulation_only") is True and upper.get("transfer_candidate") is False, "H must remain a simulation-only upper bound")

    screen = contract["context_occupancy_screen"]
    radii = tuple(float(value) for value in screen["radii_cm"])
    _require(len(radii) == len(EXPECTED_RADII), "occupancy radii differ from PRE")
    _require(all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(radii, EXPECTED_RADII)), "occupancy radii differ from PRE")
    _require(screen.get("definition") == "symmetric_radius_cm_around_each_flare_marker", "radius semantics are ambiguous")
    _require(screen.get("selection_allowed") is False and screen.get("uses_truth") is False, "smoke may not select a radius or use truth")

    controls = contract["controls"]
    _require(controls.get("frozen_baseline") == "FLARE_F0", "baseline is not frozen")
    capacity_keys = {"same_architecture", "same_parameter_count", "same_optimizer", "same_budget", "same_receptive_field"}
    _require_all_true(controls["rare_channel_disabled_ablation"], capacity_keys | {"only_rare_channel_disabled"}, "disabled-channel ablation")
    common = controls["matched_common_locus_control"]
    _require_all_true(common, capacity_keys | {"same_tensor_slots", "multiple_matched_sets_required", "scientific_pilot_blocked_until_frozen"}, "matched-common control")
    _require(common.get("construction_frozen") is False, "matched-common construction may not be falsely marked frozen")
    _require(set(common.get("matching_variables", [])) == {"genetic_position", "callability", "local_locus_density"}, "matched-common variables differ from PRE")

    sham = controls["primary_sham"]
    _require(sham.get("operation") == "permute_complete_diploid_REF_LAI_ancestry_labels", "primary sham is not label permutation by diploid person")
    expected_preserved = {
        "reference_genotype_matrices",
        "reference_missingness",
        "reference_LD_and_locus_order",
        "ancestry_group_sizes",
        "target_data",
    }
    _require(expected_preserved.issubset(set(sham.get("preserves", []))), "primary sham lacks required invariants")
    _require(sham.get("breaks_only") == "reference_genotype_to_ancestry_label_association", "primary sham changes the wrong estimand")
    _require(controls.get("phase_diagnostic") == "swap_h0_h1_at_heterozygous_target_loci_preserving_diploid_dosage", "phase diagnostic differs from PRE")

    split = contract["future_split_contract"]
    _require(split.get("independent_unit") == "complete_diploid_individual", "independent unit is invalid")
    _require(split.get("root17_root18") == "consumed_known_answer_only", "consumed roots are not sealed")
    truth_access = split.get("truth_access", {})
    expected_truth = {
        "tensor_producer": "never",
        "TRAIN": "supervised_fit_and_inner_train_selection_only",
        "CAL": "calibration_after_model_freeze_only",
        "DEV": "sealed_scorer_after_prediction_manifest_only",
        "EVAL": "sealed_final_scorer_after_prediction_manifest_only",
    }
    _require(truth_access == expected_truth, "truth access by role is unsafe or incomplete")
    _require(set(split.get("role_disjunction_required", [])) == {"simulation_root", "donor", "family", "IBD_component"}, "role disjunction is incomplete")
    _require(split.get("prospective_root_count_frozen") is False, "root count is not yet scientifically frozen")
    _require(split.get("scientific_pilot_blocked_until_root_count_frozen") is True, "pilot must remain blocked until root count is frozen")

    model = contract["future_model_screen"]
    _require(model.get("models") == ["residual_linear_local", "small_residual_cnn_1d"], "model family differs from PRE")
    _require(model.get("transformers_or_diffusion") is False, "large models are outside this gate")
    _require(model.get("maximum_cnn_configurations") == 8 and model.get("initial_training_seeds") == 3, "future screen bounds differ from PRE")
    _require(model.get("hyperparameters_frozen") is False and model.get("freeze_only_after_occupancy_and_memory_screen") is True, "hyperparameter freeze state is invalid")

    metrics = contract["future_metrics"]
    _require(metrics.get("primary") == "paired_boundary_F1_at_0.2_cM", "primary metric differs from PRE")
    _require(math.isclose(float(metrics.get("minimum_relevant_delta_F1", -1)), 0.01, abs_tol=1e-12), "SESOI differs from PRE")
    _require(set(metrics.get("required_contrasts", [])) == {"F0", "rare_channel_disabled", "matched_common_loci", "REF_label_sham"}, "required contrasts are incomplete")
    _require(metrics.get("bootstrap_unit") == "complete_diploid_individual", "bootstrap unit is invalid")
    _require(metrics.get("bootstrap_method") == "resample_people_then_recompute_TP_FP_FN_and_metrics", "bootstrap method is invalid")
    _require(metrics.get("no_retuning_after_DEV") is True, "post-DEV retuning is not prohibited")
    _require(metrics.get("scientific_run_authorized") is False, "scientific execution is not authorized")
    return contract
