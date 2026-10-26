#!/usr/bin/env python3
"""Recoverable FIT diagnostic using the nearest COMMON marker on each side.

The counting stage visits the 730 frozen chr22 person/cell fragments.  For
each fragment it streams the complete 1,460-person FIT roster in batches of at
most eight, counts target doses conditional on the two flank doses, and then
subtracts the complete authenticated family component of each query person.
Cell choice, flank choice and family exclusion never inspect target dosage.

The scoring stage opens only completed ``full`` model checkpoints.  It scores
the neural network and the flank table on the exact same FIT rows for which the
target and two flanks are observed and the leave-one-component-out table has
positive raw support.  This is a FIT diagnostic with different fitting access;
it is not SELECT, EVALUATE or validated predictive evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

import r02_masked_training_request as training
import r02_masked_training_campaign as training_campaign
from r02_context_checkpoint import CheckpointStore
from r02_masked_genotype_controls import (
    FlankWindow, fit_flanking_genotype_counts,
    leave_out_flanking_component_counts, locus_keys,
    merge_flanking_genotype_counts, predict_from_flanking_counts,
    select_common_flanks)
from r02_masked_genotype_model import (MaskedGenotypeModel, MaskedModelConfig,
                                       build_masked_adam)
from r02_masked_genotype_update import masked_update_code_hashes
from r02_masked_store import MaskedStoreReader, role_person_ids
from r02_masked_training_campaign import CellExample, fit_cell_loader


REQUEST_SCHEMA = "r02_masked_chr22_two_common_flanks_request_v2"
POST_SCHEMA = "r02_masked_chr22_two_common_flanks_post_v2"
POST_STATUS = "ACCEPTED_FOR_FIT_TWO_COMMON_FLANKS_COUNTS_AND_PAIRED_SCORE"
COUNT_SCHEMA = "r02_masked_chr22_two_common_flanks_counts_v2"
COUNT_STATUS = "COMPLETE_730_FIT_FRAGMENTS_LEAVE_ONE_COMPONENT_OUT"
COUNT_INCOMPLETE_STATUS = "TIME_LIMIT_INCOMPLETE"
FRAGMENT_SCHEMA = "r02_masked_chr22_two_common_flanks_fragment_v2"
SCORE_SCHEMA = "r02_masked_chr22_two_common_flanks_score_v2"
SCORE_STATUS = "COMPLETE_PAIRED_FIT_DIAGNOSTIC"
READINESS_SCHEMA = "r02_masked_chr22_two_common_flanks_readiness_v1"
ALPHAS = (0.5, 1.0)
FAMILIES = training.FAMILIES


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json.dumps(value, sort_keys=True, indent=2,
                                     allow_nan=False).encode() + b"\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".",
                                               dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_control_request(value, campaign_path, campaign):
    required = {"aggregation", "alpha_primary", "alpha_sensitivity", "calendar",
        "calendar_or_cell_replacement", "campaign", "chromosome",
        "denominator_audit", "denominator_audit_source", "evaluate_opened", "fit_internal_exclusion",
        "fit_roster", "flanks", "frozen_core", "implemented_primitives", "missing_for_execution",
        "network_comparison", "resources", "role", "schema", "select_opened",
        "status", "support_mask", "target_channels"}
    require(isinstance(value, dict) and set(value) == required
            and value["schema"] == REQUEST_SCHEMA and value["role"] == "FIT"
            and value["chromosome"] == 22
            and value["alpha_primary"] == 0.5
            and value["alpha_sensitivity"] == 1.0
            and value["fit_internal_exclusion"] ==
                "LEAVE_OUT_COMPLETE_AUTHENTICATED_FAMILY_COMPONENT_OF_QUERY_PERSON"
            and value["calendar_or_cell_replacement"] is False
            and value["network_comparison"] ==
                "EXACT_SAME_TARGET_MASK;_DIAGNOSTIC_FIT_DIFFERENT_TRAINING_ACCESS"
            and value["resources"] == {"cpus": 1,
                "count_memory_bytes": 2147483648,
                "score_memory_bytes": 8589934592,
                "count_max_seconds": 72000, "score_max_seconds": 14400,
                "worker_can_run_separately": True}
            and value["select_opened"] is False
            and value["evaluate_opened"] is False,
            "two-COMMON-flank request differs from reviewed FIT design")
    require(value["flanks"] == {"channel": "common", "left": 1,
        "maximum_distance_cm": 0.2, "minimum_distance_cm": 0.0,
        "right": 1, "target_cell_fully_excluded": True,
        "tie_break": "NEAREST_CM_THEN_PHYSICAL_POSITION_NO_DOSAGE"},
        "two-COMMON-flank geometry differs")
    require(value["target_channels"] == ["common", "rare"]
            and value["support_mask"] ==
                "TWO_FLANKS_OBSERVED_AND_TARGET_OBSERVED_AND_RAW_REFERENCE_SUPPORT_POSITIVE",
            "flank comparison support differs")
    for name, expected in (("campaign", campaign_path),
                           ("calendar", campaign["calendar"]["path"]),
                           ("fit_roster", campaign["fit_roster"]["path"])):
        descriptor = value[name]
        require(Path(descriptor["path"]).resolve() == Path(expected).resolve()
                and descriptor["sha256"] == sha(expected),
                name + " descriptor differs")
    audit_source = value["denominator_audit_source"]
    audit = value["denominator_audit"]
    require(Path(audit["path"]).is_file()
            and audit["sha256"] == sha(audit["path"]),
            "denominator audit descriptor differs")
    require(Path(audit_source["path"]).is_file()
            and audit_source["sha256"] == sha(audit_source["path"]),
            "denominator audit source binding differs")
    core = value["frozen_core"]
    require(set(core) == {"archive", "manifest", "training_pre", "training_post"},
            "frozen core inventory differs")
    for descriptor in core.values():
        require(Path(descriptor["path"]).is_file()
                and descriptor["sha256"] == sha(descriptor["path"]),
                "frozen core file changed")
    return value


def validate_control_post(value, campaign_path, control_path, pre_path):
    control_root = Path(control_path).resolve().parent
    package = control_root / "masked-chr22-flanks-v2.tgz"
    manifest = control_root / "flank-package-v2/MANIFEST.json"
    require(value == {"schema": POST_SCHEMA, "status": POST_STATUS,
        "campaign_sha256": sha(campaign_path),
        "control_request_sha256": sha(control_path),
        "training_pre_sha256": sha(pre_path), "authorized_role": "FIT",
        "flank_package_sha256": sha(package),
        "flank_manifest_sha256": sha(manifest),
        "alpha": [0.5, 1.0], "select_opened": False,
        "evaluate_opened": False},
        "exact POST for FIT flank counts and paired score is required")


def validate_inputs(campaign_path, control_path, pre_path, request_paths,
                    control_post_path=None):
    campaign = read_json(campaign_path)
    roster, calendar = training.validate_campaign(campaign)
    requests = [training.validate_request(read_json(path), campaign, campaign_path)
                for path in request_paths]
    require({row["fit_id"] for row in requests} == set(training.ALL_FIT_IDS),
            "eight exact training requests are required")
    training.validate_pre(read_json(pre_path), campaign_path, request_paths)
    control = validate_control_request(read_json(control_path), campaign_path, campaign)
    if control_post_path is not None:
        validate_control_post(read_json(control_post_path), campaign_path,
                              control_path, pre_path)
    return campaign, roster, calendar, requests, control


def base_execution_identity(stage, campaign_path, control_path, pre_path,
                            control_post_path, request_paths, **extra):
    require(stage in {"count", "score"}, "known flank diagnostic stage required")
    return digest({"schema": "r02_masked_chr22_flank_execution_identity_v2",
        "stage": stage, "campaign_sha256": sha(campaign_path),
        "control_request_sha256": sha(control_path),
        "training_pre_sha256": sha(pre_path),
        "control_post_sha256": sha(control_post_path),
        "request_sha256": {Path(path).name: sha(path) for path in request_paths},
        "flank_runner_sha256": sha(__file__), **extra})


def fragment_execution_identity(stage_identity, fragment_number, example):
    return digest({"schema": "r02_masked_chr22_flank_fragment_identity_v2",
                   "stage_identity_sha256": stage_identity,
                   "fragment": fragment_number, "example": example})


def calendar_fragments(calendar):
    result = []
    for update in calendar["updates"]:
        for fragment in update["fragments"]:
            result.append(fragment)
    require(len(result) == 730, "calendar must contain 730 frozen fragments")
    return result


def control_reader(campaign):
    source = campaign["source"]
    return MaskedStoreReader(source["store"], source["manifest_sha256"],
        source["axis_directory"], source["axis_receipt_sha256"], role="FIT",
        max_people=8, max_window_sites=campaign["reader_resources"]["max_window_sites"],
        max_cache_entries=campaign["reader_resources"]["max_cache_entries"])


def component_fit(fragment, channel, window, keep, components, roster):
    target = fragment.common if channel == "common" else fragment.rare
    targets = fragment.common_targets if channel == "common" else fragment.rare_targets
    return fit_flanking_genotype_counts(fragment.common, target, targets,
        window=window, reference_people=keep, person_components=components,
        reference_role="FIT", role_manifest_identity=roster["role_manifest_identity_sha256"])


def empty_arrays():
    return {"person_id": np.asarray([], dtype="U1"),
            "locus_key": np.asarray([], dtype="U1"),
            "dose": np.asarray([], dtype=np.uint8),
            "raw_support": np.asarray([], dtype=np.int64),
            "probability_alpha_0_5": np.empty((0, 3), dtype=np.float64),
            "probability_alpha_1": np.empty((0, 3), dtype=np.float64)}


def extract_comparison_rows(predictions, target, targets):
    rows, loci = torch.where(predictions.comparison_mask)
    row_np = rows.detach().cpu().numpy()
    locus_np = loci.detach().cpu().numpy()
    keys = np.asarray(locus_keys(target))
    people = np.asarray(target.person_ids)
    return {
        "person_id": people[row_np], "locus_key": keys[locus_np],
        "dose": targets.dosage[rows, loci].detach().cpu().numpy().astype(np.uint8),
        "raw_support": predictions.raw_reference_support[rows, loci].detach().cpu().numpy(),
        "probabilities": predictions.probabilities[rows, loci].detach().cpu().numpy()}


def add_component_predictions(by_alpha, common_mask, support, predictions):
    """Add one query component; raw support is independent of pseudocount."""
    primary = predictions[0.5]
    sensitivity = predictions[1.0]
    require(torch.equal(primary.comparison_mask, sensitivity.comparison_mask)
            and torch.equal(primary.raw_reference_support,
                            sensitivity.raw_reference_support),
            "pseudocount changed raw support or comparison mask")
    for alpha in ALPHAS:
        by_alpha.setdefault(alpha, []).append(predictions[alpha])
    common_mask = (primary.comparison_mask if common_mask is None
                   else common_mask | primary.comparison_mask)
    support = (primary.raw_reference_support if support is None
               else support + primary.raw_reference_support)
    return common_mask, support


def count_one_fragment(fragment_number, calendar_row, loader, roster, fit_people,
                       fit_batches, output, stage_identity):
    example = CellExample(calendar_row["visit"], calendar_row["chromosome"],
                          calendar_row["cell"], tuple(calendar_row["person_ids"]))
    receipt_path = output / "fragments" / f"fragment-{fragment_number:08d}.json"
    artifact_path = output / "fragments" / f"fragment-{fragment_number:08d}.npz"
    example_dict = {"visit": example.epoch, "chromosome": example.chromosome,
                    "cell": example.cell, "person_ids": list(example.person_ids)}
    execution_identity = fragment_execution_identity(
        stage_identity, fragment_number, example_dict)
    if receipt_path.is_file():
        value = read_json(receipt_path)
        require(value.get("schema") == FRAGMENT_SCHEMA
                and value.get("fragment") == fragment_number
                and value.get("example") == example_dict
                and value.get("execution_identity_sha256") == execution_identity
                and artifact_path.is_file()
                and value.get("artifact_sha256") == sha(artifact_path),
                "recoverable flank fragment differs from frozen calendar")
        return value

    query = loader(example)
    components_by_person = roster["person_components"]
    query_components = tuple(components_by_person[person]
                             for person in example.person_ids)
    complete_component_people = {
        component: tuple(person for person in fit_people
                         if components_by_person[person] == component)
        for component in set(query_components)}
    window = FlankWindow(example.cell, float(calendar_row["cell_left_cm"]),
        float(calendar_row["cell_right_cm"]), 0.0, 0.2)
    objective = {"common": int(query.common_targets.score_mask.sum().item()),
                 "rare": int(query.rare_targets.score_mask.sum().item())}
    try:
        markers = select_common_flanks(query.common, window)
    except ValueError as error:
        require(str(error) ==
                "no structural COMMON marker on both sides inside admitted flank ring",
                str(error))
        arrays = {}
        for channel in ("common", "rare"):
            for name, array in empty_arrays().items():
                arrays[channel + "__" + name] = array
        atomic_npz(artifact_path, **arrays)
        value = {"schema": FRAGMENT_SCHEMA, "fragment": fragment_number,
            "example": example_dict, "example_sha256": digest(example_dict),
            "execution_identity_sha256": execution_identity,
            "status": "NO_STRUCTURAL_TWO_FLANKS_RETAINED_WITH_ZERO_COMPARISON_ROWS",
            "left_locus_key": None, "right_locus_key": None,
            "objective_target_count": objective,
            "two_flanks_and_target_count": {"common": 0, "rare": 0},
            "comparison_count": {"common": 0, "rare": 0},
            "fit_people_streamed": 0, "fit_components_excluded": sorted(set(query_components)),
            "artifact": str(artifact_path.resolve()), "artifact_sha256": sha(artifact_path),
            "cell_or_person_replaced": False, "dosage_used_to_choose_flanks": False}
        atomic_json(receipt_path, value)
        return value

    # Merge immediately so memory holds one locus-by-9-by-3 table per channel
    # plus one per queried family component, never 183 duplicated locus axes.
    totals = {"common": None, "rare": None}
    excluded = {channel: {component: None for component in set(query_components)}
                for channel in ("common", "rare")}
    for people in fit_batches:
        reference = loader(CellExample(example.epoch, example.chromosome,
                                       example.cell, people))
        components = tuple(components_by_person[person] for person in people)
        all_people = torch.ones(len(people), dtype=torch.bool)
        for channel in ("common", "rare"):
            part = component_fit(reference, channel, window,
                                 all_people, components, roster)
            totals[channel] = (part if totals[channel] is None else
                               merge_flanking_genotype_counts((totals[channel], part)))
            for component in excluded[channel]:
                keep = torch.as_tensor([value == component for value in components])
                if bool(keep.any()):
                    part = component_fit(reference, channel, window, keep,
                                         components, roster)
                    excluded[channel][component] = (part
                        if excluded[channel][component] is None else
                        merge_flanking_genotype_counts(
                            (excluded[channel][component], part)))

    arrays = {}
    defined_counts, comparison_counts = {}, {}
    for channel in ("common", "rare"):
        total = totals[channel]
        require(total is not None, "complete FIT count stream is empty")
        target = query.common if channel == "common" else query.rare
        targets = query.common_targets if channel == "common" else query.rare_targets
        by_alpha = {}
        common_mask = None
        support = None
        for component in sorted(set(query_components)):
            removed = excluded[channel][component]
            require(removed is not None,
                    "query component is absent from authenticated FIT count stream")
            fitted = leave_out_flanking_component_counts(
                total, removed, component_id=component,
                expected_component_person_ids=complete_component_people[component])
            query_mask = torch.as_tensor([value == component
                                          for value in query_components])
            component_predictions = {}
            for alpha in ALPHAS:
                prediction = predict_from_flanking_counts(fitted, query.common,
                    target, targets, query_people=query_mask,
                    person_components=query_components, prediction_role="FIT",
                    role_manifest_identity=roster["role_manifest_identity_sha256"],
                    alpha=alpha)
                component_predictions[alpha] = prediction
            common_mask, support = add_component_predictions(
                by_alpha, common_mask, support, component_predictions)
        require(common_mask is not None and support is not None,
                "query component inventory is empty")
        probability = {}
        for alpha in ALPHAS:
            combined = torch.full((*common_mask.shape, 3), float("nan"),
                                  dtype=torch.float64)
            for prediction in by_alpha[alpha]:
                use = prediction.comparison_mask
                combined[use] = prediction.probabilities[use]
            probability[alpha] = combined
        rows, loci = torch.where(common_mask)
        row_np = rows.detach().cpu().numpy(); locus_np = loci.detach().cpu().numpy()
        keys = np.asarray(locus_keys(target)); people = np.asarray(target.person_ids)
        arrays[channel + "__person_id"] = people[row_np]
        arrays[channel + "__locus_key"] = keys[locus_np]
        arrays[channel + "__dose"] = targets.dosage[rows, loci].detach().cpu().numpy().astype(np.uint8)
        arrays[channel + "__raw_support"] = support[rows, loci].detach().cpu().numpy()
        arrays[channel + "__probability_alpha_0_5"] = probability[0.5][rows, loci].numpy()
        arrays[channel + "__probability_alpha_1"] = probability[1.0][rows, loci].numpy()
        # Defined includes observed target and both observed flanks before the
        # raw leave-one-component-out support restriction.
        defined = torch.zeros_like(common_mask)
        for prediction in by_alpha[0.5]:
            defined |= prediction.prediction_defined
        defined_counts[channel] = int(defined.sum().item())
        comparison_counts[channel] = int(common_mask.sum().item())
    atomic_npz(artifact_path, **arrays)
    value = {"schema": FRAGMENT_SCHEMA, "fragment": fragment_number,
        "example": example_dict, "example_sha256": digest(example_dict),
        "execution_identity_sha256": execution_identity,
        "status": "COMPLETE_LEAVE_ONE_COMPONENT_OUT_COUNTS",
        "left_locus_key": markers.left_locus_key,
        "right_locus_key": markers.right_locus_key,
        "objective_target_count": objective,
        "two_flanks_and_target_count": defined_counts,
        "comparison_count": comparison_counts,
        "fit_people_streamed": len(fit_people),
        "fit_components_excluded": sorted(set(query_components)),
        "artifact": str(artifact_path.resolve()), "artifact_sha256": sha(artifact_path),
        "cell_or_person_replaced": False, "dosage_used_to_choose_flanks": False}
    atomic_json(receipt_path, value)
    return value


def count(campaign_path, control_path, pre_path, request_paths, control_post_path,
          output):
    campaign, roster, calendar, _, _ = validate_inputs(
        campaign_path, control_path, pre_path, request_paths, control_post_path)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    reader = control_reader(campaign)
    fit_people = tuple(roster["person_ids"])
    require(set(fit_people) == role_person_ids(reader.reader, "FIT"),
            "frozen roster differs from complete authenticated FIT role")
    fit_batches = tuple(tuple(fit_people[start:start + 8])
                        for start in range(0, len(fit_people), 8))
    config = MaskedModelConfig(**campaign["denominator_audit_model"])
    require(math.isclose(config.required_halo_cm("common"), 0.6,
                         rel_tol=0, abs_tol=1e-12),
            "control reader must expose the reviewed 0.6 cM maximum network territory")
    torch.set_num_threads(1)
    loader = fit_cell_loader({22: reader}, config, context_exclusion_cm=0,
                             device="cpu")
    stage_identity = base_execution_identity(
        "count", campaign_path, control_path, pre_path, control_post_path,
        request_paths, denominator_audit_source_sha256=
        sha(read_json(control_path)["denominator_audit_source"]["path"]))
    deadline_path = output / "COUNT_DEADLINE.json"
    if deadline_path.is_file():
        deadline = read_json(deadline_path)
        require(type(deadline.get("expires_at_unix")) in (int, float)
            and math.isfinite(deadline["expires_at_unix"])
            and deadline == {"schema": "r02_masked_flank_count_deadline_v2",
            "execution_identity_sha256": stage_identity,
            "expires_at_unix": deadline["expires_at_unix"],
            "total_budget_seconds": 72000},
            "count deadline belongs to another execution or changed")
    else:
        deadline = {"schema": "r02_masked_flank_count_deadline_v2",
            "execution_identity_sha256": stage_identity,
            "expires_at_unix": time.time() + 72000,
            "total_budget_seconds": 72000}
        atomic_json(deadline_path, deadline)
    started = time.monotonic(); receipts = []
    totals = defaultdict(int)
    for number, row in enumerate(calendar_fragments(calendar), 1):
        if time.time() >= deadline["expires_at_unix"]:
            progress = {"schema": COUNT_SCHEMA,
                "status": COUNT_INCOMPLETE_STATUS,
                "execution_identity_sha256": stage_identity,
                "completed_fragments": number - 1, "declared_fragments": 730,
                "deadline_sha256": sha(deadline_path),
                "fragment_receipts": receipts,
                "elapsed_this_invocation_seconds": time.monotonic() - started,
                "restart_resets_total_budget": False,
                "select_opened": False, "evaluate_opened": False}
            atomic_json(output / "COUNT_PROGRESS.json", progress)
            return progress
        value = count_one_fragment(number, row, loader, roster, fit_people,
                                   fit_batches, output, stage_identity)
        receipts.append({"fragment": number,
            "receipt_sha256": sha(output / "fragments" / f"fragment-{number:08d}.json"),
            "artifact_sha256": value["artifact_sha256"]})
        for channel in ("common", "rare"):
            totals[channel + "_objective"] += value["objective_target_count"][channel]
            totals[channel + "_defined"] += value["two_flanks_and_target_count"][channel]
            totals[channel + "_comparison"] += value["comparison_count"][channel]
    result = {"schema": COUNT_SCHEMA, "status": COUNT_STATUS,
        "execution_identity_sha256": stage_identity,
        "campaign_sha256": sha(campaign_path), "control_request_sha256": sha(control_path),
        "control_post_sha256": sha(control_post_path), "training_pre_sha256": sha(pre_path),
        "calendar_sha256": campaign["calendar"]["sha256"],
        "deadline_sha256": sha(deadline_path),
        "fit_roster_sha256": campaign["fit_roster"]["sha256"],
        "fragments": 730, "fit_people": 1460, "reference_batch_max_people": 8,
        "alpha": [0.5, 1.0], "totals": dict(totals), "fragment_receipts": receipts,
        "elapsed_this_invocation_seconds": time.monotonic() - started,
        "cell_or_person_replaced": False, "dosage_used_to_choose_cells_or_flanks": False,
        "select_opened": False, "evaluate_opened": False}
    atomic_json(output / "FLANK_COUNTS.json", result)
    atomic_json(output / "COUNT_PROGRESS.json", result)
    return result


def validate_count_receipt(path, campaign_path, control_path, pre_path,
                           control_post_path, request_paths):
    value = read_json(path)
    expected_identity = base_execution_identity(
        "count", campaign_path, control_path, pre_path, control_post_path,
        request_paths, denominator_audit_source_sha256=
        sha(read_json(control_path)["denominator_audit_source"]["path"]))
    require(value.get("schema") == COUNT_SCHEMA and value.get("status") == COUNT_STATUS
            and value.get("campaign_sha256") == sha(campaign_path)
            and value.get("control_request_sha256") == sha(control_path)
            and value.get("control_post_sha256") == sha(control_post_path)
            and value.get("training_pre_sha256") == sha(pre_path)
            and value.get("execution_identity_sha256") == expected_identity
            and value.get("fragments") == 730 and value.get("fit_people") == 1460
            and value.get("alpha") == [0.5, 1.0]
            and value.get("cell_or_person_replaced") is False
            and value.get("dosage_used_to_choose_cells_or_flanks") is False
            and value.get("select_opened") is False
            and value.get("evaluate_opened") is False,
            "completed exact flank-count receipt is required")
    rows = value.get("fragment_receipts")
    require(isinstance(rows, list) and len(rows) == 730,
            "flank-count fragment inventory differs")
    root = Path(path).resolve().parent / "fragments"
    campaign = read_json(campaign_path)
    frozen = calendar_fragments(read_json(campaign["calendar"]["path"]))
    for number, row in enumerate(rows, 1):
        receipt = root / f"fragment-{number:08d}.json"
        require(row.get("fragment") == number and receipt.is_file()
                and sha(receipt) == row.get("receipt_sha256"),
                "flank-count fragment receipt missing or changed")
        fragment = read_json(receipt)
        calendar_row = frozen[number - 1]
        example = {"visit": calendar_row["visit"],
                   "chromosome": calendar_row["chromosome"],
                   "cell": calendar_row["cell"],
                   "person_ids": calendar_row["person_ids"]}
        require(fragment.get("schema") == FRAGMENT_SCHEMA
                and fragment.get("fragment") == number
                and fragment.get("example") == example
                and fragment.get("execution_identity_sha256") ==
                    fragment_execution_identity(expected_identity, number, example),
                "flank-count fragment execution identity differs")
        artifact = root / f"fragment-{number:08d}.npz"
        require(artifact.is_file() and sha(artifact) == row.get("artifact_sha256")
                == fragment.get("artifact_sha256"),
                "flank-count fragment artifact missing or changed")
    return value


def validate_complete_training_receipt(path, family, campaign_path, request,
                                       pre_path, audit_path,
                                       audit_source_path):
    value = read_json(path)
    fit_id = family + "__full"
    result = value.get("campaign_result", {})
    require(value.get("schema") == training.RECEIPT_SCHEMA
            and value.get("status") == "DECLARED_UPDATE_BUDGET_COMPLETE"
            and value.get("fit_id") == fit_id
            and value.get("campaign_sha256") == sha(campaign_path)
            and value.get("request_sha256") == sha(request)
            and value.get("pre_sha256") == sha(pre_path)
            and value.get("denominator_audit_sha256") == sha(audit_path)
            and value.get("denominator_audit_source_sha256") ==
                sha(audit_source_path)
            and value.get("rare_identity_audit_sha256") is None
            and result.get("status") == "DECLARED_UPDATE_BUDGET_COMPLETE"
            and result.get("updates_completed") == 353
            and result.get("updates_declared") == 353
            and value.get("select_opened") is False
            and value.get("evaluate_opened") is False
            and value.get("biological_result_claimed") is False,
            "exact completed FULL FIT checkpoint receipt is required")
    checkpoint = Path(value.get("checkpoint_root", ""))
    require(checkpoint.is_dir() and (checkpoint / "CURRENT").is_file(),
            "completed training checkpoint is absent")
    return value


def expected_full_checkpoint_identity(campaign_path, request_paths, pre_path,
                                      audit_path, audit_source_path, request):
    """Reconstruct the identity created by the frozen FULL FIT runner."""
    campaign = read_json(campaign_path)
    roster, calendar = training.validate_campaign(campaign)
    audit = training.validate_audit(audit_path, campaign_path, request_paths,
                                    pre_path, audit_source_path)
    schedule = training.schedule_for(campaign, roster, calendar)
    config = MaskedModelConfig(**request["model"])
    model = MaskedGenotypeModel(config)
    optimizer = build_masked_adam(model, **request["optimizer"])
    identity = {"schema": "r02_masked_campaign_v1",
        "schedule": schedule.identity_payload(), "model": asdict(config),
        "source_sha256": training.digest({
            "campaign_sha256": sha(campaign_path),
            "audit_sha256": sha(audit_path),
            "denominator_audit_source_sha256": sha(audit_source_path),
            "rare_identity_audit_sha256": None,
            "pre_sha256": sha(pre_path),
            "paired_request_sha256": audit["request_sha256"]}),
        "protocol_sha256": campaign["protocol"]["sha256"],
        "condition": "full", "max_updates": 353,
        "max_seconds": request["resources"]["max_seconds"],
        "optimizer": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
        "optimizer_options": [{key: value for key, value in group.items()
                               if key != "params"}
                              for group in optimizer.param_groups],
        "code_sha256": {**masked_update_code_hashes(),
            "r02_masked_training_campaign.py": sha(training_campaign.__file__)}}
    return json.loads(training_campaign._canonical(identity)), model, optimizer


def restore_complete_model(training_receipt, request, campaign_path,
                           request_paths, pre_path, audit_path,
                           audit_source_path):
    identity, model, optimizer = expected_full_checkpoint_identity(
        campaign_path, request_paths, pre_path, audit_path, audit_source_path,
        request)
    root = Path(training_receipt["checkpoint_root"])
    store = CheckpointStore(root, identity)
    try:
        step, history = store.restore(model, optimizer)
    finally:
        store.close()
    require(step == 353 and history == training_receipt["campaign_result"]["history"],
            "checkpoint state differs from completed training receipt")
    model.eval()
    return model


def metric_rows(channel, alpha, doses, flank_probability,
                network_probability, network_log_probability):
    result = []
    for dose in (0, 1, 2):
        use = doses == dose
        count = int(use.sum())
        if count:
            f = flank_probability[use]; n = network_probability[use]
            truth = np.full(count, dose, dtype=np.int64)
            f_loss = float((-np.log(f[np.arange(count), truth])).sum())
            n_log = network_log_probability[use]
            n_loss = float((-n_log[np.arange(count), truth]).sum())
            f_correct = int((f.argmax(1) == truth).sum())
            n_correct = int((n.argmax(1) == truth).sum())
        else:
            f_loss = n_loss = 0.0; f_correct = n_correct = 0
        result.append({"channel": channel, "dose": dose, "alpha": alpha,
            "count": count, "flank_cross_entropy_sum": f_loss,
            "network_cross_entropy_sum": n_loss,
            "flank_minus_network_cross_entropy_sum": f_loss - n_loss,
            "flank_correct": f_correct, "network_correct": n_correct})
    return result


def select_network_rows(logits, person_indexes, locus_indexes):
    """Select paired rows with integer indices and stable log probabilities."""
    pi = torch.as_tensor(person_indexes, dtype=torch.long)
    li = torch.as_tensor(locus_indexes, dtype=torch.long)
    log_probability = torch.log_softmax(logits[pi, li], -1).double()
    return pi, li, log_probability.exp().numpy(), log_probability.numpy()


def score(campaign_path, control_path, pre_path, request_paths, control_post_path,
          count_receipt_path, training_receipt_path, family, output):
    campaign, roster, calendar, requests, _ = validate_inputs(
        campaign_path, control_path, pre_path, request_paths, control_post_path)
    count_receipt = validate_count_receipt(
        count_receipt_path, campaign_path, control_path, pre_path,
        control_post_path, request_paths)
    require(family in FAMILIES, "reviewed network family required")
    request_path = next(path for path in request_paths
        if read_json(path)["fit_id"] == family + "__full")
    request = next(row for row in requests if row["fit_id"] == family + "__full")
    control = read_json(control_path)
    audit_path = control["denominator_audit"]["path"]
    audit_source_path = control["denominator_audit_source"]["path"]
    training_receipt = validate_complete_training_receipt(
        training_receipt_path, family, campaign_path, request_path, pre_path,
        audit_path, audit_source_path)
    model = restore_complete_model(training_receipt, request, campaign_path,
                                   request_paths, pre_path, audit_path,
                                   audit_source_path)
    reader, fit_ids = training.open_reader(campaign)
    require(set(roster["person_ids"]) == fit_ids,
            "frozen roster differs from complete authenticated FIT role")
    loader = fit_cell_loader({22: reader}, model.config, context_exclusion_cm=0,
                             device="cpu")
    torch.set_num_threads(1)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    count_root = Path(count_receipt_path).resolve().parent / "fragments"
    aggregated = defaultdict(lambda: {"count": 0,
        "flank_cross_entropy_sum": 0.0, "network_cross_entropy_sum": 0.0,
        "flank_minus_network_cross_entropy_sum": 0.0,
        "flank_correct": 0, "network_correct": 0})
    receipts = []
    score_identity = base_execution_identity(
        "score", campaign_path, control_path, pre_path, control_post_path,
        request_paths, family=family,
        count_receipt_sha256=sha(count_receipt_path),
        count_execution_identity_sha256=
            count_receipt["execution_identity_sha256"],
        training_receipt_sha256=sha(training_receipt_path),
        checkpoint_identity_sha256=digest(expected_full_checkpoint_identity(
            campaign_path, request_paths, pre_path, audit_path,
            audit_source_path, request)[0]))
    started = time.monotonic()
    for number, row in enumerate(calendar_fragments(calendar), 1):
        fragment_receipt_path = output / "fragments" / f"fragment-{number:08d}.json"
        if fragment_receipt_path.is_file():
            fragment_result = read_json(fragment_receipt_path)
            require(fragment_result.get("schema") == SCORE_SCHEMA
                    and fragment_result.get("family") == family
                    and fragment_result.get("fragment") == number
                    and fragment_result.get("execution_identity_sha256") ==
                        fragment_execution_identity(score_identity, number,
                            {"count_artifact_sha256": read_json(
                                count_root / f"fragment-{number:08d}.json")
                                ["artifact_sha256"]}),
                    "recoverable flank score belongs to another family or fragment")
        else:
            fragment = loader(CellExample(row["visit"], row["chromosome"], row["cell"],
                                          tuple(row["person_ids"])))
            with torch.no_grad():
                prediction = model(fragment.common, fragment.rare,
                                   information_condition="full")
            logits = {"common": prediction.common_logits,
                      "rare": prediction.rare_logits}
            artifact_path = count_root / f"fragment-{number:08d}.npz"
            count_fragment_receipt = read_json(
                count_root / f"fragment-{number:08d}.json")
            require(sha(artifact_path) == count_fragment_receipt["artifact_sha256"],
                    "flank comparison artifact changed")
            rows_out = []
            with np.load(artifact_path, allow_pickle=False) as artifact:
                for channel in ("common", "rare"):
                    people = artifact[channel + "__person_id"]
                    keys = artifact[channel + "__locus_key"]
                    doses = artifact[channel + "__dose"].astype(np.int64, copy=False)
                    target = fragment.common if channel == "common" else fragment.rare
                    targets = (fragment.common_targets if channel == "common"
                               else fragment.rare_targets)
                    person_index = {person: index for index, person in enumerate(target.person_ids)}
                    locus_index = {key: index for index, key in enumerate(locus_keys(target))}
                    require(all(str(person) in person_index for person in people)
                            and all(str(key) in locus_index for key in keys),
                            "saved comparison row is absent from network fragment")
                    pi, li, network_probability, network_log_probability = (
                        select_network_rows(logits[channel],
                            [person_index[str(person)] for person in people],
                            [locus_index[str(key)] for key in keys]))
                    require(bool(targets.score_mask[pi, li].all())
                            and np.array_equal(targets.dosage[pi, li].numpy(), doses),
                            "saved comparison mask or target dose differs on replay")
                    require(np.isfinite(network_probability).all()
                            and np.allclose(network_probability.sum(1), 1,
                                            rtol=0, atol=1e-6),
                            "network probabilities are invalid")
                    for alpha, suffix in ((0.5, "0_5"), (1.0, "1")):
                        flank_probability = artifact[
                            channel + "__probability_alpha_" + suffix]
                        require(flank_probability.shape == network_probability.shape
                                and np.isfinite(flank_probability).all()
                                and np.allclose(flank_probability.sum(1), 1,
                                                rtol=0, atol=1e-12),
                                "flank probabilities are invalid")
                        rows_out.extend(metric_rows(channel, alpha, doses,
                                                    flank_probability,
                                                    network_probability,
                                                    network_log_probability))
            fragment_identity = fragment_execution_identity(
                score_identity, number,
                {"count_artifact_sha256":
                    count_fragment_receipt["artifact_sha256"]})
            fragment_result = {"schema": SCORE_SCHEMA,
                "status": "COMPLETE_EXACT_FRAGMENT_MASK", "family": family,
                "fragment": number,
                "execution_identity_sha256": fragment_identity,
                "example_sha256": count_fragment_receipt["example_sha256"],
                "count_artifact_sha256": count_fragment_receipt["artifact_sha256"],
                "rows": rows_out, "select_opened": False, "evaluate_opened": False}
            atomic_json(fragment_receipt_path, fragment_result)
        for metric in fragment_result["rows"]:
            key = (metric["channel"], metric["dose"], metric["alpha"])
            for name in aggregated[key]:
                aggregated[key][name] += metric[name]
        receipts.append({"fragment": number,
                         "receipt_sha256": sha(fragment_receipt_path)})
    rows = []
    for (channel, dose, alpha), values in sorted(aggregated.items()):
        rows.append({"family": family, "channel": channel, "dose": dose,
                     "alpha": alpha, **values})
    result = {"schema": SCORE_SCHEMA, "status": SCORE_STATUS,
        "execution_identity_sha256": score_identity,
        "family": family, "condition": "full", "role": "FIT",
        "campaign_sha256": sha(campaign_path),
        "control_request_sha256": sha(control_path),
        "control_post_sha256": sha(control_post_path),
        "training_receipt_sha256": sha(training_receipt_path),
        "count_receipt_sha256": sha(count_receipt_path),
        "comparison": "EXACT_SAME_PERSON_LOCUS_TARGET_MASK",
        "fit_access_limitation": "NETWORK_TRAINED_ON_FIT_WHILE_TABLE_EXCLUDES_QUERY_FAMILY_COMPONENT",
        "rows": rows, "fragment_receipts": receipts,
        "elapsed_this_invocation_seconds": time.monotonic() - started,
        "winner_selected": False, "biological_result_claimed": False,
        "select_opened": False, "evaluate_opened": False}
    atomic_json(output / "FLANK_SCORE.json", result)
    return result


def readiness(campaign_path, full_receipts, output):
    campaign_sha = sha(campaign_path)
    expected = {family: family + "__full" for family in FAMILIES}
    ready, waiting = {}, []
    for family in FAMILIES:
        path = Path(full_receipts.get(family, ""))
        if not path.is_file():
            waiting.append(expected[family]); continue
        value = read_json(path)
        if (value.get("schema") == training.RECEIPT_SCHEMA
                and value.get("fit_id") == expected[family]
                and value.get("campaign_sha256") == campaign_sha
                and value.get("status") == "DECLARED_UPDATE_BUDGET_COMPLETE"
                and value.get("campaign_result", {}).get("updates_completed") == 353):
            ready[family] = {"path": str(path.resolve()), "sha256": sha(path)}
        else:
            waiting.append(expected[family])
    value = {"schema": READINESS_SCHEMA,
        "status": ("READY_FOR_PAIRED_FIT_SCORE" if not waiting
                   else "WAITING_FOR_EXACT_FULL_FIT_CHECKPOINTS"),
        "campaign_sha256": campaign_sha, "ready": ready,
        "waiting_fit_ids": waiting, "polling_or_watcher_started": False,
        "select_opened": False, "evaluate_opened": False}
    atomic_json(output, value)
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("count", "score"):
        command = commands.add_parser(name)
        command.add_argument("--campaign", required=True)
        command.add_argument("--control-request", required=True)
        command.add_argument("--pre-receipt", required=True)
        command.add_argument("--control-post", required=True)
        command.add_argument("--all-request", action="append", required=True)
        command.add_argument("--output", required=True)
        if name == "score":
            command.add_argument("--count-receipt", required=True)
            command.add_argument("--training-receipt", required=True)
            command.add_argument("--family", choices=FAMILIES, required=True)
    check = commands.add_parser("readiness")
    check.add_argument("--campaign", required=True)
    check.add_argument("--local-deepsets-receipt")
    check.add_argument("--local-attention-receipt")
    check.add_argument("--output", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "count":
        value = count(args.campaign, args.control_request, args.pre_receipt,
                      args.all_request, args.control_post, args.output)
    elif args.command == "score":
        value = score(args.campaign, args.control_request, args.pre_receipt,
                      args.all_request, args.control_post, args.count_receipt,
                      args.training_receipt, args.family, args.output)
    else:
        value = readiness(args.campaign, {
            "local_deepsets": args.local_deepsets_receipt,
            "local_attention": args.local_attention_receipt}, args.output)
    print(json.dumps(value, sort_keys=True))
    return 4 if value.get("status") == COUNT_INCOMPLETE_STATUS else 0


if __name__ == "__main__":
    raise SystemExit(main())
