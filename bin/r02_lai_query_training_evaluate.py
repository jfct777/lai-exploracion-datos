#!/usr/bin/env python3
"""Describe FI96 and FV step-48 probabilities on one authenticated FIT case."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np

from r02_lai_boundary_stats import boundary_report_stats
from r02_lai_error_strata import paired_strata
from r02_lai_probability_calibration import temperature_scale
from r02_lai_saved_case import (authenticated, authenticated_json,
                                load_saved_fit_case, sha256)


SCHEMA = "r02_lai_query_training_post_case_v1"
RESULT_SCHEMA = "r02_lai_query_training_post_case_result_v1"
TEMPERATURE = 1.4981986872645123
METHODS = ("FLARE", "FLARE_TEMPERATURE", "FI96", "FV")
def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_fresh(path, value):
    path = Path(path)
    require(not path.exists(), "preserve previous output: " + str(path))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError("preserve previous output: " + str(path)) from error
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_or_authenticate_identical(path, value):
    """Create a request once, or reuse the exact canonical request after failure."""
    path = Path(path)
    expected = canonical_json(value)
    if path.exists():
        require(path.is_file() and not path.is_symlink()
                and path.read_bytes() == expected,
                "preserve different previous request: " + str(path))
    else:
        write_fresh(path, value)


def prepare_output_directory(path):
    """Admit a fresh directory or the empty directory left by a failed start."""
    destination = Path(path)
    if destination.exists():
        require(destination.is_dir() and not destination.is_symlink()
                and not (destination / "RESULT.json").exists()
                and not any(destination.iterdir()),
                "preserve previous or partial POST output: " + str(destination))
    else:
        destination.mkdir()
    return destination


def authenticate_fv(request):
    contract = authenticated_json(request["fv_contract"], "FV contract")
    resolved = authenticated_json(request["fv_resolved"], "FV resolved recipe")
    receipt = authenticated_json(request["fv_training_receipt"], "FV training receipt")
    predictions = authenticated_json(request["fv_predictions"], "FV predictions index")
    mode = request["fv_query_event_mode"]
    allowed = {"LOCAL_MEAN_V1", "QUERY_EVENT_CONTENT_V1",
               "QUERY_EVENT_RELATIVE_V1"}
    initialization = contract.get("lai_weight_initialization", {})
    amendment = contract.get("query_event_resolved_amendment")
    amendment_valid = (amendment is None if mode == "LOCAL_MEAN_V1" else
        isinstance(amendment, dict)
        and amendment.get("schema") == "r02_lai_query_event_resolved_amendment_v1"
        and amendment.get("added_field") == "lai_query_event_mode"
        and amendment.get("added_value") == mode)
    require(mode in allowed and contract.get("objective") == "lai"
            and initialization.get("source_step") == 96
            and initialization.get("source_role") == "FIT"
            and initialization.get("source_arm") == "NONE"
            and initialization.get("optimizer_state_imported") is False
            and amendment_valid,
            "FV contract does not admit the FI96 query-readout trajectory")
    require(resolved.get("lai_query_event_mode", "LOCAL_MEAN_V1") == mode,
            "FV recipe query-EVENT mode differs")
    receipt_initialization = receipt.get("lai_weight_initialization", {})
    require(receipt.get("schema") == "r02_context_training_v1"
            and receipt.get("status") == "FIT_CHECKPOINT"
            and receipt.get("score_role") == "FIT"
            and receipt.get("arm") == "NONE"
            and receipt.get("scope") == "DEVELOPMENT"
            and receipt.get("select_accessed") is False
            and receipt.get("steps") == 48
            and receipt.get("contract_sha256") == request["fv_contract"]["sha256"]
            and receipt.get("resolved_sha256") == request["fv_resolved"]["sha256"]
            and receipt.get("prediction_sha256") == request["fv_predictions"]["sha256"]
            and receipt_initialization.get("source_step") == 96
            and receipt_initialization.get("source_role") == "FIT"
            and receipt_initialization.get("source_arm") == "NONE"
            and receipt_initialization.get("optimizer_state_imported") is False
            and receipt_initialization.get("target_mode") == mode,
            "FV receipt is not the frozen step-48 FIT result")
    require(isinstance(predictions, list) and len(predictions) == 48,
            "FV prediction index must contain all 48 FIT regions")
    return contract, resolved, receipt


def authenticate_temperature(request):
    result = authenticated_json(request["temperature_result"], "FD temperature result")
    require(request.get("temperature") == TEMPERATURE
            and result.get("schema") == "r02_fita_weighted_temperature_result_v1"
            and result.get("status") == "COMPLETE_FIT_TRAINING"
            and result.get("role") == "FIT"
            and result.get("task") == {
                "task_id": "flare", "family": "lai_cnn", "version": "FLARE"}
            and result.get("fit", {}).get("status") == "INTERIOR"
            and result.get("fit_is_independent_validation") is False
            and result.get("selection_performed") is False
            and result.get("select_opened") is False
            and result.get("evaluate_opened") is False
            and result.get("fit", {}).get("temperature") == TEMPERATURE,
            "frozen FIT temperature differs")


def add_rows(total, rows):
    if total is None:
        return copy.deepcopy(rows)
    require(len(total) == len(rows), "strata axes differ")
    for aggregate, current in zip(total, rows):
        require((aggregate["distance_bin"], aggregate["exposure_bin"]) ==
                (current["distance_bin"], current["exposure_bin"]),
                "strata order differs")
        for key, value in current.items():
            if key in {"distance_bin", "exposure_bin"}:
                continue
            if isinstance(value, dict):
                for inner, item in value.items():
                    aggregate[key][inner] = ((np.asarray(aggregate[key][inner]) + item).tolist()
                                             if isinstance(item, list)
                                             else aggregate[key][inner] + item)
            elif isinstance(value, list):
                aggregate[key] = (np.asarray(aggregate[key]) + value).tolist()
            else:
                aggregate[key] += value
    return total


def relabel(rows, method):
    result = copy.deepcopy(rows)
    for row in result:
        if method == "FLARE":
            require(row["NEURAL"] == row["FLARE"]
                    and row["neural_errors"] == row["flare_errors"]
                    and row["corrected"] == row["introduced"] == 0,
                    "FLARE self-comparison differs before relabeling")
        row[method] = row.pop("NEURAL")
        row[method.lower() + "_errors"] = row.pop("neural_errors")
    return result


def metric_summary(rows, method):
    observations = sum(row["count"] for row in rows)
    require(observations > 0, "no observed FIT queries")
    return {
        "observations": observations,
        "log_loss": sum(row[method]["logloss_sum"] for row in rows) / observations,
        "brier": sum(row[method]["brier_sum"] for row in rows) / observations,
        "dosage_mae_AFR_EUR_NAM": (sum(
            (np.asarray(row[method]["dosage_abs_sum"], dtype=float) for row in rows),
            start=np.zeros(3)) / observations).tolist(),
        "errors": sum(row[method.lower() + "_errors"] if method != "FLARE"
                      else row["flare_errors"] for row in rows),
        "true_zero_count": sum(row[method]["true_zero_count"] for row in rows),
    }


def run(request_path, request_sha256, output):
    started = time.monotonic()
    request = json.loads(authenticated(
        {"path": request_path, "sha256": request_sha256}, "POST request").read_text())
    require(request.get("schema") == SCHEMA and request.get("role") == "FIT"
            and request.get("select_opened") is False
            and request.get("evaluate_opened") is False
            and request.get("cuts_cm") == [0.1, 0.2, 0.5]
            and request.get("floor") == 1e-12,
            "POST request scope differs")
    for descriptor in request["code"]:
        authenticated(descriptor, "POST code")
    _, _, fv_receipt = authenticate_fv(request)
    authenticate_temperature(request)
    ft = authenticated_json(request["source_ft_request"], "source FT request")
    require(ft.get("case") == request.get("case")
            and ft.get("cuts_cm") == request["cuts_cm"]
            and ft.get("floor") == request["floor"]
            and ft.get("people") == request["people"]
            and ft.get("queries") == request["queries"],
            "POST changed the FT case or denominators")
    case = load_saved_fit_case(
        ft, np, started=started, max_seconds=request["max_seconds"],
        additional_prediction_indices={"FV": request["fv_predictions"]},
        include_fi96=True)
    probabilities = {
        "FLARE": case["flare"],
        "FI96": case["predictions"]["FI96"],
        "FV": case["predictions"]["FV"],
    }
    probabilities["FLARE_TEMPERATURE"] = temperature_scale(
        probabilities["FLARE"], TEMPERATURE)
    require(np.array_equal(probabilities["FLARE"].argmax(-1),
                           probabilities["FLARE_TEMPERATURE"].argmax(-1)),
            "positive temperature changed FLARE assignments")

    totals = {method: None for method in METHODS}
    flat_truth, flat_mask, flat_cm, groups = [], [], [], []
    hard = {method: [] for method in METHODS}
    offset = 0
    for person in range(len(case["people"])):
        n = request["queries"]
        groups.append((str(case["people"][person]), offset, offset + n)); offset += n
        flat_truth.append(case["truth"][person]); flat_mask.append(case["observed"][person])
        flat_cm.append(case["cm"][person])
        for method in METHODS:
            rows = paired_strata(
                probabilities["FLARE"][person], probabilities[method][person],
                case["truth"][person], case["observed"][person],
                case["cm"][person], case["burden"][person],
                cuts_cm=request["cuts_cm"], floor=request["floor"])
            totals[method] = add_rows(totals[method], relabel(rows, method))
            hard[method].append(probabilities[method][person].argmax(-1))
    truth = np.concatenate(flat_truth); mask = np.concatenate(flat_mask)
    cm = np.concatenate(flat_cm)
    boundary = {method: boundary_report_stats(
        truth, np.concatenate(hard[method]), mask, cm, groups,
        request["boundary_memory_bytes"]) for method in METHODS}
    metrics = {method: metric_summary(totals[method], method) for method in METHODS}
    errors = {method: {
        "corrected_vs_flare": sum(row["corrected"] for row in totals[method]),
        "introduced_vs_flare": sum(row["introduced"] for row in totals[method]),
    } for method in METHODS}
    require(errors["FLARE"] == {"corrected_vs_flare": 0, "introduced_vs_flare": 0}
            and errors["FLARE_TEMPERATURE"] == errors["FLARE"],
            "FLARE self/temperature assignments differ")

    destination = prepare_output_directory(output)
    result = {
        "schema": RESULT_SCHEMA, "status": "COMPLETE_STEP48_FIT_DESCRIPTION",
        "case": request["case"], "role": "FIT", "people": request["people"],
        "queries": request["queries"], "methods": list(METHODS),
        "metrics": metrics, "errors_vs_flare": errors,
        "strata_by_method": totals, "boundary_by_method": boundary,
        "temperature": TEMPERATURE, "source_ft_request": request["source_ft_request"],
        "fi96_training_receipt": case["training_receipt"],
        "fv_training_receipt": request["fv_training_receipt"],
        "fv_contract": request["fv_contract"], "fv_resolved": request["fv_resolved"],
        "request_sha256": request_sha256, "select_opened": False,
        "evaluate_opened": False, "seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "limits": [
            "FIT development cases share donors and are not independent evaluation",
            "Truth-boundary and rare-exposure strata are descriptive and do not identify rare causality",
            "All four methods/references are reported; POST does not select a model or checkpoint",
            "FLARE temperature was fitted previously on FIT and is only a fixed reference here",
        ],
    }
    write_fresh(destination / "RESULT.json", result)
    return result


def evaluate(new_contract_descriptor, new_resolved_descriptor, new_training_dir,
             original_ft_request_descriptor, output, *, temperature_result):
    """Materialize and execute one POST request after FV training finishes."""
    training = Path(new_training_dir)
    receipt_path, predictions_path = (training / "training.receipt.json",
                                      training / "predictions.json")
    require(receipt_path.is_file() and predictions_path.is_file(),
            "FV training directory lacks its step-48 receipt or predictions index")
    resolved = authenticated_json(new_resolved_descriptor, "FV resolved recipe")
    ft = authenticated_json(original_ft_request_descriptor, "source FT request")
    authenticated_json(temperature_result, "FD temperature result")
    source = Path(__file__).resolve().parent
    code_names = (
        "r02_lai_query_training_evaluate.py", "r02_lai_saved_case.py",
        "r02_lai_error_strata.py", "r02_lai_boundary_stats.py",
        "r02_lai_probability_calibration.py", "m39_calibration.py",
        "r02_lai_evaluate.py", "m37_trace_core.py", "m37_trace_score.py",
    )
    request = {
        "schema": SCHEMA, "role": "FIT", "case": ft["case"],
        "people": ft["people"], "queries": ft["queries"],
        "cuts_cm": [0.1, 0.2, 0.5], "floor": 1e-12,
        "temperature": TEMPERATURE, "max_seconds": 900,
        "boundary_memory_bytes": 256 * 1024**2,
        "select_opened": False, "evaluate_opened": False,
        "fv_query_event_mode": resolved.get("lai_query_event_mode", "LOCAL_MEAN_V1"),
        "fv_contract": new_contract_descriptor,
        "fv_resolved": new_resolved_descriptor,
        "fv_training_receipt": {"path": str(receipt_path.resolve()),
                                "sha256": sha256(receipt_path)},
        "fv_predictions": {"path": str(predictions_path.resolve()),
                           "sha256": sha256(predictions_path)},
        "source_ft_request": original_ft_request_descriptor,
        "temperature_result": temperature_result,
        "code": [{"path": str((source / name).resolve()),
                  "sha256": sha256(source / name)} for name in code_names],
    }
    request_path = Path(output).parent / (Path(output).name + ".request.json")
    write_or_authenticate_identical(request_path, request)
    return run(str(request_path), sha256(request_path), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request"); parser.add_argument("request_sha256")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(run(args.request, args.request_sha256, args.output),
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
