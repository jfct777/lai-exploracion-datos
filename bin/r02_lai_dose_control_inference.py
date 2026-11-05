#!/usr/bin/env python3
"""Read-only FK dose interventions with the frozen FI NONE model.

The historical FH package constructs the model and authenticates checkpoint 96.
Separately authenticated FL helpers project whole-person FK products into one
FIT region.  This program never constructs an optimizer, calls backward, opens
SELECT, or writes prediction matrices.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import resource
import signal
import sys
import time

import numpy as np


SCHEMA = "r02_lai_frozen_none_personal_dose_inference_v1"
STATUS = "FROZEN_FI_NONE_READ_ONLY_FIT_DOSE_INTERVENTIONS_COMPLETE"
REMOVED = "PERSONAL_DOSE_REMOVED_FIXED_SUPPORT"
PERMUTED = "PERSONAL_DOSE_PERMUTED_FIXED_SUPPORT"
CONDITIONS = (REMOVED, PERMUTED)
PACKAGE_SCHEMA = "r02_lai_dose_control_inference_package_v1"
PLAN_SCHEMA = "r02_lai_dose_control_inference_product_plan_v1"
FROZEN_IMAGE = ("us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/"
                "m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99")
HEX = re.compile(r"[0-9a-f]{64}")
REGION = re.compile(r"region-(0[0-9]|1[0-5])")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def object_sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pinned(path, expected_sha256):
    path = Path(path)
    require(HEX.fullmatch(expected_sha256 or "") and path.is_file()
            and sha256_file(path) == expected_sha256,
            "pinned JSON identity differs: " + str(path))
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), "pinned JSON must be an object: " + str(path))
    return value


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None,
            "cannot load authenticated module: " + str(path))
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def validate_control_package(root, manifest_path, expected_sha256):
    """Authenticate the small FL helper package without importing its trainer."""
    root = Path(root).resolve()
    manifest = read_pinned(manifest_path, expected_sha256)
    require(manifest.get("schema") == PACKAGE_SCHEMA
            and isinstance(manifest.get("files"), dict),
            "control helper package manifest differs")
    required = {
        "bin/r02_lai_control_views.py", "bin/r02_lai_dose_controls.py",
        "bin/r02_lai_dose_control_training.py", "bin/r02_lai_person_microbatch.py",
        "bin/r02_lai_fit_monitor.py", "bin/r02_lai_dose_control_inference.py",
    }
    require(set(manifest["files"]) == required,
            "control helper package inventory differs")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file()
                and HEX.fullmatch(digest or "") and sha256_file(path) == digest,
                "control helper package member differs: " + name)
    return manifest


def load_control_helpers(root):
    """Load only intervention/microbatch/audit helpers beside frozen FH code."""
    root = Path(root).resolve() / "bin"
    # control_views is byte-identical in FH and FL.  Install it only when the
    # frozen trainer did not already import it.
    if "r02_lai_control_views" not in sys.modules:
        load_module(root / "r02_lai_control_views.py", "r02_lai_control_views")
    load_module(root / "r02_lai_dose_controls.py", "r02_lai_dose_controls")
    binding = load_module(root / "r02_lai_dose_control_training.py",
                          "r02_lai_dose_control_training_inference")
    microbatch = load_module(root / "r02_lai_person_microbatch.py",
                             "r02_lai_person_microbatch_inference")
    audit = load_module(root / "r02_lai_fit_monitor.py",
                        "r02_lai_fit_monitor_inference")
    return binding, microbatch, audit


def validate_product_plan(path, expected_sha256):
    value = read_pinned(path, expected_sha256)
    require(set(value) == {"schema", "source_fk_results", "conditions", "products_by_case"}
            and value["schema"] == PLAN_SCHEMA
            and value["conditions"] == list(CONDITIONS)
            and isinstance(value["products_by_case"], dict)
            and bool(value["products_by_case"]),
            "dose inference product plan differs")
    source = value["source_fk_results"]
    require(isinstance(source, dict) and set(source) == {"path", "sha256"},
            "FK source descriptor differs")
    prepared = read_pinned(source["path"], source["sha256"])
    require(prepared.get("status") ==
            "THREE_CASES_SIX_FIT_VIEWS_PUBLISHED_LOCALLY_HASH_VERIFIED"
            and prepared.get("truth_loaded") is False
            and prepared.get("new_training") is False,
            "FK preparation result differs")
    by_case = {row["case_id"]: row for row in prepared["cases"]}
    require(set(by_case) == set(value["products_by_case"]),
            "FK result and product-plan cases differ")
    for case, product in value["products_by_case"].items():
        require(set(product) == {"receipt", "batch"}, "product fields differ: " + case)
        for name in ("receipt", "batch"):
            item = product[name]
            require(isinstance(item, dict) and set(item) == {"path", "sha256"}
                    and HEX.fullmatch(item.get("sha256", ""))
                    and Path(item["path"]).is_file()
                    and sha256_file(item["path"]) == item["sha256"],
                    "FK product differs: " + case + " " + name)
        row = by_case[case]
        require(product["receipt"]["sha256"] == row["receipt_sha256"]
                and product["batch"]["sha256"] == row["batch_sha256"],
                "FK product does not match published result: " + case)
    return value


def binding_spec(binding, plan, condition):
    require(condition in CONDITIONS, "unknown effective dose condition")
    return {
        "schema": binding.BINDING_SCHEMA,
        "base_condition": "NONE",
        "effective_condition": condition,
        "role": "FIT",
        "source_minor": binding.SOURCE_MINOR,
        "personal_dose_available": condition == PERMUTED,
        "products_by_case": plan["products_by_case"],
    }


def probability_statistics(prediction, data, observed=None):
    """Return the fixed additive metrics used by the LAI FIT monitor."""
    probability = np.asarray(prediction, dtype=np.float64)
    if observed is None:
        observed = np.asarray(data.get("truth_observed", data["query_valid"]), dtype=bool)
        observed &= np.asarray(data["query_valid"], dtype=bool)
    else:
        observed = np.asarray(observed, dtype=bool)
    labels = np.asarray(data["truth_state"])[observed]
    selected = probability[observed]
    require(probability.shape == (*observed.shape, 6) and len(labels) > 0
            and np.isfinite(selected).all() and (selected >= 0).all()
            and (selected <= 1).all()
            and np.allclose(selected.sum(1), 1, atol=5e-5, rtol=0)
            and np.isin(labels, np.arange(6)).all(),
            "six-state FIT probabilities or truth differ")
    true_probability = selected[np.arange(len(labels)), labels]
    require((true_probability > 0).all(), "zero true-state probability")
    one_hot = np.eye(6, dtype=np.float64)[labels]
    dosage = np.asarray(((2, 0, 0), (1, 1, 0), (1, 0, 1),
                         (0, 2, 0), (0, 1, 1), (0, 0, 2)), dtype=np.float64)
    expected = selected @ dosage
    actual = dosage[labels]
    count = int(len(labels))
    log_loss_sum = float(-np.log(true_probability).sum())
    brier_sum = float(np.square(selected - one_hot).sum())
    state_error_sum = int((selected.argmax(1) != labels).sum())
    dosage_abs_sum = np.abs(expected - actual).sum(0)
    return {
        "aggregation": "QUERY_POOLED_SUMS_OVER_ONE_FIXED_SCORE_MASK",
        "observed_queries": count,
        "log_loss_sum": log_loss_sum,
        "brier_sum": brier_sum,
        "state_error_sum": state_error_sum,
        "dosage_abs_sum_AFR_EUR_NAM": dosage_abs_sum.tolist(),
        "mean_log_loss": log_loss_sum / count,
        "mean_brier": brier_sum / count,
        "state_error_rate": state_error_sum / count,
        "dosage_mae_AFR_EUR_NAM": (dosage_abs_sum / count).tolist(),
    }


def prediction_change(prediction, original, data, observed=None):
    if observed is None:
        observed = np.asarray(data.get("truth_observed", data["query_valid"]), dtype=bool)
        observed &= np.asarray(data["query_valid"], dtype=bool)
    else:
        observed = np.asarray(observed, dtype=bool)
    left, right = np.asarray(prediction)[observed], np.asarray(original)[observed]
    require(left.shape == right.shape and left.ndim == 2 and left.shape[1] == 6,
            "prediction change axes differ")
    delta = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    return {
        "mean_abs_probability_change": float(np.abs(delta).mean()),
        "max_abs_probability_change": float(np.abs(delta).max()),
        "rms_probability_change": float(np.sqrt(np.square(delta).mean())),
        "argmax_state_changes": int((left.argmax(1) != right.argmax(1)).sum()),
        "observed_queries": int(len(left)),
    }


def compare_saved_original(recomputed, saved, tolerance):
    recomputed, saved = np.asarray(recomputed), np.asarray(saved)
    require(recomputed.shape == saved.shape and np.isfinite(recomputed).all()
            and np.isfinite(saved).all() and tolerance >= 0,
            "saved/recomputed FI prediction axes or tolerance differ")
    exact = np.array_equal(recomputed, saved)
    maximum = float(np.abs(recomputed.astype(np.float64) - saved.astype(np.float64)).max())
    require(exact or (tolerance > 0 and maximum <= tolerance),
            "recomputed original differs from saved FI prediction")
    return {"exact": exact, "max_abs_delta_probability": maximum,
            "admitted_absolute_tolerance": float(tolerance),
            "comparison": "BITWISE_EXACT" if exact else "MEASURED_ABSOLUTE_TOLERANCE"}


def predict_preserving_state(model, views, predict, state_digest):
    before = state_digest(model)
    predictions = {}
    for name, view in views.items():
        predictions[name] = predict(model, view)
        require(state_digest(model) == before,
                "model, gradients, modes, or RNG changed during read-only inference")
    return predictions, before


def array_sha256(value):
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(canonical(list(array.shape)))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def score_support_identity(data, observed):
    """Bind per-region sums to one unique owned-coordinate and truth mask."""
    observed = np.asarray(observed, dtype=bool)
    core = np.asarray(data["query_core"], dtype=bool)
    truth_observed = np.asarray(data["truth_observed"], dtype=bool)
    valid = np.asarray(data["query_valid"], dtype=bool)
    require(observed.shape == core.shape == truth_observed.shape == valid.shape
            and np.array_equal(observed, core & truth_observed)
            and not (truth_observed & ~valid).any(),
            "score mask is not exactly query_core intersect truth_observed")
    require(np.array_equal(core, np.broadcast_to(core[:1], core.shape)),
            "query ownership differs between people")
    chromosome = np.asarray(data["query_chrom"])
    position = np.asarray(data["query_pos_bp"])
    require(chromosome.shape == position.shape == core.shape
            and np.array_equal(chromosome, np.broadcast_to(chromosome[:1], chromosome.shape))
            and np.array_equal(position, np.broadcast_to(position[:1], position.shape)),
            "query coordinate axes differ between people")
    coordinates = [(str(chromosome[0, index]), int(position[0, index]))
                   for index in np.flatnonzero(core[0])]
    require(coordinates and len(coordinates) == len(set(coordinates)),
            "owned query coordinates are empty or repeated within a region")
    per_person = observed.sum(1).astype(np.int64)
    return {
        "owned_coordinate_count": len(coordinates),
        "owned_coordinates_sha256": object_sha256(coordinates),
        "score_mask_sha256": array_sha256(observed),
        "truth_state_on_score_mask_sha256": array_sha256(
            np.asarray(data["truth_state"])[observed]),
        "observed_query_count": int(observed.sum()),
        "per_person_observed_count_sha256": array_sha256(per_person),
        "per_person_observed_count_min": int(per_person.min()),
        "per_person_observed_count_max": int(per_person.max()),
        "mask_rule": "QUERY_CORE_INTERSECT_TRUTH_OBSERVED",
    }


def invocation_identity(args):
    names = ("package", "package_manifest_sha256", "control_package",
             "control_package_manifest", "control_package_manifest_sha256",
             "fj_diagnostic", "fj_diagnostic_sha256", "none_route",
             "none_route_sha256", "checkpoint_root", "ex_helper",
             "ex_helper_sha256", "prediction_index", "prediction_index_sha256",
             "product_plan", "product_plan_sha256", "case_id", "region_id",
             "people_per_microbatch", "threads", "original_atol",
             "max_wall_seconds", "max_rss_bytes")
    return {name: getattr(args, name) for name in names}


def write_idempotent(path, value, request_sha256):
    path = Path(path)
    content = canonical(value)
    if path.exists():
        existing = json.loads(path.read_bytes())
        require(existing.get("schema") == SCHEMA and existing.get("status") == STATUS
                and existing.get("request_sha256") == request_sha256,
                "existing region receipt belongs to another request")
        return False
    require(path.parent.is_dir(), "region receipt parent must exist")
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        created = True
    except FileExistsError:
        existing = json.loads(path.read_bytes())
        require(existing.get("schema") == SCHEMA and existing.get("status") == STATUS
                and existing.get("request_sha256") == request_sha256,
                "concurrent region receipt belongs to another request")
        created = False
    finally:
        temporary.unlink(missing_ok=True)
    return created


def validate_existing_output(path, args, request_sha256):
    """Authenticate source identities and closed-world result shape on resume."""
    path = Path(path)
    existing = json.loads(path.read_bytes())
    required = {"schema", "status", "request_sha256", "selection", "authentication",
                "score_support", "scenarios", "state", "access", "resources",
                "stored_outputs", "interpretation_limit", "dependence_limit"}
    require(set(existing) == required and existing.get("schema") == SCHEMA
            and existing.get("status") == STATUS
            and existing.get("request_sha256") == request_sha256,
            "existing region receipt schema/request differs")
    selection = existing["selection"]
    require(selection == {"case_id": args.case_id, "region_id": args.region_id,
            "role": "FIT", "people": 128, "people_per_microbatch": 8,
            "model_condition": "NONE", "checkpoint_step": 96},
            "existing region receipt selection differs")
    authentication = existing["authentication"]
    expected = {
        "FH_package_manifest_sha256": args.package_manifest_sha256,
        "FL_control_package_manifest_sha256": args.control_package_manifest_sha256,
        "execution_image": FROZEN_IMAGE,
        "NONE_route_sha256": args.none_route_sha256,
        "FI_prediction_index_sha256": args.prediction_index_sha256,
        "FK_product_plan_sha256": args.product_plan_sha256,
    }
    require(all(authentication.get(key) == value for key, value in expected.items())
            and authentication.get("original_reproduction", {}).get("exact") is True
            and set(existing["scenarios"]) == {"FLARE", "ORIGINAL", *CONDITIONS}
            and existing["state"].get("unchanged") is True
            and existing["state"].get("optimizer_constructed") is False
            and existing["state"].get("backward_called") is False
            and existing["access"] == {"fit_entries_reconstructed": 1,
                "select_entries_reconstructed": 0, "evaluate_opened": False}
            and existing["stored_outputs"] ==
                "AGGREGATE_SUMMARIES_AND_AUTHENTICATED_IDENTITIES_ONLY",
            "existing region receipt provenance or no-training guards differ")
    pinned = ((args.package + "/MANIFEST.json", args.package_manifest_sha256),
              (args.control_package_manifest, args.control_package_manifest_sha256),
              (args.fj_diagnostic, args.fj_diagnostic_sha256),
              (args.none_route, args.none_route_sha256),
              (args.ex_helper, args.ex_helper_sha256),
              (args.prediction_index, args.prediction_index_sha256),
              (args.product_plan, args.product_plan_sha256))
    require(all(Path(name).is_file() and sha256_file(name) == digest for name, digest in pinned),
            "current source identity differs from existing region receipt")
    route = read_pinned(args.none_route, args.none_route_sha256)
    contract_spec, resolved_spec = route.get("contract", {}), route.get("resolved", {})
    contract = read_pinned(contract_spec.get("path"), contract_spec.get("sha256"))
    read_pinned(resolved_spec.get("path"), resolved_spec.get("sha256"))
    require(authentication.get("NONE_contract_sha256") == contract_spec.get("sha256")
            and authentication.get("resolved_sha256") == resolved_spec.get("sha256"),
            "existing result contract/resolved identity differs")
    matches = []
    for entry in contract.get("batches", {}).get("FIT", []):
        receipt = read_pinned(entry.get("receipt_path"), entry.get("receipt_sha256"))
        if (receipt.get("case_id"), receipt.get("region_id")) == (
                args.case_id, args.region_id):
            matches.append(entry)
    require(len(matches) == 1
            and authentication.get("FIT_entry_sha256") == matches[0].get("sha256")
            and authentication.get("FIT_entry_receipt_sha256") ==
                matches[0].get("receipt_sha256"),
            "existing result FIT entry identity differs")
    checkpoint = authentication.get("checkpoint", {})
    current_path = Path(args.checkpoint_root) / "CURRENT"
    current = read_pinned(current_path, checkpoint.get("pointer_sha256"))
    manifest_path = Path(args.checkpoint_root) / current.get("manifest", "")
    manifest = read_pinned(manifest_path, checkpoint.get("manifest_sha256"))
    payload_path = Path(args.checkpoint_root) / manifest.get("payload", "")
    require(current.get("next_step") == manifest.get("next_step") == 96
            and sha256_file(payload_path) == checkpoint.get("payload_sha256"),
            "existing result checkpoint generation differs")
    prediction_index = json.loads(Path(args.prediction_index).read_bytes())
    saved = [item for item in prediction_index
             if item.get("input", {}).get("sha256") == matches[0].get("sha256")]
    require(len(saved) == 1
            and saved[0].get("sha256") == authentication.get("FI_prediction_sha256")
            and sha256_file(saved[0].get("path")) == saved[0].get("sha256"),
            "existing result saved FI prediction differs")
    for scenario in existing["scenarios"].values():
        metrics = scenario.get("metrics", {})
        count = metrics.get("observed_queries")
        require(metrics.get("aggregation") ==
                    "QUERY_POOLED_SUMS_OVER_ONE_FIXED_SCORE_MASK"
                and type(count) is int and count > 0
                and np.isfinite([metrics.get("log_loss_sum"), metrics.get("brier_sum"),
                                 metrics.get("state_error_sum")]).all()
                and len(metrics.get("dosage_abs_sum_AFR_EUR_NAM", [])) == 3,
                "existing region receipt metric sums/denominator differ")
    return existing


def parser():
    value = argparse.ArgumentParser()
    add = value.add_argument
    for name in ("package", "package-manifest-sha256", "control-package",
                 "control-package-manifest", "control-package-manifest-sha256",
                 "fj-diagnostic", "fj-diagnostic-sha256", "none-route",
                 "none-route-sha256", "checkpoint-root", "ex-helper",
                 "ex-helper-sha256", "prediction-index", "prediction-index-sha256",
                 "product-plan", "product-plan-sha256", "case-id", "region-id", "output"):
        add("--" + name, required=True)
    add("--people-per-microbatch", type=int, default=8)
    add("--threads", type=int, default=1)
    add("--original-atol", type=float, default=0.0)
    add("--max-wall-seconds", type=int, default=300)
    add("--max-rss-bytes", type=int, default=4 * 1024 ** 3)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    require(REGION.fullmatch(args.region_id or "") and args.people_per_microbatch == 8
            and args.threads == 1 and args.original_atol >= 0
            and args.max_wall_seconds > 0 and args.max_rss_bytes > 0,
            "region, microbatch, thread, tolerance, or resource limit differs")
    request = invocation_identity(args)
    request_sha256 = object_sha256(request)
    output = Path(args.output)
    if output.exists():
        validate_existing_output(output, args, request_sha256)
        print(json.dumps({"status": "EXISTING_AUTHENTICATED_RESULT", "output": str(output)}))
        return

    started = time.monotonic()
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(TimeoutError("wall limit exceeded")))
    signal.alarm(args.max_wall_seconds)
    require(sha256_file(args.ex_helper) == args.ex_helper_sha256,
            "FJ/EX frozen checkpoint helper differs")
    helper = load_module(args.ex_helper, "r02_frozen_fit_checkpoint_helper")
    require(sha256_file(args.fj_diagnostic) == args.fj_diagnostic_sha256,
            "FJ frozen dose diagnostic differs")
    fj = load_module(args.fj_diagnostic, "r02_fj_frozen_dose_diagnostic")
    frozen_python = fj.validate_fh_package(args.package, args.package_manifest_sha256, helper)
    validate_control_package(args.control_package, args.control_package_manifest,
                             args.control_package_manifest_sha256)
    plan = validate_product_plan(args.product_plan, args.product_plan_sha256)

    frozen_bin = str(Path(args.package).resolve() / "bin")
    sys.path.insert(0, frozen_bin)
    import torch
    import r02_context_checkpoint as checkpoint
    import r02_context_train as train
    import r02_lai_training_campaign as campaign
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    binding, microbatch, audit = load_control_helpers(args.control_package)

    route = helper.read_pinned(args.none_route, args.none_route_sha256)
    require(route.get("condition") == "NONE", "frozen route is not NONE")
    contract = helper.read_pinned(route["contract"]["path"], route["contract"]["sha256"])
    resolved = helper.read_pinned(route["resolved"]["path"], route["resolved"]["sha256"])
    train.validate_contract(contract)
    entry, entry_receipt = helper.find_entry(contract, "NONE", args.case_id, args.region_id)
    require(entry_receipt.get("role") == "FIT" and entry_receipt.get("condition") == "NONE",
            "selected entry is not NONE FIT")
    limit = campaign.validate_reconstruction_execution_limit(
        route["reconstruction_limit"]["path"], route["reconstruction_limit"]["sha256"],
        route["contract"]["sha256"], "lai_local_attention", "NONE")
    data, dataset = train.validate_training_entry(
        entry, contract, "FIT", resolved, max_reconstruction_bytes=limit)
    require(dataset is None and len(data["person_id"]) == 128,
            "dose inference requires all 128 people in one FIT region")

    model = train.TaskModel(resolved, contract)
    identity = train.checkpoint_identity(
        contract, route["contract"]["sha256"], route["resolved"]["sha256"],
        "NONE", 1701, "fp32", model.encoder.config.rare_pooling,
        model.encoder.config.locus_descriptor)
    checkpoint_info = helper.load_model_only(args.checkpoint_root, identity, checkpoint, model)
    model.eval()

    audit_contracts, views, bindings = {}, {"ORIGINAL": data}, {}
    for condition in CONDITIONS:
        candidate = copy.deepcopy(contract)
        candidate[binding.BLOCK] = binding_spec(binding, plan, condition)
        view = binding.apply_product(data, entry, candidate, "FIT", resolved)
        require(len(view["person_id"]) == 128
                and str(np.asarray(view["personal_dose_intervention"]).item()) == condition
                and bool(np.asarray(view["personal_dose_available"]).item()) ==
                    (condition == PERMUTED),
                "projected personal-dose view differs")
        for name in ("query_core", "truth_observed", "query_valid", "truth_state",
                     "query_chrom", "query_pos_bp"):
            require(np.array_equal(np.asarray(view[name]), np.asarray(data[name])),
                    "dose intervention changed score support or truth: " + name)
        audit_contracts[condition] = candidate
        views[condition] = view
        bindings[condition] = binding.checkpoint_binding_identity(candidate)

    predict = lambda current_model, view: microbatch.predict_person_microbatches(
        current_model, view, "NONE", args.people_per_microbatch).cpu().numpy()
    predictions, state_before = predict_preserving_state(
        model, views, predict, audit.training_state_sha256)
    state_after = audit.training_state_sha256(model)
    require(state_before == state_after, "frozen model or RNG state changed")

    index = json.loads(Path(args.prediction_index).read_bytes())
    require(sha256_file(args.prediction_index) == args.prediction_index_sha256
            and isinstance(index, list), "FI prediction index differs")
    matches = [item for item in index if item.get("input", {}).get("path") == entry["path"]
               and item.get("input", {}).get("sha256") == entry["sha256"]]
    require(len(matches) == 1 and sha256_file(matches[0]["path"]) == matches[0]["sha256"],
            "saved FI prediction identity differs")
    saved = np.load(matches[0]["path"], allow_pickle=False)
    original_reproduction = compare_saved_original(
        predictions["ORIGINAL"], saved, args.original_atol)

    score_mask = train.lai_loss_mask(data)
    support = score_support_identity(data, score_mask)
    scenarios = {"FLARE": {"metrics": probability_statistics(
        data["baseline_probabilities"], data, score_mask)}}
    for name in ("ORIGINAL", *CONDITIONS):
        scenarios[name] = {"metrics": probability_statistics(
            predictions[name], data, score_mask)}
        scenarios[name]["versus_flare"] = prediction_change(
            predictions[name], data["baseline_probabilities"], data, score_mask)
        if name != "ORIGINAL":
            scenarios[name]["versus_original"] = prediction_change(
                predictions[name], predictions["ORIGINAL"], data, score_mask)
            scenarios[name]["binding_sha256"] = object_sha256(bindings[name])

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak <= args.max_rss_bytes, "memory ceiling exceeded")
    elapsed = time.monotonic() - started
    result = {
        "schema": SCHEMA, "status": STATUS, "request_sha256": request_sha256,
        "selection": {"case_id": args.case_id, "region_id": args.region_id,
                      "role": "FIT", "people": 128, "people_per_microbatch": 8,
                      "model_condition": "NONE", "checkpoint_step": 96},
        "authentication": {
            "FH_package_manifest_sha256": args.package_manifest_sha256,
            "FH_frozen_python_sha256": frozen_python,
            "FL_control_package_manifest_sha256": args.control_package_manifest_sha256,
            "execution_image": FROZEN_IMAGE,
            "NONE_route_sha256": args.none_route_sha256,
            "NONE_contract_sha256": route["contract"]["sha256"],
            "resolved_sha256": route["resolved"]["sha256"],
            "checkpoint": checkpoint_info,
            "FIT_entry_sha256": entry["sha256"],
            "FIT_entry_receipt_sha256": entry["receipt_sha256"],
            "FI_prediction_index_sha256": args.prediction_index_sha256,
            "FI_prediction_sha256": matches[0]["sha256"],
            "FK_product_plan_sha256": args.product_plan_sha256,
            "original_reproduction": original_reproduction,
        },
        "score_support": support,
        "scenarios": scenarios,
        "state": {"training_state_sha256_before": state_before,
                  "training_state_sha256_after": state_after,
                  "unchanged": state_before == state_after,
                  "optimizer_constructed": False, "backward_called": False},
        "access": {"fit_entries_reconstructed": 1, "select_entries_reconstructed": 0,
                   "evaluate_opened": False},
        "resources": {"threads": args.threads,
                      "people_per_microbatch": args.people_per_microbatch,
                      "max_wall_seconds": args.max_wall_seconds,
                      "max_rss_bytes": args.max_rss_bytes,
                      "observed_wall_seconds": elapsed, "observed_peak_rss_bytes": peak},
        "stored_outputs": "AGGREGATE_SUMMARIES_AND_AUTHENTICATED_IDENTITIES_ONLY",
        "dependence_limit": (
            "EACH_CASE_HAS_128_MOSAICS_WITH_REUSED_DONORS;PEOPLE_QUERIES_AND_REGIONS_"
            "ARE_NOT_INDEPENDENT_BIOLOGICAL_REPLICATES"),
        "interpretation_limit": (
            "DESCRIPTIVE_FIT_INTERVENTION_ON_FROZEN_NONE_MODEL;MEASURES_MODEL_USE_"
            "NOT_RELEARNING_GENERALIZATION_BENEFIT_OR_BIOLOGICAL_CAUSALITY"),
    }
    write_idempotent(output, result, request_sha256)
    signal.alarm(0)
    print(json.dumps({"status": STATUS, "case_id": args.case_id,
                      "region_id": args.region_id, "wall_seconds": elapsed,
                      "peak_rss_bytes": peak}))


if __name__ == "__main__":
    main()
