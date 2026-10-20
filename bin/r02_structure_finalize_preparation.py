#!/usr/bin/env python3
"""Authenticate completed v8 preparation products and publish a final receipt.

This command copies only JSON preparation products. It never opens genotype or
target arrays and never imports the trainer or creates a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


HEX64 = re.compile(r"[0-9a-f]{64}")
READER_LIMITS_SHA256 = "d5fb2201e26f41d3ddc29a25fd93af7cfe4fd2cbb4042650b46e994dc9a463f8"
REQUESTS = {
    "fb706b61a27f8a65a1a5f4c9f3626f10977347e67aa9c21e454743865fd95b9f":
        (0.2, "11", 46258094, 109568),
    "d2995289f7f198491f8ddf10dcecc6ff14d6adc67ed5ed97437344c8eee492e2":
        (1.0, "3", 86933252, 187392),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + str(path))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError("nonfinite JSON value: " + token)))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")


def inventory(root):
    root = Path(root).resolve()
    return {str(path.relative_to(root)): {"bytes": path.stat().st_size,
            "sha256": sha256(path)} for path in sorted(root.rglob("*")) if path.is_file()}


def validate_recovery(root, receipt_path, receipt_sha256):
    root, receipt_path = Path(root).resolve(), Path(receipt_path).resolve()
    require(HEX64.fullmatch(receipt_sha256 or "") and sha256(receipt_path) == receipt_sha256,
            "v8 recovery receipt differs")
    receipt = load_json(receipt_path)
    require(receipt.get("schema") == "r02_structure_v8_finalization_recovery_v1" and
            receipt.get("status") ==
            "RECOVERED_COMPLETE_INDEX_TWO_MATERIALIZATIONS_TWO_READER_MEASUREMENTS_FINALIZER_FAILED_ONLY" and
            receipt.get("source_package_sha256_list") ==
            "41698cca8216736bba0bc9ea2a202f54be6c650bb669f3522cb86b2ad26bc5c3" and
            receipt.get("training_started") is False and
            receipt.get("evaluate_opened") is False,
            "v8 recovery identity/state differs")
    require(inventory(root) == receipt.get("files"), "v8 recovered file inventory differs")
    return receipt


def validate_index(path):
    index = load_json(path)
    require(index.get("schema") == "r02_structure_lazy_person_batch_index_v1" and
            index.get("status") == "PREPARED_SHARED_LAZY_INPUTS_FOR_ALL_528_TASKS_NO_TRAINING" and
            index.get("counts") == {"tasks": 528, "views": 88, "contexts": 6, "stores": 22} and
            index.get("reader_limits", {}).get("sha256") == READER_LIMITS_SHA256 and
            len(index.get("tasks", [])) == 528 and len(index.get("contexts", {})) == 6 and
            sum(row.get("recovered") is True for row in index["contexts"].values()) == 4 and
            index.get("training_allowed") is False and index.get("evaluate_allowed") is False,
            "recovered v8 index differs")
    expected_limits = sorted([100000] * 4 + [109568, 187392])
    require(sorted(row.get("max_window_sites") for row in index["contexts"].values()) ==
            expected_limits, "six reader limits differ")
    return index


def validate_materialized(root):
    directories = sorted(path for path in Path(root).iterdir() if path.is_dir())
    require({path.name.removeprefix("materialized-") for path in directories} == set(REQUESTS),
            "two materialized request identities differ")
    for directory in directories:
        result = load_json(directory / "trainer-entries.json")
        request_id = directory.name.removeprefix("materialized-")
        require(result.get("schema") == "r02_structure_materialized_lazy_trainer_entries_v1" and
                result.get("request_id") == request_id and
                result.get("family") == "structure_local_attention" and
                result.get("training_allowed") is False and
                result.get("evaluate_allowed") is False and
                set(result.get("entries", {})) == {"FIT", "SELECT"},
                "materialized trainer entry differs")
        for stage, entry in result["entries"].items():
            request = directory / (stage + ".autosomal-person-batches.json")
            require(request.is_file() and sha256(request) == entry["autosomal_request_sha256"],
                    "materialized autosomal request differs")


def validate_measurements(root):
    rows = []
    for path in sorted(Path(root).glob("*.measurement.json")):
        row = load_json(path); rows.append(row)
        require(row.get("request_id") in REQUESTS, "measurement request differs")
        radius, chromosome, core_bp, cap = REQUESTS[row["request_id"]]
        peak = row.get("peak_per_tile", {})
        require(row.get("schema") == "r02_structure_real_interval_reader_measurement_v1" and
                row.get("status") ==
                "COMPLETE_ONE_REAL_DENSE_INTERVAL_READER_MEASUREMENT_NO_MODEL_NO_TARGET" and
                row.get("family") == "structure_local_attention" and
                row.get("request_context", {}).get("rare_radius_cm") == radius and
                row.get("chromosome") == chromosome and row.get("requested_core_bp") == core_bp and
                row.get("authenticated_max_window_sites") == cap and row.get("autosomes") == 1 and
                row.get("person_batch_size") == 8 and
                max(peak.get("common_rows", cap + 1), peak.get("event_rows", cap + 1)) <= cap and
                row.get("peak_rss_bytes", 2**63) <= 2 * 1024**3 and
                row.get("target_arrays_read") is False and
                row.get("trainer_module_imported") is False and
                row.get("model_created") is False and row.get("training_started") is False and
                row.get("evaluate_opened") is False,
                "bounded real-interval measurement differs")
    require(len(rows) == 2 and {row["request_id"] for row in rows} == set(REQUESTS),
            "two measurements required")
    return rows


def finalize_staged(prepared, materialized, measurements, published_root, final_path,
                    recovered_v8_receipt=None):
    prepared, materialized = Path(prepared).resolve(), Path(materialized).resolve()
    measurements, final_path = Path(measurements).resolve(), Path(final_path).resolve()
    validate_index(prepared / "index.json")
    validate_materialized(materialized)
    rows = validate_measurements(measurements)
    require(not final_path.exists(), "fresh FINAL.json required")
    published = Path(published_root).resolve()
    for path in sorted(materialized.glob("materialized-*/trainer-entries.json")):
        result = load_json(path)
        for stage, entry in result["entries"].items():
            entry["autosomal_request_path"] = str(
                published / "materialized-final" / path.parent.name /
                (stage + ".autosomal-person-batches.json"))
        write_json(path, result)
    files = {}
    for prefix, directory in (("prepared-final", prepared),
                              ("materialized-final", materialized),
                              ("measurements-final", measurements)):
        for name, descriptor in inventory(directory).items():
            files[prefix + "/" + name] = descriptor
    final = {"schema": "r02_structure_lazy_batch_preparation_final_v1",
        "status": "COMPLETE_SHARED_528_INDEX_AND_TWO_BOUNDED_REAL_INTERVAL_READER_MEASUREMENTS_NO_TRAINING",
        "measurements": rows, "outputs_sha256": {name: row["sha256"] for name, row in files.items()},
        "training_started": False, "evaluate_opened": False}
    if recovered_v8_receipt is not None:
        final["recovered_v8_receipt"] = recovered_v8_receipt
    write_json(final_path, final)
    return final


def finalize(recovery_root, receipt_path, receipt_sha256, published_root, output):
    recovery_root, output = Path(recovery_root).resolve(), Path(output).resolve()
    receipt = validate_recovery(recovery_root, receipt_path, receipt_sha256)
    require(not output.exists(), "fresh finalization output required")
    output.mkdir(parents=True)
    shutil.copytree(recovery_root / "prepared", output / "prepared-final")
    shutil.copytree(recovery_root / "materialized", output / "materialized-final")
    shutil.copytree(recovery_root / "measurements", output / "measurements-final")
    recovered = {"path": str(Path(receipt_path).resolve()), "sha256": receipt_sha256,
                 "source_finalizer_exit": receipt["source_finalizer_exit"]}
    return finalize_staged(output / "prepared-final", output / "materialized-final",
                           output / "measurements-final", published_root,
                           output / "FINAL.json", recovered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("recover")
    recover.add_argument("--recovery-root", required=True)
    recover.add_argument("--recovery-receipt", required=True)
    recover.add_argument("--recovery-receipt-sha256", required=True)
    recover.add_argument("--published-root", required=True)
    recover.add_argument("--output", required=True)
    staged = commands.add_parser("staged")
    staged.add_argument("--prepared", required=True)
    staged.add_argument("--materialized", required=True)
    staged.add_argument("--measurements", required=True)
    staged.add_argument("--published-root", required=True)
    staged.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "recover":
        result = finalize(args.recovery_root, args.recovery_receipt,
                          args.recovery_receipt_sha256, args.published_root, args.output)
    else:
        result = finalize_staged(args.prepared, args.materialized, args.measurements,
                                 args.published_root, args.output)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
