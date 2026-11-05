"""Materialize matched-dose interventions from authenticated existing FIT arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from r02_lai_control_views import load, require, sha256, array_sha256, _descriptor
from r02_lai_density_control import authenticate_event_axis
from r02_lai_dose_controls import transform, chromosome_prefixes
from r02_lai_query_burden import build_prefix

SCHEMA = "r02_lai_personal_dose_preparation_v1"


def prepare(request_path, request_sha256, output):
    started = time.monotonic()
    require(sha256(request_path) == request_sha256, "preparation request hash differs")
    request = json.loads(Path(request_path).read_text())
    require(set(request) == {"schema", "source_link", "role", "case_id", "chromosome"}
            and request["schema"] == SCHEMA and request["role"] == "FIT", "FIT-only request required")
    link_path = _descriptor(request["source_link"], "source link")
    link = json.loads(link_path.read_text())
    require(link.get("condition") == "NONE" and all(request[k] == link.get(k)
            for k in ("case_id", "role", "chromosome")), "original real-dose FIT source required")
    descriptors = {name: link[name] for name in ("control_request", "control_receipt", "control_batch")}
    paths = {name: _descriptor(value, name) for name, value in descriptors.items()}
    arrays, receipt = load(paths["control_request"], descriptors["control_request"]["sha256"],
                           paths["control_receipt"], descriptors["control_receipt"]["sha256"], paths["control_batch"])
    require(all(request[k] == receipt.get(k) for k in ("case_id", "role", "chromosome")),
            "control identity differs from requested FIT case")
    control_request = json.loads(paths["control_request"].read_text())
    source = control_request["factorized_receipt"]
    source_path = _descriptor(source, "factorized receipt")
    source_receipt = json.loads(source_path.read_text())
    index_path = source_path.parent / "indices.npz"
    require(sha256(index_path) == source_receipt["outputs_sha256"]["indices.npz"], "index hash differs")
    with np.load(index_path, allow_pickle=False) as z:
        indices = {name: z[name] for name in z.files if name == "person_id" or name.startswith("event_")}
    authenticate_event_axis(arrays, indices, np)
    require(np.array_equal(arrays["person_id"].astype(str), indices["person_id"].astype(str)), "people axis differs")
    cm = indices["event_position_cm"].astype(np.float64)
    chrom = indices["event_chromosome"].astype(np.int64)
    original_prefix = build_prefix(arrays["NONE_event_dosage"], arrays["NONE_event_valid"], cm, chrom, "NONE", np)
    prefix_path = _descriptor(link["event_burden_prefix"], "existing original counts")
    with np.load(prefix_path, allow_pickle=False) as z:
        require(all(key in z and np.array_equal(value, z[key]) for key, value in original_prefix.items()),
                "full-chromosome original counts differ from existing authenticated product")
    fields = [arrays["event_" + name].astype(str) for name in ("chromosome", "position_bp", "ref", "alt", "counted_allele")]
    keys = np.asarray([json.dumps(row, separators=(",", ":")) for row in zip(*fields)])
    views, metrics = transform(arrays["NONE_event_dosage"], arrays["NONE_event_valid"],
        person_ids=arrays["person_id"].astype(str), expected_person_ids=indices["person_id"].astype(str),
        locus_keys=keys, role=receipt["role"], expected_role=request["role"],
        stratum_id=json.dumps([receipt["case_id"], receipt["direction"]]), seed=receipt["seed"])
    from r02_lai_dose_controls import REMOVED, PERMUTED
    observed = arrays["NONE_event_valid"]
    require(not views[REMOVED].any(), "removed personal dose channel must be constant zero")
    for locus in range(observed.shape[1]):
        mask = observed[:, locus]
        require(np.array_equal(np.sort(views[PERMUTED][mask, locus]),
                np.sort(arrays["NONE_event_dosage"][mask, locus])), "permuted locus marginal changed")
    prefixes = chromosome_prefixes(views, arrays["NONE_event_valid"], cm, chrom,
        person_ids=arrays["person_id"].astype(str), event_global_index=arrays["event_global_index"],
        expected_event_count=receipt["event_loci"])
    for prefix in prefixes.values():
        require(np.array_equal(prefix["prefix_counts"][..., 2], original_prefix["prefix_counts"][..., 2]),
                "observed-locus denominator changed")
    product = {name: value for name, value in arrays.items() if name.startswith("event_") or name == "person_id"}
    product.update(event_valid=arrays["NONE_event_valid"], event_cm=cm)
    for name, doses in views.items():
        product[name + "_event_dosage"] = doses
        product.update({name + "_" + key: value for key, value in prefixes[name].items()})
    out = Path(output)
    require(not out.exists(), "output must be new; reuse verified previous product rather than overwrite")
    out.mkdir(parents=True, mode=0o700)
    batch = out / "personal-dose-views.npz"
    np.savez_compressed(batch, **product)
    hashes = {name: array_sha256(value, np) for name, value in product.items()}
    with np.load(batch, allow_pickle=False) as z:
        require(set(z.files) == set(hashes) and all(array_sha256(z[k], np) == v for k, v in hashes.items()),
                "written arrays failed exact readback")
    report = dict(schema=SCHEMA, status="TWO_VIEWS_AND_FULL_CHROMOSOME_COUNTS_VERIFIED_PREPARE_ONLY",
        request_sha256=request_sha256, source_link=request["source_link"], source_controls=descriptors,
        source_index_sha256=sha256(index_path), source_factorized_receipt=source,
        reference_identity=receipt["reference_identity"], metrics=metrics,
        batch_sha256=sha256(batch), bytes=batch.stat().st_size, arrays_sha256=hashes,
        elapsed_seconds=time.monotonic()-started,
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
            ("r02_lai_dose_controls.py", "r02_lai_dose_control_prepare.py", "r02_lai_query_burden.py")},
        truth_loaded=False, training_authorized=False, evaluate_opened=False,
        limit="Input interventions with rare population references retained, not biological validation or an exchangeable null.")
    (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "bytes", "elapsed_seconds")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("request", "request-sha256", "output"): parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    prepare(args.request, args.request_sha256, args.output)
