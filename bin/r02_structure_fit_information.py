#!/usr/bin/env python3
"""Describe chr22 rare information in FIT without aggregating SELECT/EVALUATE.

The store keeps all roles in one C-contiguous array.  File hashing authenticates
bytes but does not interpret genotypes.  Genotype calculations index only the
frozen FIT rows.  This diagnostic does not train, select, or score a model.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np

from r02_event_mmap_reader import mapped_arrays, signature
from r02_genomic_event_inputs import ARRAY_FIELDS
from r02_lai_fit_information import counts, describe


SCHEMA = "r02_structure_chr22_fit_information_v1"
SCOPES = ("FULL_CHR22", "EXCLUDE_CHR22_PART01_MARGIN", "EXCLUDE_CHR22_PART02_MARGIN",
          "WHOLE_CHR22_EXCLUDED")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_roles(path):
    rows = list(csv.DictReader(Path(path).open(newline=""), delimiter="\t"))
    require(len(rows) == 2619 and len({r["sample_id"] for r in rows}) == len(rows),
            "Person axis differs")
    require({role: sum(r["role"] == role for r in rows)
             for role in ("FIT", "SELECT", "EVALUATE")} ==
            {"FIT": 1460, "SELECT": 476, "EVALUATE": 683}, "Role counts differ")
    fit = np.flatnonzero(np.asarray([r["role"] == "FIT" for r in rows]))
    return rows, fit


def read_exclusions(path):
    with Path(path).open(newline="") as stream:
        rows = {r["block_id"]: r for r in csv.DictReader(stream, delimiter="\t")
                if r["chromosome"] == "22"}
    require(set(rows) == {"chr22_part01", "chr22_part02"}, "Exact chr22 blocks required")
    result = {"FULL_CHR22": None, "WHOLE_CHR22_EXCLUDED": (1, 50818468)}
    for block, scope in (("chr22_part01", "EXCLUDE_CHR22_PART01_MARGIN"),
                         ("chr22_part02", "EXCLUDE_CHR22_PART02_MARGIN")):
        row = rows[block]
        result[scope] = (int(row["excluded_start_bp"]), int(row["excluded_end_bp"]))
    require(result["EXCLUDE_CHR22_PART01_MARGIN"] == (1, 34462475) and
            result["EXCLUDE_CHR22_PART02_MARGIN"] == (32288674, 50818468),
            "Chr22 exclusion margins differ")
    return result


def scope_masks(position_bp, exclusions):
    return {name: (np.ones(len(position_bp), bool) if interval is None else
                   ~((position_bp >= interval[0]) & (position_bp <= interval[1])))
            for name, interval in exclusions.items()}


def validate_headers(data, chunk, people):
    require(set(data) == ARRAY_FIELDS, "NPZ member inventory differs")
    n = chunk["n_sites"]
    for name, value in data.items():
        expected = (people, n) if name in {"dosage", "observed"} else (n,)
        require(value.shape == expected, "Array shape differs: " + name)
    require(data["dosage"].dtype == np.uint8 and data["observed"].dtype == np.bool_,
            "Dose/mask type differs")


def select_fit(dose, observed, fit_indices):
    """Copy and validate only authenticated FIT rows from mapped cohort arrays."""
    fit_dose = np.asarray(dose[fit_indices, :])
    fit_observed = np.asarray(observed[fit_indices, :])
    require(np.all(fit_dose[fit_observed] <= 2) and
            np.all(fit_dose[~fit_observed] == 255), "FIT dose/mask differs")
    return fit_dose, fit_observed


def process_chunk(path, chunk, fit_indices, exclusions, out_rows):
    before = signature(path)
    require(before[2] == chunk["bytes"] and sha256(path) == chunk["sha256"],
            "Chunk size/hash differs")
    with mapped_arrays(path) as data:
        validate_headers(data, chunk, 2619)
        pos = np.asarray(data["position_bp"])
        require(pos.dtype == np.int64 and np.all(np.diff(pos) > 0) and
                pos[0] == chunk["start_bp"] and pos[-1] == chunk["end_bp"],
                "Chunk position identity differs")
        source_ac = np.asarray(data["allele_count"])
        source_an = np.asarray(data["allele_number"])
        source_frequency = np.asarray(data["frequency"])
        counted = np.asarray(data["counted_allele"])
        require(np.all(source_an > 0) and np.all((source_ac >= 0) & (source_ac <= source_an)) and
                np.allclose(source_frequency, source_ac/source_an, rtol=0, atol=1e-12) and
                np.all(counted <= 1), "Source-minor metadata differs")
        # This is the only genotype-value access: advanced indexing copies FIT rows only.
        dose, observed = select_fit(data["dosage"], data["observed"], fit_indices)
        c = counts(dose, observed)
        masks = scope_masks(pos, exclusions)
        person = {}
        locus = {}
        for scope, mask in masks.items():
            d, o = dose[:, mask], observed[:, mask]
            person[scope] = {
                "carrier_sites": (((d > 0) & o).sum(1)).astype(int).tolist(),
                "observed_sites": o.sum(1).astype(int).tolist(),
                "minor_copies": np.where(o, d, 0).sum(1).astype(int).tolist()
            }
            cm = {key: np.asarray(value)[mask] for key, value in c.items()}
            locus[scope] = {
                "loci": int(mask.sum()), "observed": int(cm["n"].sum()),
                "missing": int(1460*mask.sum() - cm["n"].sum()),
                "unobserved_loci": int((cm["n"] == 0).sum()),
                "observed_ac_zero": int(((cm["n"] > 0) & (cm["ac"] == 0)).sum()),
                "singletons": int((cm["ac"] == 1).sum()),
                "doubletons": int((cm["ac"] == 2).sum()),
                "ac_ge_3": int((cm["ac"] >= 3).sum()),
                "carriers_ge_2": int((cm["carriers"] >= 2).sum()),
                "carriers_ge_5": int((cm["carriers"] >= 5).sum()),
                "carriers_ge_10": int((cm["carriers"] >= 10).sum()),
                "variable_loci": int(cm["variable"].sum()),
                "positive_variance_loci": int((cm["variance"] > 0).sum())
            }
        fields = ("position_bp", "counted_allele", "source_ac", "source_an", "source_frequency",
                  "fit_observed_people", "fit_n0", "fit_n1", "fit_n2", "fit_ac", "fit_an",
                  "fit_carriers", "fit_variance", "fit_variable", "retained_after_part01_margin",
                  "retained_after_part02_margin")
        with gzip.open(out_rows, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fields, delimiter="\t")
            for i in range(len(pos)):
                writer.writerow({"position_bp": int(pos[i]), "counted_allele": int(counted[i]),
                    "source_ac": int(source_ac[i]), "source_an": int(source_an[i]),
                    "source_frequency": float(source_frequency[i]),
                    "fit_observed_people": int(c["n"][i]), "fit_n0": int(c["n0"][i]),
                    "fit_n1": int(c["n1"][i]), "fit_n2": int(c["n2"][i]),
                    "fit_ac": int(c["ac"][i]), "fit_an": int(c["an"][i]),
                    "fit_carriers": int(c["carriers"][i]), "fit_variance": float(c["variance"][i]),
                    "fit_variable": bool(c["variable"][i]),
                    "retained_after_part01_margin": bool(masks["EXCLUDE_CHR22_PART01_MARGIN"][i]),
                    "retained_after_part02_margin": bool(masks["EXCLUDE_CHR22_PART02_MARGIN"][i])})
    require(signature(path) == before, "Chunk changed during diagnostic")
    return {"schema": SCHEMA + "_chunk", "chunk": chunk, "source_signature": before,
            "fit_people": len(fit_indices), "person": person, "locus": locus,
            "rows_sha256": sha256(out_rows)}


def add_people(total, current):
    if total is None:
        return {k: np.asarray(v, dtype=np.int64) for k, v in current.items()}
    for key in total:
        total[key] += np.asarray(current[key], dtype=np.int64)
    return total


def describe_people(values):
    """Reuse the LAI summary and add the pre-specified fifth percentile."""
    summary = describe(values)
    if summary is not None:
        summary["p05"] = float(np.quantile(np.asarray(values), .05))
    return summary


def run(store, expected_manifest_sha256, output, max_seconds=300, max_rss_mib=512):
    started = time.monotonic()
    root, output = Path(store).resolve(), Path(output).resolve()
    manifest_path = root/"manifest.json"
    require(sha256(manifest_path) == expected_manifest_sha256, "Manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == "r02_genomic_event_chromosome_v1" and
            manifest["chromosome"] == "22" and manifest["n_people"] == 2619 and
            manifest["role_counts"] == {"EVALUATE": 683, "FIT": 1460, "SELECT": 476},
            "Chr22 cohort manifest differs")
    require(len(manifest["channels"]["rare"]) == 816 and
            sum(c["n_sites"] for c in manifest["channels"]["rare"]) == 417601,
            "Chr22 rare chunk universe differs")
    persons = root/"source/persons.tsv"; blocks = root/"source/blocks.tsv"
    require(sha256(persons) == manifest["outputs_sha256"]["source/persons.tsv"] and
            sha256(blocks) == manifest["outputs_sha256"]["source/blocks.tsv"] and
            sha256(root/"source/materialize.json") == manifest["outputs_sha256"]["source/materialize.json"],
            "Person/block metadata differs")
    require(all(manifest["outputs_sha256"].get(c["path"]) == c["sha256"]
                for c in manifest["channels"]["rare"]), "Rare output inventory differs")
    _, fit_indices = read_roles(persons); exclusions = read_exclusions(blocks)
    output.mkdir(parents=True, exist_ok=True); chunk_root = output/"chunks"; chunk_root.mkdir(exist_ok=True)
    identity = {"schema": SCHEMA, "store": str(root), "manifest_sha256": expected_manifest_sha256,
                "persons_sha256": sha256(persons), "blocks_sha256": sha256(blocks),
                "code_sha256": sha256(__file__), "role": "FIT", "fit_people": 1460}
    identity_path = output/"identity.json"
    if identity_path.exists():
        require(json.loads(identity_path.read_text()) == identity, "Resume identity differs")
    else:
        write_json(identity_path, identity)
    final = output/"summary.json"
    if final.exists():
        prior = json.loads(final.read_text())
        require(prior["identity"] == identity and
                sha256(output/"variant-fit-information.tsv.gz") == prior["variant_table_sha256"],
                "Finished diagnostic differs")
        return {"status": "FINISHED_SKIP", "summary": str(final)}
    for index, chunk in enumerate(manifest["channels"]["rare"]):
        rows = chunk_root/f"{index:04d}.tsv.gz"; receipt = chunk_root/f"{index:04d}.receipt.json.gz"
        if receipt.exists():
            with gzip.open(receipt, "rt") as stream: saved = json.load(stream)
            require(saved["chunk"] == chunk and list(signature(root/chunk["path"])) == saved["source_signature"]
                    and sha256(rows) == saved["rows_sha256"],
                    "Chunk checkpoint differs")
            continue
        if time.monotonic()-started >= max_seconds:
            progress = {"identity": identity, "status": "INCOMPLETE_TIME_LIMIT_NO_CONCLUSION",
                "completed_chunks": index, "total_chunks": 816,
                "authenticated_chunk_bytes": sum(c["bytes"] for c in manifest["channels"]["rare"][:index]),
                "elapsed_seconds": time.monotonic()-started, "training_started": False}
            write_json(output/"progress.json", progress)
            return {"status": progress["status"], "progress": str(output/"progress.json")}
        saved = process_chunk(root/chunk["path"], chunk, fit_indices, exclusions, rows)
        tmp = receipt.with_suffix(receipt.suffix + ".tmp")
        with gzip.open(tmp, "wt") as stream: json.dump(saved, stream, separators=(",", ":"), allow_nan=False)
        os.replace(tmp, receipt)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= max_rss_mib*1024,
                "Diagnostic memory budget exhausted")
        write_json(output/"progress.json", {"identity": identity, "status": "RUNNING_NO_CONCLUSION",
            "completed_chunks": index+1, "total_chunks": 816,
            "authenticated_chunk_bytes": sum(c["bytes"] for c in manifest["channels"]["rare"][:index+1]),
            "elapsed_seconds": time.monotonic()-started, "training_started": False})
    people = {scope: None for scope in SCOPES}
    locus = {scope: {} for scope in SCOPES}
    table = output/"variant-fit-information.tsv.gz"; temporary = table.with_suffix(".tmp.gz")
    fields = ("position_bp", "counted_allele", "source_ac", "source_an", "source_frequency",
              "fit_observed_people", "fit_n0", "fit_n1", "fit_n2", "fit_ac", "fit_an",
              "fit_carriers", "fit_variance", "fit_variable", "retained_after_part01_margin",
              "retained_after_part02_margin")
    with gzip.open(temporary, "wt", newline="") as out:
        out.write("\t".join(fields) + "\n")
        for index, chunk in enumerate(manifest["channels"]["rare"]):
            receipt = chunk_root/f"{index:04d}.receipt.json.gz"
            with gzip.open(receipt, "rt") as stream: saved = json.load(stream)
            for scope in SCOPES:
                people[scope] = add_people(people[scope], saved["person"][scope])
                for key, value in saved["locus"][scope].items():
                    locus[scope][key] = locus[scope].get(key, 0) + value
            with gzip.open(chunk_root/f"{index:04d}.tsv.gz", "rt") as rows:
                shutil.copyfileobj(rows, out)
    os.replace(temporary, table)
    result = {"identity": identity, "status": "FIT_INFORMATION_ONLY_NO_TRAINING_NO_SELECT_EVALUATE_AGGREGATION",
        "chromosome": "22", "fit_people": 1460, "rare_loci": 417601,
        "counted_allele_encoding": {"0": "REF", "1": "ALT"},
        "source_rare_rule": json.loads((root/"source/materialize.json").read_text())["source_rare_rule"],
        "scopes": {}, "variant_table": str(table), "variant_table_sha256": sha256(table),
        "chunks": 816, "logical_fit_dose_mask_bytes": 1219394920,
        "elapsed_seconds": time.monotonic()-started,
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "training_started": False, "select_or_evaluate_genotypes_aggregated": False,
        "limits": ["Hashing authenticates bytes; it does not interpret genotype values.",
                   "Interleaved rows prevent a promise that the operating system reads no adjacent pages.",
                   "Counts describe FIT information, not predictive value, power, or independent loci."]}
    for scope in SCOPES:
        result["scopes"][scope] = {"locus": locus[scope], "people": {
            name: describe_people(values) for name, values in people[scope].items()}}
    write_json(final, result)
    write_json(output/"progress.json", {"identity": identity, "status": "COMPLETE",
        "completed_chunks": 816, "total_chunks": 816,
        "authenticated_chunk_bytes": sum(c["bytes"] for c in manifest["channels"]["rare"]),
        "elapsed_seconds": result["elapsed_seconds"], "training_started": False})
    return {"status": result["status"], "summary": str(final),
            "variant_table_sha256": result["variant_table_sha256"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True); p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--output", required=True); p.add_argument("--max-seconds", type=int, default=180)
    p.add_argument("--max-rss-mib", type=int, default=512)
    a = p.parse_args()
    print(json.dumps(run(a.store, a.manifest_sha256, a.output, a.max_seconds, a.max_rss_mib)))
