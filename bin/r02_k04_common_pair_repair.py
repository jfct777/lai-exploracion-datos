#!/usr/bin/env python3
"""Recover 11,872 omitted K04 input pairs from common-genotype events.

The event archives contain every FIT-eligible common marker before LD pruning.
This producer therefore intersects them with the exact ``common_markers.tsv``
written by the corresponding full-chromosome legacy partial.  It audits every
legacy pair on that chromosome and emits only the omitted same-component FIT
pairs plus narrow per-person sufficient statistics.  It never fits a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np

REQUEST_SCHEMA = "r02_k04_common_pair_repair_request_v2"
CHROM_SCHEMA = "r02_k04_common_pair_repair_chromosome_v1"
AGG_SCHEMA = "r02_k04_common_pair_repair_aggregate_v1"
ROLES = ("FIT", "SELECT", "EVALUATE")
BASES = np.array(list("ACGT"))
PARTIAL_FIELDS = {
    "pairs", "common_numerator", "common_joint", "common_zsum",
    "common_called", "common_sites", "rare_copies", "sample_ids",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path, chunk=2**20):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            value = handle.read(chunk)
            if not value:
                return h.hexdigest()
            h.update(value)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def authenticate(path, descriptor, label):
    path = Path(path)
    require(path.is_file(), f"Missing {label}")
    require(path.stat().st_size == descriptor["bytes"], f"{label} byte count differs")
    require(sha256_file(path) == descriptor["sha256"], f"{label} SHA256 differs")


def validate_descriptor(value, label):
    require(set(value) == {"uri", "generation", "bytes", "md5_base64", "sha256"},
            f"Invalid {label} descriptor fields")
    require(isinstance(value["uri"], str) and value["uri"].startswith("gs://projects-usp/"),
            f"Invalid {label} URI")
    require(re.fullmatch(r"[1-9][0-9]*", value["generation"] or "") is not None,
            f"Invalid {label} generation")
    require(type(value["bytes"]) is int and value["bytes"] > 0, f"Invalid {label} bytes")
    require(re.fullmatch(r"[0-9a-f]{64}", value["sha256"] or "") is not None,
            f"Invalid {label} SHA256")
    require(isinstance(value["md5_base64"], str) and value["md5_base64"],
            f"Invalid {label} MD5")


def validate_request(request):
    require(request.get("schema") == REQUEST_SCHEMA, "Unsupported request schema")
    require(request.get("scientific_scope") ==
            "SOURCE_REPAIR_ONLY_NO_MODEL_FIT_NO_OUTCOME_READ",
            "Wrong scientific scope")
    require(request.get("whole_target_chromosome_exclusion") is True,
            "Whole target chromosome exclusion must be explicit")
    require(request.get("evaluated_outcomes_read") is False and
            request.get("biological_validation") is False,
            "Repair cannot read outcomes or claim biological validation")
    require(set(request.get("chromosomes", {})) == {str(i) for i in range(1, 23)},
            "Exact autosomes 1..22 required")
    validate_descriptor(request.get("supplement_archive", {}), "common supplement archive")
    require(request.get("population") == {"n_people": 2619,
            "roles": {"FIT": 1460, "SELECT": 476, "EVALUATE": 683}},
            "Frozen population differs")
    require(request.get("persons_sha256") ==
            "6a9979bf9b169dd4a6a27c354463147a53c11479423baba986039ae4b6f73757",
            "Frozen persons/components digest differs")
    resources = request.get("resources", {})
    require(set(resources) == {"matrix_site_batch", "max_archive_bytes",
            "max_partial_bytes", "max_marker_bytes"},
            "Exact resource bounds required")
    require(type(resources["matrix_site_batch"]) is int and
            1 <= resources["matrix_site_batch"] <= 8192, "Invalid matrix batch")
    require(type(resources["max_archive_bytes"]) is int and
            resources["max_archive_bytes"] >= max(x["event_archive"]["bytes"]
            for x in request["chromosomes"].values()), "Archive byte bound too small")
    require(type(resources["max_partial_bytes"]) is int and
            resources["max_partial_bytes"] >= max(x["partial"]["bytes"]
            for x in request["chromosomes"].values()), "Partial byte bound too small")
    require(type(resources["max_marker_bytes"]) is int and
            resources["max_marker_bytes"] >= max(x["common_markers"]["bytes"]
            for x in request["chromosomes"].values()), "Marker byte bound too small")
    for chrom, item in request["chromosomes"].items():
        require(set(item) == {"event_archive", "partial", "partial_manifest",
                "common_markers", "partial_task_id", "partial_contract_sha256",
                "sample_ids_sha256", "fit_ids_sha256", "supplement"},
                f"Invalid chromosome {chrom} source fields")
        for key in ("event_archive", "partial", "partial_manifest", "common_markers"):
            validate_descriptor(item[key], f"chr{chrom} {key}")
        base = item["partial"]["uri"].rsplit("/", 1)[0]
        require(item["partial_manifest"]["uri"] == base + "/manifest.json" and
                item["common_markers"]["uri"] == base + "/common_markers.tsv",
                f"Chromosome {chrom} siblings do not share the pinned partial")
        require(re.fullmatch(r"[0-9a-f]{64}", item["partial_contract_sha256"] or "") and
                re.fullmatch(r"[0-9a-f]{64}", item["sample_ids_sha256"] or "") and
                re.fullmatch(r"[0-9a-f]{64}", item["fit_ids_sha256"] or ""),
                f"Invalid chromosome {chrom} inherited hashes")
        supplement = item["supplement"]
        if supplement is not None:
            require(set(supplement) == {"manifest_sha256", "array_sha256", "array_bytes",
                    "n_sites", "original_source_sha256"} and
                    all(re.fullmatch(r"[0-9a-f]{64}", supplement[x] or "")
                        for x in ("manifest_sha256", "array_sha256", "original_source_sha256")) and
                    type(supplement["array_bytes"]) is int and supplement["array_bytes"] > 0 and
                    type(supplement["n_sites"]) is int and supplement["n_sites"] > 0,
                    f"Invalid chromosome {chrom} supplement descriptor")
    return request


def load_request(path, expected_sha):
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha or ""),
            "Request SHA256 is required")
    require(sha256_file(path) == expected_sha, "Request SHA256 differs")
    return validate_request(read_json(path))


def load_persons(raw, expected_sha):
    require(sha256_bytes(raw) == expected_sha, "persons.tsv SHA256 differs")
    rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t"))
    require(rows and list(rows[0]) == ["sample_id", "component", "role"],
            "persons.tsv columns/order differ")
    require(len(rows) == 2619 and len({x["sample_id"] for x in rows}) == 2619,
            "Frozen person roster differs")
    role_indices = {}
    for role, expected in (("FIT", 1460), ("SELECT", 476), ("EVALUATE", 683)):
        role_indices[role] = np.asarray([i for i, x in enumerate(rows) if x["role"] == role],
                                        dtype=np.int64)
        require(len(role_indices[role]) == expected, f"{role} count differs")
    require(all(x["role"] in ROLES and x["component"] for x in rows),
            "Invalid role/component")
    return rows, role_indices


def expected_pairs(persons, role_indices):
    n = len(persons)
    components = np.asarray([x["component"] for x in persons])
    legacy, missing, missing_roles = [], [], []
    for role in ROLES:
        idx = role_indices[role]
        a, b = np.triu_indices(len(idx), 1)
        a, b = idx[a], idx[b]
        same = components[a] == components[b]
        legacy.append(np.column_stack((a[~same], b[~same])))
        missing.append(np.column_stack((a[same], b[same])))
        missing_roles.append(np.full(int(same.sum()), ROLES.index(role), dtype=np.uint8))
    legacy = np.concatenate(legacy)
    legacy = legacy[np.argsort(legacy[:, 0] * n + legacy[:, 1])]
    missing = np.concatenate(missing)
    missing_roles = np.concatenate(missing_roles)
    order = np.argsort(missing[:, 0] * n + missing[:, 1])
    missing, missing_roles = missing[order], missing_roles[order]
    return legacy.astype(np.int64), missing.astype(np.int64), missing_roles


def load_markers(path, chromosome):
    rows = list(csv.DictReader(Path(path).open(), delimiter="\t"))
    require(rows and list(rows[0]) ==
            ["variant_id", "chromosome", "position_bp", "fit_alt_frequency"],
            "common_markers.tsv columns/order differ")
    selected = {}
    previous = 0
    for row in rows:
        require(row["chromosome"] == chromosome, "Marker chromosome differs")
        pos = int(row["position_bp"])
        parts = row["variant_id"].split(":")
        require(len(parts) == 4 and parts[0] == chromosome and int(parts[1]) == pos and
                parts[2] in "ACGT" and parts[3] in "ACGT" and parts[2] != parts[3],
                "Marker identity differs")
        p = float(row["fit_alt_frequency"])
        require(pos > previous and 0 < p < 1 and pos not in selected,
                "Markers must be unique, sorted and polymorphic")
        previous = pos
        selected[pos] = (parts[2], parts[3], p)
    return selected


def read_event_metadata(archive, chromosome):
    manifest_raw = persons_raw = None
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            if re.fullmatch(r"published/chr0?" + re.escape(chromosome) + r"/manifest.json", member.name):
                require(manifest_raw is None, "Repeated event manifest")
                manifest_raw = handle.extractfile(member).read()
            elif re.fullmatch(r"published/chr0?" + re.escape(chromosome) + r"/source/persons.tsv", member.name):
                require(persons_raw is None, "Repeated event persons.tsv")
                persons_raw = handle.extractfile(member).read()
    require(manifest_raw is not None and persons_raw is not None,
            "Event archive lacks published manifest/persons")
    manifest = json.loads(manifest_raw)
    require(manifest.get("schema") == "r02_genomic_event_chromosome_v1" and
            manifest.get("stage") == "COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING" and
            manifest.get("chromosome") == chromosome and
            manifest.get("common_policy") == "ALL_FIT_ELIGIBLE_WITHOUT_LD_PRUNING" and
            manifest.get("training_allowed") is False,
            "Event manifest scope differs")
    require(manifest["outputs_sha256"].get("source/persons.tsv") == sha256_bytes(persons_raw),
            "Event manifest does not authenticate persons.tsv")
    channels = manifest.get("channels", {}).get("common", [])
    require(channels and len({x["path"] for x in channels}) == len(channels),
            "Invalid common event channel")
    return manifest, persons_raw, {x["path"]: x for x in channels}


def add_batch(dosages, observed, frequencies, missing_pairs, audit_pairs, n_people, totals):
    dosage = np.column_stack(dosages).astype(np.float64, copy=False)
    mask = np.column_stack(observed).astype(bool, copy=False)
    p = np.asarray(frequencies, dtype=np.float64)
    z = np.where(mask, (dosage - 2 * p) / np.sqrt(2 * p * (1 - p)), 0.0)
    totals["called"] += mask.sum(axis=1)
    totals["zsum"] += z.sum(axis=1)
    totals["zsum_abs"] += np.abs(z).sum(axis=1)
    ma, mb = missing_pairs.T
    totals["missing_numerator"] += np.sum(z[ma] * z[mb], axis=1)
    aa, ab = audit_pairs.T
    products = z[aa] * z[ab]
    totals["audit_numerator"] += np.sum(products, axis=1)
    totals["audit_abs"] += np.sum(np.abs(products), axis=1)
    # Pairwise joint callability is sparse-complement arithmetic.  For pair
    # (i,j): sites - missing_i - missing_j + both_missing_ij.  This proves all
    # legacy denominators without a dense people-by-people matrix.
    for col in range(mask.shape[1]):
        absent = np.flatnonzero(~mask[:, col])
        totals["missing_count"][absent] += 1
        if len(absent) > 1:
            a, b = np.triu_indices(len(absent), 1)
            codes = absent[a] * n_people + absent[b]
            unique, counts = np.unique(codes, return_counts=True)
            for code, count in zip(unique.tolist(), counts.tolist()):
                totals["both_missing"][code] = totals["both_missing"].get(code, 0) + count


def load_supplement(archive, chromosome, descriptor, selected, *, n_people,
                    sample_ids_sha256, source_contract_sha256):
    """Authenticate exact loci omitted only by the mapped event territory."""
    if descriptor is None:
        return []
    prefix = f"published/supplements/common_supplement_chr{chromosome}"
    with tarfile.open(archive, "r:") as handle:
        manifest_raw = handle.extractfile(handle.getmember(prefix + "/manifest.json")).read()
        array_raw = handle.extractfile(handle.getmember(prefix + "/supplement.npz")).read()
    require(sha256_bytes(manifest_raw) == descriptor["manifest_sha256"],
            "Supplement manifest SHA256 differs")
    require(len(array_raw) == descriptor["array_bytes"] and
            sha256_bytes(array_raw) == descriptor["array_sha256"],
            "Supplement array authentication differs")
    manifest = json.loads(manifest_raw)
    require(manifest.get("schema") == "r02_structure_common_supplement_v1" and
            manifest.get("stage") == "EXACT_SELECTED_COMMON_LOCI_NO_RESELECTION" and
            manifest.get("chromosome") == chromosome and
            manifest.get("n_people") == n_people and
            manifest.get("n_sites") == descriptor["n_sites"] and
            manifest.get("sample_ids_sha256") == sample_ids_sha256 and
            manifest.get("source_contract_sha256") == source_contract_sha256 and
            manifest.get("original_source_sha256") == descriptor["original_source_sha256"] and
            manifest.get("counted_allele") == "ALT" and
            manifest.get("selection_changed") is False and
            manifest.get("target_outcomes_read") is False and
            manifest.get("array", {}).get("sha256") == descriptor["array_sha256"] and
            manifest.get("array", {}).get("bytes") == descriptor["array_bytes"],
            "Supplement manifest scope/linkage differs")
    required = {"dosage", "observed", "position_bp", "position_cm", "ref", "alt",
                "counted_allele", "allele_count", "allele_number", "frequency"}
    result = []
    with np.load(io.BytesIO(array_raw), allow_pickle=False) as arrays:
        require(set(arrays.files) == required and
                arrays["dosage"].shape == arrays["observed"].shape ==
                    (n_people, descriptor["n_sites"]),
                "Supplement array schema/dimensions differ")
        positions = arrays["position_bp"]
        require(np.array_equal(positions, np.sort(positions)) and
                positions.tolist() == manifest.get("position_bp"),
                "Supplement position order differs")
        for col, pos_value in enumerate(positions):
            pos = int(pos_value)
            require(pos in selected, "Supplement contains a non-selected marker")
            ref, alt, frequency = selected[pos]
            require(BASES[int(arrays["ref"][col])] == ref and
                    BASES[int(arrays["alt"][col])] == alt and
                    int(arrays["counted_allele"][col]) == 1 and
                    np.isclose(float(arrays["frequency"][col]), frequency,
                               rtol=0, atol=1e-12),
                    "Supplement selected-marker allele/frequency differs")
            dose = arrays["dosage"][:, col]
            mask = arrays["observed"][:, col]
            require(mask.dtype == np.bool_ and
                    np.all((dose[mask] >= 0) & (dose[mask] <= 2)) and
                    np.all(dose[~mask] == 255),
                    "Supplement missing/dosage encoding differs")
            result.append((pos, dose.copy(), mask.copy(), frequency))
    return result


def event_counts(archive, chromosome, selected, event_manifest, channel, supplement,
                 role_indices, batch_size, missing_pairs, audit_pairs):
    n_people = sum(len(x) for x in role_indices.values())
    require(batch_size == 512, "Exact legacy accumulation requires 512 selected markers per batch")
    totals = {"called": np.zeros(n_people, dtype=np.int64),
              "zsum": np.zeros(n_people, dtype=np.float64),
              "zsum_abs": np.zeros(n_people, dtype=np.float64),
              "missing_count": np.zeros(n_people, dtype=np.int64),
              "missing_numerator": np.zeros(len(missing_pairs), dtype=np.float64),
              "audit_numerator": np.zeros(len(audit_pairs), dtype=np.float64),
              "audit_abs": np.zeros(len(audit_pairs), dtype=np.float64),
              "both_missing": {}}
    dosages, observed, frequencies, found = [], [], [], set()
    supplement = iter(supplement)
    pending = next(supplement, None)
    def append_marker(pos, dose, mask, frequency):
        require(pos not in found, "Selected marker repeated across authenticated sources")
        dosages.append(dose.copy()); observed.append(mask.copy()); frequencies.append(frequency)
        found.add(pos)
        if len(dosages) == batch_size:
            add_batch(dosages, observed, frequencies, missing_pairs, audit_pairs,
                      n_people, totals)
            dosages.clear(); observed.clear(); frequencies.clear()
    common_prefix = re.compile(r"published/chr0?" + re.escape(chromosome) +
                               r"/(common\.[0-9]{6}\.npz)")
    seen_chunks = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            match = common_prefix.fullmatch(member.name)
            if not match:
                continue
            name = match.group(1)
            require(name in channel and name not in seen_chunks and member.isfile(),
                    "Unexpected/repeated common event chunk")
            raw = handle.extractfile(member).read()
            desc = channel[name]
            require(len(raw) == desc["bytes"] and sha256_bytes(raw) == desc["sha256"] and
                    event_manifest["outputs_sha256"].get(name) == desc["sha256"],
                    f"Common event chunk authentication failed: {name}")
            seen_chunks.add(name)
            with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                require(set(arrays.files) == {"position_bp", "position_cm", "ref", "alt",
                        "counted_allele", "allele_count", "allele_number", "frequency",
                        "dosage", "observed"}, "Common event arrays differ")
                positions = arrays["position_bp"]
                require(len(positions) == desc["n_sites"] and
                        arrays["dosage"].shape == (n_people, len(positions)) and
                        arrays["observed"].shape == (n_people, len(positions)),
                        "Common event dimensions differ")
                for col, pos_value in enumerate(positions):
                    pos = int(pos_value)
                    wanted = selected.get(pos)
                    if wanted is None:
                        continue
                    while pending is not None and pending[0] < pos:
                        append_marker(*pending)
                        pending = next(supplement, None)
                    ref, alt, p = wanted
                    require(BASES[int(arrays["ref"][col])] == ref and
                            BASES[int(arrays["alt"][col])] == alt and
                            int(arrays["counted_allele"][col]) == 1 and
                            np.isclose(float(arrays["frequency"][col]), p, rtol=0, atol=1e-12),
                            "Selected-marker allele/frequency mapping differs")
                    dose = arrays["dosage"][:, col]
                    mask = arrays["observed"][:, col]
                    require(mask.dtype == np.bool_ and
                            np.all((dose[mask] >= 0) & (dose[mask] <= 2)) and
                            np.all(dose[~mask] == 255), "Event missing/dosage encoding differs")
                    append_marker(pos, dose, mask, p)
    while pending is not None:
        append_marker(*pending)
        pending = next(supplement, None)
    require(seen_chunks == set(channel), "Event archive common chunk inventory differs")
    require(found == set(selected), "LD-selected marker absent from event source")
    if dosages:
        add_batch(dosages, observed, frequencies, missing_pairs, audit_pairs, n_people, totals)
    return totals


def compute_joint(pairs, n_sites, missing_count, both_missing, n_people):
    a, b = pairs.T
    result = n_sites - missing_count[a] - missing_count[b]
    if both_missing:
        codes = a * n_people + b
        order = np.argsort(codes)
        sorted_codes = codes[order]
        for code, count in both_missing.items():
            pos = np.searchsorted(sorted_codes, code)
            if pos < len(sorted_codes) and sorted_codes[pos] == code:
                result[order[pos]] += count
    return result.astype(np.int64, copy=False)


def pair_rows_for_role(pair_role, role, *, evaluate_admitted=False):
    """Return one role's input rows, keeping EVALUATE closed by default."""
    require(role in ROLES, "Unknown pair role")
    values = np.asarray(pair_role)
    require(values.ndim == 1 and values.dtype.kind in "iu" and
            np.all(np.isin(values, [0, 1, 2])), "Invalid pair-role encoding")
    if role == "EVALUATE":
        require(evaluate_admitted is True,
                "EVALUATE input rows are retained but scoring is not admitted")
    else:
        require(evaluate_admitted is False,
                "EVALUATE admission flag is invalid for FIT/SELECT")
    return np.flatnonzero(values == ROLES.index(role))


def chromosome_stage(args):
    request = load_request(args.request, args.request_sha256)
    chrom = str(int(args.chromosome))
    desc = request["chromosomes"][chrom]
    for path, key in ((args.event_archive, "event_archive"), (args.partial, "partial"),
                      (args.partial_manifest, "partial_manifest"),
                      (args.common_markers, "common_markers")):
        authenticate(path, desc[key], f"chr{chrom} {key}")
    authenticate(args.supplement_archive, request["supplement_archive"],
                 "common supplement archive")
    partial_manifest = read_json(args.partial_manifest)
    source_intervals = partial_manifest.get("source_intervals", [])
    require(partial_manifest.get("schema") == "r02_structure_materialize_partial_v1" and
            partial_manifest.get("stage") == "COMPLETE_SLICE" and
            partial_manifest.get("request", {}).get("chromosome") == chrom and
            partial_manifest.get("request", {}).get("kind") == "features" and
            len(source_intervals) == 1 and source_intervals[0][:2] == [chrom, 1] and
            partial_manifest.get("request", {}).get("intervals") == [[1, source_intervals[0][2]]] and
            partial_manifest.get("request", {}).get("task_id") == desc["partial_task_id"] and
            partial_manifest.get("contract_sha256") == desc["partial_contract_sha256"] and
            partial_manifest.get("sample_ids_sha256") == desc["sample_ids_sha256"] and
            partial_manifest.get("fit_ids_sha256") == desc["fit_ids_sha256"] and
            partial_manifest.get("selection_used") == "FIT_ONLY" and
            partial_manifest.get("evaluated_outcomes_for_selection") is False and
            partial_manifest.get("outputs_sha256", {}).get("data.npz") == desc["partial"]["sha256"] and
            partial_manifest.get("outputs_sha256", {}).get("common_markers.tsv") ==
                desc["common_markers"]["sha256"],
            "Legacy partial manifest/source linkage differs")
    event_manifest, persons_raw, channel = read_event_metadata(args.event_archive, chrom)
    persons, role_indices = load_persons(persons_raw, request["persons_sha256"])
    require(event_manifest["source_contract_sha256"] == desc["partial_contract_sha256"] and
            event_manifest["sample_ids_sha256"] == desc["sample_ids_sha256"] and
            event_manifest["role_counts"] == request["population"]["roles"] and
            event_manifest["n_people"] == request["population"]["n_people"],
            "Event and legacy partial population/source differ")
    markers = load_markers(args.common_markers, chrom)
    supplement = load_supplement(
        args.supplement_archive, chrom, desc["supplement"], markers,
        n_people=len(persons), sample_ids_sha256=desc["sample_ids_sha256"],
        source_contract_sha256=desc["partial_contract_sha256"])
    legacy_pairs, missing_pairs, missing_pair_role = expected_pairs(persons, role_indices)
    require(missing_pairs.shape == (11872, 2) and
            np.count_nonzero(missing_pair_role == 0) == 360 and
            np.count_nonzero(missing_pair_role == 1) == 40 and
            np.count_nonzero(missing_pair_role == 2) == 11472,
            "Expected exactly 360 FIT, 40 SELECT and 11,472 EVALUATE same-component pairs")
    with np.load(args.partial, allow_pickle=False) as partial:
        require(PARTIAL_FIELDS <= set(partial.files), "Legacy partial required arrays absent")
        sample_ids = partial["sample_ids"].astype(str)
        require(np.array_equal(sample_ids, np.asarray([x["sample_id"] for x in persons])) and
                np.array_equal(partial["pairs"], legacy_pairs),
                "Legacy partial roster/pair universe differs")
        require(int(partial["common_sites"]) == len(markers),
                "Legacy common_sites differs from selected markers")
        legacy_joint = partial["common_joint"].astype(np.int64)
        legacy_numerator = partial["common_numerator"].astype(np.float64)
        common_zsum = partial["common_zsum"].astype(np.float64)
        common_called = partial["common_called"].astype(np.int64)
        rare_copies = partial["rare_copies"].astype(np.int64)
    audit_indices = np.linspace(0, len(legacy_pairs) - 1, 64, dtype=np.int64)
    require(len(np.unique(audit_indices)) == 64, "Cannot construct 64 fixed overlap audit pairs")
    audit_pairs = legacy_pairs[audit_indices]
    totals = event_counts(args.event_archive, chrom, markers, event_manifest, channel, supplement,
                          role_indices, request["resources"]["matrix_site_batch"],
                          missing_pairs, audit_pairs)
    overlap_joint = compute_joint(legacy_pairs, len(markers), totals["missing_count"],
                                  totals["both_missing"], len(persons))
    require(np.array_equal(overlap_joint, legacy_joint),
            "All-pair common_joint overlap audit failed")
    zdelta = np.abs(totals["zsum"] - common_zsum)
    eps = np.finfo(np.float64).eps
    gamma = len(markers) * eps / (1.0 - len(markers) * eps)
    # Each side is a floating sum of the same products.  Four gamma_n times
    # sum(abs(terms)) is a conservative roundoff envelope for the two sums and
    # minor BLAS grouping differences; it is derived from n, not chosen from data.
    zbound = 4.0 * gamma * totals["zsum_abs"] + np.finfo(np.float64).tiny
    audit_delta = np.abs(totals["audit_numerator"] - legacy_numerator[audit_indices])
    audit_bound = 4.0 * gamma * totals["audit_abs"] + np.finfo(np.float64).tiny
    require(np.array_equal(totals["called"], common_called),
            "All-person common_called audit failed")
    require(np.all(zdelta <= zbound),
            "All-person common_zsum audit failed")
    require(np.all(audit_delta <= audit_bound),
            "Deterministic 64-pair common_numerator overlap audit failed")
    missing_joint = compute_joint(missing_pairs, len(markers), totals["missing_count"],
                                  totals["both_missing"], len(persons))
    missing_num = totals["missing_numerator"]
    out = Path(args.output)
    require(not out.exists() and out.parent.is_dir(), "Fresh chromosome output required")
    out.mkdir()
    np.savez(out / "repair.npz", missing_pairs=missing_pairs,
             missing_pair_role=missing_pair_role,
             common_joint_missing=missing_joint, common_numerator_missing=missing_num,
             common_zsum=common_zsum, common_called=common_called,
             common_sites=np.asarray(len(markers), dtype=np.int64),
             rare_copies=rare_copies, sample_ids=sample_ids)
    manifest = {
        "schema": CHROM_SCHEMA, "stage": "CHROMOSOME_REPAIRED_OVERLAP_AUDITED",
        "chromosome": chrom, "request_sha256": args.request_sha256,
        "n_people": len(persons), "n_selected_markers": len(markers),
        "n_supplement_markers": len(supplement),
        "n_legacy_pairs_audited": len(legacy_pairs), "n_missing_input_pairs": len(missing_pairs),
        "n_missing_model_pairs": 400,
        "n_missing_fit_pairs": 360, "n_missing_select_pairs": 40,
        "n_missing_evaluate_pairs": 11472,
        "missing_pair_role_encoding": {"0": "FIT", "1": "SELECT", "2": "EVALUATE"},
        "evaluate_inputs_materialized": True,
        "evaluate_outcomes_read": False,
        "evaluate_scoring_allowed": False,
        "common_joint_overlap_exact": True,
        "common_numerator_overlap_allclose": False,
        "common_numerator_overlap_audit": "64 value-independent uniformly spaced indices over sorted legacy pair order; tolerance is 4*gamma_n*sum(abs(terms))",
        "common_numerator_pairs_audited": 64,
        "common_numerator_total_legacy_pairs": len(legacy_pairs),
        "common_numerator_max_abs_error": float(audit_delta.max(initial=0)),
        "common_numerator_max_roundoff_bound": float(audit_bound.max(initial=0)),
        "common_zsum_all_people_allclose": True,
        "common_zsum_max_abs_error": float(zdelta.max(initial=0)),
        "common_zsum_max_roundoff_bound": float(zbound.max(initial=0)),
        "whole_target_chromosome_exclusion_applied_here": False,
        "evaluated_outcomes_read": False, "biological_validation": False,
        "outputs_sha256": {"repair.npz": sha256_file(out / "repair.npz")},
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"stage": manifest["stage"], "chromosome": chrom,
                      "legacy_pairs_audited": len(legacy_pairs), "missing_pairs": 11872,
                      "fit": 360, "select": 40, "evaluate_inputs_only": 11472}))


def aggregate_stage(args):
    request = load_request(args.request, args.request_sha256)
    directories = [Path(x) for x in args.chromosome_directories]
    require(len(directories) == 22, "Exactly 22 chromosome outputs required")
    rows, receipts = {}, {}
    supplement_total = 0
    for directory in directories:
        manifest = read_json(directory / "manifest.json")
        chrom = manifest.get("chromosome")
        expected_supplement = request["chromosomes"].get(chrom, {}).get("supplement")
        expected_supplement_sites = 0 if expected_supplement is None else expected_supplement["n_sites"]
        require(manifest.get("schema") == CHROM_SCHEMA and
                manifest.get("stage") == "CHROMOSOME_REPAIRED_OVERLAP_AUDITED" and
                manifest.get("request_sha256") == args.request_sha256 and
                manifest.get("common_joint_overlap_exact") is True and
                manifest.get("common_zsum_all_people_allclose") is True and
                manifest.get("n_missing_input_pairs") == 11872 and
                manifest.get("n_missing_model_pairs") == 400 and
                manifest.get("n_missing_fit_pairs") == 360 and
                manifest.get("n_missing_select_pairs") == 40 and
                manifest.get("n_missing_evaluate_pairs") == 11472 and
                manifest.get("n_supplement_markers") == expected_supplement_sites and
                manifest.get("evaluate_outcomes_read") is False and
                manifest.get("evaluate_scoring_allowed") is False and
                manifest.get("evaluated_outcomes_read") is False and
                manifest.get("outputs_sha256", {}).get("repair.npz") ==
                    sha256_file(directory / "repair.npz"),
                "Chromosome repair receipt differs")
        require(chrom not in rows, "Repeated chromosome repair")
        supplement_total += expected_supplement_sites
        receipts[chrom] = sha256_file(directory / "manifest.json")
        with np.load(directory / "repair.npz", allow_pickle=False) as arrays:
            rows[chrom] = {k: arrays[k].copy() for k in arrays.files}
    require(set(rows) == set(request["chromosomes"]), "Chromosome repair set differs")
    ordered = [rows[str(c)] for c in range(1, 23)]
    pair_keys = ordered[0]["missing_pairs"]
    pair_roles = ordered[0]["missing_pair_role"]
    sample_ids = ordered[0]["sample_ids"]
    require(all(np.array_equal(x["missing_pairs"], pair_keys) and
                np.array_equal(x["missing_pair_role"], pair_roles) and
                np.array_equal(x["sample_ids"], sample_ids) for x in ordered),
            "Chromosome pair/person order differs")
    totals = {}
    for key in ("common_joint_missing", "common_numerator_missing", "common_zsum",
                "common_called", "common_sites", "rare_copies"):
        totals[key] = sum((x[key] for x in ordered), np.zeros_like(ordered[0][key]))
    target_values = {key: np.stack([totals[key] - x[key] for x in ordered])
                     for key in totals}
    out = Path(args.output)
    require(not out.exists() and out.parent.is_dir(), "Fresh aggregate output required")
    out.mkdir()
    np.savez(out / "repair.npz", target_chromosomes=np.arange(1, 23, dtype=np.int16),
             missing_pairs=pair_keys, missing_pair_role=pair_roles,
             sample_ids=sample_ids, **target_values)
    manifest = {
        "schema": AGG_SCHEMA, "stage": "ALL_22_TARGET_EXCLUSIONS_REPAIRED",
        "request_sha256": args.request_sha256, "n_people": 2619,
        "n_missing_input_pairs": 11872, "n_missing_model_pairs": 400,
        "n_missing_fit_pairs": 360, "n_missing_select_pairs": 40,
        "n_missing_evaluate_pairs": 11472, "n_target_chromosomes": 22,
        "n_authenticated_supplement_markers": supplement_total,
        "missing_pair_role_encoding": {"0": "FIT", "1": "SELECT", "2": "EVALUATE"},
        "evaluate_inputs_materialized": True,
        "evaluate_outcomes_read": False,
        "evaluate_scoring_allowed": False,
        "target_exclusion": "SUM_ALL_22_MINUS_TARGET_CHROMOSOME",
        "contains_full_common_axes": False,
        "contains_common_callability_and_quantity_sources": True,
        "contains_rare_copies": True,
        "human_c_cq_cqr_fit_admitted": False,
        "reason_human_fit_closed": "Non-axis C/CQ/CQR column membership remains unfrozen",
        "evaluated_outcomes_read": False, "biological_validation": False,
        "chromosome_receipts": {str(c): receipts[str(c)] for c in range(1, 23)},
        "outputs_sha256": {"repair.npz": sha256_file(out / "repair.npz")},
    }
    write_json(out / "manifest.json", manifest)
    print(manifest["stage"])


def verify_stage(args):
    request = load_request(args.request, args.request_sha256)
    catalogue = Path(args.catalogue)
    chromosome_root = Path(args.chromosome_root)
    manifest = read_json(catalogue / "manifest.json")
    require(manifest.get("schema") == AGG_SCHEMA and
            manifest.get("stage") == "ALL_22_TARGET_EXCLUSIONS_REPAIRED" and
            manifest.get("request_sha256") == args.request_sha256 and
            manifest.get("n_people") == 2619 and
            manifest.get("n_missing_input_pairs") == 11872 and
            manifest.get("n_missing_model_pairs") == 400 and
            manifest.get("n_missing_fit_pairs") == 360 and
            manifest.get("n_missing_select_pairs") == 40 and
            manifest.get("n_missing_evaluate_pairs") == 11472 and
            manifest.get("evaluate_outcomes_read") is False and
            manifest.get("evaluate_scoring_allowed") is False and
            manifest.get("n_target_chromosomes") == 22 and
            manifest.get("target_exclusion") == "SUM_ALL_22_MINUS_TARGET_CHROMOSOME" and
            manifest.get("contains_full_common_axes") is False and
            manifest.get("human_c_cq_cqr_fit_admitted") is False and
            manifest.get("evaluated_outcomes_read") is False and
            manifest.get("outputs_sha256", {}).get("repair.npz") ==
                sha256_file(catalogue / "repair.npz"),
            "Aggregate terminal receipt differs")
    with np.load(catalogue / "repair.npz", allow_pickle=False) as arrays:
        require(set(arrays.files) == {"target_chromosomes", "missing_pairs", "missing_pair_role", "sample_ids",
                "common_joint_missing", "common_numerator_missing", "common_zsum",
                "common_called", "common_sites", "rare_copies"},
                "Aggregate arrays differ")
        require(np.array_equal(arrays["target_chromosomes"], np.arange(1, 23)) and
                arrays["missing_pairs"].shape == (11872, 2) and
                arrays["missing_pair_role"].shape == (11872,) and
                arrays["sample_ids"].shape == (2619,) and
                arrays["common_joint_missing"].shape == (22, 11872) and
                arrays["common_numerator_missing"].shape == (22, 11872) and
                arrays["common_zsum"].shape == (22, 2619) and
                arrays["common_called"].shape == (22, 2619) and
                arrays["common_sites"].shape == (22,) and
                arrays["rare_copies"].shape == (22, 2619),
                "Aggregate dimensions differ")
        require(np.all(arrays["common_sites"] > 0) and
                np.all((arrays["common_joint_missing"] >= 0) &
                       (arrays["common_joint_missing"] <= arrays["common_sites"][:, None])) and
                np.all((arrays["common_called"] >= 0) &
                       (arrays["common_called"] <= arrays["common_sites"][:, None])) and
                np.all(arrays["rare_copies"] >= 0) and
                np.all(np.isfinite(arrays["common_numerator_missing"])) and
                np.all(np.isfinite(arrays["common_zsum"])),
                "Aggregate values violate count/finite bounds")
        require(np.count_nonzero(arrays["missing_pair_role"] == 0) == 360 and
                np.count_nonzero(arrays["missing_pair_role"] == 1) == 40 and
                np.count_nonzero(arrays["missing_pair_role"] == 2) == 11472 and
                np.all(np.isin(arrays["missing_pair_role"], [0, 1, 2])),
                "Missing pair role split differs")
        n = len(arrays["sample_ids"])
        codes = arrays["missing_pairs"][:, 0] * n + arrays["missing_pairs"][:, 1]
        require(np.all(arrays["missing_pairs"][:, 0] < arrays["missing_pairs"][:, 1]) and
                np.all(np.diff(codes) > 0), "Missing pair keys are not canonical/unique")
    for chrom in range(1, 23):
        directory = chromosome_root / f"repair_chr{chrom:02d}"
        receipt = read_json(directory / "manifest.json")
        require(receipt.get("chromosome") == str(chrom) and
                receipt.get("common_joint_overlap_exact") is True and
                receipt.get("common_numerator_pairs_audited") == 64 and
                receipt.get("common_zsum_all_people_allclose") is True and
                receipt.get("n_legacy_pairs_audited", 0) > 0 and
                receipt.get("n_missing_input_pairs") == 11872 and
                receipt.get("n_missing_model_pairs") == 400 and
                receipt.get("n_missing_fit_pairs") == 360 and
                receipt.get("n_missing_select_pairs") == 40 and
                receipt.get("n_missing_evaluate_pairs") == 11472 and
                receipt.get("n_supplement_markers") ==
                    (0 if request["chromosomes"][str(chrom)]["supplement"] is None else
                     request["chromosomes"][str(chrom)]["supplement"]["n_sites"]) and
                receipt.get("evaluate_outcomes_read") is False and
                receipt.get("evaluate_scoring_allowed") is False and
                receipt.get("outputs_sha256", {}).get("repair.npz") ==
                    sha256_file(directory / "repair.npz") and
                manifest["chromosome_receipts"].get(str(chrom)) ==
                    sha256_file(directory / "manifest.json"),
                f"Chromosome {chrom} terminal receipt differs")
    require(set(request["chromosomes"]) == set(manifest["chromosome_receipts"]),
            "Terminal chromosome/request inventory differs")
    require(manifest.get("n_authenticated_supplement_markers") == 328,
            "Terminal supplement-marker count differs")
    print("K04_COMMON_PAIR_REPAIR_V4_VERIFIED_22_CHROMOSOMES_11872_INPUT_PAIRS_328_SUPPLEMENT_LOCI_EVALUATE_INPUTS_ONLY_NO_HUMAN_FIT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate-request", "chromosome", "aggregate", "verify"), required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--chromosome")
    parser.add_argument("--event-archive")
    parser.add_argument("--partial")
    parser.add_argument("--partial-manifest")
    parser.add_argument("--common-markers")
    parser.add_argument("--supplement-archive")
    parser.add_argument("--chromosome-directories", nargs="*")
    parser.add_argument("--catalogue")
    parser.add_argument("--chromosome-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.stage == "validate-request":
        load_request(args.request, args.request_sha256)
        print("REQUEST_VALID")
    elif args.stage == "chromosome":
        require(all((args.chromosome, args.event_archive, args.partial,
                     args.partial_manifest, args.common_markers,
                     args.supplement_archive, args.output)),
                "Chromosome stage arguments required")
        chromosome_stage(args)
    elif args.stage == "aggregate":
        require(args.chromosome_directories and args.output, "Aggregate arguments required")
        aggregate_stage(args)
    else:
        require(args.catalogue and args.chromosome_root, "Verify arguments required")
        verify_stage(args)


if __name__ == "__main__":
    main()
