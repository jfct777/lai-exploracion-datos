#!/usr/bin/env python3
"""Exact chromosome pair-sharing bounds conditional on Gnomix MSP hardcalls.

The consumer retains every authenticated source-minor catalogue locus.  It
does not assign a heterozygous rare copy to a homologue.  For ancestry-copy
count A and observed minor dosage D it uses L=max(0,D+A-2), U=min(D,A), then
forms exact unweighted Gram matrices of 1{L>0} and 1{U>0}.

Pair denominators are computed without a dense locus-by-person matrix.  MSP
interval sufficient statistics give the baseline and one-sided missing terms;
an exact sparse Gram gives the both-missing correction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import tarfile
import tempfile
import time

import numpy as np

import r02_rare_ancestry_identifiability as core


INPUT_SCHEMA = "r02_rare_ancestry_chromosome_input_v1"
PRODUCT = "PAIR_SHARING_PARTIAL_IDENTIFICATION_V1"
OUTPUT_SCHEMA = "r02_rare_ancestry_pair_bounds_v1"
STATUS = "COMPLETE_CONDITIONAL_PAIR_SHARING_BOUNDS_NOT_PHASE_OR_FINE_STRUCTURE_VALIDATION"
IMPLEMENTATION_VERSION = "1.0.0"
ANCESTRY_CODES = ("AFR", "EUR", "NAM")
CODE_FILES = (*core.CODE_FILES, "r02_rare_ancestry_chromosome.py",
              "r02_rare_ancestry_pair_bounds.py")
require = core.require


def prepare_hard_windows(paths, contract, samples, mapping, scratch,
                         max_matrix_mb, min_free_mb):
    """Read selected MSP hardcalls; authenticate but do not interpret FB rows.

    FB is retained in the exact input inventory and authenticated by SHA256.
    Its header is checked jointly with MSP to bind people, homologues and the
    ancestry dictionary.  Pair bounds depend only on MSP hardcalls, so loading
    posterior rows would add memory without changing this calculation.
    """
    populations = tuple(contract["population_order"])
    source = core.ancestry.source_headers(
        paths["msp"], paths["fb"], populations,
        contract["source_samples_sha256"])
    source_index = {name: i for i, name in enumerate(source)}
    reverse = {sample: src for src, sample in mapping.items()}
    require(all(reverse[s] in source_index for s in samples),
            "Analytical person absent from MSP columns")
    select = np.asarray([2 * source_index[reverse[s]] + h
                         for s in samples for h in (0, 1)])
    hard_path = scratch / "hard.bin"
    windows = []
    previous = 0
    chrom = str(contract["chromosome"])
    with core.ancestry.text_open(paths["msp"]) as handle, hard_path.open("xb") as out:
        next(handle); next(handle)
        for row_number, line in enumerate(handle):
            fields = line.rstrip("\r\n").split("\t")
            require(len(fields) == 6 + len(source) * 2,
                    "MSP column count differs")
            require(str(core.ancestry.exact_int(fields[0])) == chrom,
                    "MSP chromosome differs")
            lo, hi = map(core.ancestry.exact_int, fields[1:3])
            g0, g1 = map(float, fields[3:5])
            require(1 <= lo <= hi and lo > previous,
                    "MSP windows overlap, regress or have invalid coordinates")
            require(math.isfinite(g0) and math.isfinite(g1) and g0 <= g1 and
                    core.ancestry.exact_int(fields[5]) >= 0,
                    "Invalid MSP genetic coordinates/marker count")
            labels = np.asarray([
                -1 if value in (".", "NA", "nan", "-1")
                else core.ancestry.exact_int(value)
                for value in fields[6:]], dtype=np.int64)
            require(np.isin(labels, (-1, 0, 1, 2)).all(),
                    "Invalid MSP ancestry code")
            labels[select].astype(np.int8).tofile(out)
            windows.append((lo, hi, g0, g1))
            previous = hi
            if row_number % 50 == 0:
                core.ancestry.reserve(scratch, min_free_mb)
    require(windows, "Empty MSP windows")
    n_people = len(samples)
    required = len(windows) * n_people * (2 + 4)
    require(required <= max_matrix_mb * 2**20,
            "MSP hardcalls plus interval missing counters exceed matrix budget")
    hard = np.memmap(hard_path, mode="r", dtype=np.int8,
                     shape=(len(windows), 2 * n_people))
    return windows, hard, dict(
        source_people=len(source), windows=len(windows),
        fb_policy="AUTHENTICATED_SHA256_AND_HEADER_ONLY_NOT_USED_BY_MSP_HARDCALL_BOUNDS",
        hardcall_bytes=len(windows) * 2 * n_people,
        interval_missing_counter_bytes=len(windows) * n_people * 4)


def locus_windows(positions, starts, ends):
    indexes = np.searchsorted(starts, positions, side="right") - 1
    covered = (indexes >= 0) & (positions <= ends[np.maximum(indexes, 0)])
    return np.where(covered, indexes, -1).astype(np.int64, copy=False)


def hard_for_loci(indexes, hard, n_people):
    """Return person x locus hardcalls, with -1 outside MSP."""
    n_loci = len(indexes)
    h0 = np.full((n_people, n_loci), -1, dtype=np.int8)
    h1 = np.full_like(h0, -1)
    covered = indexes >= 0
    if np.any(covered):
        selected = hard[indexes[covered]]
        h0[:, covered] = selected[:, 0::2].T
        h1[:, covered] = selected[:, 1::2].T
    return h0, h1


def add_sparse_gram(accumulator, values):
    """Add X.T@X exactly, retaining X as a sparse locus x person matrix."""
    from scipy import sparse

    require(values.ndim == 2 and values.shape[1] == accumulator.shape[0],
            "Sparse Gram dimensions differ")
    rows, columns = np.nonzero(values)
    if not len(rows):
        return 0
    data = np.asarray(values[rows, columns], dtype=np.int64)
    matrix = sparse.csr_matrix((data, (rows, columns)), shape=values.shape,
                               dtype=np.int64)
    gram = (matrix.T @ matrix).tocoo()
    gram.sum_duplicates()
    current = accumulator[gram.row, gram.col].astype(np.uint64)
    updated = current + gram.data.astype(np.uint64)
    require(np.all(updated <= np.iinfo(np.uint32).max),
            "Pair count exceeds uint32")
    accumulator[gram.row, gram.col] = updated.astype(np.uint32)
    return int(matrix.nnz)


def new_accumulators(n_people):
    shape = (n_people, n_people)
    matrices = {}
    for code in ANCESTRY_CODES:
        matrices[f"lower_presence_sharing_{code}"] = np.zeros(shape, np.uint32)
        matrices[f"upper_presence_sharing_{code}"] = np.zeros(shape, np.uint32)
    corrections = {"joint_valid_loci": np.zeros(shape, np.uint32)}
    corrections.update({f"ancestral_opportunities_{code}": np.zeros(shape, np.uint32)
                        for code in ANCESTRY_CODES})
    person = {
        "observed_gt": np.zeros(n_people, np.uint64),
        "missing_gt": np.zeros(n_people, np.uint64),
        "outside_msp_observed": np.zeros(n_people, np.uint64),
        "inside_msp_missing_ancestry_observed": np.zeros(n_people, np.uint64),
        "joint_valid_loci": np.zeros(n_people, np.uint64),
    }
    for code in ANCESTRY_CODES:
        for name in ("ancestry_present_loci", "ancestry_homolog_copies",
                     "lower_presence_burden", "upper_presence_burden",
                     "lower_copy_burden", "upper_copy_burden"):
            person[f"{name}_{code}"] = np.zeros(n_people, np.uint64)
    return matrices, corrections, person


def accumulate_chunk(data, indexes, hard, n_by_window, missing_by_window,
                     matrices, corrections, person):
    """Accumulate exact sparse numerators and interval denominator statistics."""
    dosage = data["dosage"]
    observed = data["observed"]
    n_people, n_loci = dosage.shape
    require(len(indexes) == n_loci, "Locus/window dimensions differ")
    h0, h1 = hard_for_loci(indexes, hard, n_people)
    valid_hard = (h0 >= 0) & (h1 >= 0)
    require(np.all(np.isin(h0[valid_hard], (0, 1, 2))) and
            np.all(np.isin(h1[valid_hard], (0, 1, 2))),
            "Invalid ancestry hardcall")
    joint = observed & valid_hard
    covered = indexes >= 0

    person["observed_gt"] += np.count_nonzero(observed, axis=1).astype(np.uint64)
    person["missing_gt"] += np.count_nonzero(~observed, axis=1).astype(np.uint64)
    person["outside_msp_observed"] += np.count_nonzero(
        observed & ~covered[np.newaxis, :], axis=1).astype(np.uint64)
    person["inside_msp_missing_ancestry_observed"] += np.count_nonzero(
        observed & covered[np.newaxis, :] & ~valid_hard,
        axis=1).astype(np.uint64)
    person["joint_valid_loci"] += np.count_nonzero(
        joint, axis=1).astype(np.uint64)

    for window in np.unique(indexes[covered]):
        columns = np.flatnonzero(indexes == window)
        n_by_window[window] += len(columns)
        missing_by_window[window] += np.count_nonzero(
            ~observed[:, columns], axis=1).astype(np.uint32)

    missing_valid = (~observed & valid_hard).T
    sparse_nnz = {"joint_missing": add_sparse_gram(
        corrections["joint_valid_loci"], missing_valid)}
    dose = dosage.astype(np.int16, copy=False)
    for ancestry, code in enumerate(ANCESTRY_CODES):
        ancestry_dose = ((h0 == ancestry).astype(np.int16) +
                         (h1 == ancestry).astype(np.int16))
        lower = np.maximum(0, dose + ancestry_dose - 2)
        upper = np.minimum(dose, ancestry_dose)
        lower_presence = joint & (lower > 0)
        upper_presence = joint & (upper > 0)
        sparse_nnz[f"lower_{code}"] = add_sparse_gram(
            matrices[f"lower_presence_sharing_{code}"], lower_presence.T)
        sparse_nnz[f"upper_{code}"] = add_sparse_gram(
            matrices[f"upper_presence_sharing_{code}"], upper_presence.T)
        weighted_missing = (ancestry_dose * (~observed) * valid_hard).T
        sparse_nnz[f"opportunity_missing_{code}"] = add_sparse_gram(
            corrections[f"ancestral_opportunities_{code}"], weighted_missing)
        person[f"ancestry_present_loci_{code}"] += np.count_nonzero(
            joint & (ancestry_dose > 0), axis=1).astype(np.uint64)
        person[f"ancestry_homolog_copies_{code}"] += np.sum(
            ancestry_dose * joint, axis=1, dtype=np.uint64)
        person[f"lower_presence_burden_{code}"] += np.count_nonzero(
            lower_presence, axis=1).astype(np.uint64)
        person[f"upper_presence_burden_{code}"] += np.count_nonzero(
            upper_presence, axis=1).astype(np.uint64)
        person[f"lower_copy_burden_{code}"] += np.sum(
            lower * joint, axis=1, dtype=np.uint64)
        person[f"upper_copy_burden_{code}"] += np.sum(
            upper * joint, axis=1, dtype=np.uint64)
    return sparse_nnz


def interval_features(hard, population, start, end):
    h0 = np.asarray(hard[start:end, 0::2])
    h1 = np.asarray(hard[start:end, 1::2])
    valid = (h0 >= 0) & (h1 >= 0)
    if population is None:
        return valid.astype(np.float64)
    return (((h0 == population).astype(np.int8) +
             (h1 == population).astype(np.int8)) * valid).astype(np.float64)


def finish_denominator(hard, population, n_by_window, missing_by_window,
                       both_missing_correction, max_array_mb):
    """Exact baseline-minus-missing denominator via batched matrix products."""
    n_windows, n_people = missing_by_window.shape
    require(hard.shape == (n_windows, 2 * n_people),
            "Interval sufficient-statistic dimensions differ")
    # f, f*missing and n*f coexist; keep those work arrays under max_array_mb.
    rows = max(1, max_array_mb * 2**20 // max(1, n_people * 8 * 3))
    total = np.zeros((n_people, n_people), dtype=np.float64)
    for start in range(0, n_windows, rows):
        end = min(n_windows, start + rows)
        feature = interval_features(hard, population, start, end)
        counts = np.asarray(n_by_window[start:end], dtype=np.float64)
        missing = np.asarray(missing_by_window[start:end], dtype=np.float64)
        weighted_missing = feature * missing
        total += feature.T @ (feature * counts[:, np.newaxis])
        cross = feature.T @ weighted_missing
        total -= cross + cross.T
    total += both_missing_correction
    rounded = np.rint(total)
    require(np.array_equal(total, rounded),
            "Denominator batch arithmetic lost integer exactness")
    require(float(rounded.min(initial=0)) >= 0 and
            float(rounded.max(initial=0)) <= np.iinfo(np.uint32).max,
            "Denominator outside uint32 range")
    return rounded.astype(np.uint32)


def read_people(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(rows and all(row.get("sample_id") and row.get("role") in
                         ("FIT", "SELECT", "EVALUATE") for row in rows),
            "Invalid persons table")
    return rows


def write_people(output, rows, metrics):
    with (output / "persons.tsv").open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("person_index", "sample_id", "role"))
        for index, row in enumerate(rows):
            writer.writerow((index, row["sample_id"], row["role"]))
    names = list(metrics)
    with (output / "person_metrics.tsv").open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("person_index", "sample_id", "role", *names))
        for index, row in enumerate(rows):
            writer.writerow((index, row["sample_id"], row["role"],
                             *[int(metrics[name][index]) for name in names]))


def run(contract, contract_sha256, inputs_dir, output_dir, scratch):
    started = time.monotonic()
    contract = Path(contract).resolve()
    output = Path(output_dir)
    scratch = Path(scratch)
    require(core.sha256(contract) == contract_sha256,
            "Chromosome specification SHA256 differs")
    specification = json.loads(contract.read_text())
    require(specification.get("schema") == INPUT_SCHEMA,
            "Unexpected chromosome input schema")
    require(specification.get("product") == PRODUCT,
            "Explicit pair-bound product discriminator required")
    expected = (set(core.FILES) - {"event_chunk"}) | {"event_archive"}
    require(set(specification.get("inputs", {})) == expected,
            "Exact complete-chromosome input inventory required")
    limits = specification.get("resources", {})
    require(set(limits) == {"max_array_mb", "max_window_matrix_mb", "min_free_mb",
                            "max_archive_bytes", "timeout_seconds"} and
            all(type(value) is int and value > 0 for value in limits.values()),
            "Explicit positive inherited resource limits required")
    paths = {key: core.authenticate(value, inputs_dir)
             for key, value in specification["inputs"].items()}
    require(paths["event_archive"].stat().st_size <= limits["max_archive_bytes"],
            "Archive exceeds bound")
    code_paths = [Path(__file__).with_name(name) for name in CODE_FILES]
    before = core.input_snapshot([contract, *paths.values(), *code_paths])
    ancestry, event, material, samples, mapping = core.validate_sources(paths)
    people = read_people(paths["persons"])
    require([row["sample_id"] for row in people] == samples,
            "Persons order changed after source validation")
    chrom = str(ancestry["chromosome"])
    prefix = specification.get("archive_member_prefix")
    require(prefix == "published/chr" + chrom,
            "Exact scientific product prefix required")
    entries = event["channels"]["rare"]
    require(entries, "Nonempty rare catalogue required")
    lookup = {prefix + "/" + Path(row["path"]).name: row for row in entries}
    require(len(lookup) == len(entries), "Duplicate rare chunk identities")
    previous = 0
    for entry in entries:
        require(type(entry["n_sites"]) is int and entry["n_sites"] > 0 and
                entry["start_bp"] > previous and
                entry["end_bp"] >= entry["start_bp"],
                "Source chunk order/overlap invalid")
        previous = entry["end_bp"]
    require(scratch.is_dir() and not output.exists(),
            "Fresh output and existing local scratch required")
    core.ancestry.reserve(scratch, limits["min_free_mb"])
    output.mkdir(parents=True, mode=0o700)

    with tempfile.TemporaryDirectory(prefix="k04-pair-bounds-", dir=scratch) as temp:
        temp = Path(temp)
        windows, hard, window_receipt = prepare_hard_windows(
            paths, ancestry, samples, mapping, temp,
            limits["max_window_matrix_mb"], limits["min_free_mb"])
        starts = np.asarray([window[0] for window in windows])
        ends = np.asarray([window[1] for window in windows])
        n_windows = len(windows)

        # Authenticate exact locus coverage before any genotype array is read.
        preflight = {}
        with tarfile.open(paths["event_archive"], "r|gz") as archive:
            for member in archive:
                if member.name not in lookup:
                    continue
                require(member.isfile() and member.name not in preflight,
                        "Repeated or nonregular preflight chunk")
                entry = lookup[member.name]
                require(member.size == entry["bytes"],
                        "Preflight chunk size differs")
                target = temp / "coordinates.npz"
                digest = hashlib.sha256()
                with archive.extractfile(member) as source, target.open("wb") as sink:
                    while block := source.read(2**20):
                        digest.update(block); sink.write(block)
                require(digest.hexdigest() == entry["sha256"],
                        "Preflight chunk SHA256 differs")
                preflight[member.name] = core.coordinate_preflight(
                    target, paths["msp"], chrom, entry, limits["max_array_mb"],
                    require_overlap=False, intervals=windows)
                require(time.monotonic() - started <= limits["timeout_seconds"],
                        "Coordinate preflight timeout exceeded")
        require(set(preflight) == set(lookup),
                "Incomplete archive: rare catalogue chunks missing before genotype read")
        require(sum(row["loci_in_msp"] for row in preflight.values()) > 0,
                "NO_COORDINATE_OVERLAP: chromosome has no joint territory")

        missing_path = temp / "missing-by-window.bin"
        missing_by_window = np.memmap(
            missing_path, mode="w+", dtype=np.uint32,
            shape=(n_windows, len(samples)))
        missing_by_window[:] = 0
        n_by_window = np.zeros(n_windows, np.uint32)
        matrices, corrections, person = new_accumulators(len(samples))
        seen = set()
        loci = 0
        loci_in_msp = 0
        sparse_nnz = {name: 0 for name in (
            "joint_missing", *[f"{bound}_{code}" for code in ANCESTRY_CODES
                               for bound in ("lower", "upper")],
            *[f"opportunity_missing_{code}" for code in ANCESTRY_CODES])}
        with (output / "chunks.jsonl").open("x") as records, \
                tarfile.open(paths["event_archive"], "r|gz") as archive:
            for member in archive:
                if member.name not in lookup:
                    continue
                require(member.isfile() and member.name not in seen,
                        "Repeated or nonregular rare chunk")
                entry = lookup[member.name]
                require(member.size == entry["bytes"], "Chunk byte count differs")
                target = temp / "chunk.npz"
                digest = hashlib.sha256()
                with archive.extractfile(member) as source, target.open("wb") as sink:
                    while block := source.read(2**20):
                        digest.update(block); sink.write(block)
                require(digest.hexdigest() == entry["sha256"],
                        "Chunk SHA256 differs")
                data = core.load_chunk(
                    target, len(samples), entry, material["source_rare_rule"],
                    material["source_n_samples"], limits["max_array_mb"])
                indexes = locus_windows(data["position_bp"], starts, ends)
                covered = int(np.count_nonzero(indexes >= 0))
                require(covered == preflight[member.name]["loci_in_msp"],
                        "Coordinate preflight changed")
                chunk_nnz = accumulate_chunk(
                    data, indexes, hard, n_by_window, missing_by_window,
                    matrices, corrections, person)
                for name, value in chunk_nnz.items():
                    sparse_nnz[name] += value
                loci += entry["n_sites"]
                loci_in_msp += covered
                seen.add(member.name)
                records.write(json.dumps(dict(
                    chunk=Path(member.name).name, sha256=entry["sha256"],
                    start_bp=entry["start_bp"], end_bp=entry["end_bp"],
                    n_loci=entry["n_sites"], loci_in_msp=covered,
                    loci_outside_msp=entry["n_sites"] - covered,
                    sparse_nonzeros=chunk_nnz), sort_keys=True) + "\n")
                records.flush()
                require(time.monotonic() - started <= limits["timeout_seconds"],
                        "Chromosome timeout exceeded")
                core.ancestry.reserve(output, limits["min_free_mb"])
        require(seen == set(lookup), "Incomplete archive: rare catalogue chunks missing")
        require(loci == sum(row["n_sites"] for row in entries),
                "Whole-catalogue locus total differs")
        require(int(n_by_window.sum()) == loci_in_msp,
                "Interval locus sufficient statistics differ")
        missing_by_window.flush()

        matrices["joint_valid_loci"] = finish_denominator(
            hard, None, n_by_window, missing_by_window,
            corrections["joint_valid_loci"], limits["max_array_mb"])
        for ancestry_index, code in enumerate(ANCESTRY_CODES):
            matrices[f"ancestral_opportunities_{code}"] = finish_denominator(
                hard, ancestry_index, n_by_window, missing_by_window,
                corrections[f"ancestral_opportunities_{code}"],
                limits["max_array_mb"])

        for code in ANCESTRY_CODES:
            lower = matrices[f"lower_presence_sharing_{code}"]
            upper = matrices[f"upper_presence_sharing_{code}"]
            opportunity = matrices[f"ancestral_opportunities_{code}"]
            require(np.all(lower <= upper), "Lower pair sharing exceeds upper")
            require(np.all(upper <= matrices["joint_valid_loci"]),
                    "Upper sharing exceeds jointly valid loci")
            require(np.all(opportunity <= 4 * matrices["joint_valid_loci"].astype(np.uint64)),
                    "Ancestral opportunity exceeds four per jointly valid locus")
            require(np.array_equal(np.diag(lower),
                    person[f"lower_presence_burden_{code}"].astype(np.uint32)) and
                    np.array_equal(np.diag(upper),
                    person[f"upper_presence_burden_{code}"].astype(np.uint32)),
                    "Pair diagonal differs from person burden")
        require(np.array_equal(np.diag(matrices["joint_valid_loci"]),
                               person["joint_valid_loci"].astype(np.uint32)),
                "Joint coverage diagonal differs from person support")

        np.savez(output / "matrices.npz", **matrices)
        write_people(output, people, person)

    seconds = time.monotonic() - started
    result = dict(
        schema=OUTPUT_SCHEMA, implementation_version=IMPLEMENTATION_VERSION,
        product=PRODUCT, status=STATUS,
        clinical_status="NOT_CLINICAL_RESEARCH_DESCRIPTIVE_INPUT_ONLY",
        chromosome=chrom, n_people=len(samples),
        source_n_people=material["source_n_samples"], n_loci=loci,
        n_chunks=len(entries), loci_in_msp=loci_in_msp,
        loci_outside_msp=loci - loci_in_msp,
        matrix_shape=[len(samples), len(samples)], matrix_dtype="uint32",
        matrix_file="matrices.npz", person_order_file="persons.tsv",
        person_metrics_file="person_metrics.tsv", chunk_audit_file="chunks.jsonl",
        matrix_keys=list(matrices), ancestry_codes=dict(zip(
            ANCESTRY_CODES, core.ancestry.POPULATIONS)),
        lower_numerator="SUM_LOCI 1{MAX(0,D_i+A_i-2)>0} 1{MAX(0,D_j+A_j-2)>0}",
        upper_numerator="SUM_LOCI 1{MIN(D_i,A_i)>0} 1{MIN(D_j,A_j)>0}",
        joint_loci_denominator="SUM_LOCI V_i*V_j; V REQUIRES OBSERVED_GT_AND_TWO_VALID_MSP_HARDCALLS",
        ancestral_opportunity_denominator="SUM_LOCI V_i*V_j*A_i*A_j",
        matrix_units=dict(
            lower_and_upper_presence_sharing="CATALOGUE_SITES",
            joint_valid_loci="CATALOGUE_SITES",
            ancestral_opportunities="ANCESTRAL_HOMOLOGUE_COPY_PAIRS_TIMES_CATALOGUE_SITES"),
        ratio_limitation="NUMERATOR_DIVIDED_BY_ANCESTRAL_OPPORTUNITIES_IS_NOT_A_HAPLOTYPIC_SHARING_PROBABILITY",
        coefficient_weight=1,
        coefficient_interpretation="RAW_SOURCE_RARE_CATALOGUE_LOCUS_COUNTS_NOT_OPTIMIZED_PARAMETER",
        missing_priority="MISSING_GT_PRECEDES_MISSING_ANCESTRY; OUTSIDE_MSP_OBSERVED_IS_MISSING_ANCESTRY",
        global_missing_counts=dict(
            missing_gt_person_locus_rows=int(person["missing_gt"].sum()),
            observed_gt_outside_msp_person_locus_rows=int(
                person["outside_msp_observed"].sum()),
            observed_gt_inside_msp_missing_ancestry_person_locus_rows=int(
                person["inside_msp_missing_ancestry_observed"].sum())),
        scope="ALL_AUTHENTICATED_SOURCE_RARE_CATALOGUE_LOCI_AND_ALL_FROZEN_ANALYTICAL_PEOPLE_NO_OUTCOME_SELECTION",
        ancestry_is="GNOMIX_ESTIMATED_AFR_EUR_NAM_HARDCALL_CONDITIONAL_NOT_TRUTH",
        bounds_are="PAIRWISE_OUTER_COMBINATORIAL_BOUNDS_NOT_CONFIDENCE_INTERVALS_AND_NOT_JOINTLY_ATTAINABLE_ACROSS_ALL_PAIRS_OR_ANCESTRIES",
        phase_accuracy="NOT_VALIDATED", no_rephasing=True, no_imputation=True,
        evidence_class="DESCRIPTIVE_INPUT_FOR_A_LATER_PRESPECIFIED_COMPARISON_NOT_FINE_STRUCTURE_EVIDENCE",
        fb_policy=window_receipt["fb_policy"], windows=window_receipt,
        sparse_nonzeros=sparse_nnz,
        software=dict(numpy=np.__version__, scipy=__import__("scipy").__version__),
        seconds=seconds,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    core.verify_unchanged([contract, *paths.values(), *code_paths], before)
    outputs = {path.name: core.sha256(path) for path in output.iterdir()
               if path.is_file() and path.name != "manifest.json"}
    manifest = dict(
        schema=OUTPUT_SCHEMA, status=STATUS, product=PRODUCT,
        contract_sha256=contract_sha256, sources=before, resources=limits,
        outputs_sha256=outputs, operational_outputs_sha256={})
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "contract-sha256", "inputs-dir", "output-dir", "scratch"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), allow_nan=False))


if __name__ == "__main__":
    main()
