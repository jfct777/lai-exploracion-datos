#!/usr/bin/env python3
"""Annotate reported IBD copies with existing Gnomix MSP/FB, without inference.

BEDTools intersects the shared MSP window geometry. Exact source-person and
homologue columns are looked up only afterwards. This avoids a Cartesian join
between people. Uncovered bases remain unknown; discordant calls and separate
posterior vectors are retained. Children never become newly inferred IBD calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time

import numpy as np

from r02_ibd_compatibility import input_snapshot, require, sha256, verify_unchanged
from r02_ibd_prepare import native_record
from r02_genomic_pair_evidence import read_samples, sample_hash


SCHEMA = "r02_ibd_local_ancestry_v1"
POPULATIONS = ("African", "European", "Native-American")
STATES = ("CONCORDANT_AFR", "CONCORDANT_EUR", "CONCORDANT_NAM", "DISCORDANT", "UNKNOWN")
NA = "NA"
PARENT_COLUMNS = ("record_id", "source_a", "hap_a", "source_b", "hap_b", "chrom",
                  "start_pos", "end_pos", "lod", "length_cm", "sample_a", "sample_b",
                  "parent_bp", *[s.lower() + "_bp" for s in STATES],
                  "both_posteriors_bp", "children_count")
CHILD_COLUMNS = ("record_id", "chrom", "start_pos", "end_pos", "length_bp",
                 "msp_window_index", "hardcall_a", "hardcall_b", "state", "posterior_status",
                 "p_a_AFR", "p_a_EUR", "p_a_NAM", "p_b_AFR", "p_b_EUR", "p_b_NAM")


def text_open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def exact_int(value):
    number = float(value)
    require(math.isfinite(number) and number.is_integer() and abs(number) < 2**53,
            "Invalid exact integer")
    return int(number)


def reserve(scratch, min_free_mb):
    require(shutil.disk_usage(scratch).free >= min_free_mb * 2**20,
            "Local disk reserve exhausted")


def source_headers(msp, fb, populations, expected_source_hash):
    """Validate full column identity, order and explicit population dictionaries."""
    with text_open(msp) as handle:
        first = handle.readline().rstrip("\r\n")
        require(first.startswith("#Subpopulation order/codes: "), "MSP dictionary missing")
        require(first.split(": ", 1)[1].split("\t") ==
                [f"{name}={i}" for i, name in enumerate(populations)], "MSP population dictionary differs")
        header = handle.readline().rstrip("\r\n").split("\t")
    require(header[:6] == ["#chm", "spos", "epos", "sgpos", "egpos", "n snps"],
            "Unexpected MSP metadata columns")
    require((len(header) - 6) % 2 == 0 and len(header) > 6, "MSP haplotype columns malformed")
    names = []
    for left, right in zip(header[6::2], header[7::2]):
        require(left.endswith(".0") and right == left[:-2] + ".1", "MSP homologue ordering differs")
        names.append(left[:-2])
    require(len(set(names)) == len(names) and all(names), "Duplicate/empty source people")
    require(sample_hash(names) == expected_source_hash, "Source sample order hash differs")
    with text_open(fb) as handle:
        pop = handle.readline().rstrip("\r\n").split("\t")
        fields = handle.readline().rstrip("\r\n").split("\t")
    require(pop == ["#reference_panel_population:", *populations], "FB population dictionary differs")
    require(fields[:4] == ["chromosome", "physical position", "genetic_position", "genetic_marker_index"],
            "Unexpected FB metadata columns")
    expected = [f"{person}:::{hap}:::{anc}" for person in names
                for hap in ("hap1", "hap2") for anc in populations]
    require(fields[4:] == expected, "FB people/homologues/population order differs")
    return names


def prepare_windows(paths, contract, samples, mapping, scratch, max_matrix_mb, min_free_mb):
    """Store only analytical columns on local memmaps; check all source columns."""
    populations = tuple(contract["population_order"])
    source = source_headers(paths["msp"], paths["fb"], populations, contract["source_samples_sha256"])
    source_index = {name: i for i, name in enumerate(source)}
    reverse = {sample: src for src, sample in mapping.items()}
    require(all(reverse[s] in source_index for s in samples), "Analytical person absent from source columns")
    select = np.array([2 * source_index[reverse[s]] + h for s in samples for h in (0, 1)])
    hard_path = scratch / "hard.bin"
    windows = []
    chrom = contract["chromosome"]
    with text_open(paths["msp"]) as handle, hard_path.open("xb") as hard, (scratch / "windows.bed").open("x") as bed:
        next(handle); next(handle)
        previous = 0
        for index, line in enumerate(handle):
            f = line.rstrip("\r\n").split("\t")
            require(len(f) == 6 + len(source) * 2, "MSP column count differs")
            require(str(exact_int(f[0])) == chrom, "MSP chromosome differs")
            lo, hi = exact_int(f[1]), exact_int(f[2])
            g0, g1 = float(f[3]), float(f[4])
            require(1 <= lo <= hi and lo > previous, "MSP windows overlap, regress or have invalid coordinates")
            require(math.isfinite(g0) and math.isfinite(g1) and g0 <= g1 and exact_int(f[5]) >= 0,
                    "Invalid MSP genetic coordinates/marker count")
            labels = np.array([-1 if v in (".", "NA", "nan", "-1") else exact_int(v) for v in f[6:]], dtype=np.int64)
            require(np.isin(labels, (-1, 0, 1, 2)).all(), "Invalid MSP ancestry code")
            labels[select].astype(np.int8).tofile(hard)
            windows.append((lo, hi, g0, g1))
            require(len(windows) * len(select) * 25 <= max_matrix_mb * 2**20, "Matrix disk budget exceeded")
            bed.write(f"{chrom}\t{lo - 1}\t{hi}\t{index}\n")
            previous = hi
            if index % 50 == 0:
                reserve(scratch, min_free_mb)
    require(windows, "Empty MSP windows")
    hard = np.memmap(hard_path, mode="r", dtype=np.int8, shape=(len(windows), len(select)))
    reserve(scratch, min_free_mb + math.ceil(len(windows) * len(select) * 24 / 2**20))
    posterior = np.memmap(scratch / "posterior.bin", mode="w+", dtype=np.float64,
                          shape=(len(windows), len(select), 3))
    posterior[:] = np.nan
    midpoint = {round((lo + hi) / 2): i for i, (lo, hi, _, _) in enumerate(windows)}
    require(len(midpoint) == len(windows), "MSP midpoint matching ambiguous")
    seen = set()
    prior_position = -1
    tolerance = contract["posterior_sum_tolerance"]
    with text_open(paths["fb"]) as handle:
        next(handle); next(handle)
        for row_number, line in enumerate(handle):
            f = line.rstrip("\r\n").split("\t")
            require(len(f) == 4 + len(source) * 6, "FB column count differs")
            require(str(exact_int(f[0])) == chrom, "FB chromosome differs")
            pos, gp = exact_int(f[1]), float(f[2])
            require(pos > prior_position and pos in midpoint, "FB position is unordered or not an exact MSP midpoint")
            index = midpoint[pos]
            require(index not in seen, "Duplicate FB window")
            expected_gp = (windows[index][2] + windows[index][3]) / 2
            require(math.isfinite(gp) and math.isclose(gp, expected_gp, rel_tol=0, abs_tol=1e-8),
                    "FB/MSP genetic midpoint mismatch")
            values = np.array([float("nan") if v in (".", "NA", "nan") else float(v) for v in f[4:]],
                              dtype=np.float64).reshape(len(source) * 2, 3)
            missing = np.isnan(values).all(axis=1)
            require(np.all(np.isnan(values).any(axis=1) == missing), "Partial missing posterior vector")
            observed = values[~missing]
            require(np.isfinite(observed).all() and ((observed >= 0) & (observed <= 1)).all(), "Invalid FB probabilities")
            require(np.all(np.abs(observed.sum(axis=1) - 1) <= tolerance), "FB vector does not sum to one")
            posterior[index] = values[select]
            seen.add(index); prior_position = pos
            if row_number % 50 == 0:
                reserve(scratch, min_free_mb)
    posterior.flush()
    return windows, hard, posterior, dict(source_people=len(source), windows=len(windows),
        fb_rows=len(seen), windows_without_fb=len(windows) - len(seen))


def annotation(hard_a, hard_b, p_a, p_b):
    state = "UNKNOWN" if hard_a < 0 or hard_b < 0 else (
        STATES[hard_a] if hard_a == hard_b else "DISCORDANT")
    known_a, known_b = np.isfinite(p_a).all(), np.isfinite(p_b).all()
    posterior_status = "BOTH_AVAILABLE" if known_a and known_b else (
        "A_ONLY" if known_a else "B_ONLY" if known_b else "MISSING_FB")
    labels = tuple(NA if x < 0 else POPULATIONS[x] for x in (hard_a, hard_b))
    values = [format(float(v), ".17g") if math.isfinite(v) else NA for v in (*p_a, *p_b)]
    return state, posterior_status, labels, values


def annotate(contract_path, expected_sha, output, scratch, min_free_mb=12288,
             sort_memory_mb=64, max_matrix_mb=1024, bedtools="bedtools"):
    started = time.monotonic()
    contract_path, output, scratch = map(Path, (contract_path, output, scratch))
    require(sha256(contract_path) == expected_sha, "Local ancestry contract SHA mismatch")
    contract = json.loads(contract_path.read_text())
    require(contract.get("schema") == SCHEMA, "Unsupported ancestry contract")
    chrom = str(contract.get("chromosome")); contract["chromosome"] = chrom
    require(chrom in {str(i) for i in range(1, 23)}, "Autosome required")
    require(contract.get("genome_build") == "GRCh38" and contract.get("coordinate_system") == "1-based-inclusive",
            "GRCh38/inclusive source coordinates required")
    require(tuple(contract.get("population_order", ())) == POPULATIONS, "Explicit population order differs")
    require(contract.get("fb_midpoint_rule") == "round_half_to_even", "FB midpoint convention unsupported")
    tolerance = contract.get("posterior_sum_tolerance")
    require(not isinstance(tolerance, bool) and isinstance(tolerance, (int, float)) and
            0 <= tolerance <= 1e-3, "Invalid numerical probability tolerance")
    phase = contract.get("phase_declared_source", {})
    require(phase.get("same_source") is True and phase.get("haplotype_mapping") == {"1": 0, "2": 1}
            and all(phase.get(key) for key in ("ibd_input_vcf", "gnomix_rephased_vcf", "evidence")),
            "Explicit same-source phase declaration and homologue correspondence required")
    paths = {}
    for name in ("native", "msp", "fb", "crosswalk", "samples"):
        entry = contract[name]
        path = contract_path.parent / entry["path"]
        require(path.is_file() and sha256(path) == entry["sha256"], "Input authentication failed: " + name)
        paths[name] = path
    code_paths = [Path(__file__).resolve().parent / name for name in
                  ("r02_ibd_local_ancestry.py", "r02_ibd_prepare.py", "r02_ibd_compatibility.py",
                   "r02_genomic_pair_evidence.py", "r02_segment_evidence.py")]
    snapshots = input_snapshot([contract_path, *paths.values(), *code_paths])
    require(snapshots[str(contract_path.resolve())]["sha256"] == expected_sha and
            all(snapshots[str(path.resolve())]["sha256"] == contract[key]["sha256"] for key, path in paths.items()),
            "Input changed during authentication")
    samples = read_samples(paths["samples"])
    require(sample_hash(samples) == contract.get("analytical_samples_sha256"), "Analytical sample order hash differs")
    order = {name: i for i, name in enumerate(samples)}
    mapping = {}
    with paths["crosswalk"].open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require({"source_id", "sample_id"} <= set(reader.fieldnames or []), "Crosswalk fields missing")
        for row in reader:
            require(row["source_id"] and row["source_id"] not in mapping and row["sample_id"] in order,
                    "Invalid/duplicate crosswalk person")
            mapping[row["source_id"]] = row["sample_id"]
    require(len(mapping) == len(samples) and set(mapping.values()) == set(samples), "Crosswalk not bijective")
    require(min_free_mb >= 0 and sort_memory_mb > 0 and max_matrix_mb > 0 and scratch.is_dir(), "Invalid scratch/resources")
    reserve(scratch, min_free_mb)
    bedtools_version = subprocess.check_output([bedtools, "--version"], text=True, timeout=20).strip()
    require(bedtools_version.startswith("bedtools v"), "Expected BEDTools executable")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    counts = Counter(source_records=0, analytical_records=0, excluded_nonanalytical_records=0, children=0,
                     record_bp=0, both_posteriors_bp=0, **{state + "_bp": 0 for state in STATES})
    with tempfile.TemporaryDirectory(prefix="ibd-lai-", dir=scratch) as directory:
        work = Path(directory)
        windows, hard, posterior, window_report = prepare_windows(paths, contract, samples, mapping, work,
                                                                  max_matrix_mb, min_free_mb)
        db = sqlite3.connect(work / "parents.sqlite")
        try:
            db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA cache_size=-8192")
            db.execute("CREATE TABLE parents(id INTEGER PRIMARY KEY, native TEXT)")
            with text_open(paths["native"]) as handle, (work / "ibd.bed").open("x") as bed:
                for record_id, line in enumerate(handle, 1):
                    x, h1, y, h2, lo, hi, _, _ = native_record(line, chrom)
                    counts["source_records"] += 1
                    if x not in mapping or y not in mapping:
                        counts["excluded_nonanalytical_records"] += 1
                        continue
                    db.execute("INSERT INTO parents VALUES(?,?)", (record_id, line.strip()))
                    bed.write(f"{chrom}\t{lo - 1}\t{hi}\t{record_id}\n")
                    counts["analytical_records"] += 1
                    if record_id % 10000 == 0:
                        reserve(work, min_free_mb)
            db.commit()
            with (work / "ibd.sorted.bed").open("x") as sorted_file:
                subprocess.run(["sort", "-k1,1", "-k2,2n", "-k3,3n", "-k4,4n", "--parallel=1",
                    "-S", f"{sort_memory_mb}M", "-T", str(work), str(work / "ibd.bed")],
                    stdout=sorted_file, check=True, env={**os.environ, "LC_ALL": "C"})
            command = [bedtools, "intersect", "-sorted", "-wao", "-a", str(work / "ibd.sorted.bed"),
                       "-b", str(work / "windows.bed")]
            with (work / "bedtools.stderr").open("w+") as errors, \
                    gzip.open(output / "parents.tsv.gz", "wt", newline="") as parents_handle, \
                    gzip.open(output / "children.tsv.gz", "wt", newline="") as children_handle:
                parents = csv.writer(parents_handle, delimiter="\t"); parents.writerow(PARENT_COLUMNS)
                children = csv.writer(children_handle, delimiter="\t"); children.writerow(CHILD_COLUMNS)
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
                emitted = 0
                try:
                    for record_id, rows in itertools.groupby(process.stdout, key=lambda line: int(line.split("\t")[3])):
                        source_row = db.execute("SELECT native FROM parents WHERE id=?", (record_id,)).fetchone()
                        require(source_row is not None, "BEDTools returned unknown parent")
                        native = source_row[0].split()
                        x, h1, y, h2, lo, hi, _, _ = native_record(source_row[0], chrom)
                        ha, hb = 2 * order[mapping[x]] + int(h1) - 1, 2 * order[mapping[y]] + int(h2) - 1
                        cursor, tally, child_count = lo, Counter(), 0

                        def emit(start, end, window_index):
                            nonlocal child_count
                            if window_index is None:
                                result = annotation(-1, -1, np.full(3, np.nan), np.full(3, np.nan))
                            else:
                                result = annotation(int(hard[window_index, ha]), int(hard[window_index, hb]),
                                                    posterior[window_index, ha], posterior[window_index, hb])
                            state, posterior_status, labels, values = result
                            length = end - start + 1
                            require(length > 0, "Nonpositive child length")
                            children.writerow((record_id, chrom, start, end, length,
                                window_index if window_index is not None else NA, *labels, state, posterior_status, *values))
                            tally[state + "_bp"] += length
                            if posterior_status == "BOTH_AVAILABLE":
                                tally["both_posteriors_bp"] += length
                            child_count += 1

                        for line in rows:
                            f = line.rstrip("\n").split("\t")
                            require(len(f) == 9 and f[0] == chrom and int(f[1]) == lo - 1 and int(f[2]) == hi,
                                    "Unexpected BEDTools parent geometry")
                            overlap = int(f[8])
                            if overlap == 0:
                                require(f[4] == ".", "Unexpected zero-overlap feature")
                                continue
                            index = int(f[7]); left, right, _, _ = windows[index]
                            require(f[4] == chrom and int(f[5]) == left - 1 and int(f[6]) == right,
                                    "BEDTools window differs")
                            start, end = max(lo, left), min(hi, right)
                            require(end - start + 1 == overlap and start >= cursor, "Invalid/overlapping intersection")
                            if cursor < start:
                                emit(cursor, start - 1, None)
                            emit(start, end, index); cursor = end + 1
                        if cursor <= hi:
                            emit(cursor, hi, None)
                        length = hi - lo + 1
                        require(sum(tally[s + "_bp"] for s in STATES) == length, "Children fail parent bp conservation")
                        parents.writerow((record_id, *native, mapping[x], mapping[y], length,
                            *[tally[s + "_bp"] for s in STATES], tally["both_posteriors_bp"], child_count))
                        counts.update(tally); counts["record_bp"] += length; counts["children"] += child_count
                        emitted += 1
                        if emitted % 10000 == 0:
                            reserve(work, min_free_mb)
                    require(process.wait() == 0, "BEDTools intersection failed")
                    require(emitted == counts["analytical_records"], "BEDTools omitted or duplicated parents")
                finally:
                    process.stdout.close()
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait(timeout=30)
            del hard, posterior
        finally:
            db.close()
    verify_unchanged([contract_path, *paths.values(), *code_paths], snapshots)
    reserve(scratch, min_free_mb)
    report = dict(schema=SCHEMA, status="ANNOTATED_REPORTED_IBD_WITH_EXISTING_GNOMIX_NOT_VALIDATION",
        chromosome=chrom, genome_build=contract["genome_build"], analytical_samples=len(samples),
        **counts, **window_report, input_contract_sha256=expected_sha,
        input_sha256={key: contract[key]["sha256"] for key in paths},
        code_sha256={p.name: snapshots[str(p.resolve())]["sha256"] for p in code_paths},
        phase_declared_source=phase, phase_accuracy="NOT_VALIDATED", probabilities="SEPARATE_NOT_MULTIPLIED",
        child_length_cm="NOT_ASSIGNED_PARENT_CM_PRESERVED", absence_semantics="UNKNOWN_NOT_BIOLOGICAL_ABSENCE",
        counting_scope="SUM_OF_ORIGINAL_RECORD_BP_NOT_UNION_BY_PAIR_OR_HAPLOTYPE",
        local_ancestry_source="EXISTING_GNOMIX_MSP_FB_NO_NEW_INFERENCE",
        posterior_sum_tolerance=tolerance, bedtools_version=bedtools_version, numpy_version=np.__version__,
        posterior_missing_semantics="KEEP_MSP_LABEL_AND_NA_POSTERIOR_NO_FORWARD_FILL",
        resources=dict(sort_memory_mb=sort_memory_mb, max_matrix_mb=max_matrix_mb, min_free_mb=min_free_mb),
        seconds=time.monotonic() - started, comparison_with_m14_executed=False,
        contains_individual_identifiers=True,
        outputs_sha256={name: sha256(output / name) for name in ("parents.tsv.gz", "children.tsv.gz")})
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--min-free-mb", type=int, default=12288)
    parser.add_argument("--sort-memory-mb", type=int, default=64)
    parser.add_argument("--max-matrix-mb", type=int, default=1024)
    parser.add_argument("--bedtools", default="bedtools")
    args = parser.parse_args()
    os.umask(0o077)
    result = annotate(args.contract, args.contract_sha256, args.output_dir, args.scratch_dir,
                      args.min_free_mb, args.sort_memory_mb, args.max_matrix_mb, args.bedtools)
    print(json.dumps({key: result[key] for key in ("status", "chromosome", "analytical_records", "children")}))


if __name__ == "__main__":
    main()
