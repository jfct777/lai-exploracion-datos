#!/usr/bin/env python3
"""Audit ALT-versus-minor-allele semantics on one chromosome.

The script parses the selected M14 cohort once, verifies REF against the
declared reference, runs the production M14 segment/window algorithms under
three explicit carrier policies, and compares the historical arm against the
published per-chromosome artefacts.  It does not train a model or read a held-
out label as a target.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rare_allele_orientation import (
    MODES,
    SiteOrientation,
)
from rare_allele_sharing_painter import (
    compute_sharing_windows,
    detect_pairwise_segments_direct,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit rare-allele orientation and its chr-level M14 sensitivity."
    )
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--canonical-summary", required=True)
    parser.add_argument("--canonical-windows", required=True)
    parser.add_argument("--canonical-segments", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--train-label", default="TRAIN")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--outdir", default=".")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_reference_contig(fasta_path: str | Path, chrom: str) -> tuple[str, str]:
    """Load one indexed FASTA contig without requiring pysam in the container."""

    fai_path = Path(f"{fasta_path}.fai")
    if not fai_path.exists():
        raise SystemExit(f"Missing FASTA index: {fai_path}")
    entries = {}
    with fai_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise SystemExit(f"Malformed FASTA index line: {line[:200]!r}")
            entries[fields[0]] = tuple(map(int, fields[1:5]))

    candidates = (chrom, f"chr{chrom}" if not chrom.startswith("chr") else chrom[3:])
    for candidate in candidates:
        if candidate in entries:
            length, offset, line_bases, line_width = entries[candidate]
            n_lines = (length + line_bases - 1) // line_bases
            bytes_to_read = n_lines * line_width
            with open(fasta_path, "rb") as fasta_handle:
                fasta_handle.seek(offset)
                raw = fasta_handle.read(bytes_to_read)
            sequence = raw.replace(b"\n", b"").replace(b"\r", b"")[:length].decode("ascii").upper()
            if len(sequence) != length:
                raise SystemExit(
                    f"Could not load complete FASTA contig {candidate}: {len(sequence)} != {length}"
                )
            return candidate, sequence
    raise SystemExit(
        f"Reference FASTA has neither {candidates[0]!r} nor {candidates[1]!r}; "
        f"first contigs: {list(entries)[:5]}"
    )


def bcftools_version() -> str:
    proc = subprocess.run(
        ["bcftools", "--version"], capture_output=True, text=True, check=True
    )
    return proc.stdout.splitlines()[0]


def open_bcftools_query(
    vcf_path: str | Path,
    selected_samples: list[str],
) -> tuple[subprocess.Popen, Path]:
    """Open a binary, fixed-width GT stream in canonical sample order."""

    sample_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
    sample_file.write("\n".join(selected_samples) + "\n")
    sample_file.close()
    sample_path = Path(sample_file.name)

    listed = subprocess.run(
        ["bcftools", "query", "-l", "-S", str(sample_path), str(vcf_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if listed != selected_samples:
        sample_path.unlink(missing_ok=True)
        raise SystemExit("bcftools did not preserve the canonical M14 sample identity/order")

    proc = subprocess.Popen(
        [
            "bcftools",
            "query",
            "-S",
            str(sample_path),
            "-f",
            r"%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n",
            str(vcf_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout is None or proc.stderr is None:
        sample_path.unlink(missing_ok=True)
        raise SystemExit("Could not open bcftools query subprocess")
    return proc, sample_path


def load_canonical_summary(path: str | Path, chrom: str) -> tuple[list[str], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    observed = str(payload.get("chrom", "")).removeprefix("chr")
    expected = str(chrom).removeprefix("chr")
    if observed != expected:
        raise SystemExit(f"Canonical summary chromosome is {observed!r}, expected {expected!r}")
    samples = payload.get("selected_samples")
    if not isinstance(samples, list) or not samples or len(samples) != len(set(samples)):
        raise SystemExit("Canonical summary lacks a unique, non-empty selected_samples list")

    required = {
        "window_size_bp",
        "step_size_bp",
        "min_shared_variants",
        "min_jaccard",
        "max_gap_bp",
        "min_segment_bp",
    }
    params = payload.get("parameters_used", {})
    missing = sorted(required - set(params))
    if missing:
        raise SystemExit(f"Canonical M14 summary lacks parameters_used fields: {missing}")
    return [str(sample) for sample in samples], params


def load_train_mask(
    path: str | Path,
    selected_samples: list[str],
    sample_id_col: str,
    split_col: str,
    train_label: str,
) -> np.ndarray:
    split = pd.read_csv(path, sep="\t", dtype={sample_id_col: str, split_col: str})
    if split[sample_id_col].duplicated().any():
        raise SystemExit("split_manifest contains duplicate sample IDs")
    role = split.set_index(sample_id_col)[split_col]
    missing = [sample for sample in selected_samples if sample not in role.index]
    if missing:
        raise SystemExit(f"split_manifest lacks {len(missing)} M14 samples; first: {missing[:5]}")
    return np.array([role.at[sample] == train_label for sample in selected_samples], dtype=bool)


def _sorted_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "chrom" in normalized.columns:
        normalized["chrom"] = normalized["chrom"].astype(str).str.replace(
            r"^chr", "", regex=True
        )
    for column in normalized.select_dtypes(include=["object", "string"]).columns:
        normalized[column] = normalized[column].fillna("").astype(str)
    if normalized.empty:
        return normalized.reset_index(drop=True)
    return normalized.sort_values(list(normalized.columns), kind="stable").reset_index(drop=True)


def frames_equal_exact(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, str | None]:
    try:
        pd.testing.assert_frame_equal(
            _sorted_frame(left),
            _sorted_frame(right),
            check_dtype=False,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        return False, str(exc)[:2000]
    return True, None


def pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    return set(zip(df["sample_a"].astype(str), df["sample_b"].astype(str)))


def segment_set(df: pd.DataFrame) -> set[tuple]:
    if df.empty:
        return set()
    return set(map(tuple, df.itertuples(index=False, name=None)))


def interval_set(df: pd.DataFrame) -> set[tuple[str, str, int, int]]:
    if df.empty:
        return set()
    return set(
        zip(
            df["sample_a"].astype(str),
            df["sample_b"].astype(str),
            df["start_pos"].astype(int),
            df["end_pos"].astype(int),
        )
    )


def _merged_intervals(df: pd.DataFrame) -> dict[tuple[str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        grouped[(str(row.sample_a), str(row.sample_b))].append(
            (int(row.start_pos), int(row.end_pos))
        )

    merged: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for pair, intervals in grouped.items():
        current: list[list[int]] = []
        for start, end in sorted(intervals):
            if not current or start > current[-1][1] + 1:
                current.append([start, end])
            else:
                current[-1][1] = max(current[-1][1], end)
        merged[pair] = [(start, end) for start, end in current]
    return merged


def interval_overlap_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compare pair-specific genomic coverage without requiring identical borders."""

    reference_intervals = _merged_intervals(reference)
    current_intervals = _merged_intervals(current)

    def total_bp(grouped: dict[tuple[str, str], list[tuple[int, int]]]) -> int:
        return sum(end - start + 1 for intervals in grouped.values() for start, end in intervals)

    overlap_bp = 0
    for pair in reference_intervals.keys() & current_intervals.keys():
        left = reference_intervals[pair]
        right = current_intervals[pair]
        left_idx = right_idx = 0
        while left_idx < len(left) and right_idx < len(right):
            left_start, left_end = left[left_idx]
            right_start, right_end = right[right_idx]
            overlap_bp += max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
            if left_end <= right_end:
                left_idx += 1
            else:
                right_idx += 1

    reference_bp = total_bp(reference_intervals)
    current_bp = total_bp(current_intervals)
    return {
        "pairwise_interval_overlap_bp": int(overlap_bp),
        "historical_pairwise_bp_fraction_overlapped": (
            overlap_bp / reference_bp if reference_bp else 1.0
        ),
        "current_pairwise_bp_fraction_overlapped": (
            overlap_bp / current_bp if current_bp else 1.0
        ),
        "interval_set_jaccard_vs_historical": jaccard(
            interval_set(reference), interval_set(current)
        ),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    left_series = pd.Series(left)
    right_series = pd.Series(right)
    if left_series.nunique(dropna=True) < 2 or right_series.nunique(dropna=True) < 2:
        return None
    return float(left_series.corr(right_series, method=method))


def window_comparison(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    keys = ["chrom", "start_pos", "end_pos"]
    columns = keys + ["n_sharing_pairs"]
    merged = reference[columns].merge(
        current[columns], on=keys, how="outer", suffixes=("_historical", "_current"),
        validate="one_to_one", indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise SystemExit("Window coordinates differ between historical and sensitivity modes")
    historical = merged["n_sharing_pairs_historical"].to_numpy(dtype=float)
    current_values = merged["n_sharing_pairs_current"].to_numpy(dtype=float)
    return {
        "windows_with_changed_pair_count": int(np.count_nonzero(historical != current_values)),
        "total_window_pair_count_historical": int(historical.sum()),
        "total_window_pair_count_current": int(current_values.sum()),
        "window_pair_count_pearson_vs_historical": _safe_correlation(
            historical, current_values, "pearson"
        ),
        "window_pair_count_spearman_vs_historical": _safe_correlation(
            historical, current_values, "spearman"
        ),
    }


def jaccard(left: set, right: set) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def mode_summary(mode: str, variants: list, windows: pd.DataFrame, segments: pd.DataFrame) -> dict:
    pairs = pair_set(segments)
    return {
        "mode": mode,
        "n_variants_with_at_least_two_carriers": len(variants),
        "n_windows": int(len(windows)),
        "n_segments": int(len(segments)),
        "n_pairs": len(pairs),
        "total_shared_bp": int(segments["length_bp"].sum()) if not segments.empty else 0,
    }


def main() -> None:
    args = parse_args()
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be >=1")

    chrom = str(args.chrom).removeprefix("chr")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selected_samples, production_params = load_canonical_summary(args.canonical_summary, chrom)
    train_mask = load_train_mask(
        args.split_manifest,
        selected_samples,
        args.sample_id_col,
        args.split_col,
        args.train_label,
    )

    variants_by_mode: dict[str, list[tuple[int, frozenset[int]]]] = {
        mode: [] for mode in MODES
    }
    n_samples = len(selected_samples)
    dosage_by_mode = {mode: np.zeros(n_samples, dtype=np.int64) for mode in MODES}
    carrier_sites_by_mode = {mode: np.zeros(n_samples, dtype=np.int64) for mode in MODES}
    callable_sites_by_mode = {mode: np.zeros(n_samples, dtype=np.int64) for mode in MODES}

    counts = defaultdict(int)
    incidence = defaultdict(int)
    train_incidence = defaultdict(int)
    train_dosage = defaultdict(int)
    chrom_min = None
    chrom_max = None

    reference_contig, reference_sequence = load_reference_contig(args.reference_fasta, chrom)
    query_proc, sample_path = open_bcftools_query(args.vcf, selected_samples)

    sites_path = outdir / f"chr{chrom}.orientation_sites.tsv.gz"
    columns = [
        "chrom", "pos", "ref", "alt", "ref_matches_hg38", "alt_count", "allele_number",
        "alt_frequency", "minor_allele", "alt_is_major", "n_alt_carriers",
        "n_minor_carriers", "n_alt_carriers_train", "alt_dosage_train",
    ]
    with gzip.open(sites_path, "wt", encoding="utf-8", newline="") as sites_handle:
        sites_handle.write("\t".join(columns) + "\n")
        for raw_line in query_proc.stdout:
            counts["total_sites"] += 1
            parts = raw_line.rstrip(b"\n").split(b"\t", 4)
            if len(parts) != 5:
                raise SystemExit(f"Malformed bcftools query row: {raw_line[:200]!r}")
            chrom_b, pos_b, ref_b, alt_b, genotype_bytes = parts
            row_chrom = chrom_b.decode("ascii")
            ref = ref_b.decode("ascii")
            alt = alt_b.decode("ascii")
            observed_chrom = row_chrom.removeprefix("chr")
            if observed_chrom != chrom:
                raise SystemExit(
                    f"VCF chromosome mismatch at {row_chrom}:{pos_b.decode()}; expected chr{chrom}"
                )
            if "," in alt or len(ref) != 1 or len(alt) != 1:
                raise SystemExit(
                    f"Expected biallelic SNV, found {ref}>{alt} at {row_chrom}:{pos_b.decode()}"
                )
            # Every autosomal biallelic GT is three bytes (e.g. 0/1, 1|1,
            # ./.) separated by one tab. Prefixing one tab gives a fixed
            # N×4 byte matrix and avoids 1.2 billion Python string splits.
            fixed_width = b"\t" + genotype_bytes
            if len(fixed_width) != 4 * n_samples:
                raise SystemExit(
                    f"Non-diploid or malformed GT width at {row_chrom}:{pos_b.decode()}: "
                    f"{len(fixed_width)} bytes != {4 * n_samples}"
                )
            gt_bytes = np.frombuffer(fixed_width, dtype=np.uint8).reshape(n_samples, 4)
            alleles = gt_bytes[:, (1, 3)]
            called = alleles != ord(".")
            orientation = SiteOrientation(
                alt_count=int(np.count_nonzero(alleles == ord("1"))),
                allele_number=int(called.sum()),
            )
            pos = int(pos_b)
            chrom_min = pos if chrom_min is None else min(chrom_min, pos)
            chrom_max = pos if chrom_max is None else max(chrom_max, pos)
            if pos > len(reference_sequence):
                raise SystemExit(
                    f"Position {pos} exceeds reference contig length {len(reference_sequence)}"
                )
            ref_observed = reference_sequence[pos - 1]
            ref_matches = ref_observed == ref.upper()
            counts["ref_matches_hg38" if ref_matches else "ref_mismatches_hg38"] += 1
            if orientation.allele_number == 0:
                counts["all_missing_sites"] += 1
            elif orientation.is_tie:
                counts["tie_sites"] += 1
            elif orientation.alt_is_major:
                counts["alt_major_sites"] += 1
            else:
                counts["alt_minor_sites"] += 1

            fully_called = called.all(axis=1)
            alt_carrier_mask = np.any(alleles == ord("1"), axis=1)
            minor_carrier_mask = (
                np.any(alleles == ord("0"), axis=1)
                if orientation.alt_is_major
                else alt_carrier_mask
            )
            mode_masks = {
                "historical_alt": alt_carrier_mask,
                "minor_allele": None if orientation.is_tie else minor_carrier_mask,
                "exclude_alt_major": None if orientation.alt_is_major else alt_carrier_mask,
            }
            for mode, carrier_mask in mode_masks.items():
                if carrier_mask is None or orientation.allele_number == 0:
                    continue
                carrier_idx = np.flatnonzero(carrier_mask)
                carrier_set = frozenset(map(int, carrier_idx))
                incidence[mode] += int(carrier_mask.sum())
                train_incidence[mode] += int(np.count_nonzero(carrier_mask & train_mask))
                if len(carrier_set) >= 2:
                    variants_by_mode[mode].append((pos, carrier_set))

                counted_allele = ord("0") if mode == "minor_allele" and orientation.alt_is_major else ord("1")
                dosage = np.count_nonzero(alleles == counted_allele, axis=1).astype(np.int8)
                dosage = np.where(fully_called, dosage, 0)
                callable_sites_by_mode[mode] += fully_called
                dosage_by_mode[mode] += dosage
                carrier_sites_by_mode[mode] += dosage > 0
                train_dosage[mode] += int(dosage[train_mask].sum())

            alt_carriers = frozenset(map(int, np.flatnonzero(alt_carrier_mask)))
            minor_carriers = frozenset(map(int, np.flatnonzero(minor_carrier_mask)))
            alt_train_carriers = int(np.count_nonzero(alt_carrier_mask & train_mask))
            alt_dosage = np.count_nonzero(alleles == ord("1"), axis=1).astype(np.int8)
            alt_dosage = np.where(fully_called, alt_dosage, 0)
            alt_train_dosage = int(alt_dosage[train_mask].sum())
            alt_frequency_text = (
                "NA" if orientation.alt_frequency is None
                else f"{orientation.alt_frequency:.12g}"
            )
            sites_handle.write(
                "\t".join(
                    map(
                        str,
                        (
                            chrom,
                            pos,
                            ref,
                            alt,
                            int(ref_matches),
                            orientation.alt_count,
                            orientation.allele_number,
                            alt_frequency_text,
                            orientation.minor_allele,
                            int(orientation.alt_is_major),
                            len(alt_carriers),
                            len(minor_carriers),
                            alt_train_carriers,
                            alt_train_dosage,
                        ),
                    )
                )
                + "\n"
            )

    query_proc.stdout.close()
    stderr = query_proc.stderr.read().decode("utf-8", errors="replace")
    query_proc.stderr.close()
    return_code = query_proc.wait()
    sample_path.unlink(missing_ok=True)
    if return_code != 0:
        raise SystemExit(f"bcftools query failed ({return_code}): {stderr.strip()}")

    if counts["ref_mismatches_hg38"]:
        raise SystemExit(
            f"REF_QC_FAIL: {counts['ref_mismatches_hg38']} sites disagree with the declared hg38 FASTA"
        )

    canonical_windows = pd.read_csv(args.canonical_windows, sep="\t")
    canonical_segments = pd.read_csv(args.canonical_segments, sep="\t")
    mode_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    mode_summaries: dict[str, dict] = {}
    historical_reproduction = None

    for mode in MODES:
        variants = variants_by_mode[mode]
        segments = detect_pairwise_segments_direct(
            chrom,
            variants,
            selected_samples,
            int(production_params["max_gap_bp"]),
            int(production_params["min_segment_bp"]),
            int(production_params["min_shared_variants"]),
            n_jobs=args.n_jobs,
        )
        windows = compute_sharing_windows(
            chrom,
            variants,
            selected_samples,
            int(production_params["window_size_bp"]),
            int(production_params["step_size_bp"]),
            int(production_params["min_shared_variants"]),
            float(production_params["min_jaccard"]),
        )
        segments.to_csv(
            outdir / f"chr{chrom}.{mode}.pairwise_segments.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
        windows.to_csv(
            outdir / f"chr{chrom}.{mode}.sharing_windows.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
        summary = mode_summary(mode, variants, windows, segments)
        write_json(outdir / f"chr{chrom}.{mode}.summary.json", summary)
        mode_frames[mode] = (windows, segments)
        mode_summaries[mode] = summary

        if mode == "historical_alt":
            windows_equal, windows_diff = frames_equal_exact(windows, canonical_windows)
            segments_equal, segments_diff = frames_equal_exact(segments, canonical_segments)
            historical_reproduction = {
                "windows_equal": windows_equal,
                "segments_equal": segments_equal,
                "windows_difference": windows_diff,
                "segments_difference": segments_diff,
            }

    if not historical_reproduction or not all(
        historical_reproduction[key] for key in ("windows_equal", "segments_equal")
    ):
        raise SystemExit(
            "HISTORICAL_REPRODUCTION_FAIL: recomputed historical_alt does not match canonical chr output; "
            + json.dumps(historical_reproduction, sort_keys=True)
        )

    hist_segments = mode_frames["historical_alt"][1]
    hist_pairs = pair_set(hist_segments)
    hist_segment_set = segment_set(hist_segments)
    comparison_rows = []
    comparisons = {}
    for mode in MODES:
        current_windows = mode_frames[mode][0]
        current_segments = mode_frames[mode][1]
        current_pairs = pair_set(current_segments)
        current_segment_set = segment_set(current_segments)
        row = {
            **mode_summaries[mode],
            "pair_set_jaccard_vs_historical": jaccard(hist_pairs, current_pairs),
            "segment_set_jaccard_vs_historical": jaccard(hist_segment_set, current_segment_set),
            "pairs_added_vs_historical": len(current_pairs - hist_pairs),
            "pairs_removed_vs_historical": len(hist_pairs - current_pairs),
            "segments_added_vs_historical": len(current_segment_set - hist_segment_set),
            "segments_removed_vs_historical": len(hist_segment_set - current_segment_set),
            "delta_total_shared_bp_vs_historical": (
                mode_summaries[mode]["total_shared_bp"]
                - mode_summaries["historical_alt"]["total_shared_bp"]
            ),
            **interval_overlap_summary(hist_segments, current_segments),
            **window_comparison(mode_frames["historical_alt"][0], current_windows),
        }
        for scope, mask in (("all", np.ones(n_samples, dtype=bool)), ("train", train_mask)):
            for measure, values_by_mode in (
                ("dosage", dosage_by_mode),
                ("carrier_sites", carrier_sites_by_mode),
            ):
                historical_values = values_by_mode["historical_alt"][mask]
                current_values = values_by_mode[mode][mask]
                row[f"{scope}_{measure}_pearson_vs_historical"] = _safe_correlation(
                    historical_values, current_values, "pearson"
                )
                row[f"{scope}_{measure}_spearman_vs_historical"] = _safe_correlation(
                    historical_values, current_values, "spearman"
                )
        comparison_rows.append(row)
        comparisons[mode] = row
    pd.DataFrame(comparison_rows).to_csv(
        outdir / f"chr{chrom}.mode_comparison.tsv", sep="\t", index=False
    )

    burden = pd.DataFrame({"sample_id": selected_samples, "is_train": train_mask.astype(int)})
    for mode in MODES:
        burden[f"{mode}_dosage_sum"] = dosage_by_mode[mode]
        burden[f"{mode}_carrier_site_count"] = carrier_sites_by_mode[mode]
        burden[f"{mode}_callable_sites"] = callable_sites_by_mode[mode]
    burden.to_csv(
        outdir / f"chr{chrom}.sample_burden_by_mode.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    orientation_summary = {
        "chrom": chrom,
        "n_samples": n_samples,
        "n_train_samples": int(train_mask.sum()),
        "counts": dict(sorted(counts.items())),
        "carrier_incidences": {mode: int(incidence[mode]) for mode in MODES},
        "train_carrier_incidences": {mode: int(train_incidence[mode]) for mode in MODES},
        "train_dosage_sum": {mode: int(train_dosage[mode]) for mode in MODES},
        "sensitivity_fractions": {
            "alt_major_site_fraction": counts["alt_major_sites"] / counts["total_sites"],
            "historical_carrier_incidence_at_alt_major_fraction": (
                (incidence["historical_alt"] - incidence["exclude_alt_major"])
                / incidence["historical_alt"]
            ),
            "historical_train_dosage_at_alt_major_fraction": (
                (train_dosage["historical_alt"] - train_dosage["exclude_alt_major"])
                / train_dosage["historical_alt"]
            ),
        },
        "production_parameters_from_canonical_summary": production_params,
        "reference_contig": reference_contig,
        "field_definitions": {
            "historical_alt": "Cuenta ALT tal como hicieron M14/M20/M23, sin reinterpretar cuál alelo es menor.",
            "minor_allele": "Cuenta el alelo menos frecuente entre los cromosomas llamados; excluye empates sin alelo menor único.",
            "exclude_alt_major": "Conserva el conteo ALT histórico, pero excluye sitios donde ALT es mayoritario.",
            "alt_is_major": "ALT tiene más de la mitad de las copias alélicas llamadas en las 2619 muestras de M14.",
            "carrier_site_count": "Número de sitios retenidos donde el individuo porta al menos una copia del alelo contado.",
            "dosage_sum": "Suma de copias del alelo contado en sitios retenidos con genotipo diploide completo.",
            "callable_sites": "Número de sitios retenidos con las dos copias del genotipo observadas; no es callability por base.",
            "is_train": "Marca de pertenencia a TRAIN usada solo para resúmenes de carga, nunca como etiqueta objetivo.",
        },
    }
    write_json(outdir / f"chr{chrom}.orientation_summary.json", orientation_summary)

    audit = {
        "status": "PASS",
        "scope": "single_chromosome_sensitivity_no_model_training",
        "chrom": chrom,
        "historical_reproduction": historical_reproduction,
        "reference_qc": {
            "matches": int(counts["ref_matches_hg38"]),
            "mismatches": int(counts["ref_mismatches_hg38"]),
        },
        "orientation_summary": orientation_summary,
        "mode_comparisons": comparisons,
        "interpretation_limits": [
            "This audit measures sensitivity of chr-level M14 outputs; it is not a genome-wide M16.5 community rerun.",
            "M14-derived labels are internal to the same rare-variant structure and are not independent biological truth.",
            "No M22/M23 model was retrained and the held-out TEST fold was not used as a target.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "bcftools": bcftools_version(),
        },
        "inputs": {
            "vcf_sha256": sha256(args.vcf),
            "reference_fasta_sha256": sha256(args.reference_fasta),
            "canonical_summary_sha256": sha256(args.canonical_summary),
            "canonical_windows_sha256": sha256(args.canonical_windows),
            "canonical_segments_sha256": sha256(args.canonical_segments),
            "split_manifest_sha256": sha256(args.split_manifest),
        },
        "analysis_date": datetime.now(timezone.utc).isoformat(),
    }
    write_json(outdir / f"chr{chrom}.audit_report.json", audit)


if __name__ == "__main__":
    main()
