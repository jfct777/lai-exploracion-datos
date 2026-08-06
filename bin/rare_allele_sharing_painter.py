#!/usr/bin/env python3

import argparse
import gzip
import html
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from rare_allele_orientation import SiteOrientation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARING_WINDOW_COLUMNS = [
    "chrom",
    "window_id",
    "start_pos",
    "end_pos",
    "mid_pos",
    "window_bp",
    "n_rare_variants",
    "n_carriers_per_variant_mean",
    "jaccard_mean",
    "jaccard_max",
    "n_sharing_pairs",
    "sharing_pair_list",
]

PAIRWISE_SEGMENT_COLUMNS = [
    "chrom",
    "sample_a",
    "sample_b",
    "segment_id",
    "start_pos",
    "end_pos",
    "length_bp",
    "n_shared_variants",
    "jaccard",
]

SCAN_SUMMARY_COLUMNS = [
    "chrom",
    "n_samples",
    "n_rare_variants",
    "n_windows",
    "n_sharing_pairs",
    "n_segments",
    "total_shared_bp",
    "status",
]


# ---------------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------------

def _parse_region(region_str):
    """Parse a region string like 'chr22:30000000-40000000' or '30000000-40000000'."""
    if not region_str:
        return None, None
    region_str = region_str.strip()
    # Strip optional chromosome prefix (e.g. "chr22:")
    if ":" in region_str:
        region_str = region_str.split(":", 1)[1]
    region_str = region_str.replace("_", "")
    parts = region_str.split("-")
    if len(parts) != 2:
        _fail(f"Invalid region format '{region_str}'. Use chr:START-END or START-END, e.g. 'chr22:30000000-40000000'")
    try:
        start = int(float(parts[0]))
        end   = int(float(parts[1]))
    except ValueError:
        _fail(f"Cannot parse region '{region_str}' as numeric START-END")
    if start >= end:
        _fail(f"Region start ({start}) must be < end ({end})")
    return start, end


def _filter_variants_by_region(variants, start_bp, end_bp):
    """Keep only variants whose position falls within [start_bp, end_bp]."""
    return [(pos, cs) for pos, cs in variants if start_bp <= pos <= end_bp]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Identify and paint shared rare allele segments between individuals."
    )
    parser.add_argument("--mode", required=True, choices=["scan", "aggregate"])

    # --- scan mode ---
    parser.add_argument("--input")
    parser.add_argument("--input-format", default="vcf_rare")
    parser.add_argument("--chr")
    parser.add_argument("--sample-ids-file")
    parser.add_argument(
        "--canonical-summary",
        help=(
            "Canonical M14 per-chromosome summary. Required for minor_allele; "
            "its selected_samples and load-bearing parameters are validated."
        ),
    )
    parser.add_argument(
        "--carrier-allele-mode",
        choices=["historical_alt", "minor_allele"],
        default="historical_alt",
        help=(
            "Allele whose carriers define M14. historical_alt preserves the "
            "published behavior; minor_allele flips ALT-major sites to REF and "
            "excludes frequency ties."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--region",
                        help="Genomic region to analyse, e.g. '16000000-26000000' "
                             "or '16e6-26e6'.  Only variants inside this interval "
                             "are kept.  Generates one painting per region.")
    parser.add_argument("--region-size-bp", type=float, default=None,
                        help="If set (e.g. 10e6), automatically splits the "
                             "chromosome into consecutive windows of this size "
                             "and produces one painting per window.")
    parser.add_argument("--window-size-bp", type=int, default=500000)
    parser.add_argument("--step-size-bp", type=int, default=250000)
    parser.add_argument("--min-shared-variants", type=int, default=2)
    parser.add_argument("--min-jaccard", type=float, default=0.05)
    parser.add_argument("--max-gap-bp", type=int, default=500000)
    parser.add_argument("--min-segment-bp", type=int, default=100000)
    parser.add_argument("--max-block-gap-bp", type=int, default=200000,
                        help="Max distance (bp) between consecutive carrier "
                             "variants to keep them in the same block.  "
                             "Default 200 kb, consistent with LD decay in humans.")
    parser.add_argument("--min-block-snps", type=int, default=2,
                        help="Minimum number of consecutive rare-variant carrier "
                             "positions (per individual) required to form a "
                             "carrier block in the individual painting, block-"
                             "group heatmap and span-distribution plots. Runs "
                             "shorter than this threshold are treated as isolated "
                             "carriers (rendered as faint dots). Default 2; "
                             "increase to 3+ to be more conservative and suppress "
                             "single-pair tracts of length 2.")
    parser.add_argument("--out-sharing-windows")
    parser.add_argument("--out-pairwise-segments")
    parser.add_argument("--out-summary-json")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Workers for the pairwise-segment detection loop "
                             "(detect_pairwise_segments_direct). 1 = serial "
                             "(no joblib overhead, byte-identical to pre-2026 "
                             "behaviour). >1 = joblib 'loky' multiprocess; "
                             "speedup is near-linear up to ~8 cores on "
                             "DNABR-scale chromosomes (2M+ pairs/chr) and "
                             "plateaus beyond that as numpy.intersect1d on "
                             "shared CSR rows becomes the bottleneck. The "
                             "Nextflow module forwards task.cpus here so the "
                             "scan respects the Slurm cpus reservation.")

    # --- aggregate mode ---
    parser.add_argument("--pairwise-segments", action="append", default=[])
    parser.add_argument("--per-chr-summary", action="append", default=[])
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--aggregate-chunk-rows", type=int, default=1_000_000,
                        help="Chunk size (rows) used when streaming per-chromosome "
                             "pairwise-segment TSVs in aggregate mode. Lower values "
                             "reduce peak RAM at the cost of more Python-level "
                             "iterations; 1M rows is ~100 MB of pandas working memory "
                             "per chunk. Previous implementation loaded every chunk "
                             "into a single concatenated DataFrame and OOMed at "
                             "N ~= 2500 with 24 chromosomes (~70 GB uncompressed, "
                             "~200-400 GB pandas resident).")

    # --- plotting (shared) ---
    parser.add_argument("--plot-dpi", type=int, default=600)
    parser.add_argument("--plot-palette", choices=["journal", "colorblind"], default="journal")
    parser.add_argument("--plot-font-family", default="DejaVu Sans")
    parser.add_argument("--plot-width-inches", type=float, default=16.0)
    parser.add_argument("--plot-height-inches", type=float, default=10.0)
    parser.add_argument("--plot-max-height-inches", type=float, default=40.0,
                        help="Upper bound for the individual-painting figure height. "
                             "Prevents OOM when rendering large cohorts at high DPI: "
                             "at 600 dpi a 16x40 in figure allocates ~0.9 GB for the RGBA "
                             "raster buffer, whereas an uncapped 16x800 in figure would "
                             "allocate ~18 GB and get SIGKILLed.")
    parser.add_argument("--plot-export-pdf", type=_parse_bool, default=True)
    parser.add_argument("--plot-export-svg", type=_parse_bool, default=False)
    parser.add_argument("--plot-mode", choices=["individual", "pairwise", "both"], default="individual")
    parser.add_argument("--plot-max-pairs-legend", type=int, default=30)
    parser.add_argument("--plot-raster-bp-per-col", type=int, default=10000,
                        help="Base-pairs per column of the rasterised individual-painting "
                             "image. Smaller = higher genomic resolution but larger raster "
                             "(memory and file size scale linearly). Default 10000 (10 kb) "
                             "gives ~13k columns for chr1, crisp block boundaries and <60 MB "
                             "of int32 raster for N=1000.")
    parser.add_argument("--skip-plots", type=_parse_bool, default=False,
                        help="If true, skip all plot generation and only produce TSV/JSON "
                             "outputs. Useful when plotting dominates cost or fails on "
                             "very large cohorts; downstream modules only consume the TSVs.")
    parser.add_argument("--output-dir", default=".")

    return parser.parse_args()


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _natural_chr_key(chrom):
    clean = str(chrom).replace("chr", "")
    if clean.isdigit():
        return (0, int(clean))
    if clean == "X":
        return (1, 0)
    if clean == "Y":
        return (1, 1)
    if clean == "MT":
        return (1, 2)
    return (2, clean)


def _chrom_key(chrom):
    return str(chrom).replace("chr", "")


def _chrom_label(chrom):
    key = _chrom_key(chrom)
    return str(chrom) if str(chrom).startswith("chr") else f"chr{key}"


def _fail(message):
    raise SystemExit(message)


def _log(message):
    print(f"[rare_allele_sharing] {message}", file=sys.stderr, flush=True)


def _write_json(path, payload):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# ---------------------------------------------------------------------------
# VCF reading
# ---------------------------------------------------------------------------

def _read_vcf_header_lines(input_path):
    opener = gzip.open if str(input_path).endswith(".gz") else open
    header_lines = []
    with opener(input_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header_lines.append(line.rstrip("\n"))
    return header_lines


def _read_header_samples(input_path):
    header_lines = _read_vcf_header_lines(input_path)
    if not header_lines:
        _fail(f"Could not read VCF header from: {input_path}")

    chrom_line = None
    for line in header_lines:
        if line.startswith("#CHROM\t"):
            chrom_line = line
            break
    if chrom_line is None:
        _fail(f"Missing #CHROM header line in {input_path}")

    parts = chrom_line.split("\t")
    if len(parts) < 10:
        _fail(f"VCF has no sample columns in {input_path}")
    return parts[9:]


def validate_input_schema(input_path, input_format):
    """Comprueba el formato y las columnas requeridas del archivo de variantes."""
    if input_format != "vcf_rare":
        _fail(
            f"Unsupported input format '{input_format}'. "
            "This module expects upstream rare-only VCF.gz inputs."
        )

    in_path = Path(input_path)
    if not in_path.exists():
        _fail(f"Missing rare VCF input: {in_path}")

    header_lines = _read_vcf_header_lines(in_path)
    if not header_lines:
        _fail(f"Could not read VCF header from: {in_path}")

    if not any(line.startswith("##FORMAT=<ID=GT,") for line in header_lines):
        _fail(
            "Rare allele sharing analysis requires FORMAT/GT in the upstream rare VCF. "
            f"Missing GT format declaration in {in_path}"
        )

    _read_header_samples(in_path)


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def _read_sample_ids_file(path):
    ids = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            sample_id = line.split()[0]
            if sample_id not in seen:
                ids.append(sample_id)
                seen.add(sample_id)
    return ids


def load_selected_samples(header_samples, sample_ids_file, max_samples):
    """Selecciona muestras conservando el orden del encabezado."""
    header_set = set(header_samples)

    if sample_ids_file:
        ids = _read_sample_ids_file(sample_ids_file)
        if not ids:
            _fail(f"Sample selection file is empty or invalid: {sample_ids_file}")
        missing = [s for s in ids if s not in header_set]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            _fail(
                "Some selected sample IDs were not found in the rare VCF header: "
                f"{preview}{suffix}"
            )
        selected = ids
    else:
        selected = list(header_samples)

    if max_samples is not None:
        if max_samples <= 0:
            _fail("max-samples must be positive when provided.")
        selected = selected[:max_samples]

    if not selected:
        _fail("No samples were selected for rare allele sharing analysis.")

    return selected


def load_and_validate_canonical_summary(path, chrom, args):
    """Load the immutable M14 cohort and verify load-bearing scan parameters."""

    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    observed_chrom = str(summary.get("chrom", "")).removeprefix("chr")
    expected_chrom = str(chrom).removeprefix("chr")
    if observed_chrom != expected_chrom:
        _fail(
            f"Canonical summary chromosome mismatch: {observed_chrom!r} != {expected_chrom!r}"
        )
    samples = summary.get("selected_samples")
    if not isinstance(samples, list) or not samples:
        _fail("Canonical summary lacks a non-empty selected_samples list")
    if len(samples) != len(set(map(str, samples))):
        _fail("Canonical summary selected_samples contains duplicates")
    samples = list(map(str, samples))
    if args.expected_samples is not None and len(samples) != args.expected_samples:
        _fail(
            f"Canonical cohort size {len(samples)} != expected {args.expected_samples}"
        )

    canonical_params = summary.get("parameters_used", {})
    checks = {
        "window_size_bp": int(args.window_size_bp),
        "step_size_bp": int(args.step_size_bp),
        "min_shared_variants": int(args.min_shared_variants),
        "min_jaccard": float(args.min_jaccard),
        "max_gap_bp": int(args.max_gap_bp),
        "min_segment_bp": int(args.min_segment_bp),
    }
    for key, current in checks.items():
        if key not in canonical_params:
            _fail(f"Canonical summary lacks load-bearing parameter {key!r}")
        canonical = type(current)(canonical_params[key])
        if canonical != current:
            _fail(
                f"Parameter drift for {key}: requested {current!r}, canonical {canonical!r}"
            )
    return samples


# ---------------------------------------------------------------------------
# Genotype parsing: build carrier sets per variant
# ---------------------------------------------------------------------------

def _is_carrier_gt(gt_value):
    gt = str(gt_value).strip()
    if gt in {"", ".", "./.", ".|."}:
        return False
    alleles = gt.replace("|", "/").split("/")
    return "1" in alleles


def parse_genotypes_carrier_sets(
    input_path,
    chrom,
    selected_samples,
    carrier_allele_mode="historical_alt",
    return_orientation_qc=False,
):
    """Return list of (pos, carrier_set) for each variant where carrier_set
    is a frozenset of sample indices that carry the requested allele.
    Also returns chromosome positional extent (min_pos, max_pos) from ALL
    variants for proper plot scaling."""
    query_cmd = [
        "bcftools", "query", "-f", r"%CHROM\t%POS\t%ALT[\t%GT]\n",
    ]

    temp_samples = None
    if selected_samples:
        temp_samples = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        temp_samples.write("\n".join(selected_samples))
        temp_samples.write("\n")
        temp_samples.close()
        query_cmd.extend(["-S", temp_samples.name])

    query_cmd.append(str(input_path))

    if carrier_allele_mode not in {"historical_alt", "minor_allele"}:
        _fail(f"Unsupported carrier allele mode: {carrier_allele_mode}")

    proc = subprocess.Popen(query_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None or proc.stderr is None:
        _fail("Could not open bcftools query subprocess for rare allele sharing analysis.")

    variants = []  # list of (pos, frozenset_of_carrier_indices) — only those with 2+ carriers
    prev_pos = -1
    total_variants = 0
    total_with_any_carrier = 0
    observed_chrom = None
    chrom_min_pos = None
    chrom_max_pos = None
    orientation_qc = defaultdict(int)

    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip(b"\n")
            if not line:
                continue

            parts = line.split(b"\t", 3)
            if len(parts) != 4:
                _fail(
                    "Unexpected bcftools query output for rare allele sharing: "
                    f"{line[:200]!r}"
                )

            row_chrom = parts[0].decode("ascii")
            pos_s = parts[1].decode("ascii")
            alt_s = parts[2].decode("ascii")
            genotype_bytes = parts[3]
            total_variants += 1
            orientation_qc["total_sites"] += 1
            observed_chrom = observed_chrom or row_chrom

            if row_chrom != chrom and row_chrom != f"chr{chrom}":
                _fail(
                    f"Chromosome mismatch in {input_path}: expected {chrom}, "
                    f"observed {row_chrom}"
                )
            if "," in alt_s:
                _fail(
                    f"Found multiallelic site at {row_chrom}:{pos_s}. "
                    "This module expects already-biallelic rare VCFs."
                )

            try:
                pos = int(pos_s)
            except ValueError as exc:
                _fail(f"Invalid POS in {input_path} at {row_chrom}:{pos_s}: {exc}")

            if pos <= 0:
                _fail(f"Invalid genomic position at {row_chrom}:{pos_s}")
            if pos < prev_pos:
                _fail(
                    f"Input rare VCF not sorted at {row_chrom}:{pos} after {prev_pos}"
                )
            prev_pos = pos

            # Track chromosome extent from ALL variants
            if chrom_min_pos is None or pos < chrom_min_pos:
                chrom_min_pos = pos
            if chrom_max_pos is None or pos > chrom_max_pos:
                chrom_max_pos = pos

            # Diploid autosomal GTs are exactly three bytes (0/0, 0|1, ./.)
            # separated by tabs. The fixed-width view avoids creating billions
            # of Python strings at DNABR scale.
            fixed_width = b"\t" + genotype_bytes
            expected_width = 4 * len(selected_samples)
            if len(fixed_width) != expected_width:
                _fail(
                    f"Non-diploid or malformed GT width at {row_chrom}:{pos_s}: "
                    f"{len(fixed_width)} bytes != {expected_width}"
                )
            gt_bytes = np.frombuffer(fixed_width, dtype=np.uint8).reshape(
                len(selected_samples), 4
            )
            alleles = gt_bytes[:, (1, 3)]
            called = alleles != ord(".")
            orientation = SiteOrientation(
                alt_count=int(np.count_nonzero(alleles == ord("1"))),
                allele_number=int(called.sum()),
            )
            if orientation.allele_number == 0:
                orientation_qc["all_missing_sites"] += 1
                continue
            if orientation.is_tie:
                orientation_qc["tie_sites"] += 1
                if carrier_allele_mode == "minor_allele":
                    continue
            elif orientation.alt_is_major:
                orientation_qc["alt_major_sites"] += 1
            else:
                orientation_qc["alt_minor_sites"] += 1

            orientation_qc["partially_missing_genotypes"] += int(
                np.count_nonzero(called.sum(axis=1) == 1)
            )
            carrier_code = (
                ord("0")
                if carrier_allele_mode == "minor_allele" and orientation.alt_is_major
                else ord("1")
            )
            carriers = frozenset(
                map(int, np.flatnonzero(np.any(alleles == carrier_code, axis=1)))
            )
            if len(carriers) >= 1:
                total_with_any_carrier += 1
            if len(carriers) >= 2:
                variants.append((pos, carriers))
    finally:
        if temp_samples is not None:
            Path(temp_samples.name).unlink(missing_ok=True)

    proc.stdout.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace")
    proc.stderr.close()
    return_code = proc.wait()
    if return_code != 0:
        _fail(f"bcftools query failed for {input_path}: {stderr.strip()}")

    _log(f"chr{chrom}: {total_variants} total rare variants in VCF")
    _log(f"chr{chrom}: {total_with_any_carrier} variants with >=1 carrier among {len(selected_samples)} selected samples")
    _log(f"chr{chrom}: {len(variants)} variants with >=2 carriers (usable for sharing detection)")
    _log(f"chr{chrom}: positional extent {chrom_min_pos}-{chrom_max_pos}")

    result = (observed_chrom or chrom, variants, total_variants, chrom_min_pos, chrom_max_pos)
    if return_orientation_qc:
        return (*result, dict(sorted(orientation_qc.items())))
    return result


# ---------------------------------------------------------------------------
# Windowed Jaccard sharing
# ---------------------------------------------------------------------------

def _build_carrier_matrix(variants, n_samples):
    """Build a sparse N×V binary carrier matrix from the (pos, carrier_set)
    variant list, preserving column order = variant order (already sorted
    by position upstream).

    Returns
    -------
    C : scipy.sparse.csr_matrix of shape (N_samples, V)
        ``C[s, v] == 1`` iff sample ``s`` carries variant ``v``.  Row
        indices are sorted via ``sort_indices()`` so callers can rely on
        ``C.indices`` being ascending per row.
    positions : np.ndarray[int64] of shape (V,)
        Genomic position for each variant column.
    """
    n_variants = len(variants)
    positions = np.empty(n_variants, dtype=np.int64)
    total_incidences = 0
    for v_idx, (pos, carrier_set) in enumerate(variants):
        positions[v_idx] = pos
        total_incidences += len(carrier_set)

    if total_incidences == 0:
        C_empty = sp.csr_matrix((n_samples, n_variants), dtype=np.int32)
        return C_empty, positions

    sample_idx_flat = np.empty(total_incidences, dtype=np.int32)
    var_idx_flat = np.empty(total_incidences, dtype=np.int32)
    offset = 0
    for v_idx, (_pos, carrier_set) in enumerate(variants):
        sz = len(carrier_set)
        if sz == 0:
            continue
        sample_idx_flat[offset:offset + sz] = list(carrier_set)
        var_idx_flat[offset:offset + sz] = v_idx
        offset += sz

    data = np.ones(total_incidences, dtype=np.int32)
    C = sp.csr_matrix(
        (data, (sample_idx_flat, var_idx_flat)),
        shape=(n_samples, n_variants),
    )
    C.sort_indices()
    return C, positions


def compute_sharing_windows(chrom, variants, selected_samples, window_size_bp,
                            step_size_bp, min_shared_variants, min_jaccard,
                            max_pairs_listed=500):
    """Slide fixed-bp windows over the chromosome and compute pairwise Jaccard
    for all sample pairs that share >= min_shared_variants in each window.

    Vectorised implementation
    -------------------------
    For each window we slice the global sparse carrier matrix ``C`` to the
    variants whose position falls inside the window and compute
    ``S = C_win @ C_win.T`` (an N×N sparse shared-variant count matrix) in
    a single scipy.sparse matmul.  Jaccard is then a vectorised operation
    on the non-zero entries of the upper triangle.  This replaces the
    previous ``for i in range(...): for j in range(...):`` Python double
    loop, dropping per-window cost from O(N²) Python iterations to a
    single sparse matmul and O(P) numpy work, where P is the number of
    qualifying pairs.

    The ``sharing_pair_list`` column in the output TSV is truncated to at
    most ``max_pairs_listed`` pairs per window (kept in descending Jaccard
    order) with a ``...+N_more`` suffix, so the per-row string size stays
    bounded regardless of cohort size.  Pass ``max_pairs_listed <= 0`` to
    recover the previous unbounded behaviour.
    """
    if not variants:
        return pd.DataFrame(columns=SHARING_WINDOW_COLUMNS)

    n_samples = len(selected_samples)
    C, positions = _build_carrier_matrix(variants, n_samples)
    if positions.size == 0:
        return pd.DataFrame(columns=SHARING_WINDOW_COLUMNS)

    min_pos = int(positions[0])
    max_pos = int(positions[-1])

    # Access CSC view once so column slicing in the window loop is cheap.
    C_csc = C.tocsc()

    window_rows = []
    window_id = 0

    for win_start in range(min_pos, max_pos + 1, step_size_bp):
        win_end = win_start + window_size_bp - 1
        lo_idx = int(np.searchsorted(positions, win_start, side="left"))
        hi_idx = int(np.searchsorted(positions, win_end, side="right"))
        if hi_idx <= lo_idx:
            continue

        n_rare = hi_idx - lo_idx
        window_id += 1

        # Slice the carrier submatrix for this window (O(nnz) on CSC).
        C_win = C_csc[:, lo_idx:hi_idx].tocsr()

        n_carriers_per_variant = np.asarray(C_win.sum(axis=0)).ravel()
        mean_carriers = float(n_carriers_per_variant.mean()) if n_rare else 0.0

        # Pairwise shared-variant count via a single sparse matmul.
        S = (C_win @ C_win.T).tocsr()
        sample_counts = np.asarray(S.diagonal()).astype(np.int64)

        S_triu = sp.triu(S, k=1).tocoo()
        cnt_mask = S_triu.data >= min_shared_variants
        pair_a = S_triu.row[cnt_mask].astype(np.int64)
        pair_b = S_triu.col[cnt_mask].astype(np.int64)
        shared = S_triu.data[cnt_mask].astype(np.int64)

        denom = sample_counts[pair_a] + sample_counts[pair_b] - shared
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard_values = np.where(denom > 0, shared / denom, 0.0)

        jac_mask = jaccard_values >= min_jaccard
        pair_a = pair_a[jac_mask]
        pair_b = pair_b[jac_mask]
        jaccard_values = jaccard_values[jac_mask]

        n_sharing = int(pair_a.size)
        mid_pos = (win_start + win_end) // 2

        if n_sharing == 0:
            pair_str = ""
            jac_mean = 0.0
            jac_max = 0.0
        else:
            if (max_pairs_listed is not None
                    and max_pairs_listed > 0
                    and n_sharing > max_pairs_listed):
                top_idx = np.argsort(-jaccard_values, kind="stable")[:max_pairs_listed]
                pair_str = ";".join(
                    f"{selected_samples[int(pair_a[i])]}:{selected_samples[int(pair_b[i])]}"
                    for i in top_idx
                ) + f";...+{n_sharing - max_pairs_listed}_more"
            else:
                pair_str = ";".join(
                    f"{selected_samples[int(a)]}:{selected_samples[int(b)]}"
                    for a, b in zip(pair_a, pair_b)
                )
            jac_mean = float(jaccard_values.mean())
            jac_max = float(jaccard_values.max())

        window_rows.append({
            "chrom": chrom,
            "window_id": window_id,
            "start_pos": win_start,
            "end_pos": win_end,
            "mid_pos": mid_pos,
            "window_bp": window_size_bp,
            "n_rare_variants": n_rare,
            "n_carriers_per_variant_mean": round(mean_carriers, 3),
            "jaccard_mean": round(jac_mean, 4),
            "jaccard_max": round(jac_max, 4),
            "n_sharing_pairs": n_sharing,
            "sharing_pair_list": pair_str,
        })

    return pd.DataFrame(window_rows, columns=SHARING_WINDOW_COLUMNS)


# ---------------------------------------------------------------------------
# Pair-centric segment detection (primary method)
# ---------------------------------------------------------------------------

def _detect_segments_for_pair(shared_positions, max_gap_bp, min_segment_bp,
                              min_shared_variants):
    """Given the sorted carrier positions shared by one sample pair, yield
    ``(start_pos, end_pos, length_bp, n_shared)`` for each segment that
    passes the size filters.

    Segments are maximal runs of shared positions separated by gaps no
    larger than ``max_gap_bp``.  Detection is vectorised with ``np.diff``
    + ``np.where`` so it stays O(|shared_positions|) in numpy-level work.
    """
    n = shared_positions.size
    if n < min_shared_variants:
        return

    if n == 1:
        # Cannot form a multi-variant segment unless min_shared_variants == 1.
        if min_shared_variants <= 1:
            pos = int(shared_positions[0])
            if 1 >= min_segment_bp:
                yield pos, pos, 1, 1
        return

    break_pts = np.where(np.diff(shared_positions) > max_gap_bp)[0]
    if break_pts.size == 0:
        starts_idx = (0,)
        ends_idx = (n - 1,)
    else:
        starts_idx = np.concatenate([[0], break_pts + 1])
        ends_idx = np.concatenate([break_pts, [n - 1]])

    for s, e in zip(starts_idx, ends_idx):
        n_shared = int(e - s + 1)
        if n_shared < min_shared_variants:
            continue
        seg_start = int(shared_positions[s])
        seg_end = int(shared_positions[e])
        length_bp = seg_end - seg_start + 1
        if length_bp < min_segment_bp:
            continue
        yield seg_start, seg_end, length_bp, n_shared


def _segments_for_pair_batch(pair_chunk, pair_a, pair_b, indptr, indices,
                              positions, max_gap_bp, min_segment_bp,
                              min_shared_variants):
    """Worker: detect segments for one contiguous slice of qualifying pairs.

    Pure function over its inputs (no shared mutable state), so it parallelises
    cleanly via joblib 'loky' multiprocess. Returns five column buffers that
    the caller concatenates across workers; the order of pair_chunk is
    preserved within each worker, and overall determinism is guaranteed
    because the chunks are dispatched in deterministic order and joblib's
    'loky' backend returns results in submission order.
    """
    loc_pair_idx, loc_start, loc_end, loc_length, loc_n_shared = [], [], [], [], []
    for pair_i in pair_chunk:
        a = int(pair_a[pair_i])
        b = int(pair_b[pair_i])
        shared_vars = np.intersect1d(
            indices[indptr[a]:indptr[a + 1]],
            indices[indptr[b]:indptr[b + 1]],
            assume_unique=True,
        )
        if shared_vars.size < min_shared_variants:
            continue
        for seg_start, seg_end, length_bp, n_shared in _detect_segments_for_pair(
            positions[shared_vars], max_gap_bp, min_segment_bp, min_shared_variants
        ):
            loc_pair_idx.append(pair_i)
            loc_start.append(seg_start)
            loc_end.append(seg_end)
            loc_length.append(length_bp)
            loc_n_shared.append(n_shared)
    return loc_pair_idx, loc_start, loc_end, loc_length, loc_n_shared


def detect_pairwise_segments_direct(chrom, variants, selected_samples,
                                    max_gap_bp, min_segment_bp,
                                    min_shared_variants, n_jobs=1):
    """Detect shared rare-allele segments for every sample pair on one
    chromosome.

    Algorithm
    ---------
    1. Build sparse carrier matrix ``C`` of shape ``(N_samples, V)`` where
       ``C[s, v] == 1`` iff sample ``s`` carries variant ``v``.
    2. Compute ``S = C @ C.T`` (N×N sparse) in a single scipy.sparse matmul
       to obtain per-pair shared-variant counts; only pairs with count
       ``>= min_shared_variants`` can produce a segment.
    3. For each qualifying pair intersect their carrier variant indices via
       ``np.intersect1d`` on the sorted CSR rows and run vectorised segment
       detection on the resulting position array.

    Per-pair segment detection keeps peak memory bounded to one pair's
    worth of shared positions (kilobytes) rather than the concatenation of
    all pair-variant incidences (which on dense cohorts like chr18 with
    N=2000 reached ~15 GB and triggered SLURM OOM kills under 32 GB).  The
    inner numpy work remains fully vectorised, so the ~2M iteration loop
    finishes in minutes even for the worst-case chromosomes.

    The output schema (``PAIRWISE_SEGMENT_COLUMNS``) is unchanged.  The
    ``segment_id`` column is a monotonically increasing counter; its exact
    ordering carries no downstream meaning.
    """
    if not variants:
        return pd.DataFrame(columns=PAIRWISE_SEGMENT_COLUMNS)

    n_samples = len(selected_samples)
    _log(f"chr{chrom}: detecting shared segments for {n_samples} samples ...")

    C, positions = _build_carrier_matrix(variants, n_samples)
    if C.nnz == 0:
        _log(f"chr{chrom}: no carrier incidences; nothing to segment")
        return pd.DataFrame(columns=PAIRWISE_SEGMENT_COLUMNS)

    # One sparse matmul gives per-pair shared-variant counts.  We extract
    # the qualifying upper-triangular pairs and immediately drop the N×N
    # intermediate products to keep peak memory bounded before the loop.
    S = (C @ C.T).tocsr()
    S_triu = sp.triu(S, k=1).tocoo()
    qualifying = S_triu.data >= min_shared_variants
    pair_a = S_triu.row[qualifying].astype(np.int32)
    pair_b = S_triu.col[qualifying].astype(np.int32)
    n_pairs_any = int(S_triu.data.size)
    n_pairs = int(pair_a.size)
    del S, S_triu, qualifying

    _log(
        f"chr{chrom}: {n_pairs_any} pairs share >=1 rare variant, "
        f"{n_pairs} meet min_shared_variants={min_shared_variants}"
    )

    if n_pairs == 0:
        return pd.DataFrame(columns=PAIRWISE_SEGMENT_COLUMNS)

    # CSR internals for O(1) per-sample row slicing; ``C.sort_indices()``
    # during construction guarantees ``indices`` is ascending per row,
    # which in turn makes ``intersect1d`` output ascending (so shared
    # positions are already sorted by genome order).
    indptr = C.indptr
    indices = C.indices

    # Column-oriented output buffers.  Using parallel Python lists and
    # converting to numpy once at the end is both faster and more memory
    # efficient than building a list of dicts and letting pandas infer
    # per-row types.
    out_pair_idx = []
    out_start = []
    out_end = []
    out_length = []
    out_n_shared = []

    # Serial path preserves byte-identical behaviour to pre-joblib revisions
    # and avoids worker-startup overhead when the user runs the script outside
    # the pipeline with the default --n-jobs 1.  The parallel path slices the
    # qualifying pairs into ``4 * n_jobs`` chunks so a slow chunk does not
    # starve the rest (each worker grabs the next chunk as soon as it finishes).
    if n_jobs <= 1:
        chunks = [range(n_pairs)]
        results = [_segments_for_pair_batch(
            chunks[0], pair_a, pair_b, indptr, indices, positions,
            max_gap_bp, min_segment_bp, min_shared_variants,
        )]
        _log(f"chr{chrom}: segment scan serial over {n_pairs} pairs")
    else:
        target_chunks = max(n_jobs * 4, n_jobs)
        chunk_size = max(1, (n_pairs + target_chunks - 1) // target_chunks)
        chunks = [
            range(i, min(i + chunk_size, n_pairs))
            for i in range(0, n_pairs, chunk_size)
        ]
        _log(
            f"chr{chrom}: segment scan parallel n_jobs={n_jobs} over "
            f"{n_pairs} pairs in {len(chunks)} chunks of ~{chunk_size}"
        )
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_segments_for_pair_batch)(
                chunk, pair_a, pair_b, indptr, indices, positions,
                max_gap_bp, min_segment_bp, min_shared_variants,
            )
            for chunk in chunks
        )

    # Concatenate worker outputs into the column buffers.  Each worker keeps
    # the original pair_i identity, so the final segment_id ordering matches
    # the deterministic chunk dispatch order (== serial order when chunks are
    # contiguous ranges, as constructed above).
    for loc_pair_idx, loc_start, loc_end, loc_length, loc_n_shared in results:
        out_pair_idx.extend(loc_pair_idx)
        out_start.extend(loc_start)
        out_end.extend(loc_end)
        out_length.extend(loc_length)
        out_n_shared.extend(loc_n_shared)

    n_segs = len(out_pair_idx)
    if n_segs == 0:
        _log(f"chr{chrom}: detected 0 shared segments (all below thresholds)")
        return pd.DataFrame(columns=PAIRWISE_SEGMENT_COLUMNS)

    # Batch-convert column buffers and resolve sample names in one pass.
    pair_idx = np.asarray(out_pair_idx, dtype=np.int32)
    sample_arr = np.asarray(selected_samples, dtype=object)

    segment_df = pd.DataFrame({
        "chrom": np.full(n_segs, chrom, dtype=object),
        "sample_a": sample_arr[pair_a[pair_idx]],
        "sample_b": sample_arr[pair_b[pair_idx]],
        "segment_id": np.array(
            [f"{chrom}_{i:06d}" for i in range(1, n_segs + 1)], dtype=object
        ),
        "start_pos": np.asarray(out_start, dtype=np.int64),
        "end_pos": np.asarray(out_end, dtype=np.int64),
        "length_bp": np.asarray(out_length, dtype=np.int64),
        "n_shared_variants": np.asarray(out_n_shared, dtype=np.int64),
        # Constant 1.0 by design — NOT a placeholder.
        #
        # The Jaccard index of a *segment* (a maximal run of shared rare-
        # variant positions for one sample pair) is trivially 1 by
        # construction: the segment IS the intersection of the two carrier
        # sets restricted to that run, so |A ∩ B| / |A ∪ B| = n_shared /
        # n_shared = 1.  Reporting it per row keeps the schema stable for
        # downstream consumers (M16, M16.5, audits) and lets future
        # producers that compute a different per-segment similarity
        # (e.g. weighted Jaccard over MAF) populate the same column
        # without a schema migration.
        #
        # Genuine pair-level Jaccard (intersection / union of carrier
        # variant sets across the whole window) is reported by
        # ``compute_sharing_windows`` and the per-pair aggregates.
        "jaccard": np.ones(n_segs, dtype=np.float64),
    }, columns=PAIRWISE_SEGMENT_COLUMNS)

    n_pairs_with_segments = int(np.unique(pair_idx).size)
    _log(
        f"chr{chrom}: detected {n_segs} shared segments across "
        f"{n_pairs_with_segments} pairs"
    )
    return segment_df


# ---------------------------------------------------------------------------
# Plotting: palettes & style
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Neutral-tone palette shared by all block-group plots
# ---------------------------------------------------------------------------
#
# The individual painting distinguishes three "neutral" categories that are
# not positional subgroups.  A previous revision used three similar greys
# (#d5d5d5 / #e0e0e0 / #888888) which were visually indistinguishable in
# the legend.  The colours below separate them by *hue* as well as by
# lightness, so the reader can tell them apart at a glance even in greyscale
# printouts:
#
#   UNSHARED_BLOCK_COLOR   — cool medium grey (single-individual block)
#   ISOLATED_CARRIER_COLOR — warm pale beige  (< min_block_snps consec.)
#   MORE_REGIONS_COLOR     — dark charcoal    (legend swatch "+N more")
#   BACKGROUND_COLOR       — near-white ivory (chromosome backdrop)
UNSHARED_BLOCK_COLOR   = "#b8bdc2"   # soft cool grey (single-individual)
ISOLATED_CARRIER_COLOR = "#ece3cf"   # very pale warm beige (barely visible)
MORE_REGIONS_COLOR     = "#4d4d4d"   # dark charcoal (legend swatch only)
BACKGROUND_COLOR       = "#fafafa"   # near-white ivory backdrop


def _journal_palette(name):
    """Return a short list of maximally distinct, high-contrast colours
    used for block-group colouring.  When more groups than palette entries
    exist, the caller extends with HSV-derived colours."""
    palettes = {
        "journal": [
            "#e6194b",  # red
            "#3cb44b",  # green
            "#4363d8",  # blue
            "#f58231",  # orange
            "#911eb4",  # purple
            "#42d4f4",  # cyan
            "#f032e6",  # magenta
            "#bfef45",  # lime
            "#fabed4",  # pink
            "#469990",  # teal
            "#dcbeff",  # lavender
            "#9A6324",  # brown
        ],
        "colorblind": [
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#009E73",  # bluish green
            "#CC79A7",  # reddish purple
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#F0E442",  # yellow
            "#000000",  # black
            "#882255",  # wine
            "#44AA99",  # teal
            "#AA4499",  # plum
            "#332288",  # indigo
        ],
    }
    return palettes.get(name, palettes["journal"])


def _set_plot_style(plot_config):
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": int(plot_config["dpi"]),
        "figure.dpi": 150,
        "font.family": plot_config["font_family"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
    })


def _save_figure(fig, base_path, plot_config):
    png_path = base_path.parent / f"{base_path.name}.png"
    fig.savefig(png_path, bbox_inches="tight")
    if plot_config.get("export_pdf", False):
        fig.savefig(base_path.parent / f"{base_path.name}.pdf", bbox_inches="tight")
    if plot_config.get("export_svg", False):
        fig.savefig(base_path.parent / f"{base_path.name}.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ordering individuals by shared-region count (hierarchical clustering)
# ---------------------------------------------------------------------------

def order_individuals_by_sharing(segment_df, all_samples):
    """Order individuals so that those sharing more regions are adjacent.
    Uses hierarchical clustering on a sharing-count distance matrix.

    Memory / performance note
    -------------------------
    The sharing matrix is accumulated with ``numpy.add.at`` on vectorised
    index arrays.  This replaces a previous ``segment_df.iterrows()`` loop
    which, on dense cohorts that produce tens of millions of segment rows
    (e.g. chr1 with N=1000 → ~10^8 segments), allocated ~5 KB of transient
    ``pd.Series`` per row and drove the process into the SLURM memory cap.
    ``np.add.at`` performs unbuffered accumulation, so repeated pair
    indices are summed correctly even though it is vectorised.
    """
    n = len(all_samples)
    if n <= 2 or segment_df.empty:
        return list(all_samples)

    sample_to_idx = {s: i for i, s in enumerate(all_samples)}
    sharing_matrix = np.zeros((n, n), dtype=np.float64)

    # Keep only rows whose samples are registered in ``all_samples`` (the
    # common case is that every row qualifies, so this mask is cheap).
    mask = (
        segment_df["sample_a"].isin(sample_to_idx)
        & segment_df["sample_b"].isin(sample_to_idx)
    )
    if not mask.any():
        return list(all_samples)

    a_idx = segment_df.loc[mask, "sample_a"].map(sample_to_idx).to_numpy(
        dtype=np.int64, copy=False
    )
    b_idx = segment_df.loc[mask, "sample_b"].map(sample_to_idx).to_numpy(
        dtype=np.int64, copy=False
    )
    lengths = segment_df.loc[mask, "length_bp"].to_numpy(
        dtype=np.float64, copy=False
    )

    np.add.at(sharing_matrix, (a_idx, b_idx), lengths)
    np.add.at(sharing_matrix, (b_idx, a_idx), lengths)

    # Convert similarity to distance
    max_val = sharing_matrix.max()
    if max_val > 0:
        dist_matrix = max_val - sharing_matrix
    else:
        return list(all_samples)

    np.fill_diagonal(dist_matrix, 0)
    # Make symmetric and ensure non-negative
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    dist_matrix = np.clip(dist_matrix, 0, None)

    try:
        # Greedy nearest-neighbor ordering (no scipy needed)
        remaining = set(range(n))
        # Start from the sample with smallest total distance (most sharing)
        current = int(np.argmin(dist_matrix.sum(axis=1)))
        order = [current]
        remaining.discard(current)
        while remaining:
            row = dist_matrix[current]
            # Find nearest unvisited
            best = None
            best_dist = np.inf
            for j in remaining:
                if row[j] < best_dist:
                    best_dist = row[j]
                    best = j
            order.append(best)
            remaining.discard(best)
            current = best
        return [all_samples[i] for i in order]
    except Exception:
        return list(all_samples)


# ---------------------------------------------------------------------------
# Individual painting plot  —  block-identity style
# ---------------------------------------------------------------------------

def generate_individual_painting(variants, chrom, selected_samples,
                                 all_samples_ordered,
                                 plot_config, output_path,
                                 max_pairs_legend=30,
                                 chrom_extent=None,
                                 min_block_snps=2,
                                 max_block_gap_bp=200000):
    """Multi-individual block-identity painting.

    All individuals are shown as horizontal tracks.  For each individual,
    runs of ≥ *min_block_snps* consecutive rare-variant positions where they
    carry the ALT allele form *carrier blocks*, provided the gap between
    successive carrier variants does not exceed *max_block_gap_bp* (default
    200 kb — consistent with typical LD decay in humans; Gabriel et al. 2002).

    Blocks that share ≥ 2 rare-variant positions with blocks in OTHER
    individuals are clustered together (Union-Find on shared positions),
    then sub-divided into *positional sub-groups* by genomic overlap.
    Each positional sub-group receives a distinct colour so that:

      - Same colour at the same position across rows = shared IBD proxy
      - Different positions = different colours (even within one UF cluster)
      - Shorter/longer blocks at the same position → recombination

    Blocks carried by a single individual only → light grey.
    Isolated carrier positions (< min_block_snps consecutive) → faint dots.
    """
    import colorsys

    palette_colors = _journal_palette(plot_config["palette"])
    n_samples = len(all_samples_ordered)
    chrom_label = _chrom_label(chrom)
    name_to_idx = {s: i for i, s in enumerate(selected_samples)}

    # ------------------------------------------------------------------
    # 1.  Per-individual carrier blocks
    #     Consecutive = adjacent variant index AND gap ≤ max_block_gap_bp
    # ------------------------------------------------------------------
    all_vpos = [pos for pos, _ in variants]
    n_variants = len(all_vpos)

    block_list = []
    sample_block_ids = defaultdict(list)
    sample_isolates = defaultdict(list)

    for sample_name in all_samples_ordered:
        idx = name_to_idx.get(sample_name)
        if idx is None:
            continue

        carrier_vi = [vi for vi, (_, cs) in enumerate(variants) if idx in cs]
        if not carrier_vi:
            continue

        run = [carrier_vi[0]]
        for k in range(1, len(carrier_vi)):
            prev_vi = carrier_vi[k - 1]
            curr_vi = carrier_vi[k]
            gap_ok = (curr_vi == prev_vi + 1
                      and all_vpos[curr_vi] - all_vpos[prev_vi] <= max_block_gap_bp)
            if gap_ok:
                run.append(curr_vi)
            else:
                if len(run) >= min_block_snps:
                    pl = [all_vpos[vi] for vi in run]
                    bid = len(block_list)
                    block_list.append((idx, pl[0], pl[-1],
                                       frozenset(run), pl))
                    sample_block_ids[idx].append(bid)
                else:
                    sample_isolates[idx].extend(all_vpos[vi] for vi in run)
                run = [curr_vi]
        if len(run) >= min_block_snps:
            pl = [all_vpos[vi] for vi in run]
            bid = len(block_list)
            block_list.append((idx, pl[0], pl[-1],
                               frozenset(run), pl))
            sample_block_ids[idx].append(bid)
        else:
            sample_isolates[idx].extend(all_vpos[vi] for vi in run)

    n_blocks = len(block_list)
    _log(f"chr{chrom}: {n_blocks} carrier blocks found "
         f"(min {min_block_snps} consec. SNPs, "
         f"max gap {max_block_gap_bp/1e3:.0f} kb)")

    # ------------------------------------------------------------------
    # 2.  Empty check
    # ------------------------------------------------------------------
    has_anything = n_blocks > 0 or any(
        len(v) > 0 for v in sample_isolates.values())
    if not has_anything:
        fig, ax = plt.subplots(
            figsize=(plot_config["width_inches"],
                     max(3, plot_config["height_inches"])),
            constrained_layout=True)
        ax.set_title(f"Rare allele sharing painting — {chrom_label}",
                      loc="left", pad=16, fontweight="bold")
        ax.text(0.5, 0.5, "No carrier blocks detected",
                ha="center", va="center", fontsize=13,
                transform=ax.transAxes)
        ax.set_xlabel("Genomic position (Mb)")
        ax.set_ylabel("Individual")
        _save_figure(fig, output_path, plot_config)
        return

    # ------------------------------------------------------------------
    # 3.  Windowed Union-Find + positional sub-grouping
    #     Clustering is performed independently within windows of
    #     UF_WINDOW_BP so that long-range transitive chains do not
    #     merge distant regions into a single colour.
    # ------------------------------------------------------------------
    UF_WINDOW_BP = 10_000_000  # 10 Mb — same visual resolution as region runs

    def _run_uf_on_bids(bid_subset):
        """Run Union-Find + positional sub-grouping on a subset of block ids."""
        if len(bid_subset) < 2:
            return []
        local_parent = {b: b for b in bid_subset}
        local_rank = {b: 0 for b in bid_subset}

        def _find(x):
            while local_parent[x] != x:
                local_parent[x] = local_parent[local_parent[x]]
                x = local_parent[x]
            return x

        def _union(x, y):
            rx, ry = _find(x), _find(y)
            if rx == ry:
                return
            if local_rank[rx] < local_rank[ry]:
                rx, ry = ry, rx
            local_parent[ry] = rx
            if local_rank[rx] == local_rank[ry]:
                local_rank[rx] += 1

        vi_to_bids = defaultdict(list)
        for bid in bid_subset:
            for vi in block_list[bid][3]:
                vi_to_bids[vi].append(bid)

        pair_cnt = defaultdict(int)
        for vi, bids in vi_to_bids.items():
            for i in range(len(bids)):
                for j in range(i + 1, len(bids)):
                    bi, bj = bids[i], bids[j]
                    if block_list[bi][0] != block_list[bj][0]:
                        key = (min(bi, bj), max(bi, bj))
                        pair_cnt[key] += 1

        MIN_SHARED_FOR_CLUSTER = 2
        for (bi, bj), cnt in pair_cnt.items():
            if cnt >= MIN_SHARED_FOR_CLUSTER:
                _union(bi, bj)

        groups = defaultdict(list)
        for bid in bid_subset:
            groups[_find(bid)].append(bid)

        subgroups = []
        for root, members in groups.items():
            if len(set(block_list[bid][0] for bid in members)) < 2:
                continue
            sorted_m = sorted(members, key=lambda bid: block_list[bid][1])
            cur_group = [sorted_m[0]]
            cur_end = block_list[sorted_m[0]][2]
            for bid in sorted_m[1:]:
                if block_list[bid][1] <= cur_end:
                    cur_group.append(bid)
                    cur_end = max(cur_end, block_list[bid][2])
                else:
                    if len(set(block_list[b][0] for b in cur_group)) >= 2:
                        subgroups.append(cur_group)
                    cur_group = [bid]
                    cur_end = block_list[bid][2]
            if len(set(block_list[b][0] for b in cur_group)) >= 2:
                subgroups.append(cur_group)
        return subgroups

    # Partition blocks into windows by midpoint
    all_block_starts = [block_list[bid][1] for bid in range(n_blocks)]
    if all_block_starts:
        win_origin = min(all_block_starts)
    else:
        win_origin = 0

    window_bids = defaultdict(list)
    for bid in range(n_blocks):
        mid = (block_list[bid][1] + block_list[bid][2]) // 2
        win_idx = (mid - win_origin) // UF_WINDOW_BP
        window_bids[win_idx].append(bid)

    positional_subgroups = []
    for win_idx in sorted(window_bids.keys()):
        positional_subgroups.extend(_run_uf_on_bids(window_bids[win_idx]))

    def _subgroup_span(sg):
        return (max(block_list[b][2] for b in sg)
                - min(block_list[b][1] for b in sg))

    positional_subgroups.sort(key=lambda sg: (-_subgroup_span(sg), -len(sg)))

    _log(f"chr{chrom}: {len(positional_subgroups)} positional shared regions "
         f"({n_blocks} total blocks across {n_samples} individuals)")

    # ------------------------------------------------------------------
    # 4.  Assign colour CODES (int) — one per positional sub-group
    # ------------------------------------------------------------------
    # Integer codes are the cells of the raster image rendered by imshow.
    # Layout:
    #   0                       → background (no carrier)
    #   1                       → unshared carrier block (single individual)
    #   2                       → isolated carrier position
    #   3 .. n_subgroups + 2    → positional shared sub-groups (rank 1..N)
    BG_CODE, UNSHARED_CODE, ISOLATE_CODE = 0, 1, 2
    SUBGROUP_CODE_OFFSET = 3

    n_subgroups = len(positional_subgroups)
    subgroup_colors = []
    for i in range(n_subgroups):
        if i < len(palette_colors):
            subgroup_colors.append(palette_colors[i])
        else:
            hue = (i * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.82)
            subgroup_colors.append(
                f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")

    subgroup_meta = []  # (rank, color, n_indiv, n_blocks, span_bp, members)
    block_to_code = {}
    for rank_idx, sg in enumerate(positional_subgroups):
        code = SUBGROUP_CODE_OFFSET + rank_idx
        c = subgroup_colors[rank_idx]
        n_ind = len(set(block_list[bid][0] for bid in sg))
        span_bp = _subgroup_span(sg)
        subgroup_meta.append((rank_idx + 1, c, n_ind, len(sg), span_bp, sg))
        for bid in sg:
            block_to_code[bid] = code
    for bid in range(n_blocks):
        block_to_code.setdefault(bid, UNSHARED_CODE)

    # ------------------------------------------------------------------
    # 5.  Genomic range
    # ------------------------------------------------------------------
    if (chrom_extent and chrom_extent[0] is not None
            and chrom_extent[1] is not None):
        global_min = int(chrom_extent[0])
        global_max = int(chrom_extent[1])
    else:
        global_min = all_vpos[0]
        global_max = all_vpos[-1]
    span_bp_total = max(1, global_max - global_min)
    pad_bp = max(int(span_bp_total * 0.02), 10000)
    x_min = max(0, global_min - pad_bp)
    x_max = global_max + pad_bp
    raster_span_bp = max(1, x_max - x_min)

    # ------------------------------------------------------------------
    # 6.  Rasterise to a 2D int32 grid (N_samples × N_cols)
    #
    # Rendering as a single raster via ``imshow`` scales to 10^6+ carrier
    # blocks with O(n_cols × n_samples) numpy memory and one Agg call,
    # versus O(n_blocks) matplotlib ``Rectangle`` patches which peaked at
    # ~70 GB for chr12/N=1000.  Memory footprint now:
    #   painting   : N × C × 4 bytes   (int32)   ≈ 52 MB for N=1000, C=13k
    #   RGBA canvas: unchanged, capped by --plot-max-height-inches
    # ------------------------------------------------------------------
    bp_per_col = int(plot_config.get("raster_bp_per_col", 10_000) or 10_000)
    bp_per_col = max(1, bp_per_col)
    MAX_RASTER_COLS = 40_000   # upper cap to bound file size & render time
    n_cols = min(MAX_RASTER_COLS,
                 max(500, int(np.ceil(raster_span_bp / bp_per_col))))
    effective_bp_per_col = raster_span_bp / n_cols

    sample_to_ypos = {s: i for i, s in enumerate(all_samples_ordered)}

    # Vertical expansion: each sample occupies ROW_EXPANSION raster rows,
    # but only the centre row is painted.  The top and bottom rows stay as
    # background, giving the classic "horizontal bar with gap" look that
    # makes individuals visually distinct even when N is large (the same
    # effect the previous ``barh(height=0.33)`` implementation produced).
    ROW_EXPANSION = 3
    PAINT_ROW = 1                          # centre row within each strip
    n_rows = n_samples * ROW_EXPANSION
    painting = np.full((n_rows, n_cols), BG_CODE, dtype=np.int32)

    def _row_for(sample_y):
        return sample_y * ROW_EXPANSION + PAINT_ROW

    # Paint blocks.  A block at genomic [spos, epos] maps to column range
    # [col_a, col_b] inclusive; we write the subgroup code (shared) or
    # UNSHARED_CODE (single individual) directly with a slice assignment.
    # Priority: shared subgroup code overwrites unshared code if two blocks
    # of the same sample overlap the same column (should not happen in
    # practice because carrier blocks are disjoint runs within a sample,
    # but we assign shared first then unshared second as a safe default).
    unshared_assignments = []   # (image_row, col_a, col_b)
    for sample_name in all_samples_ordered:
        sidx = name_to_idx.get(sample_name)
        if sidx is None:
            continue
        y = sample_to_ypos[sample_name]
        row = _row_for(y)

        for bid in sample_block_ids.get(sidx, []):
            _sidx, spos, epos, _vi_set, _positions_list = block_list[bid]
            col_a = int((spos - x_min) * n_cols / raster_span_bp)
            col_b = int(np.ceil((epos - x_min) * n_cols / raster_span_bp))
            col_a = max(0, min(n_cols - 1, col_a))
            col_b = max(col_a, min(n_cols - 1, col_b))
            code = block_to_code[bid]
            if code == UNSHARED_CODE:
                unshared_assignments.append((row, col_a, col_b))
            else:
                painting[row, col_a:col_b + 1] = code

    # Unshared blocks: only paint where still background (shared has priority).
    for row, col_a, col_b in unshared_assignments:
        row_slice = painting[row, col_a:col_b + 1]
        mask = (row_slice == BG_CODE)
        if mask.any():
            row_slice[mask] = UNSHARED_CODE

    # Isolated carriers: single-column marks on the centre row only,
    # never overwriting blocks.
    for sample_name in all_samples_ordered:
        sidx = name_to_idx.get(sample_name)
        if sidx is None:
            continue
        y = sample_to_ypos[sample_name]
        row = _row_for(y)
        iso_positions = sample_isolates.get(sidx, [])
        if not iso_positions:
            continue
        cols = np.asarray(iso_positions, dtype=np.int64)
        cols = ((cols - x_min) * n_cols / raster_span_bp).astype(np.int64)
        cols = np.clip(cols, 0, n_cols - 1)
        row_slice = painting[row, :]
        # vectorised "paint only if background" via boolean indexing
        valid = row_slice[cols] == BG_CODE
        if valid.any():
            row_slice[cols[valid]] = ISOLATE_CODE

    # ------------------------------------------------------------------
    # 7.  Figure + imshow
    # ------------------------------------------------------------------
    per_row_in_ideal = 0.40
    margin_in = 2.5
    fig_height_uncapped = max(4.5, per_row_in_ideal * n_samples + margin_in)
    max_height_in = float(plot_config.get("max_height_inches", 40.0) or 40.0)
    fig_height = min(fig_height_uncapped, max_height_in)
    fig_width = plot_config["width_inches"]
    if fig_height_uncapped > max_height_in:
        _log(f"chr{chrom}: capping individual-painting figure height "
             f"{fig_height_uncapped:.1f} in -> {fig_height:.1f} in "
             f"(n_samples={n_samples}, dpi={plot_config['dpi']}). "
             f"Override with --plot-max-height-inches.")
    per_row_in_eff = max(1e-3, (fig_height - margin_in) / max(1, n_samples))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height),
                            constrained_layout=True)

    # Colormap: codes 0..N_subgroups+2 → colours.
    cmap_colors = [BACKGROUND_COLOR, UNSHARED_BLOCK_COLOR,
                   ISOLATED_CARRIER_COLOR] + subgroup_colors
    cmap = ListedColormap(cmap_colors, name="rare_allele_painting_cmap")

    ax.imshow(
        painting,
        cmap=cmap,
        vmin=0,
        vmax=len(cmap_colors) - 1,
        aspect="auto",
        interpolation="nearest",
        extent=(x_min, x_max, n_samples - 0.5, -0.5),
        origin="upper",
        zorder=1,
    )
    _log(f"chr{chrom}: rasterised painting grid {n_samples} x {n_cols} "
         f"(~{effective_bp_per_col/1e3:.1f} kb/col).")

    # ------------------------------------------------------------------
    # 8.  Axes (sample labels always visible — adaptive subsampling)
    # ------------------------------------------------------------------
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(n_samples - 0.5, -0.5)   # invert_yaxis via limits

    # Always show sample IDs.  When per-row height is too small to fit a
    # label per row, sub-sample to fit ``max_labels`` evenly-spaced rows
    # so the user can still read a representative subset of VCF sample
    # names (e.g. 1001, 1050, 1100, ...).
    LABEL_PT_MIN = 4.0
    label_pt = max(LABEL_PT_MIN, min(9.0, per_row_in_eff * 72.0 * 0.85))
    label_inch = label_pt / 72.0
    available_in = max(0.1, fig_height - margin_in)
    max_labels = max(10, int(available_in / max(label_inch * 1.15, 0.05)))

    if n_samples <= max_labels:
        tick_positions = np.arange(n_samples)
        label_step = 1
    else:
        label_step = max(1, int(np.ceil(n_samples / max_labels)))
        tick_positions = np.arange(0, n_samples, label_step)
        if tick_positions[-1] != n_samples - 1:
            tick_positions = np.append(tick_positions, n_samples - 1)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels([all_samples_ordered[i] for i in tick_positions],
                       fontsize=label_pt)

    if n_samples > max_labels:
        _log(f"chr{chrom}: showing {len(tick_positions)}/{n_samples} sample "
             f"labels (every {label_step} rows) due to limited vertical space "
             f"(per-row {per_row_in_eff*72:.2f} pt).")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}"))
    ax.set_xlabel("Genomic position (Mb)")
    ax.set_ylabel("Individuals")
    ax.grid(True, axis="x", alpha=0.2, zorder=0)

    # Main title + descriptive subtitle: use fig.suptitle (main) and
    # ax.set_title (subtitle).  Leaving ``y`` unset (do NOT force va="top")
    # lets constrained_layout reserve vertical space above the axes so the
    # title never overlaps the plot area or the legend.
    fig.suptitle(f"Rare allele sharing painting — {chrom_label}",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(
        f"n_samples = {n_samples}   |   {n_subgroups} shared regions   |   "
        f"{n_blocks} carrier blocks   |   raster {n_cols} cols "
        f"(~{effective_bp_per_col/1e3:.1f} kb/col)",
        loc="left", fontsize=8, color="#555555", style="italic", pad=6,
    )

    # ------------------------------------------------------------------
    # 9.  Legend — top positional shared regions + neutral categories
    # ------------------------------------------------------------------
    MAX_LEGEND_GROUPS = 20
    legend_handles = []
    for sg_rank, sg_c, sg_nind, sg_nblk, sg_span, sg_members in \
            subgroup_meta[:MAX_LEGEND_GROUPS]:
        lengths_kb = [max(1, block_list[bid][2] - block_list[bid][1]) / 1e3
                      for bid in sg_members]
        min_kb = min(lengths_kb)
        max_kb = max(lengths_kb)
        legend_handles.append(
            mpatches.Patch(facecolor=sg_c, edgecolor="none",
                           label=f"#{sg_rank}  {sg_nind} indiv, "
                                 f"{sg_nblk} blocks "
                                 f"({min_kb:.0f} kb; {max_kb:.0f} kb)"))

    n_remaining = max(0, n_subgroups - MAX_LEGEND_GROUPS)
    if n_remaining > 0:
        legend_handles.append(
            mpatches.Patch(facecolor=MORE_REGIONS_COLOR, edgecolor="none",
                           label=f"+ {n_remaining} more shared region"
                                 f"{'s' if n_remaining != 1 else ''}"))
    legend_handles.append(
        mpatches.Patch(facecolor=UNSHARED_BLOCK_COLOR, edgecolor="none",
                       label="Carrier block (not shared)"))
    legend_handles.append(
        mpatches.Patch(facecolor=ISOLATED_CARRIER_COLOR, edgecolor="none",
                       label=f"Isolated carrier (< {min_block_snps} consec.)"))

    legend = ax.legend(
        handles=legend_handles,
        title="Shared regions",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        title_fontsize=8,
        frameon=True,
        framealpha=0.95,
        edgecolor="#d8dee3",
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=1.5,
        handleheight=1.0)
    legend.get_frame().set_linewidth(0.8)

    _save_figure(fig, output_path, plot_config)


# ---------------------------------------------------------------------------
# Block group analysis helper  (shared by all block-group plots)
# ---------------------------------------------------------------------------

def _build_block_groups(variants, selected_samples, all_samples_ordered,
                        palette_name="journal", min_block_snps=2,
                        max_block_gap_bp=200000):
    """Build carrier blocks per individual, cluster via Union-Find, then
    split into positional sub-groups (same logic as generate_individual_painting).

    Returns a dict with:
      block_list, sample_block_ids, sample_isolates, n_blocks,
      positional_subgroups (sorted list of member-lists),
      subgroup_meta (rank, color, n_indiv, n_blocks, span_bp, members),
      block_to_color, UNSHARED_BLK, ISOLATE_CLR.
    """
    import colorsys

    palette_colors = _journal_palette(palette_name)
    name_to_idx = {s: i for i, s in enumerate(selected_samples)}
    all_vpos = [pos for pos, _ in variants]

    block_list = []
    sample_block_ids = defaultdict(list)
    sample_isolates = defaultdict(list)

    for sample_name in all_samples_ordered:
        idx = name_to_idx.get(sample_name)
        if idx is None:
            continue
        carrier_vi = [vi for vi, (_, cs) in enumerate(variants) if idx in cs]
        if not carrier_vi:
            continue
        run = [carrier_vi[0]]
        for k in range(1, len(carrier_vi)):
            prev_vi = carrier_vi[k - 1]
            curr_vi = carrier_vi[k]
            gap_ok = (curr_vi == prev_vi + 1
                      and all_vpos[curr_vi] - all_vpos[prev_vi] <= max_block_gap_bp)
            if gap_ok:
                run.append(curr_vi)
            else:
                if len(run) >= min_block_snps:
                    pl = [all_vpos[vi] for vi in run]
                    bid = len(block_list)
                    block_list.append((idx, pl[0], pl[-1], frozenset(run), pl))
                    sample_block_ids[idx].append(bid)
                else:
                    sample_isolates[idx].extend(all_vpos[vi] for vi in run)
                run = [curr_vi]
        if len(run) >= min_block_snps:
            pl = [all_vpos[vi] for vi in run]
            bid = len(block_list)
            block_list.append((idx, pl[0], pl[-1], frozenset(run), pl))
            sample_block_ids[idx].append(bid)
        else:
            sample_isolates[idx].extend(all_vpos[vi] for vi in run)

    n_blocks = len(block_list)

    # Windowed Union-Find + positional sub-grouping -----------------------
    UF_WINDOW_BP = 10_000_000

    def _run_uf_on_bids(bid_subset):
        if len(bid_subset) < 2:
            return []
        local_parent = {b: b for b in bid_subset}
        local_rank = {b: 0 for b in bid_subset}

        def _find(x):
            while local_parent[x] != x:
                local_parent[x] = local_parent[local_parent[x]]
                x = local_parent[x]
            return x

        def _union(x, y):
            rx, ry = _find(x), _find(y)
            if rx == ry:
                return
            if local_rank[rx] < local_rank[ry]:
                rx, ry = ry, rx
            local_parent[ry] = rx
            if local_rank[rx] == local_rank[ry]:
                local_rank[rx] += 1

        vi_to_bids = defaultdict(list)
        for bid in bid_subset:
            for vi in block_list[bid][3]:
                vi_to_bids[vi].append(bid)

        pair_cnt = defaultdict(int)
        for vi, bids in vi_to_bids.items():
            for i in range(len(bids)):
                for j in range(i + 1, len(bids)):
                    bi, bj = bids[i], bids[j]
                    if block_list[bi][0] != block_list[bj][0]:
                        key = (min(bi, bj), max(bi, bj))
                        pair_cnt[key] += 1

        MIN_SHARED_FOR_CLUSTER = 2
        for (bi, bj), cnt in pair_cnt.items():
            if cnt >= MIN_SHARED_FOR_CLUSTER:
                _union(bi, bj)

        groups = defaultdict(list)
        for bid in bid_subset:
            groups[_find(bid)].append(bid)

        subgroups = []
        for root, members in groups.items():
            if len(set(block_list[bid][0] for bid in members)) < 2:
                continue
            sorted_m = sorted(members, key=lambda bid: block_list[bid][1])
            cur_group = [sorted_m[0]]
            cur_end = block_list[sorted_m[0]][2]
            for bid in sorted_m[1:]:
                if block_list[bid][1] <= cur_end:
                    cur_group.append(bid)
                    cur_end = max(cur_end, block_list[bid][2])
                else:
                    if len(set(block_list[b][0] for b in cur_group)) >= 2:
                        subgroups.append(cur_group)
                    cur_group = [bid]
                    cur_end = block_list[bid][2]
            if len(set(block_list[b][0] for b in cur_group)) >= 2:
                subgroups.append(cur_group)
        return subgroups

    all_block_starts = [block_list[bid][1] for bid in range(n_blocks)]
    win_origin = min(all_block_starts) if all_block_starts else 0

    window_bids = defaultdict(list)
    for bid in range(n_blocks):
        mid = (block_list[bid][1] + block_list[bid][2]) // 2
        win_idx = (mid - win_origin) // UF_WINDOW_BP
        window_bids[win_idx].append(bid)

    positional_subgroups = []
    for win_idx in sorted(window_bids.keys()):
        positional_subgroups.extend(_run_uf_on_bids(window_bids[win_idx]))

    def _span(sg):
        return (max(block_list[b][2] for b in sg)
                - min(block_list[b][1] for b in sg))

    positional_subgroups.sort(key=lambda sg: (-_span(sg), -len(sg)))

    # Colours — one per positional sub-group ------------------------------
    block_to_color = {}
    UNSHARED_BLK = "#d5d5d5"
    ISOLATE_CLR = "#e0e0e0"
    n_sg = len(positional_subgroups)
    sg_colors = []
    for i in range(n_sg):
        if i < len(palette_colors):
            sg_colors.append(palette_colors[i])
        else:
            hue = (i * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.82)
            sg_colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")

    subgroup_meta = []
    for rank, sg in enumerate(positional_subgroups):
        c = sg_colors[rank % len(sg_colors)]
        n_ind = len(set(block_list[bid][0] for bid in sg))
        span_bp = _span(sg)
        subgroup_meta.append((rank + 1, c, n_ind, len(sg), span_bp, sg))
        for bid in sg:
            block_to_color[bid] = c

    for bid in range(n_blocks):
        if bid not in block_to_color:
            block_to_color[bid] = UNSHARED_BLK

    return {
        "block_list": block_list,
        "sample_block_ids": dict(sample_block_ids),
        "sample_isolates": dict(sample_isolates),
        "n_blocks": n_blocks,
        "positional_subgroups": positional_subgroups,
        "subgroup_meta": subgroup_meta,
        "block_to_color": block_to_color,
        "UNSHARED_BLK": UNSHARED_BLK,
        "ISOLATE_CLR": ISOLATE_CLR,
    }


# ---------------------------------------------------------------------------
# Block group co-membership heatmap
# ---------------------------------------------------------------------------

def generate_block_group_heatmap(variants, chrom, selected_samples,
                                  all_samples_ordered, plot_config,
                                  output_path, min_block_snps=2):
    """Co-membership heatmap: how many block groups each pair shares.

    Each cell (i, j) = number of Union-Find clusters in which both
    individuals have at least one carrier block.  Higher values suggest
    more shared ancestral carrier segments.
    """
    bg = _build_block_groups(variants, selected_samples, all_samples_ordered,
                             plot_config["palette"], min_block_snps)
    block_list = bg["block_list"]
    subgroups = bg["positional_subgroups"]
    n = len(all_samples_ordered)

    if n < 2 or not subgroups:
        return

    sample_to_disp = {s: i for i, s in enumerate(all_samples_ordered)}
    sel_to_disp = {}
    for s_idx, s_name in enumerate(selected_samples):
        if s_name in sample_to_disp:
            sel_to_disp[s_idx] = sample_to_disp[s_name]

    co_membership = np.zeros((n, n), dtype=int)
    for members in subgroups:
        disp_set = set()
        for bid in members:
            d = sel_to_disp.get(block_list[bid][0])
            if d is not None:
                disp_set.add(d)
        disp_list = sorted(disp_set)
        for i in range(len(disp_list)):
            for j in range(i + 1, len(disp_list)):
                co_membership[disp_list[i], disp_list[j]] += 1
                co_membership[disp_list[j], disp_list[i]] += 1

    chrom_label = _chrom_label(chrom)
    fig_size = max(6, min(16, 0.3 * n + 3))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)

    im = ax.imshow(co_membership, cmap="YlOrRd", aspect="equal",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Shared block groups (co-membership count)", fontsize=10)

    # Adaptive tick labelling: show all VCF sample IDs when they fit,
    # otherwise pick evenly-spaced subset so the axis remains readable.
    tick_fs = max(5, min(9, 180 // max(n, 1)))
    max_ticks = max(10, int(fig_size / (tick_fs / 72.0 * 1.3)))
    if n <= max_ticks:
        tick_pos = np.arange(n)
    else:
        step = max(1, int(np.ceil(n / max_ticks)))
        tick_pos = np.arange(0, n, step)
        if tick_pos[-1] != n - 1:
            tick_pos = np.append(tick_pos, n - 1)
    tick_labels = [all_samples_ordered[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=tick_fs)
    ax.set_yticklabels(tick_labels, fontsize=tick_fs)
    ax.set_xlabel("VCF sample ID")
    ax.set_ylabel("VCF sample ID")

    # fig.suptitle = main title; ax.set_title = italic subtitle below it.
    # constrained_layout reserves space for both automatically so they
    # never overlap with the colorbar, axis ticks, or plot area.
    fig.suptitle(f"Block group co-membership — {chrom_label}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    ax.set_title(
        "Each cell = number of shared block groups between two individuals. "
        "Higher values suggest more shared ancestral carrier segments.",
        loc="left", fontsize=8, color="#555555", style="italic", pad=6,
    )

    _save_figure(fig, output_path, plot_config)


# ---------------------------------------------------------------------------
# Block group span distribution
# ---------------------------------------------------------------------------

def generate_block_group_span_distribution(variants, chrom, selected_samples,
                                            all_samples_ordered, plot_config,
                                            output_path, min_block_snps=2):
    """Per-individual block lengths within shared groups.

    Left panel : histogram of individual block lengths in shared groups.
    Right panel: for each top group, the range of per-individual block
                 lengths (spread reveals recombination-driven shortening).
    """
    bg = _build_block_groups(variants, selected_samples, all_samples_ordered,
                             plot_config["palette"], min_block_snps)
    block_list = bg["block_list"]
    sg_meta = bg["subgroup_meta"]

    if not sg_meta:
        return

    chrom_label = _chrom_label(chrom)

    shared_lengths = []
    group_data = []
    for sg_rank, sg_c, _sg_nind, _sg_nblk, _sg_span, sg_members in sg_meta:
        lengths = [block_list[bid][2] - block_list[bid][1] for bid in sg_members]
        shared_lengths.extend(lengths)
        group_data.append((sg_rank, lengths, sg_c))

    if not shared_lengths:
        return

    fig, axes = plt.subplots(1, 2, figsize=(plot_config["width_inches"], 6.0),
                              constrained_layout=True)

    # Left panel: histogram of block lengths -----------------------------
    ax = axes[0]
    log_len = np.log10(np.array(shared_lengths, dtype=float).clip(min=1))
    ax.hist(log_len, bins=30, color="#4363d8", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Block length (log\u2081\u2080 bp)")
    ax.set_ylabel("Count (individual blocks in shared groups)")
    ax.set_title("Shared block length distribution",
                 loc="left", fontweight="bold", fontsize=11, pad=6)
    ax.axvline(x=np.log10(50000), color="#e6194b", linestyle="--",
               linewidth=1, alpha=0.7, label="50 kb")
    ax.axvline(x=np.log10(500000), color="#3cb44b", linestyle="--",
               linewidth=1, alpha=0.7, label="500 kb")
    ax.axvline(x=np.log10(1e6), color="#f58231", linestyle="--",
               linewidth=1, alpha=0.7, label="1 Mb")
    ax.legend(fontsize=8, title="Reference lengths", title_fontsize=8,
              loc="upper left", frameon=True, framealpha=0.9)

    # Right panel: per-group length variability --------------------------
    ax2 = axes[1]
    MAX_SHOWN = min(20, len(group_data))
    shown = group_data[:MAX_SHOWN]
    for i, (grank, lengths, color) in enumerate(shown):
        lengths_kb = [l / 1e3 for l in lengths]
        ax2.scatter(lengths_kb, [i] * len(lengths_kb), s=25, color=color,
                    edgecolors="white", linewidths=0.3, alpha=0.8, zorder=3)
        if len(lengths_kb) >= 2:
            ax2.plot([min(lengths_kb), max(lengths_kb)], [i, i],
                     color=color, linewidth=1.5, alpha=0.5, zorder=2)

    ax2.set_yticks(range(MAX_SHOWN))
    ax2.set_yticklabels([f"Group #{gd[0]}" for gd in shown], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Block length per individual (kb)")
    ax2.set_title(
        "Length variability by group"
        "\nEach dot = one individual's block; spread = recombination-driven differences.",
        loc="left", fontweight="bold", fontsize=11, pad=6,
    )
    ax2.grid(True, axis="x", alpha=0.2)

    # Main title via suptitle so it never collides with ax titles; the
    # constrained_layout engine reserves vertical space automatically.
    fig.suptitle(f"Block group span distribution — {chrom_label}",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")

    _save_figure(fig, output_path, plot_config)


# ---------------------------------------------------------------------------
# Scan mode
# ---------------------------------------------------------------------------

def _run_region_analysis(variants_region, chrom_key, chrom_label, region_extent,
                         selected_samples, all_samples_ordered, plot_config,
                         plots_dir, args, file_tag):
    """Generate all block-identity plots for one genomic region.

    *variants_region* must already be filtered to the region.
    *region_extent*   = (start_bp, end_bp) used for x-axis limits.
    *file_tag*        = string used in output filenames (e.g. "chr22" or
                        "chr22_16.0-26.0Mb").
    """
    if args.plot_mode in ("individual", "both"):
        _log(f"Generating block-identity painting for {file_tag} ...")
        generate_individual_painting(
            variants_region, chrom_key, selected_samples,
            all_samples_ordered, plot_config,
            plots_dir / f"{file_tag}.individual_painting",
            max_pairs_legend=args.plot_max_pairs_legend,
            chrom_extent=region_extent,
            min_block_snps=args.min_block_snps,
            max_block_gap_bp=args.max_block_gap_bp,
        )

    _log(f"Generating block group co-membership heatmap for {file_tag} ...")
    generate_block_group_heatmap(
        variants_region, chrom_key, selected_samples,
        all_samples_ordered, plot_config,
        plots_dir / f"{file_tag}.block_group_heatmap",
        min_block_snps=args.min_block_snps,
    )

    _log(f"Generating block group span distribution for {file_tag} ...")
    generate_block_group_span_distribution(
        variants_region, chrom_key, selected_samples,
        all_samples_ordered, plot_config,
        plots_dir / f"{file_tag}.block_group_span_distribution",
        min_block_snps=args.min_block_snps,
    )


def scan_mode(args):
    """Analiza un cromosoma y publica segmentos, resúmenes y figuras."""
    if not args.input or not args.chr:
        _fail("scan mode requires --input and --chr")

    validate_input_schema(args.input, args.input_format)
    header_samples = _read_header_samples(args.input)
    if args.carrier_allele_mode == "minor_allele":
        if not args.canonical_summary:
            _fail("minor_allele mode requires --canonical-summary")
        if args.sample_ids_file or args.max_samples is not None:
            _fail(
                "minor_allele mode is fail-closed: do not combine --canonical-summary "
                "with --sample-ids-file or --max-samples"
            )
        selected_samples = load_and_validate_canonical_summary(
            args.canonical_summary, args.chr, args
        )
        header_set = set(header_samples)
        missing = [sample for sample in selected_samples if sample not in header_set]
        if missing:
            _fail(
                f"Canonical summary contains {len(missing)} samples absent from VCF; "
                f"first: {missing[:10]}"
            )
    else:
        selected_samples = load_selected_samples(
            header_samples, args.sample_ids_file, args.max_samples
        )
    _log(f"Selected {len(selected_samples)} samples for analysis")

    (
        observed_chrom,
        variants,
        total_variants,
        chrom_min_pos,
        chrom_max_pos,
        orientation_qc,
    ) = parse_genotypes_carrier_sets(
        args.input,
        args.chr,
        selected_samples,
        carrier_allele_mode=args.carrier_allele_mode,
        return_orientation_qc=True,
    )
    chrom_key = _chrom_key(observed_chrom or args.chr)
    chrom_label = _chrom_label(observed_chrom or args.chr)
    chrom_extent = (chrom_min_pos, chrom_max_pos)

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_config = {
        "dpi": int(args.plot_dpi),
        "width_inches": float(args.plot_width_inches),
        "height_inches": float(args.plot_height_inches),
        "max_height_inches": float(args.plot_max_height_inches),
        "palette": args.plot_palette,
        "font_family": args.plot_font_family,
        "export_pdf": bool(args.plot_export_pdf),
        "export_svg": bool(args.plot_export_svg),
        "raster_bp_per_col": int(args.plot_raster_bp_per_col),
    }
    _set_plot_style(plot_config)

    # ------------------------------------------------------------------
    # Full-chromosome detection (always, for TSV outputs + summary)
    # ------------------------------------------------------------------
    segment_df = detect_pairwise_segments_direct(
        chrom_key, variants, selected_samples,
        args.max_gap_bp, args.min_segment_bp,
        args.min_shared_variants,
        n_jobs=args.n_jobs,
    )

    window_df = compute_sharing_windows(
        chrom_key, variants, selected_samples,
        args.window_size_bp, args.step_size_bp,
        args.min_shared_variants, args.min_jaccard,
    )

    all_samples_ordered = order_individuals_by_sharing(segment_df, selected_samples)

    # Write TSV outputs (full chromosome)
    if args.out_sharing_windows:
        out_path = Path(args.out_sharing_windows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        window_df.to_csv(out_path, sep="\t", index=False, compression="gzip")

    if args.out_pairwise_segments:
        out_path = Path(args.out_pairwise_segments)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        segment_df.to_csv(out_path, sep="\t", index=False, compression="gzip")

    # ------------------------------------------------------------------
    # Determine regions to paint
    # ------------------------------------------------------------------
    regions = []   # list of (start_bp, end_bp, file_tag)

    if args.region:
        r_start, r_end = _parse_region(args.region)
        tag = (f"{chrom_label}_{r_start/1e6:.1f}-{r_end/1e6:.1f}Mb")
        regions.append((r_start, r_end, tag))
        _log(f"Region mode: {tag}")

    elif args.region_size_bp:
        win = int(args.region_size_bp)
        lo = chrom_min_pos if chrom_min_pos is not None else 0
        hi = chrom_max_pos if chrom_max_pos is not None else lo + win
        for w_start in range(lo, hi, win):
            w_end = min(w_start + win, hi)
            if w_end <= w_start:
                continue
            tag = (f"{chrom_label}_{w_start/1e6:.1f}-{w_end/1e6:.1f}Mb")
            regions.append((w_start, w_end, tag))
        _log(f"Region-size mode: {len(regions)} windows of "
             f"{win/1e6:.1f} Mb across {chrom_label}")

    if args.skip_plots:
        _log(f"{chrom_label}: --skip-plots=true, skipping all plot generation "
             f"(TSV and JSON outputs still produced).")
    elif regions:
        for r_start, r_end, tag in regions:
            vr = _filter_variants_by_region(variants, r_start, r_end)
            _log(f"  {tag}: {len(vr)} variants in region")
            _run_region_analysis(
                vr, chrom_key, chrom_label, (r_start, r_end),
                selected_samples, all_samples_ordered, plot_config,
                plots_dir, args, tag,
            )
    else:
        # Full-chromosome painting (default)
        _run_region_analysis(
            variants, chrom_key, chrom_label, chrom_extent,
            selected_samples, all_samples_ordered, plot_config,
            plots_dir, args, chrom_label,
        )

    # ------------------------------------------------------------------
    # Summary JSON (always full chromosome)
    # ------------------------------------------------------------------
    n_sharing_pairs = int(segment_df.groupby(["sample_a", "sample_b"]).ngroups) if not segment_df.empty else 0
    total_shared_bp = int(segment_df["length_bp"].sum()) if not segment_df.empty else 0

    summary = {
        "chrom": str(chrom_key),
        "input_file": str(Path(args.input).resolve()),
        "total_variants_in_input": int(total_variants),
        "n_shared_carrier_variants": len(variants),
        "n_samples": len(selected_samples),
        "carrier_allele_mode": args.carrier_allele_mode,
        "orientation_universe": "selected_samples",
        "orientation_qc": orientation_qc,
        "selected_samples": selected_samples,
        "ordered_samples": all_samples_ordered,
        "chrom_extent": [chrom_min_pos, chrom_max_pos],
        "n_windows": int(window_df.shape[0]),
        "n_sharing_pairs": n_sharing_pairs,
        "n_segments": int(segment_df.shape[0]),
        "total_shared_bp": total_shared_bp,
        "status": "analyzed" if not segment_df.empty else "no_sharing_detected",
        "parameters_used": {
            "input_format": args.input_format,
            "window_size_bp": int(args.window_size_bp),
            "step_size_bp": int(args.step_size_bp),
            "min_shared_variants": int(args.min_shared_variants),
            "min_jaccard": float(args.min_jaccard),
            "max_gap_bp": int(args.max_gap_bp),
            "min_segment_bp": int(args.min_segment_bp),
            "plot_dpi": int(args.plot_dpi),
            "plot_palette": args.plot_palette,
            "plot_mode": args.plot_mode,
        },
        "analysis_date": datetime.now(timezone.utc).isoformat(),
    }
    if args.region:
        summary["region"] = args.region
    if args.region_size_bp:
        summary["region_size_bp"] = int(args.region_size_bp)

    if args.out_summary_json:
        _write_json(args.out_summary_json, summary)

    scan_summary_row = {
        "chrom": str(chrom_key),
        "n_samples": len(selected_samples),
        "n_rare_variants": len(variants),
        "n_windows": int(window_df.shape[0]),
        "n_sharing_pairs": n_sharing_pairs,
        "n_segments": int(segment_df.shape[0]),
        "total_shared_bp": total_shared_bp,
        "status": summary["status"],
    }

    scan_summary_path = output_dir / f"{chrom_label}.scan_summary.tsv"
    pd.DataFrame([scan_summary_row], columns=SCAN_SUMMARY_COLUMNS).to_csv(
        scan_summary_path, sep="\t", index=False,
    )
    _log(f"Scan complete for {chrom_label}: {len(segment_df)} segments, {n_sharing_pairs} pairs, {total_shared_bp/1e6:.2f} Mb shared")


# ---------------------------------------------------------------------------
# Aggregate mode
# ---------------------------------------------------------------------------

def _add_individual_partner(ind_acc, sample, partner, n_seg, bp, chr_bit):
    """Update the per-individual accumulator with one pair contribution.

    Layout: ``ind_acc[sample] = [partners_set, n_segments, total_bp,
    chr_bitmask]``.  The chromosome set is stored as a Python ``int``
    bitmask so that ``n_chromosomes_with_sharing`` is a popcount rather
    than an ``O(S)`` string-set containment check.
    """
    entry = ind_acc.get(sample)
    if entry is None:
        ind_acc[sample] = [{partner}, n_seg, bp, chr_bit]
    else:
        entry[0].add(partner)
        entry[1] += n_seg
        entry[2] += bp
        entry[3] |= chr_bit


def _accumulate_segment_chunk(chunk, pair_acc, ind_acc, chrom_to_bit):
    """Fold one chunk of segment rows into ``pair_acc`` and ``ind_acc``.

    Layout: ``pair_acc[(sample_a, sample_b)] = [n_segments, total_bp,
    sum_jaccard, chr_bitmask, n_shared_variants_total, max_segment_bp]``.
    A chromosome bitmask (Python int, grown by ``chrom_to_bit``) replaces
    ``set[str]`` so pair / individual residency stays constant in
    chromosome count and downstream ``n_chromosomes`` is a popcount.
    ``n_shared_variants_total`` (sum of carried rare variants across the
    pair's segments) and ``max_segment_bp`` (longest single segment, a
    recency/kinship proxy whose length tracks the age of the most recent
    common ancestor) are accumulated so the lightweight
    ``pair_sharing_summary.tsv`` carries enough density signal to apply a
    refined IBS cutoff (total_shared_Mb / n_shared_variants / max_segment)
    without re-streaming ``all_pairwise_segments.tsv.gz``.
    """
    # Per-chromosome inputs yield a single group; mixed inputs are still
    # handled correctly because we aggregate per (chrom, sample_a,
    # sample_b) before touching the accumulators.
    for chrom_str, sub in chunk.groupby("chrom", sort=False):
        c = str(chrom_str)
        if c not in chrom_to_bit:
            chrom_to_bit[c] = len(chrom_to_bit)
        chr_bit = 1 << chrom_to_bit[c]
        pair_agg = sub.groupby(["sample_a", "sample_b"], sort=False).agg(
            n_seg=("length_bp", "size"),
            bp=("length_bp", "sum"),
            jac=("jaccard", "sum"),
            nsv=("n_shared_variants", "sum"),
            max_seg=("length_bp", "max"),
        ).reset_index()
        for row in pair_agg.itertuples(index=False):
            sa = str(row.sample_a)
            sb = str(row.sample_b)
            key = (sa, sb)
            n_seg = int(row.n_seg)
            bp = int(row.bp)
            jac = float(row.jac)
            nsv = int(row.nsv)
            max_seg = int(row.max_seg)
            entry = pair_acc.get(key)
            if entry is None:
                pair_acc[key] = [n_seg, bp, jac, chr_bit, nsv, max_seg]
            else:
                entry[0] += n_seg
                entry[1] += bp
                entry[2] += jac
                entry[3] |= chr_bit
                entry[4] += nsv
                if max_seg > entry[5]:
                    entry[5] = max_seg
            _add_individual_partner(ind_acc, sa, sb, n_seg, bp, chr_bit)
            _add_individual_partner(ind_acc, sb, sa, n_seg, bp, chr_bit)


def aggregate_mode(args):
    """Streaming aggregator over per-chromosome pairwise-segment TSVs.

    Memory / performance note
    -------------------------
    The previous implementation loaded every input TSV into pandas and
    concatenated them before running two groupby passes on the combined
    DataFrame (including a full *duplicating* long-form concat for the
    per-individual summary).  For N ~= 2500 individuals across 24
    chromosomes this reached ~70 GB uncompressed / ~200-400 GB pandas
    resident and was consistently OOM-killed under any realistic SLURM
    memory cap.

    This streaming implementation reads each per-chromosome TSV in
    chunks (default 1 M rows ~= 100 MB pandas), incrementally updates
    two accumulator dicts (pair and individual), and re-streams every
    chunk into the merged ``all_pairwise_segments.tsv.gz`` via a gzip
    text handle.  Peak RAM is dominated by the accumulator dicts,
    approximately::

        pair_acc ~ |pairs| * ~230 B        -> ~830 MB at N=2723, fully dense
        ind_acc  ~ N * (|partners|+scalars) -> ~250 MB at N=2723, fully dense

    giving a total well under 2 GB for N ~= 2500, versus 200+ GB before.
    The global JSON and individual summary schemas are unchanged;
    ``pair_sharing_summary.tsv`` gains two appended columns
    (``n_shared_variants_total``, ``max_segment_bp``) that downstream
    consumers select by name (the HTML report and Module 16.5 both
    column-select, so neither breaks), enabling a refined IBS density
    cutoff on the lightweight pair file without re-streaming the segments.
    """
    if not args.pairwise_segments or not args.per_chr_summary:
        _fail("aggregate mode requires --pairwise-segments and --per-chr-summary files.")

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_config = {
        "dpi": int(args.plot_dpi),
        "width_inches": float(args.plot_width_inches),
        "height_inches": float(args.plot_height_inches),
        "max_height_inches": float(args.plot_max_height_inches),
        "palette": args.plot_palette,
        "font_family": args.plot_font_family,
        "export_pdf": bool(args.plot_export_pdf),
        "export_svg": bool(args.plot_export_svg),
        "raster_bp_per_col": int(args.plot_raster_bp_per_col),
    }
    _set_plot_style(plot_config)

    # ---- Load per-chromosome JSON summaries (tiny) ----------------------
    chr_summaries = []
    for path in args.per_chr_summary:
        p = Path(path)
        if not p.exists():
            _fail(f"Missing per-chr summary: {p}")
        with open(p, "r", encoding="utf-8") as handle:
            chr_summaries.append(json.load(handle))
    chr_summaries.sort(key=lambda s: _natural_chr_key(s["chrom"]))
    carrier_modes = {
        summary.get("carrier_allele_mode", "historical_alt")
        for summary in chr_summaries
    }
    if len(carrier_modes) != 1:
        _fail(f"Per-chromosome summaries mix carrier allele modes: {sorted(carrier_modes)}")
    sample_counts = {int(summary["n_samples"]) for summary in chr_summaries}
    if len(sample_counts) != 1:
        _fail(f"Per-chromosome summaries mix cohort sizes: {sorted(sample_counts)}")

    # ---- Streaming pass over every per-chr pairwise-segments TSV --------
    pair_acc: dict = {}
    ind_acc: dict = {}
    chrom_to_bit: dict = {}
    total_segments = 0
    total_shared_bp = 0

    # Typed read cuts memory roughly in half versus the default ``object``
    # dtype for sample-ID columns and silences the mixed-dtype warning that
    # ``pd.read_csv`` emits on multi-chunk streaming.
    dtype_map = {
        "chrom": "string",
        "sample_a": "string",
        "sample_b": "string",
        "segment_id": "string",
        "start_pos": "Int64",
        "end_pos": "Int64",
        "length_bp": "Int64",
        "n_shared_variants": "Int64",
        "jaccard": "float64",
    }

    merged_path = output_dir / "all_pairwise_segments.tsv.gz"
    chunk_size = int(args.aggregate_chunk_rows)

    with gzip.open(merged_path, "wt", compresslevel=6, encoding="utf-8") as out_handle:
        header_written = False
        for path in args.pairwise_segments:
            p = Path(path)
            if not p.exists():
                _fail(f"Missing pairwise segments file: {p}")
            reader = pd.read_csv(
                p, sep="\t", compression="gzip",
                chunksize=chunk_size, dtype=dtype_map,
                low_memory=False,
            )
            for chunk in reader:
                if chunk.empty:
                    continue
                chunk.to_csv(
                    out_handle, sep="\t", index=False,
                    header=(not header_written),
                )
                header_written = True
                _accumulate_segment_chunk(chunk, pair_acc, ind_acc, chrom_to_bit)
                total_segments += int(len(chunk))
                total_shared_bp += int(chunk["length_bp"].fillna(0).sum())

    # ---- Chromosome summary (derived from per-chr JSONs, unchanged) -----
    chrom_summary_rows = []
    for summary in chr_summaries:
        chrom_summary_rows.append({
            "chrom": summary["chrom"],
            "n_samples": summary["n_samples"],
            "n_rare_variants": summary["n_shared_carrier_variants"],
            "n_windows": summary["n_windows"],
            "n_sharing_pairs": summary["n_sharing_pairs"],
            "n_segments": summary["n_segments"],
            "total_shared_bp": summary["total_shared_bp"],
            "status": summary["status"],
        })
    chrom_df = pd.DataFrame(chrom_summary_rows, columns=SCAN_SUMMARY_COLUMNS)
    chrom_df.to_csv(output_dir / "chromosome_sharing_summary.tsv",
                     sep="\t", index=False)

    # ---- Pair summary (emit sorted by descending total_shared_bp) -------
    pair_rows = []
    for (sa, sb), (n_seg, bp, jac_sum, chr_bit, nsv, max_seg) in pair_acc.items():
        mean_jac = (jac_sum / n_seg) if n_seg > 0 else 0.0
        pair_rows.append((sa, sb, n_seg, bp, mean_jac,
                          bin(chr_bit).count("1"), nsv, max_seg))
    pair_df = pd.DataFrame(
        pair_rows,
        columns=["sample_a", "sample_b", "n_segments",
                 "total_shared_bp", "mean_jaccard", "n_chromosomes",
                 "n_shared_variants_total", "max_segment_bp"],
    )
    if not pair_df.empty:
        pair_df = pair_df.sort_values("total_shared_bp", ascending=False)
        pair_df.to_csv(output_dir / "pair_sharing_summary.tsv",
                       sep="\t", index=False)

    # ---- Sample ordering (hierarchical greedy) --------------------------
    # ``order_individuals_by_sharing`` only consults (sample_a, sample_b,
    # length_bp) and sums within pairs internally, so feeding it one
    # synthetic row per pair with length_bp = total_shared_bp is
    # mathematically equivalent while using O(|pairs|) memory instead of
    # O(|segments|).
    all_samples_universe = set(ind_acc.keys())
    for summary in chr_summaries:
        all_samples_universe.update(summary.get("ordered_samples", []))

    if pair_acc:
        synthetic = pd.DataFrame({
            "sample_a": [k[0] for k in pair_acc.keys()],
            "sample_b": [k[1] for k in pair_acc.keys()],
            "length_bp": [v[1] for v in pair_acc.values()],
        })
    else:
        synthetic = pd.DataFrame(columns=["sample_a", "sample_b", "length_bp"])
    all_samples_ordered = order_individuals_by_sharing(
        synthetic, sorted(all_samples_universe)
    )

    # ---- Individual summary (written in ordered-samples order) ----------
    ind_rows = []
    for sample in all_samples_ordered:
        entry = ind_acc.get(sample)
        if entry is None:
            continue
        partners, n_seg_ind, bp_ind, bit_ind = entry
        ind_rows.append((
            sample,
            len(partners),
            int(n_seg_ind),
            int(bp_ind),
            bin(bit_ind).count("1"),
        ))
    if ind_rows:
        ind_df = pd.DataFrame(
            ind_rows,
            columns=["sample_id", "n_sharing_partners",
                     "n_segments_involved", "total_shared_bp",
                     "n_chromosomes_with_sharing"],
        )
        ind_df.to_csv(output_dir / "individual_sharing_summary.tsv",
                      sep="\t", index=False)

    # ---- Global JSON summary --------------------------------------------
    global_summary = {
        "carrier_allele_mode": next(iter(carrier_modes)),
        "n_chromosomes_analyzed": len(chr_summaries),
        "chromosomes": [s["chrom"] for s in chr_summaries],
        "n_samples": len(all_samples_universe),
        "ordered_samples": all_samples_ordered,
        "total_segments": int(total_segments),
        "total_shared_bp": int(total_shared_bp),
        "total_sharing_pairs": len(pair_acc),
        "parameters_used": chr_summaries[0].get("parameters_used", {}) if chr_summaries else {},
        "input_dir": str(args.input_dir),
        "analysis_date": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "global_sharing_summary.json", global_summary)

    # ---- HTML report (uses precomputed top-20 to avoid a 10^8 groupby) --
    top_pairs_df = pair_df.head(20) if not pair_df.empty else None
    write_aggregate_report(chrom_df, top_pairs_df, all_samples_ordered,
                           global_summary, output_dir, args.plot_mode)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def write_aggregate_report(chrom_df, top_pairs_df, all_samples_ordered,
                           global_summary, output_dir, plot_mode):
    """Render the aggregate HTML report.

    The ``top_pairs_df`` argument replaces the previous ``all_segments``
    DataFrame: it is the already-sorted pair summary produced by
    ``aggregate_mode`` (columns: sample_a, sample_b, n_segments,
    total_shared_bp, mean_jaccard, n_chromosomes), truncated to the rows
    to display.  Passing the precomputed top-20 avoids a second
    ``all_segments.groupby`` pass that is O(|segments|) on 10^8+ rows.
    """
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"

    chrom_html = (
        chrom_df.to_html(index=False, classes="compact-table", border=0)
        if not chrom_df.empty
        else "<p>No chromosome summaries available.</p>"
    )

    if top_pairs_df is not None and not top_pairs_df.empty:
        top_pairs = top_pairs_df.copy()
        top_pairs["total_shared_Mb"] = (top_pairs["total_shared_bp"] / 1e6).round(3)
        top_pairs_html = top_pairs[["sample_a", "sample_b", "total_shared_Mb",
                                     "n_segments", "mean_jaccard"]].to_html(
            index=False, classes="compact-table", border=0
        )
    else:
        top_pairs_html = "<p>No sharing pairs detected.</p>"

    summary_json = json.dumps(global_summary, indent=2)

    # Image cards — block-group plots are per-chromosome only (scan mode)
    image_cards = []

    # Per-chrom painting images
    per_chr_cards = []
    for chrom in chrom_df["chrom"].astype(str).tolist() if not chrom_df.empty else []:
        safe = html.escape(str(chrom))
        cl = _chrom_label(chrom)
        if plot_mode in ("individual", "both"):
            per_chr_cards.append(f"""
    <div class="card">
      <h3>{cl} — individual painting</h3>
      <img src="plots/{cl}.individual_painting.png" alt="{cl} individual painting" />
    </div>""")

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Rare allele sharing analysis report</title>
  <style>
    body {{
      font-family: "DejaVu Sans", Arial, sans-serif;
      margin: 2rem auto;
      max-width: 1400px;
      color: #1f1f1f;
      line-height: 1.45;
      padding: 0 1.5rem 3rem;
      background: #ffffff;
    }}
    h1, h2, h3 {{
      color: #0b3c5d;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 1.2rem;
      margin: 1.2rem 0 2rem;
    }}
    .card {{
      border: 1px solid #d8dee3;
      border-radius: 10px;
      padding: 1rem 1.1rem;
      background: #fafbfd;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #d8dee3;
      border-radius: 8px;
      background: #fff;
      margin-bottom: 0.8rem;
    }}
    .compact-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.92rem;
    }}
    .compact-table th, .compact-table td {{
      border-bottom: 1px solid #e5e5e5;
      padding: 0.42rem 0.55rem;
      text-align: left;
    }}
    pre {{
      overflow-x: auto;
      padding: 1rem;
      background: #f3f6f8;
      border-radius: 8px;
      border: 1px solid #d8dee3;
      font-size: 0.88rem;
    }}
  </style>
</head>
<body>
  <h1>Rare allele sharing analysis</h1>
  <p>This module identifies carrier blocks of rare alleles shared across individuals, as a proxy for Identity by Descent (IBD). Carrier blocks are runs of consecutive rare-variant positions where an individual carries the ALT allele. Blocks from different individuals that overlap at the same variant positions are clustered together (Union-Find), revealing shared ancestral haplotype segments. Data are genotypic (not phased); results should be interpreted as exploratory.</p>

  <div class="grid">
    <div class="card">
      <h2>Operational definition</h2>
      <p><strong>Carrier:</strong> individual genotype contains at least one ALT allele at a rare variant site.</p>
      <p><strong>Carrier block:</strong> run of &ge;2 consecutive rare-variant positions where an individual is a carrier.</p>
      <p><strong>Block group:</strong> set of carrier blocks from different individuals that share &ge;2 variant positions (Union-Find clustering).</p>
    </div>
    <div class="card">
      <h2>Interpretation</h2>
      <p>Same colour at the same genomic coordinates across different individuals indicates a shared block group (potential IBD). Differences in block length between individuals reveal recombination breakpoints.</p>
      <p>The co-membership heatmap shows how many block groups each pair of individuals shares. The span distribution reveals recombination-driven length variation within groups.</p>
    </div>
  </div>

  <h2>Chromosome summary</h2>
  {chrom_html}

  <h2>Top sharing pairs</h2>
  {top_pairs_html}

  <h2>Global summary</h2>
  <pre>{html.escape(summary_json)}</pre>

  <h2>Genome-wide visualizations</h2>
  <div class="grid">
    {''.join(image_cards)}
  </div>

  <h2>Per-chromosome paintings</h2>
  <div class="grid">
    {''.join(per_chr_cards)}
  </div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Selecciona el modo solicitado y ejecuta el análisis."""
    args = parse_args()
    if args.mode == "scan":
        scan_mode(args)
    elif args.mode == "aggregate":
        aggregate_mode(args)
    else:
        _fail(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
