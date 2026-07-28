#!/usr/bin/env python3

import argparse
import gzip
import html
import json
import math
import re
import subprocess
import tempfile
from array import array
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INDIVIDUAL_SUMMARY_COLUMNS = [
    "sample_id",
    "sample_dir",
    "chrom",
    "n_carrier_snps",
    "n_pairs_raw",
    "n_pairs_used",
    "min_distance_bp",
    "mean_distance_bp",
    "median_distance_bp",
    "std_distance_bp",
    "p05_distance_bp",
    "p25_distance_bp",
    "p50_distance_bp",
    "p75_distance_bp",
    "p95_distance_bp",
    "mean_log10_distance_bp",
    "median_log10_distance_bp",
    "status",
]

CHROMOSOME_SUMMARY_COLUMNS = [
    "chrom",
    "n_selected_samples",
    "n_analyzed_samples",
    "n_insufficient_samples",
    "n_carrier_snps_total",
    "n_pairs_raw_total",
    "n_pairs_used_total",
    "min_distance_bp",
    "mean_distance_bp",
    "median_distance_bp",
    "std_distance_bp",
    "p05_distance_bp",
    "p25_distance_bp",
    "p50_distance_bp",
    "p75_distance_bp",
    "p95_distance_bp",
    "mean_log10_distance_bp",
    "median_log10_distance_bp",
    "status",
]

HISTOGRAM_COLUMNS = [
    "sample_id",
    "chrom",
    "bin_left_log10_bp",
    "bin_right_log10_bp",
    "bin_center_log10_bp",
    "count",
]


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Analyze per-individual pairwise distances between carrier rare SNPs by chromosome."
    )
    parser.add_argument("--mode", required=True, choices=["scan", "aggregate"])

    parser.add_argument("--input")
    parser.add_argument("--input-format", default="vcf_rare")
    parser.add_argument("--chr")
    parser.add_argument("--sample-ids-file")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-carrier-snps-per-individual-chr", type=int, default=2)
    parser.add_argument("--distance-units", choices=["bp"], default="bp")
    parser.add_argument(
        "--pair-selection",
        choices=["all", "within_max_bp", "nearest_neighbor_k"],
        default="all",
    )
    parser.add_argument("--max-pair-distance-bp", type=int, default=None)
    parser.add_argument("--nearest-neighbor-k", type=int, default=1)
    parser.add_argument("--pair-block-size-snps", type=int, default=2000)
    parser.add_argument("--hist-min-log10-bp", type=float, default=1.0)
    parser.add_argument("--hist-max-log10-bp", type=float, default=8.4)
    parser.add_argument("--hist-n-bins", type=int, default=74)
    parser.add_argument("--abort-if-pairs-exceed", type=int, default=None)
    parser.add_argument("--plot-hist", type=_parse_bool, default=True)
    parser.add_argument("--plot-dpi", type=int, default=600)
    parser.add_argument("--plot-width-inches", type=float, default=8.8)
    parser.add_argument("--plot-height-inches", type=float, default=5.6)
    parser.add_argument("--plot-palette", choices=["journal"], default="journal")
    parser.add_argument("--plot-font-family", default="DejaVu Sans")
    parser.add_argument("--plot-export-pdf", type=_parse_bool, default=True)
    parser.add_argument("--plot-export-svg", type=_parse_bool, default=False)
    parser.add_argument("--output-dir", default=".")

    parser.add_argument("--individual-summary", action="append", default=[])
    parser.add_argument("--cohort-summary", action="append", default=[])
    parser.add_argument("--input-dir", default=".")

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
            f"Unsupported input format '{input_format}'. This module expects upstream rare-only VCF.gz inputs."
        )

    in_path = Path(input_path)
    if not in_path.exists():
        _fail(f"Missing rare VCF input: {in_path}")

    header_lines = _read_vcf_header_lines(in_path)
    if not header_lines:
        _fail(f"Could not read VCF header from: {in_path}")

    if not any(line.startswith("##FORMAT=<ID=GT,") for line in header_lines):
        _fail(
            "Per-individual rare SNP distance analysis requires FORMAT/GT in the upstream rare VCF. "
            f"Missing GT format declaration in {in_path}"
        )

    _read_header_samples(in_path)


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
    """Selecciona muestras conservando el orden del encabezado del archivo."""
    header_set = set(header_samples)

    if sample_ids_file:
        ids = _read_sample_ids_file(sample_ids_file)
        if not ids:
            _fail(f"Sample selection file is empty or invalid: {sample_ids_file}")
        missing = [sample_id for sample_id in ids if sample_id not in header_set]
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
        _fail("No samples were selected for the per-individual distance analysis.")

    return selected


def _sanitize_sample_dirnames(sample_ids):
    counts = {}
    mapping = {}
    for sample_id in sample_ids:
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._")
        if not base:
            base = "sample"
        counts[base] = counts.get(base, 0) + 1
        mapping[sample_id] = base if counts[base] == 1 else f"{base}__{counts[base]}"
    return mapping


def _is_carrier_gt(gt_value):
    gt = str(gt_value).strip()
    if gt in {"", ".", "./.", ".|."}:
        return False
    alleles = gt.replace("|", "/").split("/")
    return "1" in alleles


def parse_genotypes_by_sample(input_path, chrom, selected_samples):
    """Extrae por muestra las posiciones de las variantes portadas."""
    positions_by_sample = {sample_id: array("I") for sample_id in selected_samples}
    query_cmd = [
        "bcftools",
        "query",
        "-f",
        r"%CHROM\t%POS\t%ALT[\t%GT]\n",
    ]

    temp_samples = None
    if selected_samples:
        temp_samples = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        temp_samples.write("\n".join(selected_samples))
        temp_samples.write("\n")
        temp_samples.close()
        query_cmd.extend(["-S", temp_samples.name])

    query_cmd.append(str(input_path))

    proc = subprocess.Popen(
        query_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.stdout is None or proc.stderr is None:
        _fail("Could not open bcftools query subprocess for per-individual distance analysis.")

    prev_pos = -1
    total_variants = 0
    observed_chrom = None
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 3 + len(selected_samples):
                _fail(
                    "Unexpected bcftools query output while reading GTs for per-individual distance analysis: "
                    f"{line[:160]}"
                )

            row_chrom, pos_s, alt_s = parts[:3]
            gts = parts[3:]
            total_variants += 1
            observed_chrom = observed_chrom or row_chrom

            if row_chrom != chrom and row_chrom != f"chr{chrom}":
                _fail(
                    f"Chromosome mismatch while reading {input_path}: expected {chrom}, observed {row_chrom}"
                )
            if "," in alt_s:
                _fail(
                    f"Found multiallelic site in upstream rare VCF at {row_chrom}:{pos_s}. "
                    "This module expects already-biallelic rare VCFs."
                )

            try:
                pos = int(pos_s)
            except ValueError as exc:
                _fail(f"Invalid POS annotation in {input_path} at {row_chrom}:{pos_s}: {exc}")

            if pos <= 0:
                _fail(f"Invalid genomic position in {input_path} at {row_chrom}:{pos_s}")
            if pos < prev_pos:
                _fail(
                    f"Input rare VCF is not sorted by position at {row_chrom}:{pos} after {prev_pos}: {input_path}"
                )
            prev_pos = pos

            for idx, gt_value in enumerate(gts):
                if _is_carrier_gt(gt_value):
                    arr = positions_by_sample[selected_samples[idx]]
                    if len(arr) == 0 or arr[-1] != pos:
                        arr.append(pos)
    finally:
        if temp_samples is not None:
            Path(temp_samples.name).unlink(missing_ok=True)

    stderr = proc.stderr.read()
    return_code = proc.wait()
    if return_code != 0:
        _fail(f"bcftools query failed for {input_path}: {stderr.strip()}")
    if total_variants == 0:
        _fail(f"Rare VCF has no variants to analyze: {input_path}")

    return observed_chrom or chrom, positions_by_sample, total_variants


def _journal_palette(_name):
    return {
        "primary": "#1b4f72",
        "primary_fill": "#7fb3d5",
        "accent": "#922b21",
        "neutral": "#4d4d4d",
        "grid": "#d9d9d9",
    }


def _set_plot_style(plot_config):
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": int(plot_config["dpi"]),
            "figure.dpi": 150,
            "font.family": plot_config["font_family"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
        }
    )


def _save_figure(fig, base_path, plot_config):
    png_path = base_path.parent / f"{base_path.name}.png"
    fig.savefig(png_path, bbox_inches="tight")
    if plot_config["export_pdf"]:
        fig.savefig(base_path.parent / f"{base_path.name}.pdf", bbox_inches="tight")
    if plot_config["export_svg"]:
        fig.savefig(base_path.parent / f"{base_path.name}.svg", bbox_inches="tight")
    plt.close(fig)


def _write_placeholder_plot(base_path, plot_config, title, message):
    fig, ax = plt.subplots(
        figsize=(plot_config["width_inches"], plot_config["height_inches"]),
        constrained_layout=True,
    )
    ax.axis("off")
    ax.text(0.5, 0.60, title, ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=11, wrap=True)
    _save_figure(fig, base_path, plot_config)


def build_histogram_edges(min_log10_bp, max_log10_bp, n_bins):
    """Construye los límites logarítmicos del histograma de distancias."""
    if not math.isfinite(min_log10_bp) or not math.isfinite(max_log10_bp):
        _fail("Histogram min/max log10 distance must be finite.")
    if max_log10_bp <= min_log10_bp:
        _fail("hist-max-log10-bp must be greater than hist-min-log10-bp.")
    if n_bins <= 0:
        _fail("hist-n-bins must be positive.")
    return np.linspace(float(min_log10_bp), float(max_log10_bp), int(n_bins) + 1, dtype=float)


def _empty_accumulator(n_pairs_raw):
    return {
        "n_pairs_raw": int(n_pairs_raw),
        "n_pairs_used": 0,
        "min_distance_bp": None,
        "sum_distance_bp": 0.0,
        "sum_sq_distance_bp": 0.0,
        "sum_log10_distance_bp": 0.0,
        "underflow_pairs": 0,
        "overflow_pairs": 0,
    }


def _validate_pair_selection_args(args):
    if args.pair_selection == "within_max_bp":
        if args.max_pair_distance_bp is None or int(args.max_pair_distance_bp) <= 0:
            _fail("pair-selection=within_max_bp requires --max-pair-distance-bp > 0")
    elif args.max_pair_distance_bp is not None and int(args.max_pair_distance_bp) <= 0:
        _fail("max-pair-distance-bp must be positive when provided.")

    if int(args.nearest_neighbor_k) <= 0:
        _fail("nearest-neighbor-k must be positive.")
    if args.pair_selection != "nearest_neighbor_k" and int(args.nearest_neighbor_k) != 1:
        _fail("nearest-neighbor-k is only meaningful when pair-selection=nearest_neighbor_k.")


def _update_histogram_from_distances(distances_bp, hist_edges, hist_counts, accumulator):
    if distances_bp.size == 0:
        return

    valid = distances_bp[distances_bp > 0]
    if valid.size == 0:
        return

    valid_f = valid.astype(np.float64, copy=False)
    log_distances = np.log10(valid_f)

    accumulator["n_pairs_used"] += int(valid.size)
    accumulator["sum_distance_bp"] += float(valid_f.sum())
    accumulator["sum_sq_distance_bp"] += float(np.square(valid_f).sum())
    accumulator["sum_log10_distance_bp"] += float(log_distances.sum())
    accumulator["underflow_pairs"] += int(np.count_nonzero(log_distances < hist_edges[0]))
    accumulator["overflow_pairs"] += int(np.count_nonzero(log_distances > hist_edges[-1]))

    current_min = int(valid.min())
    if accumulator["min_distance_bp"] is None or current_min < accumulator["min_distance_bp"]:
        accumulator["min_distance_bp"] = current_min

    clipped = np.clip(log_distances, hist_edges[0], np.nextafter(hist_edges[-1], hist_edges[0]))
    hist_counts += np.histogram(clipped, bins=hist_edges)[0].astype(np.int64)


def _update_histogram_from_matrix_distances(distance_matrix, hist_edges, hist_counts, accumulator, max_pair_distance_bp=None):
    if distance_matrix.size == 0:
        return
    values = distance_matrix.reshape(-1)
    if max_pair_distance_bp is not None:
        values = values[values <= int(max_pair_distance_bp)]
    _update_histogram_from_distances(values, hist_edges, hist_counts, accumulator)


def stream_pairwise_histogram_counts(
    positions,
    hist_edges,
    block_size_snps,
    pair_selection,
    max_pair_distance_bp=None,
    nearest_neighbor_k=1,
    abort_if_pairs_exceed=None,
):
    """Acumula distancias por bloques sin materializar todos los pares en memoria."""
    pos = np.asarray(positions, dtype=np.int64)
    n_positions = pos.size
    raw_pairs = int(n_positions * (n_positions - 1) // 2)
    hist_counts = np.zeros(hist_edges.size - 1, dtype=np.int64)
    accumulator = _empty_accumulator(raw_pairs)

    if raw_pairs == 0:
        return hist_counts, accumulator

    if abort_if_pairs_exceed is not None and raw_pairs > int(abort_if_pairs_exceed):
        _fail(
            "Per-individual pair count exceeds the configured hard limit for exact batch processing: "
            f"{raw_pairs} > {int(abort_if_pairs_exceed)}"
        )

    block_size = int(block_size_snps)
    if block_size <= 0:
        _fail("pair-block-size-snps must be positive.")

    if pair_selection == "nearest_neighbor_k":
        k = min(int(nearest_neighbor_k), max(0, n_positions - 1))
        for offset in range(1, k + 1):
            distances = pos[offset:] - pos[:-offset]
            _update_histogram_from_distances(distances, hist_edges, hist_counts, accumulator)
        return hist_counts, accumulator

    for start_a in range(0, n_positions, block_size):
        end_a = min(start_a + block_size, n_positions)
        block_a = pos[start_a:end_a]

        if block_a.size > 1:
            tri_i, tri_j = np.triu_indices(block_a.size, k=1)
            internal_distances = block_a[tri_j] - block_a[tri_i]
            if pair_selection == "within_max_bp":
                internal_distances = internal_distances[internal_distances <= int(max_pair_distance_bp)]
            _update_histogram_from_distances(internal_distances, hist_edges, hist_counts, accumulator)

        for start_b in range(end_a, n_positions, block_size):
            end_b = min(start_b + block_size, n_positions)
            block_b = pos[start_b:end_b]
            cross_distances = block_b[np.newaxis, :] - block_a[:, np.newaxis]
            if pair_selection == "within_max_bp":
                _update_histogram_from_matrix_distances(
                    cross_distances,
                    hist_edges,
                    hist_counts,
                    accumulator,
                    max_pair_distance_bp=max_pair_distance_bp,
                )
            else:
                _update_histogram_from_matrix_distances(
                    cross_distances,
                    hist_edges,
                    hist_counts,
                    accumulator,
                )

    expected_pairs = raw_pairs if pair_selection == "all" else None
    if expected_pairs is not None and accumulator["n_pairs_used"] != expected_pairs:
        _fail(
            "Internal consistency error while streaming pairwise distances: "
            f"expected {expected_pairs} pairs, accumulated {accumulator['n_pairs_used']}."
        )

    return hist_counts, accumulator


def _histogram_quantile_log10(hist_counts, hist_edges, quantile):
    total = int(np.sum(hist_counts))
    if total == 0:
        return None

    q = min(max(float(quantile), 0.0), 1.0)
    if q <= 0.0:
        idx = int(np.flatnonzero(hist_counts > 0)[0])
        return float(hist_edges[idx])
    if q >= 1.0:
        idx = int(np.flatnonzero(hist_counts > 0)[-1])
        return float(hist_edges[idx + 1])

    target = q * total
    cumulative = np.cumsum(hist_counts, dtype=np.int64)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(max(idx, 0), hist_counts.size - 1)
    prev_cum = int(cumulative[idx - 1]) if idx > 0 else 0
    bin_count = int(hist_counts[idx])
    if bin_count <= 0:
        return float(0.5 * (hist_edges[idx] + hist_edges[idx + 1]))
    frac = (target - prev_cum) / bin_count
    frac = min(max(frac, 0.0), 1.0)
    return float(hist_edges[idx] + frac * (hist_edges[idx + 1] - hist_edges[idx]))


def _bp_from_log10(log10_value):
    if log10_value is None:
        return None
    return float(10.0 ** float(log10_value))


def summarize_distance_distribution_from_histogram(
    sample_id,
    sample_dir,
    chrom,
    n_carrier_snps,
    hist_counts,
    hist_edges,
    accumulator,
    status,
):
    """Calcula conteos, cuantiles y momentos a partir de un histograma."""
    used_pairs = int(accumulator["n_pairs_used"])
    raw_pairs = int(accumulator["n_pairs_raw"])

    if n_carrier_snps < 2 or used_pairs == 0:
        return {
            "sample_id": sample_id,
            "sample_dir": sample_dir,
            "chrom": str(chrom),
            "n_carrier_snps": int(n_carrier_snps),
            "n_pairs_raw": raw_pairs,
            "n_pairs_used": used_pairs,
            "min_distance_bp": None,
            "mean_distance_bp": None,
            "median_distance_bp": None,
            "std_distance_bp": None,
            "p05_distance_bp": None,
            "p25_distance_bp": None,
            "p50_distance_bp": None,
            "p75_distance_bp": None,
            "p95_distance_bp": None,
            "mean_log10_distance_bp": None,
            "median_log10_distance_bp": None,
            "status": status,
        }

    mean_distance_bp = accumulator["sum_distance_bp"] / used_pairs
    variance_bp = max(0.0, (accumulator["sum_sq_distance_bp"] / used_pairs) - (mean_distance_bp ** 2))
    std_distance_bp = math.sqrt(variance_bp)
    mean_log10_bp = accumulator["sum_log10_distance_bp"] / used_pairs

    q05_log10 = _histogram_quantile_log10(hist_counts, hist_edges, 0.05)
    q25_log10 = _histogram_quantile_log10(hist_counts, hist_edges, 0.25)
    q50_log10 = _histogram_quantile_log10(hist_counts, hist_edges, 0.50)
    q75_log10 = _histogram_quantile_log10(hist_counts, hist_edges, 0.75)
    q95_log10 = _histogram_quantile_log10(hist_counts, hist_edges, 0.95)

    return {
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "chrom": str(chrom),
        "n_carrier_snps": int(n_carrier_snps),
        "n_pairs_raw": raw_pairs,
        "n_pairs_used": used_pairs,
        "min_distance_bp": None if accumulator["min_distance_bp"] is None else int(accumulator["min_distance_bp"]),
        "mean_distance_bp": float(mean_distance_bp),
        "median_distance_bp": _bp_from_log10(q50_log10),
        "std_distance_bp": float(std_distance_bp),
        "p05_distance_bp": _bp_from_log10(q05_log10),
        "p25_distance_bp": _bp_from_log10(q25_log10),
        "p50_distance_bp": _bp_from_log10(q50_log10),
        "p75_distance_bp": _bp_from_log10(q75_log10),
        "p95_distance_bp": _bp_from_log10(q95_log10),
        "mean_log10_distance_bp": float(mean_log10_bp),
        "median_log10_distance_bp": q50_log10,
        "status": status,
    }


def write_histogram_table(path, sample_id, chrom, hist_counts, hist_edges):
    """Escribe los intervalos y conteos del histograma de una muestra."""
    df = pd.DataFrame(
        {
            "sample_id": [sample_id] * hist_counts.size,
            "chrom": [str(chrom)] * hist_counts.size,
            "bin_left_log10_bp": hist_edges[:-1],
            "bin_right_log10_bp": hist_edges[1:],
            "bin_center_log10_bp": 0.5 * (hist_edges[:-1] + hist_edges[1:]),
            "count": hist_counts.astype(np.int64),
        },
        columns=HISTOGRAM_COLUMNS,
    )
    df.to_csv(path, sep="\t", index=False)


def generate_hist_plot(hist_counts, hist_edges, base_path, plot_config, title, note_lines=None):
    """Genera la figura del histograma con la configuración indicada."""
    palette = _journal_palette(plot_config["palette"])
    if int(np.sum(hist_counts)) == 0:
        _write_placeholder_plot(base_path, plot_config, title, "No valid distance pairs were available.")
        return

    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    widths = np.diff(hist_edges)

    fig, ax = plt.subplots(
        figsize=(plot_config["width_inches"], plot_config["height_inches"]),
        constrained_layout=True,
    )
    ax.bar(
        centers,
        hist_counts,
        width=widths * 0.96,
        color=palette["primary"],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
    )
    ax.plot(centers, hist_counts, color=palette["accent"], linewidth=1.25, marker="o", markersize=2.8)
    ax.set_title(title, loc="left", pad=16)
    ax.set_xlabel("log10 pairwise distance between carrier rare SNPs (bp)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(float(hist_edges[0]), float(hist_edges[-1]))
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    if note_lines:
        fig.text(
            0.985,
            0.985,
            "\n".join(note_lines),
            ha="right",
            va="top",
            fontsize=9,
            color=palette["neutral"],
            bbox={"facecolor": "white", "edgecolor": "#d8dee3", "boxstyle": "round,pad=0.3"},
        )
    _save_figure(fig, base_path, plot_config)


def _write_row_tsv(path, row, columns):
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(path, sep="\t", index=False)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def aggregate_histograms_by_chromosome(histograms):
    """Suma histogramas individuales por cromosoma."""
    if not histograms:
        return None
    total = np.zeros_like(histograms[0], dtype=np.int64)
    for histogram in histograms:
        total += np.asarray(histogram, dtype=np.int64)
    return total


def _cohort_summary_from_rows(chrom, selected_samples, rows, hist_counts, hist_edges, accumulator):
    analyzed_rows = [row for row in rows if row["status"] != "insufficient_snps"]
    summary_metrics = summarize_distance_distribution_from_histogram(
        sample_id="cohort",
        sample_dir="cohort",
        chrom=chrom,
        n_carrier_snps=sum(row["n_carrier_snps"] for row in rows),
        hist_counts=hist_counts,
        hist_edges=hist_edges,
        accumulator=accumulator,
        status="analyzed" if analyzed_rows else "insufficient_pairs",
    )

    return {
        "chrom": str(chrom),
        "n_selected_samples": int(len(selected_samples)),
        "n_analyzed_samples": int(len(analyzed_rows)),
        "n_insufficient_samples": int(len(rows) - len(analyzed_rows)),
        "n_carrier_snps_total": int(sum(row["n_carrier_snps"] for row in rows)),
        "n_pairs_raw_total": int(summary_metrics["n_pairs_raw"]),
        "n_pairs_used_total": int(summary_metrics["n_pairs_used"]),
        "min_distance_bp": summary_metrics["min_distance_bp"],
        "mean_distance_bp": summary_metrics["mean_distance_bp"],
        "median_distance_bp": summary_metrics["median_distance_bp"],
        "std_distance_bp": summary_metrics["std_distance_bp"],
        "p05_distance_bp": summary_metrics["p05_distance_bp"],
        "p25_distance_bp": summary_metrics["p25_distance_bp"],
        "p50_distance_bp": summary_metrics["p50_distance_bp"],
        "p75_distance_bp": summary_metrics["p75_distance_bp"],
        "p95_distance_bp": summary_metrics["p95_distance_bp"],
        "mean_log10_distance_bp": summary_metrics["mean_log10_distance_bp"],
        "median_log10_distance_bp": summary_metrics["median_log10_distance_bp"],
        "status": "analyzed" if analyzed_rows else "insufficient_pairs",
    }


def scan_mode(args):
    """Analiza un cromosoma y escribe resultados por muestra y cohorte."""
    if not args.input or not args.chr:
        _fail("scan mode requires --input and --chr")
    if args.min_carrier_snps_per_individual_chr < 2:
        _fail("min-carrier-snps-per-individual-chr must be at least 2")
    if args.distance_units != "bp":
        _fail("This implementation currently supports only bp distances.")
    _validate_pair_selection_args(args)
    if args.pair_block_size_snps <= 0:
        _fail("pair-block-size-snps must be positive.")
    if args.abort_if_pairs_exceed is not None and args.abort_if_pairs_exceed <= 0:
        _fail("abort-if-pairs-exceed must be positive when provided.")

    validate_input_schema(args.input, args.input_format)
    header_samples = _read_header_samples(args.input)
    selected_samples = load_selected_samples(header_samples, args.sample_ids_file, args.max_samples)
    sample_dir_map = _sanitize_sample_dirnames(selected_samples)

    observed_chrom, positions_by_sample, total_variants = parse_genotypes_by_sample(
        args.input,
        args.chr,
        selected_samples,
    )
    chrom_key = _chrom_key(observed_chrom or args.chr)
    chrom_label = _chrom_label(observed_chrom or args.chr)
    hist_edges = build_histogram_edges(args.hist_min_log10_bp, args.hist_max_log10_bp, args.hist_n_bins)

    output_dir = Path(args.output_dir)
    per_individual_dir = output_dir / "per_individual"
    per_chr_dir = output_dir / "per_chr"
    summary_dir = output_dir / "summary"
    per_individual_dir.mkdir(parents=True, exist_ok=True)
    per_chr_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    plot_config = {
        "dpi": int(args.plot_dpi),
        "width_inches": float(args.plot_width_inches),
        "height_inches": float(args.plot_height_inches),
        "palette": args.plot_palette,
        "font_family": args.plot_font_family,
        "export_pdf": bool(args.plot_export_pdf),
        "export_svg": bool(args.plot_export_svg),
        "plot_hist": bool(args.plot_hist),
    }
    _set_plot_style(plot_config)

    individual_rows = []
    cohort_histograms = []
    cohort_accumulator = _empty_accumulator(0)

    for sample_id in selected_samples:
        positions = positions_by_sample[sample_id]
        sample_dir = sample_dir_map[sample_id]
        sample_output_dir = per_individual_dir / sample_dir
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        hist_counts = np.zeros(hist_edges.size - 1, dtype=np.int64)

        if len(positions) < int(args.min_carrier_snps_per_individual_chr):
            accumulator = _empty_accumulator(0)
            row = summarize_distance_distribution_from_histogram(
                sample_id,
                sample_dir,
                chrom_key,
                len(positions),
                hist_counts,
                hist_edges,
                accumulator,
                "insufficient_snps",
            )
            note_lines = [f"carrier SNPs: {len(positions)}", "status: insufficient_snps"]
        else:
            hist_counts, accumulator = stream_pairwise_histogram_counts(
                positions,
                hist_edges,
                args.pair_block_size_snps,
                args.pair_selection,
                max_pair_distance_bp=args.max_pair_distance_bp,
                nearest_neighbor_k=args.nearest_neighbor_k,
                abort_if_pairs_exceed=args.abort_if_pairs_exceed,
            )
            row = summarize_distance_distribution_from_histogram(
                sample_id,
                sample_dir,
                chrom_key,
                len(positions),
                hist_counts,
                hist_edges,
                accumulator,
                "analyzed",
            )
            note_lines = [
                f"carrier SNPs: {len(positions)}",
                f"pairs used: {accumulator['n_pairs_used']:,}",
            ]
            if args.pair_selection != "all":
                note_lines.append(f"pair mode: {args.pair_selection}")
            if accumulator["underflow_pairs"] > 0 or accumulator["overflow_pairs"] > 0:
                note_lines.append(
                    f"clipped outside range: {accumulator['underflow_pairs'] + accumulator['overflow_pairs']:,}"
                )
            cohort_histograms.append(hist_counts)
            cohort_accumulator["n_pairs_raw"] += accumulator["n_pairs_raw"]
            cohort_accumulator["n_pairs_used"] += accumulator["n_pairs_used"]
            cohort_accumulator["sum_distance_bp"] += accumulator["sum_distance_bp"]
            cohort_accumulator["sum_sq_distance_bp"] += accumulator["sum_sq_distance_bp"]
            cohort_accumulator["sum_log10_distance_bp"] += accumulator["sum_log10_distance_bp"]
            cohort_accumulator["underflow_pairs"] += accumulator["underflow_pairs"]
            cohort_accumulator["overflow_pairs"] += accumulator["overflow_pairs"]
            if accumulator["min_distance_bp"] is not None:
                if cohort_accumulator["min_distance_bp"] is None or accumulator["min_distance_bp"] < cohort_accumulator["min_distance_bp"]:
                    cohort_accumulator["min_distance_bp"] = accumulator["min_distance_bp"]

        write_histogram_table(
            sample_output_dir / f"{chrom_label}.distance_hist_log10_bp.tsv",
            sample_id,
            chrom_key,
            hist_counts,
            hist_edges,
        )
        _write_row_tsv(
            sample_output_dir / f"{chrom_label}.distance_summary.tsv",
            row,
            INDIVIDUAL_SUMMARY_COLUMNS,
        )
        if plot_config["plot_hist"]:
            generate_hist_plot(
                hist_counts,
                hist_edges,
                sample_output_dir / f"{chrom_label}.distance_hist_log10_bp",
                plot_config,
                f"{sample_id} {chrom_label}: frequency",
                note_lines=note_lines,
            )

        individual_rows.append(row)

    cohort_hist_counts = aggregate_histograms_by_chromosome(cohort_histograms)
    if cohort_hist_counts is None:
        cohort_hist_counts = np.zeros(hist_edges.size - 1, dtype=np.int64)

    cohort_summary = _cohort_summary_from_rows(
        chrom_key,
        selected_samples,
        individual_rows,
        cohort_hist_counts,
        hist_edges,
        cohort_accumulator,
    )

    write_histogram_table(
        per_chr_dir / f"{chrom_label}.cohort_distance_hist_log10_bp.tsv",
        "cohort",
        chrom_key,
        cohort_hist_counts,
        hist_edges,
    )
    _write_row_tsv(
        per_chr_dir / f"{chrom_label}.cohort_distance_summary.tsv",
        cohort_summary,
        CHROMOSOME_SUMMARY_COLUMNS,
    )
    if plot_config["plot_hist"]:
        cohort_note_lines = [
            f"analyzed samples: {cohort_summary['n_analyzed_samples']}",
            f"pairs: {cohort_summary['n_pairs_used_total']:,}",
        ]
        if cohort_accumulator["underflow_pairs"] > 0 or cohort_accumulator["overflow_pairs"] > 0:
            cohort_note_lines.append(
                f"clipped outside range: {cohort_accumulator['underflow_pairs'] + cohort_accumulator['overflow_pairs']:,}"
            )
        generate_hist_plot(
            cohort_hist_counts,
            hist_edges,
            per_chr_dir / f"{chrom_label}.cohort_distance_hist_log10_bp",
            plot_config,
            f"Cohort {chrom_label}: frequency",
            note_lines=cohort_note_lines,
        )

    individual_summary_df = pd.DataFrame(individual_rows, columns=INDIVIDUAL_SUMMARY_COLUMNS)
    individual_summary_df.to_csv(
        summary_dir / f"{chrom_label}.individual_distance_summary.tsv",
        sep="\t",
        index=False,
    )

    cohort_summary_json = {
        "chrom": str(chrom_key),
        "input_file": str(Path(args.input).resolve()),
        "total_variants_in_input": int(total_variants),
        "selected_samples": selected_samples,
        "n_selected_samples": int(len(selected_samples)),
        "parameters_used": {
            "input_format": args.input_format,
            "min_carrier_snps_per_individual_chr": int(args.min_carrier_snps_per_individual_chr),
            "distance_units": args.distance_units,
            "pair_selection": args.pair_selection,
            "max_pair_distance_bp": (
                None if args.max_pair_distance_bp is None else int(args.max_pair_distance_bp)
            ),
            "nearest_neighbor_k": int(args.nearest_neighbor_k),
            "pair_block_size_snps": int(args.pair_block_size_snps),
            "hist_min_log10_bp": float(args.hist_min_log10_bp),
            "hist_max_log10_bp": float(args.hist_max_log10_bp),
            "hist_n_bins": int(args.hist_n_bins),
            "abort_if_pairs_exceed": (
                None if args.abort_if_pairs_exceed is None else int(args.abort_if_pairs_exceed)
            ),
            "plot_hist": bool(args.plot_hist),
            "plot_dpi": int(args.plot_dpi),
            "plot_width_inches": float(args.plot_width_inches),
            "plot_height_inches": float(args.plot_height_inches),
            "plot_palette": args.plot_palette,
            "plot_font_family": args.plot_font_family,
            "plot_export_pdf": bool(args.plot_export_pdf),
            "plot_export_svg": bool(args.plot_export_svg),
        },
        "cohort_summary": cohort_summary,
        "histogram_range_summary": {
            "underflow_pairs_total": int(cohort_accumulator["underflow_pairs"]),
            "overflow_pairs_total": int(cohort_accumulator["overflow_pairs"]),
        },
        "analysis_date": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(
        summary_dir / f"{chrom_label}.cohort_distance_summary.json",
        cohort_summary_json,
    )


def _merge_summary_tables(paths):
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            _fail(f"Missing summary file for aggregate mode: {p}")
        frames.append(pd.read_csv(p, sep="\t"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_cohort_jsons(paths):
    items = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            _fail(f"Missing cohort summary JSON for aggregate mode: {p}")
        with open(p, "r", encoding="utf-8") as handle:
            items.append(json.load(handle))
    if not items:
        _fail("No cohort summary JSON files were provided to aggregate mode.")
    return items


def write_report(individual_df, chromosome_df, cohort_summary, output_dir):
    """Escribe el informe HTML de distancias individuales y agregadas."""
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"

    top_individuals = (
        individual_df[individual_df["status"] != "insufficient_snps"]
        .sort_values(["n_pairs_used", "n_carrier_snps"], ascending=[False, False])
        .head(20)
    ) if not individual_df.empty else pd.DataFrame()

    chromosome_df = chromosome_df.sort_values("chrom", key=lambda col: col.map(_natural_chr_key)) if not chromosome_df.empty else chromosome_df

    top_individuals_html = (
        top_individuals.to_html(index=False, classes="compact-table", border=0)
        if not top_individuals.empty
        else "<p>No analyzable individuals were available.</p>"
    )
    chromosome_html = (
        chromosome_df.to_html(index=False, classes="compact-table", border=0)
        if not chromosome_df.empty
        else "<p>No chromosome summaries were available.</p>"
    )
    cohort_json = json.dumps(cohort_summary, indent=2)

    image_rows = []
    for chrom in chromosome_df["chrom"].astype(str).tolist() if not chromosome_df.empty else []:
        safe = html.escape(str(chrom))
        image_rows.append(
            f"""
    <div class="card">
      <h3>chr{safe}</h3>
      <img src="per_chr/chr{safe}.cohort_distance_hist_log10_bp.png" alt="chr{safe} histogram" />
    </div>
"""
        )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Per-individual rare SNP distance report</title>
  <style>
    body {{
      font-family: "DejaVu Sans", Arial, sans-serif;
      margin: 2rem auto;
      max-width: 1320px;
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
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
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
  <h1>Per-individual rare SNP distance analysis</h1>
  <p>This module summarizes the distribution of selected pairwise physical distances between carrier rare SNPs for each individual and chromosome. It is an exploratory genotype-based distance analysis and does not infer genealogy directly.</p>

  <div class="grid">
    <div class="card">
      <h2>Operational definition</h2>
      <p><strong>Carrier SNP:</strong> individual genotype contains at least one ALT allele.</p>
      <p><strong>Distance:</strong> selected pairwise physical distances in bp between carrier rare SNPs within the same chromosome.</p>
      <p><strong>Main plot:</strong> frequency vs log10(pairwise distance in bp), using fixed hg38 bins for comparability.</p>
      <p><strong>Pair selection:</strong> {html.escape(str(cohort_summary.get("parameters_used", {}).get("pair_selection", "all")))}</p>
    </div>
    <div class="card">
      <h2>Interpretation</h2>
      <p>Modes toward the left indicate repeated short-range clustering of carrier rare SNPs, while modes toward the right indicate broader spacing patterns across the same chromosome. These summaries are exploratory and can reflect demographic structure, selection, recombination heterogeneity, or technical properties of the rare-only input.</p>
      <p>The full per-individual files are available under <code>per_individual/</code>.</p>
    </div>
  </div>

  <h2>Chromosome summaries</h2>
  {chromosome_html}

  <h2>Individuals with the largest number of distance pairs</h2>
  {top_individuals_html}

  <h2>Cohort summary</h2>
  <pre>{html.escape(cohort_json)}</pre>

  <h2>Cohort histograms by chromosome</h2>
  <div class="grid">
    {''.join(image_rows)}
  </div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html_text)


def aggregate_mode(args):
    """Combina las salidas cromosómicas en un resultado genómico."""
    if not args.individual_summary or not args.cohort_summary:
        _fail("aggregate mode requires --individual-summary and --cohort-summary files.")

    output_dir = Path(args.output_dir)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    individual_df = _merge_summary_tables(args.individual_summary)
    if individual_df.empty:
        _fail("No individual summary rows were available for aggregate mode.")

    cohort_items = _load_cohort_jsons(args.cohort_summary)
    chromosome_rows = []
    underflow_total = 0
    overflow_total = 0
    for item in cohort_items:
        chromosome_rows.append(item["cohort_summary"].copy())
        range_summary = item.get("histogram_range_summary", {})
        underflow_total += int(range_summary.get("underflow_pairs_total", 0))
        overflow_total += int(range_summary.get("overflow_pairs_total", 0))

    chromosome_df = pd.DataFrame(chromosome_rows, columns=CHROMOSOME_SUMMARY_COLUMNS)
    chromosome_df = chromosome_df.sort_values("chrom", key=lambda col: col.map(_natural_chr_key)).reset_index(drop=True)

    total_selected_samples = int(individual_df["sample_id"].nunique())
    total_analyzed_rows = int((individual_df["status"] != "insufficient_snps").sum())
    total_pairs_used = int(individual_df["n_pairs_used"].fillna(0).sum())
    total_pairs_raw = int(individual_df["n_pairs_raw"].fillna(0).sum())
    carrier_counts = individual_df["n_carrier_snps"].fillna(0).to_numpy(dtype=float)
    pair_counts = individual_df.loc[individual_df["status"] != "insufficient_snps", "n_pairs_used"].fillna(0).to_numpy(dtype=float)

    carrier_metrics = {
        "mean": None if carrier_counts.size == 0 else float(np.mean(carrier_counts)),
        "median": None if carrier_counts.size == 0 else float(np.median(carrier_counts)),
        "std": None if carrier_counts.size == 0 else float(np.std(carrier_counts, ddof=0)),
        "p05": None if carrier_counts.size == 0 else float(np.percentile(carrier_counts, 5)),
        "p25": None if carrier_counts.size == 0 else float(np.percentile(carrier_counts, 25)),
        "p50": None if carrier_counts.size == 0 else float(np.percentile(carrier_counts, 50)),
        "p75": None if carrier_counts.size == 0 else float(np.percentile(carrier_counts, 75)),
        "p95": None if carrier_counts.size == 0 else float(np.percentile(carrier_counts, 95)),
    }
    pair_metrics = {
        "mean": None if pair_counts.size == 0 else float(np.mean(pair_counts)),
        "median": None if pair_counts.size == 0 else float(np.median(pair_counts)),
        "std": None if pair_counts.size == 0 else float(np.std(pair_counts, ddof=0)),
        "p05": None if pair_counts.size == 0 else float(np.percentile(pair_counts, 5)),
        "p25": None if pair_counts.size == 0 else float(np.percentile(pair_counts, 25)),
        "p50": None if pair_counts.size == 0 else float(np.percentile(pair_counts, 50)),
        "p75": None if pair_counts.size == 0 else float(np.percentile(pair_counts, 75)),
        "p95": None if pair_counts.size == 0 else float(np.percentile(pair_counts, 95)),
    }

    cohort_summary = {
        "n_selected_samples": total_selected_samples,
        "n_individual_chromosome_rows": int(individual_df.shape[0]),
        "n_analyzed_individual_chromosome_rows": total_analyzed_rows,
        "n_chromosomes_analyzed": int(chromosome_df.shape[0]),
        "n_pairs_used_total": total_pairs_used,
        "n_pairs_raw_total": total_pairs_raw,
        "carrier_snps_per_row": carrier_metrics,
        "distance_pairs_per_row": pair_metrics,
        "histogram_range_summary": {
            "underflow_pairs_total": underflow_total,
            "overflow_pairs_total": overflow_total,
        },
        "input_dir": str(args.input_dir),
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "chromosomes": chromosome_df["chrom"].astype(str).tolist(),
        "parameters_used": cohort_items[0].get("parameters_used", {}),
    }

    individual_df.to_csv(summary_dir / "individual_distance_summary.tsv", sep="\t", index=False)
    chromosome_df.to_csv(summary_dir / "chromosome_distance_summary.tsv", sep="\t", index=False)
    _write_json(summary_dir / "cohort_distance_summary.json", cohort_summary)
    write_report(individual_df, chromosome_df, cohort_summary, output_dir)


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
