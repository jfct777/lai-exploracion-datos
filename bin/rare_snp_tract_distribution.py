#!/usr/bin/env python3

import argparse
import gzip
import json
import math
import subprocess
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_INFO_FIELDS = ("AC", "AN", "AF")
WINDOW_SCORE_COLUMNS = [
    "chrom",
    "window_id",
    "start_snp_idx",
    "end_snp_idx",
    "start_pos",
    "end_pos",
    "mid_pos",
    "window_bp",
    "n_snps",
    "rare_snp_count",
    "rare_snp_density_per_mb",
]
TRACT_COLUMNS = [
    "chrom",
    "tract_id",
    "start_pos",
    "end_pos",
    "length_bp",
    "start_cm",
    "end_cm",
    "length_cm",
    "n_snps",
    "rare_snp_count",
    "rare_snp_density",
    "mean_window_score",
]


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Detect and summarize enriched tracts of rare SNPs from upstream rare-only VCFs."
    )
    parser.add_argument("--mode", required=True, choices=["scan", "aggregate"])

    parser.add_argument("--input")
    parser.add_argument("--input-format", default="vcf_rare")
    parser.add_argument("--chr")
    parser.add_argument("--maf-threshold", type=float, default=0.01)
    parser.add_argument("--window-size-snps", type=int, default=2000)
    parser.add_argument("--step-size-snps", type=int, default=500)
    parser.add_argument("--min-chrom-rare-snps", type=int, default=5000)
    parser.add_argument("--out-window-scores")
    parser.add_argument("--out-summary-json")

    parser.add_argument("--window-scores", action="append", default=[])
    parser.add_argument("--per-chr-summary", action="append", default=[])
    parser.add_argument("--metadata")
    parser.add_argument("--genetic-map")
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--enrichment-percentile", type=float, default=95.0)
    parser.add_argument("--threshold-scope", choices=["chromosome", "global"], default="chromosome")
    parser.add_argument("--max-gap-windows", type=int, default=1)
    parser.add_argument("--min-tract-snps", type=int, default=2000)
    parser.add_argument("--min-tract-bp", type=int, default=50000)
    parser.add_argument("--use-cm-if-available", type=_parse_bool, default=True)
    parser.add_argument("--plot-dpi", type=int, default=600)
    parser.add_argument("--plot-palette", choices=["journal"], default="journal")
    parser.add_argument("--plot-bins-method", choices=["fd", "sqrt"], default="fd")
    parser.add_argument("--plot-kde-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--plot-ecdf", type=_parse_bool, default=True)
    parser.add_argument("--plot-kde", type=_parse_bool, default=True)
    parser.add_argument("--plot-scatter", type=_parse_bool, default=True)
    parser.add_argument("--plot-boxplot-by-chrom", type=_parse_bool, default=True)
    parser.add_argument("--plot-violin-by-chrom", type=_parse_bool, default=True)
    parser.add_argument(
        "--plot-chrom-metric",
        choices=["log10_length_bp", "length_bp", "n_snps", "rare_snp_density"],
        default="log10_length_bp",
    )
    parser.add_argument("--plot-chrom-min-tracts", type=int, default=3)
    parser.add_argument("--plot-chrom-max", type=int, default=24)
    parser.add_argument("--plot-scatter-log-x", type=_parse_bool, default=False)
    parser.add_argument("--plot-scatter-log-y", type=_parse_bool, default=False)
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


def _fail(message):
    raise SystemExit(message)


def _write_json(path, payload):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_vcf_header_lines(input_path):
    opener = gzip.open if str(input_path).endswith(".gz") else open
    header_lines = []
    with opener(input_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header_lines.append(line.rstrip("\n"))
    return header_lines


def validate_input_schema(input_path, input_format):
    """Comprueba que el archivo de entrada tenga el formato y las columnas esperadas."""
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

    missing = []
    for field in REQUIRED_INFO_FIELDS:
        token = f"##INFO=<ID={field},"
        if not any(line.startswith(token) for line in header_lines):
            missing.append(field)

    if missing:
        _fail(
            "Rare SNP tract analysis requires upstream rare VCFs to preserve INFO/AC, INFO/AN and INFO/AF. "
            f"Missing header declaration(s) in {in_path}: {', '.join(missing)}"
        )


def compute_maf(*_args, **_kwargs):
    """Rechaza el cálculo de MAF cuando el flujo requiere una frecuencia ya definida."""
    _fail(
        "Rare SNP tract analysis does not recalculate rareza or MAF. "
        "Provide upstream rare-only VCFs with INFO/AC, INFO/AN and INFO/AF."
    )


def load_input_data(input_path, input_format, chrom, maf_threshold):
    """Carga las variantes del cromosoma y devuelve posiciones y frecuencias filtradas."""
    validate_input_schema(input_path, input_format)

    positions = array("Q")
    query_cmd = [
        "bcftools",
        "query",
        "-f",
        r"%CHROM\t%POS\t%INFO/AC\t%INFO/AN\t%INFO/AF\n",
        str(input_path),
    ]

    proc = subprocess.Popen(
        query_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.stdout is None or proc.stderr is None:
        _fail("Could not open bcftools query subprocess for rare SNP tract analysis.")

    total_variants = 0
    min_af = float("inf")
    max_af = float("-inf")
    prev_pos = -1
    observed_chrom = None

    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 5:
            _fail(f"Unexpected bcftools query output for {input_path}: {line}")

        row_chrom, pos_s, ac_s, an_s, af_s = parts
        total_variants += 1
        observed_chrom = observed_chrom or row_chrom

        if row_chrom != chrom and row_chrom != f"chr{chrom}":
            _fail(
                f"Chromosome mismatch while reading {input_path}: expected {chrom}, observed {row_chrom}"
            )

        if any(value in {"", "."} for value in (pos_s, ac_s, an_s, af_s)):
            _fail(
                "Rare SNP tract analysis requires INFO/AC, INFO/AN and INFO/AF on every variant. "
                f"Found missing annotation in {input_path} at {row_chrom}:{pos_s or 'NA'}."
            )

        if "," in ac_s or "," in af_s:
            _fail(
                f"Found multiallelic annotation in upstream rare VCF at {row_chrom}:{pos_s}. "
                "This module expects already-biallelic rare VCFs."
            )

        try:
            pos = int(pos_s)
            ac = int(ac_s)
            an = int(an_s)
            af = float(af_s)
        except ValueError as exc:
            _fail(f"Invalid numeric annotation in {input_path} at {row_chrom}:{pos_s}: {exc}")

        if pos <= 0:
            _fail(f"Invalid genomic position in {input_path} at {row_chrom}:{pos_s}")
        if an <= 0 or ac < 0 or ac > an:
            _fail(f"Invalid AC/AN annotation in {input_path} at {row_chrom}:{pos_s}: AC={ac}, AN={an}")
        if not math.isfinite(af) or af < 0.0 or af > 1.0:
            _fail(f"Invalid AF annotation in {input_path} at {row_chrom}:{pos_s}: AF={af}")
        if pos < prev_pos:
            _fail(
                f"Input rare VCF is not sorted by position at {row_chrom}:{pos} after {prev_pos}: {input_path}"
            )

        positions.append(pos)
        prev_pos = pos
        min_af = min(min_af, af)
        max_af = max(max_af, af)

    stderr = proc.stderr.read()
    return_code = proc.wait()
    if return_code != 0:
        _fail(f"bcftools query failed for {input_path}: {stderr.strip()}")

    if total_variants == 0:
        _fail(f"Rare VCF has no variants to analyze: {input_path}")

    return {
        "chrom": observed_chrom or chrom,
        "positions": np.asarray(positions, dtype=np.int64),
        "total_variants": total_variants,
        "min_info_af": float(min_af),
        "max_info_af": float(max_af),
    }


def identify_rare_snps(data, maf_threshold):
    """Selecciona las variantes cuya frecuencia es menor o igual al umbral."""
    return data["positions"]


def build_window_scores(chrom, positions, window_size_snps, step_size_snps):
    """Construye ventanas por número de SNP y calcula su densidad de variantes raras."""
    if positions.size < window_size_snps:
        return pd.DataFrame(columns=WINDOW_SCORE_COLUMNS)

    starts = np.arange(0, positions.size - window_size_snps + 1, step_size_snps, dtype=np.int64)
    ends = starts + window_size_snps - 1

    start_pos = positions[starts]
    end_pos = positions[ends]
    span_bp = end_pos - start_pos + 1
    density = window_size_snps * 1_000_000.0 / span_bp
    mid_pos = ((start_pos + end_pos) / 2.0).round().astype(np.int64)
    window_ids = np.arange(1, starts.size + 1, dtype=np.int64)

    return pd.DataFrame(
        {
            "chrom": chrom,
            "window_id": window_ids,
            "start_snp_idx": starts + 1,
            "end_snp_idx": ends + 1,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "mid_pos": mid_pos,
            "window_bp": span_bp,
            "n_snps": window_size_snps,
            "rare_snp_count": window_size_snps,
            "rare_snp_density_per_mb": density,
        }
    )


def call_enriched_windows(window_df, enrichment_percentile, threshold_scope):
    """Marca las ventanas que superan el percentil de enriquecimiento configurado."""
    if window_df.empty:
        return window_df.assign(enrichment_threshold=np.nan, is_enriched=False), {}

    scored = window_df.copy()
    thresholds = {}

    if threshold_scope == "global":
        threshold = float(np.percentile(scored["rare_snp_density_per_mb"], enrichment_percentile))
        thresholds = {chrom: threshold for chrom in scored["chrom"].astype(str).unique()}
        scored["enrichment_threshold"] = threshold
    else:
        scored["enrichment_threshold"] = np.nan
        for chrom, sub_df in scored.groupby("chrom", sort=False):
            threshold = float(np.percentile(sub_df["rare_snp_density_per_mb"], enrichment_percentile))
            thresholds[str(chrom)] = threshold
            scored.loc[sub_df.index, "enrichment_threshold"] = threshold

    scored["is_enriched"] = scored["rare_snp_density_per_mb"] >= scored["enrichment_threshold"]
    return scored, thresholds


def merge_windows_into_tracts(window_df, max_gap_windows):
    """Une ventanas enriquecidas cercanas en tractos continuos."""
    rows = []
    for chrom, sub_df in window_df.groupby("chrom", sort=False):
        enriched = sub_df[sub_df["is_enriched"]].sort_values("window_id").reset_index(drop=True)
        if enriched.empty:
            continue

        current = None
        tract_counter = 0
        for row in enriched.itertuples(index=False):
            if current is None:
                current = {
                    "chrom": chrom,
                    "start_pos": int(row.start_pos),
                    "end_pos": int(row.end_pos),
                    "start_snp_idx": int(row.start_snp_idx),
                    "end_snp_idx": int(row.end_snp_idx),
                    "window_ids": [int(row.window_id)],
                    "scores": [float(row.rare_snp_density_per_mb)],
                    "windows": 1,
                }
                continue

            gap_windows = int(row.window_id) - current["window_ids"][-1] - 1
            if gap_windows <= max_gap_windows:
                current["end_pos"] = int(row.end_pos)
                current["end_snp_idx"] = int(row.end_snp_idx)
                current["window_ids"].append(int(row.window_id))
                current["scores"].append(float(row.rare_snp_density_per_mb))
                current["windows"] += 1
            else:
                tract_counter += 1
                rows.append(_finalize_tract(current, tract_counter))
                current = {
                    "chrom": chrom,
                    "start_pos": int(row.start_pos),
                    "end_pos": int(row.end_pos),
                    "start_snp_idx": int(row.start_snp_idx),
                    "end_snp_idx": int(row.end_snp_idx),
                    "window_ids": [int(row.window_id)],
                    "scores": [float(row.rare_snp_density_per_mb)],
                    "windows": 1,
                }

        if current is not None:
            tract_counter += 1
            rows.append(_finalize_tract(current, tract_counter))

    if not rows:
        return pd.DataFrame(columns=TRACT_COLUMNS)

    tract_df = pd.DataFrame(rows)
    tract_df = tract_df[
        [
            "chrom",
            "tract_id",
            "start_pos",
            "end_pos",
            "length_bp",
            "start_cm",
            "end_cm",
            "length_cm",
            "n_snps",
            "rare_snp_count",
            "rare_snp_density",
            "mean_window_score",
        ]
    ]
    return tract_df


def _finalize_tract(state, tract_counter):
    n_snps = state["end_snp_idx"] - state["start_snp_idx"] + 1
    length_bp = state["end_pos"] - state["start_pos"] + 1
    density = n_snps * 1_000_000.0 / length_bp
    return {
        "chrom": state["chrom"],
        "tract_id": f"{state['chrom']}_{tract_counter:06d}",
        "start_pos": state["start_pos"],
        "end_pos": state["end_pos"],
        "length_bp": int(length_bp),
        "start_cm": np.nan,
        "end_cm": np.nan,
        "length_cm": np.nan,
        "n_snps": int(n_snps),
        "rare_snp_count": int(n_snps),
        "rare_snp_density": float(density),
        "mean_window_score": float(np.mean(state["scores"])),
    }


def load_genetic_map(path):
    """Carga y normaliza el mapa genético opcional."""
    if not path:
        return {}

    map_path = Path(path)
    if not map_path.exists():
        _fail(f"Requested genetic map does not exist: {map_path}")

    df = pd.read_csv(map_path, sep=None, engine="python")
    required = {"chrom", "pos", "cm"}
    missing = required - set(df.columns.astype(str))
    if missing:
        _fail(f"Genetic map is missing required columns {sorted(missing)}: {map_path}")

    df = df[["chrom", "pos", "cm"]].copy()
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df["cm"] = pd.to_numeric(df["cm"], errors="coerce")
    df = df.dropna()

    genetic_map = {}
    for chrom, sub_df in df.groupby("chrom", sort=False):
        ordered = sub_df.sort_values("pos").drop_duplicates("pos", keep="first")
        if ordered.shape[0] < 2:
            continue
        if np.any(np.diff(ordered["cm"].to_numpy()) < 0):
            _fail(f"Genetic map must be non-decreasing in cM for chromosome {chrom}: {map_path}")
        genetic_map[str(chrom)] = ordered.reset_index(drop=True)

    return genetic_map


def interpolate_cm(chrom, positions, genetic_map):
    """Interpola posiciones genéticas en centimorgans para un cromosoma."""
    chrom_key = str(chrom)
    if chrom_key not in genetic_map:
        return np.full(len(positions), np.nan, dtype=float)

    map_df = genetic_map[chrom_key]
    map_pos = map_df["pos"].to_numpy(dtype=float)
    map_cm = map_df["cm"].to_numpy(dtype=float)
    values = np.full(len(positions), np.nan, dtype=float)

    positions_arr = np.asarray(positions, dtype=float)
    in_range = (positions_arr >= map_pos[0]) & (positions_arr <= map_pos[-1])
    values[in_range] = np.interp(positions_arr[in_range], map_pos, map_cm)
    return values


def compute_tract_lengths(tract_df, genetic_map, use_cm_if_available):
    """Añade longitudes físicas y genéticas a cada tracto."""
    if tract_df.empty:
        return tract_df

    tract_df = tract_df.copy()
    if not use_cm_if_available or not genetic_map:
        tract_df["start_cm"] = np.nan
        tract_df["end_cm"] = np.nan
        tract_df["length_cm"] = np.nan
        return tract_df

    start_cm = []
    end_cm = []
    for row in tract_df.itertuples(index=False):
        cm_values = interpolate_cm(row.chrom, [row.start_pos, row.end_pos], genetic_map)
        start_val = float(cm_values[0]) if not np.isnan(cm_values[0]) else np.nan
        end_val = float(cm_values[1]) if not np.isnan(cm_values[1]) else np.nan
        start_cm.append(start_val)
        end_cm.append(end_val)

    tract_df["start_cm"] = start_cm
    tract_df["end_cm"] = end_cm
    tract_df["length_cm"] = tract_df["end_cm"] - tract_df["start_cm"]
    tract_df.loc[tract_df["length_cm"] < 0, ["start_cm", "end_cm", "length_cm"]] = np.nan
    return tract_df


def summarize_distributions(
    tract_df,
    window_df,
    per_chr_summaries,
    thresholds,
    threshold_scope,
    maf_threshold,
    input_dir,
    use_cm_if_available,
):
    """Resume por cromosoma y genoma las distribuciones de ventanas y tractos."""
    summary_by_chrom = {}
    for item in per_chr_summaries:
        summary_by_chrom[str(item["chrom"])] = {
            "status": item["status"],
            "n_rare_snps": item["n_rare_snps"],
            "n_windows": item["n_windows"],
            "threshold": thresholds.get(str(item["chrom"])),
            "n_enriched_windows": int(
                window_df[(window_df["chrom"] == item["chrom"]) & (window_df["is_enriched"])].shape[0]
            )
            if not window_df.empty
            else 0,
            "n_tracts": int(tract_df[tract_df["chrom"] == item["chrom"]].shape[0]) if not tract_df.empty else 0,
            "total_tract_bp": float(tract_df.loc[tract_df["chrom"] == item["chrom"], "length_bp"].sum())
            if not tract_df.empty
            else 0.0,
            "max_tract_bp": float(tract_df.loc[tract_df["chrom"] == item["chrom"], "length_bp"].max())
            if not tract_df.empty and (tract_df["chrom"] == item["chrom"]).any()
            else None,
        }

    bp_metrics = _distribution_metrics(tract_df["length_bp"].to_numpy(dtype=float) if not tract_df.empty else np.array([]))
    log_bp_metrics = _distribution_metrics(
        np.log10(tract_df["length_bp"].to_numpy(dtype=float)) if not tract_df.empty else np.array([])
    )

    if not tract_df.empty and tract_df["length_cm"].notna().any():
        cm_values = tract_df.loc[tract_df["length_cm"].notna(), "length_cm"].to_numpy(dtype=float)
        cm_metrics = _distribution_metrics(cm_values)
        log_cm_metrics = _distribution_metrics(np.log10(np.clip(cm_values, 1e-6, None)))
        cm_available = True
    else:
        cm_metrics = _distribution_metrics(np.array([]))
        log_cm_metrics = _distribution_metrics(np.array([]))
        cm_available = False

    summary = {
        "total_tracts": int(tract_df.shape[0]),
        "total_chromosomes_analyzed": int(sum(item["status"] == "analyzed" for item in per_chr_summaries)),
        "total_chromosomes_skipped": int(sum(item["status"] != "analyzed" for item in per_chr_summaries)),
        "mean_length_bp": bp_metrics["mean"],
        "median_length_bp": bp_metrics["median"],
        "std_length_bp": bp_metrics["std"],
        "quantiles_length_bp": bp_metrics["quantiles"],
        "mean_log10_length_bp": log_bp_metrics["mean"],
        "median_log10_length_bp": log_bp_metrics["median"],
        "std_log10_length_bp": log_bp_metrics["std"],
        "mean_length_cm": cm_metrics["mean"],
        "median_length_cm": cm_metrics["median"],
        "std_length_cm": cm_metrics["std"],
        "quantiles_length_cm": cm_metrics["quantiles"],
        "mean_log10_length_cm": log_cm_metrics["mean"],
        "median_log10_length_cm": log_cm_metrics["median"],
        "std_log10_length_cm": log_cm_metrics["std"],
        "summary_by_chromosome": dict(sorted(summary_by_chrom.items(), key=lambda item: _natural_chr_key(item[0]))),
        "global_enrichment_threshold": thresholds if threshold_scope == "chromosome" else next(iter(thresholds.values()), None),
        "threshold_scope": threshold_scope,
        "parameters_used": {
            "maf_threshold": maf_threshold,
            "enrichment_percentile": None,
            "threshold_scope": threshold_scope,
            "use_cm_if_available": use_cm_if_available,
        },
        "input_dir": str(input_dir),
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "cm_available": cm_available,
    }
    return summary


def _distribution_metrics(values):
    if values.size == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "quantiles": {
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
            },
        }

    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "quantiles": {
            "p05": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        },
    }


def load_metadata(path):
    """Carga metadatos opcionales para anotaciones y gráficos."""
    if not path:
        return {}

    metadata_path = Path(path)
    if not metadata_path.exists():
        _fail(f"Metadata file does not exist: {metadata_path}")

    df = pd.read_csv(metadata_path, sep=None, engine="python")
    normalized = {str(col).lower(): str(col) for col in df.columns}
    chrom_col = normalized.get("chrom") or normalized.get("chr")
    length_col = normalized.get("length_bp") or normalized.get("chrom_length_bp") or normalized.get("length")
    if not chrom_col or not length_col:
        return {}

    out = {}
    for row in df[[chrom_col, length_col]].dropna().itertuples(index=False):
        out[str(row[0])] = int(row[1])
    return out


def _journal_palette(name):
    palettes = {
        "journal": {
            "primary": "#1b4f72",
            "primary_fill": "#7fb3d5",
            "secondary": "#0b5345",
            "secondary_fill": "#73c6b6",
            "accent": "#922b21",
            "neutral": "#4d4d4d",
            "baseline": "#bdbdbd",
            "grid": "#d9d9d9",
        }
    }
    return palettes[name]


def _metric_label(metric):
    labels = {
        "log10_length_bp": "log10 tract length (bp)",
        "length_bp": "Tract length (bp)",
        "n_snps": "SNPs per tract",
        "rare_snp_density": "Rare SNP density (per Mb)",
    }
    return labels[metric]


def _extract_metric_by_chrom(tract_df, metric):
    if tract_df.empty:
        return pd.DataFrame(columns=["chrom", "metric_value"])

    values = tract_df.loc[:, ["chrom", "length_bp", "n_snps", "rare_snp_density"]].copy()
    if metric == "log10_length_bp":
        values = values[values["length_bp"] > 0].copy()
        values["metric_value"] = np.log10(values["length_bp"].astype(float))
    elif metric == "length_bp":
        values["metric_value"] = values["length_bp"].astype(float)
    elif metric == "n_snps":
        values["metric_value"] = values["n_snps"].astype(float)
    elif metric == "rare_snp_density":
        values["metric_value"] = values["rare_snp_density"].astype(float)
    else:
        _fail(f"Unsupported chromosome plot metric: {metric}")

    values = values.replace([np.inf, -np.inf], np.nan).dropna(subset=["metric_value"])
    return values[["chrom", "metric_value"]].copy()


def _select_chromosomes_for_distribution_plot(metric_df, min_tracts, max_chromosomes):
    if metric_df.empty:
        return []

    counts = (
        metric_df.groupby("chrom", sort=False)
        .size()
        .reset_index(name="n_tracts")
    )
    counts = counts[counts["n_tracts"] >= int(min_tracts)].copy()
    if counts.empty:
        return []

    if counts.shape[0] > int(max_chromosomes):
        counts = counts.sort_values(["n_tracts", "chrom"], ascending=[False, True]).head(int(max_chromosomes))

    return sorted(counts["chrom"].astype(str).tolist(), key=_natural_chr_key)


def _set_plot_style(plot_dpi):
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": int(plot_dpi),
            "figure.dpi": 150,
            "font.family": "DejaVu Sans",
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


def generate_plots(tract_df, window_df, thresholds, metadata, output_dir, plot_config):
    """Genera las figuras configuradas para ventanas y tractos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_plot_style(plot_config["dpi"])
    palette = _journal_palette(plot_config["palette"])

    bp_values = tract_df["length_bp"].to_numpy(dtype=float) if not tract_df.empty else np.array([])
    _plot_histogram(
        bp_values,
        output_dir / "tract_length_hist_bp.png",
        title="Rare SNP tract lengths",
        xlabel="Tract length (bp)",
        log_scale=False,
        palette=palette,
        bins_method=plot_config["bins_method"],
    )
    if plot_config["kde"]:
        _plot_density(
            bp_values,
            output_dir / "tract_length_density_bp.png",
            title="Rare SNP tract length density",
            xlabel="Tract length (bp)",
            palette=palette,
            bandwidth_scale=plot_config["kde_bandwidth_scale"],
        )
    else:
        write_placeholder_plot(
            output_dir / "tract_length_density_bp.png",
            "Rare SNP tract length density",
            "KDE plot disabled by configuration.",
        )

    log_bp_values = np.log10(bp_values) if bp_values.size else np.array([])
    _plot_histogram(
        log_bp_values,
        output_dir / "tract_length_hist_log10_bp.png",
        title="Rare SNP tract lengths",
        xlabel="log10 tract length (bp)",
        log_scale=False,
        palette=palette,
        bins_method=plot_config["bins_method"],
    )
    _plot_frequency_vs_log_size(
        log_bp_values,
        output_dir / "rare_snp_group_frequency_vs_log10_size_bp.png",
        title="Frequency of detected rare-SNP tracts by log10 size",
        xlabel="log10 tract length (bp)",
        palette=palette,
        bins_method=plot_config["bins_method"],
    )
    _plot_snp_burden_vs_log_size(
        tract_df,
        output_dir / "rare_snp_group_snps_vs_log10_size_bp.png",
        title="Total rare SNPs contained in detected tracts by log10 size",
        xlabel="log10 tract length (bp)",
        palette=palette,
        bins_method=plot_config["bins_method"],
    )
    if plot_config["ecdf"]:
        _plot_ecdf(
            log_bp_values,
            output_dir / "tract_length_ecdf_log10_bp.png",
            title="Empirical cumulative distribution of rare-SNP tract sizes",
            xlabel="log10 tract length (bp)",
            palette=palette,
        )
    else:
        write_placeholder_plot(
            output_dir / "tract_length_ecdf_log10_bp.png",
            "Empirical cumulative distribution of rare-SNP tract sizes",
            "ECDF plot disabled by configuration.",
        )
    if plot_config["kde"]:
        _plot_density(
            log_bp_values,
            output_dir / "tract_length_density_log10_bp.png",
            title="Rare SNP tract log-length density",
            xlabel="log10 tract length (bp)",
            palette=palette,
            bandwidth_scale=plot_config["kde_bandwidth_scale"],
        )
        _plot_density(
            log_bp_values,
            output_dir / "tract_length_kde_log10_bp.png",
            title="Kernel density estimate of rare-SNP tract sizes",
            xlabel="log10 tract length (bp)",
            palette=palette,
            bandwidth_scale=plot_config["kde_bandwidth_scale"],
        )
    else:
        write_placeholder_plot(
            output_dir / "tract_length_density_log10_bp.png",
            "Rare SNP tract log-length density",
            "KDE plot disabled by configuration.",
        )
        write_placeholder_plot(
            output_dir / "tract_length_kde_log10_bp.png",
            "Kernel density estimate of rare-SNP tract sizes",
            "KDE plot disabled by configuration.",
        )

    if plot_config["scatter"]:
        _plot_scatter_length_vs_snps(
            tract_df,
            output_dir / "tract_length_bp_vs_n_snps_scatter.png",
            palette=palette,
            log_x=plot_config["scatter_log_x"],
            log_y=plot_config["scatter_log_y"],
        )
    else:
        write_placeholder_plot(
            output_dir / "tract_length_bp_vs_n_snps_scatter.png",
            "Tract length versus SNP count",
            "Scatter plot disabled by configuration.",
        )

    chrom_metric_df = _extract_metric_by_chrom(tract_df, plot_config["chrom_metric"])
    selected_chroms = _select_chromosomes_for_distribution_plot(
        chrom_metric_df,
        min_tracts=plot_config["chrom_min_tracts"],
        max_chromosomes=plot_config["chrom_max"],
    )
    if plot_config["boxplot_by_chrom"]:
        _plot_boxplot_by_chromosome(
            chrom_metric_df,
            selected_chroms,
            output_dir / "tract_metric_boxplot_by_chromosome.png",
            metric_label=_metric_label(plot_config["chrom_metric"]),
            palette=palette,
        )
    else:
        write_placeholder_plot(
            output_dir / "tract_metric_boxplot_by_chromosome.png",
            "Tract distribution by chromosome",
            "Boxplot by chromosome disabled by configuration.",
        )
    if plot_config["violin_by_chrom"]:
        _plot_violin_by_chromosome(
            chrom_metric_df,
            selected_chroms,
            output_dir / "tract_metric_violin_by_chromosome.png",
            metric_label=_metric_label(plot_config["chrom_metric"]),
            palette=palette,
        )
    else:
        write_placeholder_plot(
            output_dir / "tract_metric_violin_by_chromosome.png",
            "Tract distribution by chromosome",
            "Violin plot by chromosome disabled by configuration.",
        )

    _plot_chromosome_tract_map(tract_df, metadata, output_dir / "chromosome_tract_map.png", palette)
    _plot_rare_density_by_chromosome(window_df, thresholds, output_dir / "rare_snp_density_by_chromosome.png", palette)

    if not tract_df.empty and tract_df["length_cm"].notna().any():
        cm_values = tract_df.loc[tract_df["length_cm"].notna(), "length_cm"].to_numpy(dtype=float)
        _plot_histogram(
            cm_values,
            output_dir / "tract_length_hist_cm.png",
            title="Rare SNP tract lengths",
            xlabel="Tract length (cM)",
            log_scale=False,
            palette=palette,
            bins_method=plot_config["bins_method"],
        )
        _plot_histogram(
            np.log10(np.clip(cm_values, 1e-6, None)),
            output_dir / "tract_length_hist_log10_cm.png",
            title="Rare SNP tract lengths",
            xlabel="log10 tract length (cM)",
            log_scale=False,
            palette=palette,
            bins_method=plot_config["bins_method"],
        )


def _plot_histogram(values, out_path, title, xlabel, log_scale=False, palette=None, bins_method="fd"):
    if values.size == 0:
        write_placeholder_plot(out_path, title, "No tracts detected under the current calling criteria.")
        return

    palette = palette or _journal_palette("journal")
    bins = _determine_bins(values, method=bins_method)
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax.hist(values, bins=bins, color=palette["primary"], edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.5)
    if log_scale:
        ax.set_xscale("log")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_density(values, out_path, title, xlabel, palette=None, bandwidth_scale=1.0):
    if values.size == 0:
        write_placeholder_plot(out_path, title, "No tracts detected under the current calling criteria.")
        return

    palette = palette or _journal_palette("journal")
    grid, density = _kde(values, bandwidth_scale=bandwidth_scale)
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    if grid is None:
        ax.axvline(values[0], color=palette["primary"], linewidth=2.0)
    else:
        ax.plot(grid, density, color=palette["primary"])
        ax.fill_between(grid, density, color=palette["primary_fill"], alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, axis="both", alpha=0.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_frequency_vs_log_size(values, out_path, title, xlabel, palette=None, bins_method="fd"):
    if values.size == 0:
        write_placeholder_plot(out_path, title, "No tracts detected under the current calling criteria.")
        return

    palette = palette or _journal_palette("journal")
    bins = _determine_bins(values, method=bins_method)
    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    ax.bar(
        centers,
        counts,
        width=widths * 0.95,
        color=palette["primary"],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
        align="center",
        label="Binned tract frequency",
    )
    ax.plot(
        centers,
        counts,
        color=palette["accent"],
        linewidth=1.6,
        marker="o",
        markersize=3.8,
        zorder=3,
        label="Frequency polygon",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency of detected tracts")
    ax.grid(True, axis="y", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_snp_burden_vs_log_size(tract_df, out_path, title, xlabel, palette=None, bins_method="fd"):
    if tract_df.empty:
        write_placeholder_plot(out_path, title, "No tracts detected under the current calling criteria.")
        return

    palette = palette or _journal_palette("journal")
    plot_df = tract_df.loc[:, ["length_bp", "n_snps"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        write_placeholder_plot(out_path, title, "No valid tract size and SNP-count values were available.")
        return

    plot_df = plot_df[plot_df["length_bp"] > 0]
    if plot_df.empty:
        write_placeholder_plot(out_path, title, "All detected tracts had non-positive lengths.")
        return

    log_lengths = np.log10(plot_df["length_bp"].to_numpy(dtype=float))
    bins = _determine_bins(log_lengths, method=bins_method)
    snp_totals, edges = np.histogram(
        log_lengths,
        bins=bins,
        weights=plot_df["n_snps"].to_numpy(dtype=float),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    ax.bar(
        centers,
        snp_totals,
        width=widths * 0.95,
        color=palette["secondary"],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
        align="center",
        label="Summed SNPs per size bin",
    )
    ax.plot(
        centers,
        snp_totals,
        color=palette["accent"],
        linewidth=1.6,
        marker="o",
        markersize=3.8,
        zorder=3,
        label="SNP burden polygon",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total SNPs in detected tracts")
    ax.grid(True, axis="y", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_ecdf(values, out_path, title, xlabel, palette=None):
    if values.size == 0:
        write_placeholder_plot(out_path, title, "No tracts detected under the current calling criteria.")
        return

    palette = palette or _journal_palette("journal")
    arr = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, arr.size + 1, dtype=float) / arr.size

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    ax.step(arr, y, where="post", color=palette["primary"], linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction of detected tracts")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="both", alpha=0.45)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_length_vs_snps(tract_df, out_path, palette=None, log_x=False, log_y=False):
    if tract_df.empty:
        write_placeholder_plot(out_path, "Tract length versus SNP count", "No tracts detected under the current calling criteria.")
        return

    palette = palette or _journal_palette("journal")
    plot_df = tract_df.loc[:, ["length_bp", "n_snps", "rare_snp_density"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[(plot_df["length_bp"] > 0) & (plot_df["n_snps"] > 0)]
    if plot_df.empty:
        write_placeholder_plot(out_path, "Tract length versus SNP count", "No valid tract lengths and SNP counts were available.")
        return

    density_for_color = plot_df["rare_snp_density"].to_numpy(dtype=float)
    x = plot_df["length_bp"].to_numpy(dtype=float)
    y = plot_df["n_snps"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    scatter = ax.scatter(
        x,
        y,
        c=density_for_color,
        cmap="Blues",
        s=28,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.35,
    )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_title("Tract length versus SNP count")
    ax.set_xlabel("Tract length (bp)")
    ax.set_ylabel("SNPs per tract")
    ax.grid(True, axis="both", alpha=0.35)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Rare SNP density (per Mb)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_boxplot_by_chromosome(metric_df, chroms, out_path, metric_label, palette=None):
    if metric_df.empty or not chroms:
        write_placeholder_plot(out_path, "Tract distribution by chromosome", "Too few chromosome-specific tracts were available for a boxplot.")
        return

    palette = palette or _journal_palette("journal")
    data = [
        metric_df.loc[metric_df["chrom"].astype(str) == chrom, "metric_value"].to_numpy(dtype=float)
        for chrom in chroms
    ]
    data = [arr for arr in data if arr.size > 0]
    if not data:
        write_placeholder_plot(out_path, "Tract distribution by chromosome", "No valid chromosome-wise tract values were available for a boxplot.")
        return

    fig_width = max(9.2, 0.42 * len(chroms) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), constrained_layout=True)
    box = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=chroms,
        medianprops={"color": palette["accent"], "linewidth": 1.5},
        whiskerprops={"color": palette["neutral"], "linewidth": 1.0},
        capprops={"color": palette["neutral"], "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": palette["accent"],
            "markeredgecolor": "white",
            "alpha": 0.8,
        },
    )
    for patch in box["boxes"]:
        patch.set_facecolor(palette["primary_fill"])
        patch.set_edgecolor(palette["primary"])
        patch.set_linewidth(1.0)

    ax.set_title("Chromosome-wise tract distribution")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(metric_label)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.35)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_violin_by_chromosome(metric_df, chroms, out_path, metric_label, palette=None):
    if metric_df.empty or not chroms:
        write_placeholder_plot(out_path, "Tract distribution by chromosome", "Too few chromosome-specific tracts were available for a violin plot.")
        return

    palette = palette or _journal_palette("journal")
    data = [
        metric_df.loc[metric_df["chrom"].astype(str) == chrom, "metric_value"].to_numpy(dtype=float)
        for chrom in chroms
    ]
    data = [arr for arr in data if arr.size > 0]
    if not data:
        write_placeholder_plot(out_path, "Tract distribution by chromosome", "No valid chromosome-wise tract values were available for a violin plot.")
        return

    fig_width = max(9.2, 0.42 * len(chroms) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), constrained_layout=True)
    violins = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for body in violins["bodies"]:
        body.set_facecolor(palette["secondary_fill"])
        body.set_edgecolor(palette["secondary"])
        body.set_alpha(0.9)
        body.set_linewidth(0.8)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        violins[part_name].set_color(palette["accent"] if part_name == "cmedians" else palette["neutral"])
        violins[part_name].set_linewidth(1.0)

    ax.set_xticks(np.arange(1, len(chroms) + 1))
    ax.set_xticklabels(chroms, rotation=45, ha="right")
    ax.set_title("Chromosome-wise tract distribution")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(metric_label)
    ax.grid(True, axis="y", alpha=0.35)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _kde(values, bandwidth_scale=1.0):
    arr = np.asarray(values, dtype=float)
    if arr.size < 2 or np.allclose(arr.min(), arr.max()):
        return None, None

    std = np.std(arr, ddof=1)
    iqr = np.subtract(*np.percentile(arr, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = max(std, 1e-6)
    bandwidth = 0.9 * sigma * (arr.size ** (-1.0 / 5.0)) * float(bandwidth_scale)
    bandwidth = max(float(bandwidth), 1e-6)

    grid = np.linspace(arr.min(), arr.max(), 512)
    diff = (grid[:, None] - arr[None, :]) / bandwidth
    density = np.exp(-0.5 * diff * diff).sum(axis=1) / (arr.size * bandwidth * math.sqrt(2.0 * math.pi))
    return grid, density


def _determine_bins(values, method="fd"):
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1 or np.allclose(arr.min(), arr.max()):
        return 1

    if method == "fd":
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            bin_width = 2.0 * iqr / np.cbrt(arr.size)
            if math.isfinite(bin_width) and bin_width > 0:
                estimated = int(np.ceil((arr.max() - arr.min()) / bin_width))
                return max(12, min(36, estimated))
    elif method == "sqrt":
        estimated = int(np.ceil(np.sqrt(arr.size)))
        return max(12, min(36, estimated))
    else:
        _fail(f"Unsupported binning method: {method}")

    fallback = int(np.ceil(np.sqrt(arr.size)))
    return max(12, min(36, fallback))


def _plot_chromosome_tract_map(tract_df, metadata, out_path, palette=None):
    if tract_df.empty:
        write_placeholder_plot(out_path, "Rare SNP tract map", "No tracts detected under the current calling criteria.")
        return

    palette = palette or _journal_palette("journal")
    chroms = sorted(tract_df["chrom"].astype(str).unique(), key=_natural_chr_key)
    fig_height = max(4.5, 0.42 * len(chroms) + 1.5)
    fig, ax = plt.subplots(figsize=(12.0, fig_height), constrained_layout=True)

    for idx, chrom in enumerate(chroms):
        chrom_df = tract_df[tract_df["chrom"].astype(str) == chrom].sort_values("start_pos")
        y = len(chroms) - idx
        for row in chrom_df.itertuples(index=False):
            ax.plot(
                [row.start_pos / 1e6, row.end_pos / 1e6],
                [y, y],
                color=palette["primary"],
                linewidth=4.0,
                solid_capstyle="butt",
            )
        chrom_length = metadata.get(str(chrom))
        if chrom_length:
            ax.plot([0, chrom_length / 1e6], [y, y], color=palette["baseline"], linewidth=0.8, alpha=0.6, zorder=0)

    ax.set_yticks(range(1, len(chroms) + 1))
    ax.set_yticklabels(list(reversed(chroms)))
    ax.set_xlabel("Physical position (Mb)")
    ax.set_ylabel("Chromosome")
    ax.set_title("Spatial distribution of enriched rare-SNP tracts")
    ax.grid(True, axis="x", alpha=0.35)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rare_density_by_chromosome(window_df, thresholds, out_path, palette=None):
    if window_df.empty:
        write_placeholder_plot(out_path, "Rare SNP density by chromosome", "No analyzable windows were produced.")
        return

    palette = palette or _journal_palette("journal")
    chroms = sorted(window_df["chrom"].astype(str).unique(), key=_natural_chr_key)
    ncols = 4
    nrows = math.ceil(len(chroms) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(17.0, max(5.0, 2.9 * nrows)),
        constrained_layout=True,
        squeeze=False,
    )

    max_points = 3000
    for idx, chrom in enumerate(chroms):
        ax = axes[idx // ncols][idx % ncols]
        chrom_df = window_df[window_df["chrom"].astype(str) == chrom].sort_values("mid_pos")
        if chrom_df.shape[0] > max_points:
            keep_idx = np.linspace(0, chrom_df.shape[0] - 1, max_points).round().astype(int)
            chrom_df = chrom_df.iloc[keep_idx]

        x = chrom_df["mid_pos"].to_numpy(dtype=float) / 1e6
        y = chrom_df["rare_snp_density_per_mb"].to_numpy(dtype=float)
        ax.plot(x, y, color=palette["neutral"], linewidth=1.0)
        threshold = thresholds.get(str(chrom))
        if threshold is not None:
            ax.axhline(float(threshold), color=palette["accent"], linestyle="--", linewidth=1.0)
        ax.set_title(f"chr{chrom}" if not str(chrom).startswith("chr") else str(chrom))
        ax.set_xlabel("Mb")
        ax.set_ylabel("Rare SNPs / Mb")
        ax.grid(True, alpha=0.3)

    total_axes = nrows * ncols
    for idx in range(len(chroms), total_axes):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Rare-variant local density by chromosome", fontsize=15, y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_placeholder_plot(out_path, title, message):
    """Escribe una figura informativa cuando no hay datos suficientes para graficar."""
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_report(summary, tract_df, output_dir, maf_threshold, threshold_scope, enrichment_percentile):
    """Escribe el informe HTML con métodos, resultados y limitaciones."""
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"

    top_chrom_rows = []
    for chrom, chrom_summary in summary["summary_by_chromosome"].items():
        top_chrom_rows.append(
            {
                "chrom": chrom,
                "n_tracts": chrom_summary["n_tracts"],
                "total_tract_bp": chrom_summary["total_tract_bp"],
                "max_tract_bp": chrom_summary["max_tract_bp"],
            }
        )
    top_df = pd.DataFrame(top_chrom_rows)
    if not top_df.empty:
        top_df = top_df.sort_values(["n_tracts", "total_tract_bp"], ascending=[False, False]).head(8)

    if tract_df.empty:
        tracts_html = "<p>No enriched tracts passed the current calling criteria.</p>"
    else:
        tracts_html = tract_df.head(20).to_html(index=False, classes="compact-table", border=0)

    top_html = top_df.to_html(index=False, classes="compact-table", border=0) if not top_df.empty else "<p>No chromosome-level tract enrichment to display.</p>"
    summary_json = json.dumps(summary, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Rare SNP tract report</title>
  <style>
    body {{
      font-family: "DejaVu Sans", Arial, sans-serif;
      margin: 2rem auto;
      max-width: 1200px;
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
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
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
    }}
    .compact-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.94rem;
    }}
    .compact-table th, .compact-table td {{
      border-bottom: 1px solid #e5e5e5;
      padding: 0.45rem 0.55rem;
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
  <h1>Rare SNP tract analysis</h1>
  <p>This module summarizes spatial clustering of already-selected rare variants. It does not infer genealogy, SMC histories, or local ancestry.</p>

  <div class="grid">
    <div class="card">
      <h2>Operational definition</h2>
      <p><strong>Rare SNP input:</strong> upstream rare-only VCF with INFO/AC, INFO/AN and INFO/AF preserved.</p>
      <p><strong>Rare threshold declared upstream:</strong> AF/MAF cutoff = {maf_threshold:.4g}</p>
      <p><strong>Window score:</strong> rare SNP density per Mb in sliding windows of fixed rare-variant count.</p>
      <p><strong>Tract calling:</strong> windows above the {enrichment_percentile:.1f}th percentile ({threshold_scope}-specific threshold), merged across gaps of up to the configured number of windows.</p>
    </div>
    <div class="card">
      <h2>Interpretation</h2>
      <p>Long tracts or non-random local accumulations of rare variants can be consistent with shared demographic history, selection, bottlenecks, or population structure.</p>
      <p>These outputs are exploratory summaries only and should be treated as inputs for downstream genealogical or ancestry-aware analyses, not as direct genealogical inference.</p>
    </div>
  </div>

  <h2>High-level summary</h2>
  <pre>{summary_json}</pre>

  <h2>Chromosomes with strongest aggregation</h2>
  {top_html}

  <h2>Representative tracts</h2>
  {tracts_html}

  <h2>Figures</h2>
  <div class="grid">
    <div><img src="plots/tract_length_hist_bp.png" alt="tract length histogram in bp" /></div>
    <div><img src="plots/tract_length_density_bp.png" alt="tract length density in bp" /></div>
    <div><img src="plots/tract_length_hist_log10_bp.png" alt="tract length histogram log10 bp" /></div>
    <div><img src="plots/rare_snp_group_frequency_vs_log10_size_bp.png" alt="frequency of detected rare SNP tracts by log10 size" /></div>
    <div><img src="plots/rare_snp_group_snps_vs_log10_size_bp.png" alt="total SNPs in detected rare SNP tracts by log10 size" /></div>
    <div><img src="plots/tract_length_ecdf_log10_bp.png" alt="empirical cumulative distribution of tract lengths in log10 bp" /></div>
    <div><img src="plots/tract_length_density_log10_bp.png" alt="tract length density log10 bp" /></div>
    <div><img src="plots/tract_length_kde_log10_bp.png" alt="kernel density estimate of tract lengths in log10 bp" /></div>
    <div><img src="plots/tract_length_bp_vs_n_snps_scatter.png" alt="scatter of tract length in bp versus SNP count" /></div>
    <div><img src="plots/tract_metric_boxplot_by_chromosome.png" alt="boxplot of tract metric by chromosome" /></div>
    <div><img src="plots/tract_metric_violin_by_chromosome.png" alt="violin plot of tract metric by chromosome" /></div>
    <div><img src="plots/chromosome_tract_map.png" alt="chromosome tract map" /></div>
    <div><img src="plots/rare_snp_density_by_chromosome.png" alt="rare SNP density by chromosome" /></div>
  </div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def _ensure_output_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def scan_mode(args):
    """Ejecuta el análisis de un cromosoma y publica sus resultados."""
    if not args.input or not args.chr or not args.out_window_scores or not args.out_summary_json:
        _fail("scan mode requires --input, --chr, --out-window-scores and --out-summary-json")
    if args.window_size_snps <= 1 or args.step_size_snps <= 0:
        _fail("window-size-snps must be >1 and step-size-snps must be >0")
    if args.min_chrom_rare_snps <= 0:
        _fail("min-chrom-rare-snps must be positive")

    data = load_input_data(args.input, args.input_format, args.chr, args.maf_threshold)
    positions = identify_rare_snps(data, args.maf_threshold)

    if positions.size < args.min_chrom_rare_snps or positions.size < args.window_size_snps:
        window_df = pd.DataFrame(columns=WINDOW_SCORE_COLUMNS)
        status = "skipped_low_snp_count"
    else:
        window_df = build_window_scores(args.chr, positions, args.window_size_snps, args.step_size_snps)
        status = "analyzed" if not window_df.empty else "skipped_low_snp_count"

    _ensure_output_parent(args.out_window_scores)
    _ensure_output_parent(args.out_summary_json)
    window_df.to_csv(args.out_window_scores, sep="\t", index=False, compression="gzip")

    positions_span = int(positions[-1] - positions[0] + 1) if positions.size else 0
    inter_pos = np.diff(positions) if positions.size > 1 else np.array([], dtype=np.int64)
    payload = {
        "chrom": str(args.chr),
        "input_file": str(Path(args.input).resolve()),
        "status": status,
        "n_rare_snps": int(positions.size),
        "observed_span_bp": positions_span,
        "median_inter_snp_distance_bp": float(np.median(inter_pos)) if inter_pos.size else None,
        "mean_inter_snp_distance_bp": float(np.mean(inter_pos)) if inter_pos.size else None,
        "n_windows": int(window_df.shape[0]),
        "maf_threshold_declared_upstream": float(args.maf_threshold),
        "min_info_af_seen": data["min_info_af"],
        "max_info_af_seen": data["max_info_af"],
        "window_size_snps": int(args.window_size_snps),
        "step_size_snps": int(args.step_size_snps),
        "min_chrom_rare_snps": int(args.min_chrom_rare_snps),
        "analysis_date": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(args.out_summary_json, payload)


def aggregate_mode(args):
    """Combina resultados cromosómicos en tablas, figuras e informe genómico."""
    if not args.window_scores or not args.per_chr_summary:
        _fail(
            "aggregate mode requires per-chromosome window scores and summaries. "
            "Confirm that upstream scan jobs produced outputs."
        )

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    window_dfs = []
    for path in args.window_scores:
        in_path = Path(path)
        if not in_path.exists():
            _fail(f"Missing per-chromosome window scores file: {in_path}")
        df = pd.read_csv(in_path, sep="\t", compression="gzip")
        if not df.empty:
            window_dfs.append(df)

    per_chr_summaries = []
    for path in args.per_chr_summary:
        summary_path = Path(path)
        if not summary_path.exists():
            _fail(f"Missing per-chromosome summary JSON: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as handle:
            per_chr_summaries.append(json.load(handle))

    if not per_chr_summaries:
        _fail("No per-chromosome summaries were provided to aggregate rare SNP tracts.")

    if window_dfs:
        window_df = pd.concat(window_dfs, ignore_index=True)
    else:
        window_df = pd.DataFrame(columns=WINDOW_SCORE_COLUMNS)

    if not window_df.empty:
        window_df["chrom"] = window_df["chrom"].astype(str)

    metadata = load_metadata(args.metadata) if args.metadata else {}
    genetic_map = load_genetic_map(args.genetic_map) if args.genetic_map and args.use_cm_if_available else {}

    scored_windows, thresholds = call_enriched_windows(
        window_df,
        enrichment_percentile=args.enrichment_percentile,
        threshold_scope=args.threshold_scope,
    )

    tract_df = merge_windows_into_tracts(scored_windows, args.max_gap_windows)
    if not tract_df.empty:
        tract_df["chrom"] = tract_df["chrom"].astype(str)
        tract_df = tract_df[
            (tract_df["n_snps"] >= int(args.min_tract_snps))
            & (tract_df["length_bp"] >= int(args.min_tract_bp))
        ].copy()

    tract_df = compute_tract_lengths(tract_df, genetic_map, args.use_cm_if_available)
    tract_df = tract_df.sort_values(["chrom", "start_pos"], key=lambda col: col.map(_natural_chr_key) if col.name == "chrom" else col).reset_index(drop=True) if not tract_df.empty else pd.DataFrame(columns=TRACT_COLUMNS)

    if tract_df.empty:
        tract_df = pd.DataFrame(columns=TRACT_COLUMNS)

    summary = summarize_distributions(
        tract_df=tract_df,
        window_df=scored_windows,
        per_chr_summaries=per_chr_summaries,
        thresholds=thresholds,
        threshold_scope=args.threshold_scope,
        maf_threshold=args.maf_threshold,
        input_dir=args.input_dir,
        use_cm_if_available=args.use_cm_if_available,
    )
    summary["parameters_used"].update(
        {
            "enrichment_percentile": float(args.enrichment_percentile),
            "threshold_scope": args.threshold_scope,
            "max_gap_windows": int(args.max_gap_windows),
            "min_tract_snps": int(args.min_tract_snps),
            "min_tract_bp": int(args.min_tract_bp),
            "use_cm_if_available": bool(args.use_cm_if_available),
            "plot_dpi": int(args.plot_dpi),
            "plot_palette": args.plot_palette,
            "plot_bins_method": args.plot_bins_method,
            "plot_kde_bandwidth_scale": float(args.plot_kde_bandwidth_scale),
            "plot_ecdf": bool(args.plot_ecdf),
            "plot_kde": bool(args.plot_kde),
            "plot_scatter": bool(args.plot_scatter),
            "plot_boxplot_by_chrom": bool(args.plot_boxplot_by_chrom),
            "plot_violin_by_chrom": bool(args.plot_violin_by_chrom),
            "plot_chrom_metric": args.plot_chrom_metric,
            "plot_chrom_min_tracts": int(args.plot_chrom_min_tracts),
            "plot_chrom_max": int(args.plot_chrom_max),
            "plot_scatter_log_x": bool(args.plot_scatter_log_x),
            "plot_scatter_log_y": bool(args.plot_scatter_log_y),
        }
    )
    summary["input_files"] = [str(Path(path).resolve()) for path in args.window_scores]

    tract_path = output_dir / "tracts.tsv"
    summary_path = output_dir / "tract_summary.json"
    tract_df.to_csv(tract_path, sep="\t", index=False)
    _write_json(summary_path, summary)

    plot_config = {
        "dpi": int(args.plot_dpi),
        "palette": args.plot_palette,
        "bins_method": args.plot_bins_method,
        "kde_bandwidth_scale": float(args.plot_kde_bandwidth_scale),
        "ecdf": bool(args.plot_ecdf),
        "kde": bool(args.plot_kde),
        "scatter": bool(args.plot_scatter),
        "boxplot_by_chrom": bool(args.plot_boxplot_by_chrom),
        "violin_by_chrom": bool(args.plot_violin_by_chrom),
        "chrom_metric": args.plot_chrom_metric,
        "chrom_min_tracts": int(args.plot_chrom_min_tracts),
        "chrom_max": int(args.plot_chrom_max),
        "scatter_log_x": bool(args.plot_scatter_log_x),
        "scatter_log_y": bool(args.plot_scatter_log_y),
    }

    generate_plots(tract_df, scored_windows, thresholds, metadata, plots_dir, plot_config)
    write_report(
        summary=summary,
        tract_df=tract_df,
        output_dir=output_dir,
        maf_threshold=args.maf_threshold,
        threshold_scope=args.threshold_scope,
        enrichment_percentile=args.enrichment_percentile,
    )


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
