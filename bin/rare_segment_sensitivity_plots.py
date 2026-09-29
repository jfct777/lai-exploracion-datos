#!/usr/bin/env python3
"""Single-chromosome descriptive plots from precomputed distance/grid TSVs.

No genotypes are read and no segment detection is performed. Distance universes
stay separate. Sensitivity heatmaps use actual effective support counts, mask
unevaluated cells, and share a colour scale across gap panels for each metric.
These figures describe sensitivity, not biological validation or an optimum.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

from rare_segment_plots import _chrom, _integer, _new_file


UNIVERSES = {
    "catalogue_all_rare": "Rare-site catalogue",
    "individual_carrier_all_rare": "Individual carrier positions",
    "pair_shared_before_segment_filter": "Pair-shared positions before segment filtering",
}
METRICS = {
    "n_segments": ("Retained segments", "Segment count"),
    "n_pairs": ("Pairs with retained segments", "Pair count"),
    "total_shared_bp": ("Summed segment length across pairs", "Pair × bp (not genomic union)"),
}


def _frame(value, columns):
    import pandas as pd
    if isinstance(value, (str, os.PathLike)):
        frame = pd.read_csv(value, sep="\t", dtype=str, keep_default_na=False)
    elif isinstance(value, pd.DataFrame):
        frame = value.copy(deep=True)
    else:
        raise TypeError("Expected a DataFrame or TSV path")
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Expected a DataFrame or TSV with unique columns")
    if not set(columns) <= set(frame.columns):
        raise ValueError("Missing required plotting columns: " + ", ".join(sorted(set(columns) - set(frame.columns))))
    return frame


def _distance_data(summary, histogram):
    summary = _frame(summary, ("universe", "n_units", "n_positions", "n_distances", "median_distance_bp", "p90_distance_bp", "p95_distance_bp"))
    histogram = _frame(histogram, ("universe", "bin_start_bp", "bin_end_bp", "count"))
    if summary["universe"].duplicated().any() or set(summary["universe"]) != set(UNIVERSES):
        raise ValueError("Distance summary must contain each of the three declared universes once")
    if set(histogram["universe"]) - set(UNIVERSES):
        raise ValueError("Unexpected distance universe")
    results = {}
    for raw in summary.to_dict("records"):
        universe = raw["universe"]
        row = {k: _integer(raw[k], k) for k in ("n_units", "n_positions", "n_distances")}
        row["quantiles_bp"] = {}
        if row["n_distances"]:
            for field in ("median_distance_bp", "p90_distance_bp", "p95_distance_bp"):
                value = float(raw[field])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Positive finite distance quantiles required for nonempty universes")
                row["quantiles_bp"][field] = value
            if list(row["quantiles_bp"].values()) != sorted(row["quantiles_bp"].values()):
                raise ValueError("Distance quantiles must be monotone")
        bins = []
        for item in histogram[histogram["universe"] == universe].to_dict("records"):
            left, right = float(item["bin_start_bp"]), float(item["bin_end_bp"])
            count = _integer(item["count"], "histogram count")
            if not math.isfinite(left) or left < 0 or math.isnan(right) or right <= left:
                raise ValueError("Invalid histogram interval")
            bins.append((left, right, count))
        bins.sort()
        if any(a[1] > b[0] for a, b in zip(bins, bins[1:])):
            raise ValueError("Histogram bins overlap")
        if sum(b[2] for b in bins) != row["n_distances"]:
            raise ValueError("Histogram counts do not reconcile with n_distances")
        results[universe] = dict(row, bins=bins)
    return results


def _grid_data(value):
    columns = ("max_gap_bp", "min_length_bp", "min_shared_effective", *METRICS)
    frame = _frame(value, columns)
    rows, seen = [], set()
    for raw in frame.to_dict("records"):
        row = {k: _integer(raw[k], k, 1 if k in columns[:3] else 0) for k in columns}
        key = tuple(row[k] for k in columns[:3])
        if key in seen:
            raise ValueError("Duplicate effective grid cell")
        if row["n_pairs"] > row["n_segments"]:
            raise ValueError("Pair count exceeds segment count")
        if row["total_shared_bp"] < row["n_segments"] * row["min_length_bp"]:
            raise ValueError("Summed segment lengths violate min_length_bp")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("Sensitivity grid is empty")
    return sorted(rows, key=lambda r: (r["max_gap_bp"], r["min_length_bp"], r["min_shared_effective"]))


def _distance_figure(data, chrom):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(13.5, 10.5), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1)
    figure.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.13, hspace=0.6)
    finite_bounds = [v for row in data.values() for left, right, _ in row["bins"] for v in (left, right) if math.isfinite(v) and v > 0]
    quantiles = [v for row in data.values() for v in row["quantiles_bp"].values()]
    lower = min(finite_bounds, default=1)
    upper = max(finite_bounds + quantiles, default=10)
    lower = min(lower, 0.5) if any(b[0] == 0 for r in data.values() for b in r["bins"]) else lower
    upper = max(upper, lower * 10)
    for ax, (universe, title) in zip(axes, UNIVERSES.items()):
        row = data[universe]
        overflow = 0
        for left, right, count in row["bins"]:
            if math.isinf(right):
                overflow += count
                continue
            left = max(left, 0.5)
            ax.bar(left, count, width=right-left, align="edge", color="#286D93", edgecolor="#174760", linewidth=0.4)
        for (field, value), style, color in zip(row["quantiles_bp"].items(), ("-", "--", ":"), ("#444444", "#B36619", "#8D4C75")):
            label = {"median_distance_bp": "Median", "p90_distance_bp": "P90", "p95_distance_bp": "P95"}[field]
            ax.axvline(value, color=color, linestyle=style, linewidth=1.1, label=f"{label} = {value:g} bp")
        ax.set_xscale("log")
        ax.set_xlim(lower, upper)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{title} | {row['n_units']:,} units | {row['n_distances']:,} distances", fontsize=10, loc="left")
        ax.set_ylabel("Pooled interval count", fontsize=9)
        ax.set_xlabel("Consecutive-position distance (bp; logarithmic axis)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DCE0E3", linewidth=0.5)
        if row["quantiles_bp"]:
            ax.legend(fontsize=8, loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="none")
        else:
            ax.text(0.5, 0.5, "No distances", transform=ax.transAxes, ha="center")
        if overflow:
            ax.text(0.01, 0.9, f"Overflow beyond final finite bin: {overflow:,} (not drawn)", transform=ax.transAxes, fontsize=8)
    figure.suptitle(f"chr{chrom}: consecutive rare-position distances", fontsize=13)
    figure.text(0.10, 0.025, "Three different denominators; pooled counts, not mean individual/pair distributions.\nUnequal-width bins are counts, not density; quantiles use the full supplied distribution. No IBD or ancestry validation implied.", fontsize=8)
    return figure


def _grid_figure(rows, metric, chrom):
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib import colormaps
    gaps = sorted({row["max_gap_bp"] for row in rows})
    lengths = sorted({row["min_length_bp"] for row in rows})
    counts = sorted({row["min_shared_effective"] for row in rows})
    figure = Figure(figsize=(max(13.5, 0.55 * len(counts) + 3), max(5, 2.8 * len(gaps) + 1.5)), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(gaps), 1, squeeze=False).ravel()
    figure.subplots_adjust(left=0.09, right=0.87, bottom=0.11, top=0.91, hspace=0.7)
    norm = Normalize(vmin=0, vmax=max(1, max(row[metric] for row in rows)))
    cmap = colormaps["Blues"].copy()
    cmap.set_bad("#D9D9D9")
    lookup = {(r["max_gap_bp"], r["min_length_bp"], r["min_shared_effective"]): r[metric] for r in rows}
    for ax, gap in zip(axes, gaps):
        matrix = np.array([[lookup.get((gap, length, count), np.nan) for count in counts] for length in lengths])
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(counts)), labels=counts, fontsize=8)
        ax.set_yticks(range(len(lengths)), labels=[f"{x/1e6:g}" for x in lengths], fontsize=9)
        ax.set_xlabel("Minimum shared sites (effective; discrete, nonuniform grid)", fontsize=9)
        ax.set_ylabel("Minimum length (Mb)", fontsize=9)
        ax.set_title(f"Maximum consecutive gap = {gap/1000:g} kb", fontsize=10, loc="left")
        for i, j in zip(*np.where(np.isfinite(matrix))):
            value = int(matrix[i, j])
            label = f"{value:,}" if metric != "total_shared_bp" else (f"{value / 1e9:.2f}G" if value >= 1e9 else f"{value / 1e6:.2f}M")
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if norm(value) > 0.6 else "#222222")
    bar = figure.colorbar(image, cax=figure.add_axes((0.90, 0.20, 0.02, 0.60)))
    bar.set_label(METRICS[metric][1], fontsize=9)
    figure.suptitle(f"chr{chrom}: {METRICS[metric][0]}", fontsize=13)
    figure.text(0.09, 0.025, "Grey = configuration not evaluated, never zero. Same colour scale across gap panels; linear scale.\nEffective support thresholds may collapse several nominal values. Descriptive sensitivity, not validation or an optimum."
                + (" Cell labels M/G = million/billion pair×bp." if metric == "total_shared_bp" else ""), fontsize=8)
    return figure


def _chain_data(value):
    frame = _frame(value, ("max_gap_bp", "metric", "bin_start", "bin_end", "count"))
    result = {}
    for row in frame[frame["metric"] == "length_bp"].to_dict("records"):
        gap = _integer(row["max_gap_bp"], "max_gap_bp", 1)
        left, right = float(row["bin_start"]), float(row["bin_end"])
        count = _integer(row["count"], "chain histogram count")
        if not math.isfinite(left) or left < 0 or math.isnan(right) or right <= left:
            raise ValueError("Invalid chain histogram interval")
        result.setdefault(gap, []).append((left, right, count))
    if not result:
        raise ValueError("Chain histogram has no length_bp rows")
    for bins in result.values():
        bins.sort()
        if any(a[1] > b[0] for a, b in zip(bins, bins[1:])):
            raise ValueError("Chain histogram bins overlap")
    return dict(sorted(result.items()))


def _chain_figure(data, chrom):
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(14, max(5, 3.0 * len(data) + 1.5)), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(data), 2, squeeze=False)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.13, hspace=0.65, wspace=0.22)
    bounds = [v for bins in data.values() for a, b, _ in bins for v in (a, b) if math.isfinite(v) and v > 0]
    lower, upper = min(bounds, default=1), max([1e6, *bounds])
    lower = min(lower, 0.5) if any(b[0] == 0 for bins in data.values() for b in bins) else lower
    max_count = max((b[2] for bins in data.values() for b in bins), default=1)
    for (hist, cdf), (gap, bins) in zip(axes, data.items()):
        finite = [b for b in bins if math.isfinite(b[1])]
        total = sum(b[2] for b in bins)
        overflow = total - sum(b[2] for b in finite)
        for left, right, count in finite:
            left = max(left, 0.5)
            if count:
                hist.bar(left, count, width=right-left, align="edge", color="#286D93", edgecolor="#174760", linewidth=0.35)
        if total and finite:
            cdf.step([max(finite[0][0], 0.5), *(b[1] for b in finite)],
                     [0, *(np.cumsum([b[2] for b in finite]) / total)], where="post", color="#286D93", linewidth=1.2,
                     marker="o", markersize=2)
        for ax in (hist, cdf):
            ax.set_xscale("log")
            ax.set_xlim(lower, max(upper, lower * 10) * 1.05)
            ax.axvline(1e6, color="#B36619", linestyle="--", linewidth=1.0, label="1 Mb reference")
            ax.set_xlabel("Chain length (bp; logarithmic axis)", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#DCE0E3", linewidth=0.4)
            ax.legend(fontsize=8, loc="upper right", frameon=True, facecolor="white", edgecolor="none")
        hist.set_yscale("log")
        hist.set_ylim(0.8, max(2, max_count * 1.3))
        hist.set_ylabel("Chain count (log scale)", fontsize=9)
        hist.set_title(f"G = {gap/1000:g} kb | {total:,} chains before filters", fontsize=10, loc="left")
        cdf.set_ylim(0, 1.03)
        cdf.set_ylabel("Cumulative fraction at bin upper edges", fontsize=9)
        cdf.set_title("Binned CDF; no interpolation within bins", fontsize=10, loc="left")
        if overflow:
            hist.text(0.02, 0.9, f"Overflow not drawn: {overflow:,}", transform=hist.transAxes, fontsize=8)
        if not total:
            hist.text(0.5, 0.5, "No chains", transform=hist.transAxes, ha="center")
            cdf.text(0.5, 0.5, "No chains", transform=cdf.transAxes, ha="center")
    figure.suptitle(f"chr{chrom}: lengths of all pair-shared chains before segment filters", fontsize=13)
    figure.text(0.075, 0.025, "All supplied chains, including short/single-site chains; not only published segments or retained candidates.\nHistogram bins are [left,right); CDF gives P(length < upper edge). The 1 Mb line is a reference, not validation.\nAn exact fraction below 1 Mb cannot be recovered unless it is an explicit bin boundary. Overflow remains in the CDF denominator.", fontsize=8)
    return figure


def render_sensitivity_views(distance_summary, distance_histogram, configuration_summary,
                             output_dir, chrom, dpi=300, formats=("png", "pdf"), prefix=None,
                             chain_histogram=None):
    """Render existing aggregate TSVs; return/write an ID-free figure manifest."""
    chrom, dpi = _chrom(chrom), _integer(dpi, "dpi", 1)
    formats = (formats,) if isinstance(formats, str) else tuple(formats)
    if not formats or len(set(formats)) != len(formats) or set(formats) - {"png", "pdf"}:
        raise ValueError("formats must contain distinct png and/or pdf values")
    prefix = f"chr{chrom}" if prefix is None else prefix
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("prefix must be a safe filename stem")
    distances, grid = _distance_data(distance_summary, distance_histogram), _grid_data(configuration_summary)
    chains = None if chain_histogram is None else _chain_data(chain_histogram)
    directory = Path(output_dir)
    names = ["distance_histograms", *(f"sensitivity_{key}" for key in METRICS)]
    if chains is not None:
        names.append("chain_lengths")
    paths = {name: {fmt: directory / f"{prefix}.{name}.{fmt}" for fmt in formats} for name in names}
    manifest_path = directory / f"{prefix}.sensitivity_plots.manifest.json"
    if any(os.path.lexists(p) for p in [manifest_path, *(p for group in paths.values() for p in group.values())]):
        raise FileExistsError("A sensitivity-plot destination already exists; use a new prefix or directory")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "1.0", "chrom": chrom, "n_effective_configurations": len(grid),
                "universes": list(UNIVERSES), "dpi": dpi, "formats": list(formats),
                "configuration_cells": grid, "unevaluated_cells": "masked grey, not zero",
                "distance_quantiles_and_counts": {k: {field: value for field, value in v.items() if field != "bins"} for k, v in distances.items()},
                "source_distances_sha256": hashlib.sha256(json.dumps(distances, sort_keys=True).encode()).hexdigest(),
                "source_grid_sha256": hashlib.sha256(json.dumps(grid, sort_keys=True).encode()).hexdigest(),
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Descriptive sensitivity only; no optimality, ancestry or IBD validation claimed", "figures": {}}
    if chains is not None:
        manifest["chain_histogram_counts_by_gap"] = {str(gap): sum(b[2] for b in bins) for gap, bins in chains.items()}
        manifest["chains_below_1mb_exact_if_bin_boundary"] = {
            str(gap): sum(b[2] for b in bins if b[1] <= 1000000)
            if any(b[1] == 1000000 for b in bins) else None for gap, bins in chains.items()
        }
        manifest["source_chain_histogram_sha256"] = hashlib.sha256(json.dumps(chains, sort_keys=True).encode()).hexdigest()
        manifest["chain_cdf"] = "Binned P(length < bin upper edge), no within-bin interpolation; overflow included in denominator"
    import matplotlib
    manifest["matplotlib_version"] = matplotlib.__version__
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.usetex": False}):
        for name in names:
            if name == "distance_histograms":
                figure = _distance_figure(distances, chrom)
            elif name == "chain_lengths":
                figure = _chain_figure(chains, chrom)
            else:
                figure = _grid_figure(grid, name.removeprefix("sensitivity_"), chrom)
            try:
                for fmt, path in paths[name].items():
                    metadata = {"Creator": "rare_segment_sensitivity_plots", "CreationDate": None, "ModDate": None} if fmt == "pdf" else {"Software": "rare_segment_sensitivity_plots"}
                    with _new_file(path) as handle:
                        figure.savefig(handle, format=fmt, dpi=dpi, metadata=metadata, facecolor="white")
            finally:
                figure.clear()
            manifest["figures"][name] = {fmt: path.name for fmt, path in paths[name].items()}
    with _new_file(manifest_path) as handle:
        handle.write((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-summary", required=True)
    parser.add_argument("--distance-histogram", required=True)
    parser.add_argument("--configuration-summary", required=True)
    parser.add_argument("--chain-histogram", help="Histogram of ALL chains before segment filters, including single-site chains")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chr", required=True, dest="chrom")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--prefix")
    args = parser.parse_args(argv)
    try:
        result = render_sensitivity_views(args.distance_summary, args.distance_histogram,
                args.configuration_summary, args.output_dir, args.chrom, dpi=args.dpi,
                formats=args.formats, prefix=args.prefix, chain_histogram=args.chain_histogram)
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"Sensitivity plotting failed: {error}\n")
    print(json.dumps({"chrom": result["chrom"], "n_effective_configurations": result["n_effective_configurations"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
