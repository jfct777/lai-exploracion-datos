#!/usr/bin/env python3
"""Faithful, pseudonymised single-chromosome views of M14 segment records.

This module does not detect, merge, filter, cluster or interpret segments as
ancestry/IBD. The API accepts a pandas DataFrame or a TSV[.gz] path. Each input
interval becomes one pairwise bar and two individual bars. ``all_samples``
defines the pseudonym universe; only people with input segments are plotted.
Pseudonyms are stable for the same universe, not across changing universes.

Coordinates are 1-based inclusive. A base-centred rectangle spans
``[start_pos - 0.5, end_pos + 0.5]``, preserving width ``end-start+1``.
The private label map is written with mode 0600. Do not publish that map.
Existing destination files are never overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from collections import defaultdict
from numbers import Integral, Real


REQUIRED_COLUMNS = (
    "chrom", "sample_a", "sample_b", "start_pos", "end_pos",
    "length_bp", "n_shared_variants",
)
BAR_COLUMNS = (
    "record_id", "chrom", "sample", "partner", "row_index", "lane_index",
    "lane_count", "start_pos", "end_pos", "length_bp", "n_shared_variants",
)
BLUE = "#286D93"


def _integer(value, field, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer >= {minimum}")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real) and math.isfinite(value) and float(value).is_integer():
        result = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        result = int(value)
    else:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    if result < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return result


def _chrom(value):
    text = str(value)
    if text.lower().startswith("chr"):
        text = text[3:]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text) or text.lower() in {"nan", "none"}:
        raise ValueError("chrom must be a nonempty chromosome identifier")
    return text


def _sample(value):
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\t\r\n"):
        raise ValueError("Sample identifiers must be nonempty strings without control separators")
    return value


def _load_segments(segment_df, chrom):
    import pandas as pd

    if isinstance(segment_df, (str, os.PathLike)):
        frame = pd.read_csv(segment_df, sep="\t", dtype=str, keep_default_na=False)
    elif isinstance(segment_df, pd.DataFrame):
        frame = segment_df.copy(deep=True)
    else:
        raise TypeError("segment_df must be a pandas DataFrame or TSV path")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate input column names")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Missing segment columns: " + ", ".join(sorted(missing)))
    records, seen = [], set()
    for row in frame.to_dict("records"):
        if _chrom(row["chrom"]) != chrom:
            raise ValueError("Input must contain exactly the requested chromosome")
        a, b = sorted((_sample(row["sample_a"]), _sample(row["sample_b"])))
        if a == b:
            raise ValueError("Self-pair segments are not allowed")
        start = _integer(row["start_pos"], "start_pos", 1)
        end = _integer(row["end_pos"], "end_pos", 1)
        length = _integer(row["length_bp"], "length_bp", 1)
        count = _integer(row["n_shared_variants"], "n_shared_variants", 1)
        if end < start or length != end - start + 1:
            raise ValueError("Segment length must equal end_pos-start_pos+1 and be positive")
        key = (a, b, start, end)
        if key in seen:
            raise ValueError("Duplicate unordered pair/interval record")
        seen.add(key)
        records.append(dict(sample_a=a, sample_b=b, start_pos=start, end_pos=end,
                            length_bp=length, n_shared_variants=count, chrom=chrom))
    return sorted(records, key=lambda r: (r["sample_a"], r["sample_b"], r["start_pos"], r["end_pos"]))


def _lanes(records):
    """Keep overlapping intervals distinct within a logical row; never merge."""
    ends, assignments = [], []
    for row in sorted(records, key=lambda r: (r["start_pos"], r["end_pos"], r["record_id"])):
        lane = next((i for i, end in enumerate(ends) if end < row["start_pos"]), len(ends))
        if lane == len(ends):
            ends.append(row["end_pos"])
        else:
            ends[lane] = row["end_pos"]
        assignments.append((row, lane))
    return [(row, lane, len(ends)) for row, lane in assignments]


def _prepare_views(records, samples):
    width = max(4, len(str(len(samples))))
    mapping = {sample: f"S{i:0{width}d}" for i, sample in enumerate(samples, 1)}
    pairs = defaultdict(list)
    for i, record in enumerate(records, 1):
        a, b = mapping[record["sample_a"]], mapping[record["sample_b"]]
        pairs[a, b].append({k: v for k, v in dict(record, record_id=f"SEG{i:06d}").items()
                           if k not in {"sample_a", "sample_b"}})
    pair_rows, pair_bars, individual_groups = [], [], defaultdict(dict)
    for row_index, ((a, b), intervals) in enumerate(sorted(pairs.items())):
        pair_rows.append({"row_index": row_index, "sample": a, "partner": b})
        for record, lane, lanes in _lanes(intervals):
            pair_bars.append(dict(record, sample=a, partner=b, row_index=row_index,
                                  lane_index=lane, lane_count=lanes))
        individual_groups[a][b] = intervals
        individual_groups[b][a] = intervals
    individual_rows, individual_bars = [], []
    for sample, partners in sorted(individual_groups.items()):
        for partner, intervals in sorted(partners.items()):
            index = len(individual_rows)
            individual_rows.append({"row_index": index, "sample": sample, "partner": partner})
            for record, lane, lanes in _lanes(intervals):
                individual_bars.append(dict(record, sample=sample, partner=partner,
                                            row_index=index, lane_index=lane, lane_count=lanes))
    return mapping, {"pairwise_segments": (pair_rows, pair_bars),
                     "individual_segments": (individual_rows, individual_bars)}


def _new_file(path, private=False):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    return os.fdopen(fd, "wb")


def _tsv_bytes(rows, fields):
    import io
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in rows)
    return handle.getvalue().encode("utf-8")


def _make_figure(kind, rows, bars, chrom, chromosome_length, maximum, n_segments, n_people):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    # Increase canvas height, never shrink fonts or omit individual/pair labels.
    heights = [max(1, max((b["lane_count"] for b in bars if b["row_index"] == i), default=1))
               for i in range(len(rows))]
    centers, cursor = [], 0
    for height in heights:
        centers.append(cursor + height / 2)
        cursor += height
    figure = Figure(figsize=(13.5, max(4.2, 1.9 + 0.18 * cursor)), facecolor="white")
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    figure.subplots_adjust(left=0.19, right=0.985, top=1 - 0.8 / figure.get_figheight(),
                           bottom=0.8 / figure.get_figheight())
    active_sample = None
    band_number = 0
    band_start = 0
    for i, row in enumerate(rows + [{"sample": None}]):
        sample = row["sample"]
        if kind == "individual_segments" and sample != active_sample:
            if active_sample is not None:
                lo = centers[band_start] - heights[band_start] / 2
                hi = centers[i - 1] + heights[i - 1] / 2
                ax.axhspan(lo, hi, facecolor="#F1F3F4" if band_number % 2 == 0 else "white", zorder=0)
                ax.axhline(hi, color="#C3C8CC", linewidth=0.45, zorder=1)
                band_number += 1
            active_sample, band_start = sample, i
    for bar in bars:
        index = bar["row_index"]
        y = centers[index] - heights[index] / 2 + bar["lane_index"] + 0.18
        patch = Rectangle((bar["start_pos"] - 0.5, y), bar["length_bp"], 0.64,
                          facecolor=BLUE, edgecolor="#174760", linewidth=0.35, zorder=3)
        patch.set_gid(bar["record_id"])
        ax.add_patch(patch)
    ax.set_yticks(centers)
    separator = " / " if kind == "pairwise_segments" else "  with  "
    ax.set_yticklabels([r["sample"] + separator + r["partner"] for r in rows], fontsize=8)
    ax.set_ylim(max(cursor, 1), 0)
    upper = chromosome_length if chromosome_length is not None else max(maximum, 1)
    ax.set_xlim(0.5, upper + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}"))
    ax.set_xlabel(f"Chromosome {chrom} position (Mb; input coordinates 1-based inclusive)", fontsize=10)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", color="#DCE0E3", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    title = "Pairwise rare-sharing segments" if kind == "pairwise_segments" else "Individual rare-sharing segments by partner"
    subtitle = (f"chr{chrom} | {n_segments} input intervals | {len(rows)} "
                + ("pairs" if kind == "pairwise_segments" else "individual-partner rows")
                + f" | {n_people} active individuals | {len(bars)} bars")
    figure.suptitle(title + "\n" + subtitle, fontsize=12, y=1 - 0.10 / figure.get_figheight())
    extent_note = "Whole chromosome extent supplied." if chromosome_length is not None else "Extent ends at last supplied interval; chromosome length not supplied."
    figure.text(0.19, 0.12 / figure.get_figheight(),
                "One bar per input interval" + (" per participant" if kind == "individual_segments" else "")
                + "; no merging, communities or ancestry inferred.\n" + extent_note, fontsize=8, color="#444444")
    if not rows:
        ax.text(0.5, 0.5, "No input segments", transform=ax.transAxes, ha="center", va="center")
    return figure


def render_segment_views(segment_df, output_dir, chrom, all_samples=None, dpi=300,
                         formats=("png", "pdf"), chromosome_length=None, prefix=None):
    """Write both segment views, anonymous row ledgers and a private ID map.

    Returns the JSON-serialisable manifest also saved alongside the figures.
    All target names are checked before writing; existing files, including
    symlinks, raise FileExistsError. No overwrite option is intentionally offered.
    Input DataFrames and original segment files are never modified.
    """
    chrom = _chrom(chrom)
    dpi = _integer(dpi, "dpi", 1)
    formats = (formats,) if isinstance(formats, str) else tuple(formats)
    if not formats or len(set(formats)) != len(formats) or set(formats) - {"png", "pdf"}:
        raise ValueError("formats must contain distinct png and/or pdf values")
    prefix = f"chr{chrom}" if prefix is None else prefix
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("prefix must be a safe filename stem, not a path")
    records = _load_segments(segment_df, chrom)
    active = {row[key] for row in records for key in ("sample_a", "sample_b")}
    if isinstance(all_samples, (str, bytes)):
        raise ValueError("all_samples must be an iterable of sample identifiers, not a string")
    supplied = list(active) if all_samples is None else [_sample(s) for s in all_samples]
    if len(supplied) != len(set(supplied)) or not active <= set(supplied):
        raise ValueError("all_samples must be unique and include every segment participant")
    samples = sorted(supplied)
    maximum = max((row["end_pos"] for row in records), default=0)
    if chromosome_length is not None:
        chromosome_length = _integer(chromosome_length, "chromosome_length", 1)
        if maximum > chromosome_length:
            raise ValueError("A segment extends beyond chromosome_length")
    mapping, views = _prepare_views(records, samples)
    directory = Path(output_dir)
    figures = {kind: {fmt: directory / f"{prefix}.{kind}.{fmt}" for fmt in formats} for kind in views}
    ledgers = {kind: directory / f"{prefix}.{kind}.rows.tsv" for kind in views}
    private_path = directory / f"{prefix}.sample_labels.private.tsv"
    manifest_path = directory / f"{prefix}.segment_views.manifest.json"
    targets = [private_path, manifest_path, *ledgers.values(),
               *(p for paths in figures.values() for p in paths.values())]
    if any(os.path.lexists(path) for path in targets):
        raise FileExistsError("A segment-view destination already exists; use a new prefix or directory")
    directory.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema_version": "1.0", "chrom": chrom, "n_segments": len(records),
        "n_pairs": len(views["pairwise_segments"][0]), "n_active_individuals": len(active),
        "n_samples_in_label_universe": len(samples), "n_samples_without_input_segments": len(samples) - len(active),
        "coordinates": "1-based inclusive; rectangles [start-0.5,end+0.5]",
        "chromosome_length": chromosome_length, "last_segment_end": maximum,
        "row_order": "lexicographic sample ID universe -> S0001...; pairs sorted; individual bands sorted then partner",
        "pseudonym_scope": "same all_samples universe; only active participants plotted",
        "source_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_sample_map": private_path.name, "private_map_warning": "Contains original identifiers; do not publish",
        "interpretation": "Unmerged input intervals only; not ancestry, inferred communities or validated IBD",
        "dpi": dpi, "formats": list(formats), "views": {},
    }
    import matplotlib
    manifest["matplotlib_version"] = matplotlib.__version__
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                                "path.simplify": False, "text.usetex": False}):
        for kind, (rows, bars) in views.items():
            figure = _make_figure(kind, rows, bars, chrom, chromosome_length, maximum, len(records), len(active))
            try:
                for fmt, path in figures[kind].items():
                    metadata = {"Creator": "rare_segment_plots", "CreationDate": None, "ModDate": None} if fmt == "pdf" else {"Software": "rare_segment_plots"}
                    with _new_file(path) as handle:
                        figure.savefig(handle, format=fmt, dpi=dpi, metadata=metadata, facecolor="white")
            finally:
                figure.clear()
            with _new_file(ledgers[kind]) as handle:
                handle.write(_tsv_bytes(bars, BAR_COLUMNS))
            manifest["views"][kind] = {"n_rows": len(rows), "n_bars": len(bars), "rows": rows,
                                       "row_ledger": ledgers[kind].name,
                                       "figures": {fmt: path.name for fmt, path in figures[kind].items()}}
    with _new_file(private_path, private=True) as handle:
        handle.write(_tsv_bytes([{"label": label, "sample_id": sample} for sample, label in mapping.items()],
                                ("label", "sample_id")))
    with _new_file(manifest_path) as handle:
        handle.write((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def main(argv=None):
    """CLI used by the Nextflow wrapper; sample IDs are one identifier per line."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", required=True, help="Single-chromosome pairwise_segments TSV or TSV.gz")
    parser.add_argument("--sample-ids-file", help="Optional complete label universe: one sample identifier per line, no header")
    parser.add_argument("--chr", required=True, dest="chrom")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--chromosome-length", type=int)
    parser.add_argument("--prefix")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    args = parser.parse_args(argv)
    samples = None
    if args.sample_ids_file:
        with Path(args.sample_ids_file).open() as handle:
            samples = [line.rstrip("\r\n") for line in handle if line.rstrip("\r\n")]
    try:
        result = render_segment_views(args.segments, args.output_dir, args.chrom, all_samples=samples,
                dpi=args.dpi, formats=args.formats, chromosome_length=args.chromosome_length, prefix=args.prefix)
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"Segment plotting failed: {error}\n")
    print(json.dumps({key: result[key] for key in ("chrom", "n_segments", "n_pairs", "n_active_individuals")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
