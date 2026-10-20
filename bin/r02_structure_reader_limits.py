#!/usr/bin/env python3
"""Measure conservative STRUCTURE preparation row limits from frozen metadata.

This reads chromosome manifests and genetic maps only.  It does not open a
genotype chunk, target, EVALUATE record, or model.  The resulting limits bound
the whole-chunk rows that the preparation reader may encounter for a one-base
core; the tiler then keeps every larger core under the same limit.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

AUTOSOMES = tuple(str(value) for value in range(1, 23))
HEX64 = __import__("re").compile(r"[0-9a-f]{64}")
SCHEMA = "r02_structure_reader_row_limits_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_unique(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + str(path))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError("nonfinite JSON value: " + token)))


def map_points(path, chromosome):
    rows = []
    with Path(path).open() as stream:
        require(stream.readline().rstrip("\n") == "chrom\tpos\tcm", "map header differs")
        for line in stream:
            chrom, bp, cm = line.split()
            require(chrom == chromosome, "map chromosome differs")
            rows.append((int(bp), float(cm)))
    require(rows and all(a[0] < b[0] and a[1] <= b[1] for a, b in zip(rows, rows[1:])),
            "map order differs")
    return rows


def interpolate(points, position):
    index = bisect.bisect_left(points, (position, -math.inf))
    if points[index][0] == position:
        return points[index][1]
    left, right = points[index - 1], points[index]
    return left[1] + (position - left[0]) * (right[1] - left[1]) / (right[0] - left[0])


def inverse(points, cms, value, side):
    index = bisect.bisect_left(cms, value)
    if cms[index] == value:
        edge = index if side == "left" else bisect.bisect_right(cms, value) - 1
        return float(points[edge][0])
    left, right = points[index - 1], points[index]
    return left[0] + (value - left[1]) * (right[0] - left[0]) / (right[1] - left[1])


def bounds(points, cms, position, halo):
    center = interpolate(points, position)
    lower = max(cms[0], center - halo)
    upper = min(cms[-1], center + halo)
    return math.floor(inverse(points, cms, lower, "left")), math.ceil(inverse(points, cms, upper, "right"))


def channel_index(chunks):
    starts, ends, prefix = [], [], [0]
    for chunk in chunks:
        starts.append(chunk["start_bp"]); ends.append(chunk["end_bp"])
        prefix.append(prefix[-1] + chunk["n_sites"])
    return starts, ends, prefix


def upper_sites(index, lower, upper):
    starts, ends, prefix = index
    first = bisect.bisect_left(ends, lower)
    after = bisect.bisect_right(starts, upper)
    return prefix[after] - prefix[first]


def measure(requests_path, requests_sha256, store_root, output):
    require(HEX64.fullmatch(requests_sha256 or "") and sha256(requests_path) == requests_sha256,
            "request catalogue differs")
    requests = load_unique(requests_path)
    require(requests.get("request_count") == 528 and requests.get("context_count") == 6,
            "request universe differs")
    stores, store_descriptors = {}, []
    root = Path(store_root).resolve()
    for chromosome in AUTOSOMES:
        directory = root / ("chr" + chromosome + ".neural-input-validation") / "store"
        manifest_path, map_path = directory / "manifest.json", directory / "map.tsv"
        manifest = load_unique(manifest_path)
        require(str(manifest.get("chromosome")) == chromosome and
                set(manifest.get("channels", {})) == {"common", "rare"}, "store manifest differs")
        points = map_points(map_path, chromosome); cms = [row[1] for row in points]
        channels = {kind: channel_index(manifest["channels"][kind]) for kind in ("common", "rare")}
        stores[chromosome] = (manifest, points, cms, channels)
        store_descriptors.append({"chromosome": chromosome,
                                  "manifest_sha256": sha256(manifest_path),
                                  "map_sha256": sha256(map_path)})
    results = []
    for row in requests["contexts"]:
        context, context_sha = row["request_context"], row["context_sha256"]
        event_halo = (context["rare_radius_cm"] or 0) * context["attention_depth"]
        halos = {"common": event_halo + context["common_radius_cm"], "rare": event_halo}
        maxima = {kind: {"rows": 0} for kind in halos}
        for chromosome, (manifest, points, cms, channels) in stores.items():
            candidates = {points[0][0], points[-1][0]}
            for chunks in manifest["channels"].values():
                for chunk in chunks:
                    candidates.update((max(points[0][0], chunk["start_bp"] - 1),
                                       min(points[-1][0], chunk["end_bp"])))
            for position in candidates:
                for kind, halo in halos.items():
                    lower, upper = bounds(points, cms, position, halo)
                    rows_count = upper_sites(channels[kind], lower, upper)
                    if rows_count > maxima[kind]["rows"]:
                        maxima[kind] = {"rows": rows_count, "chromosome": chromosome,
                                        "core_bp": position, "read_bounds_bp": [lower, upper]}
        required = max(item["rows"] for item in maxima.values())
        # 100,000 preserves byte-identical compatible v6 tile inventories.
        effective = max(100000, required)
        results.append({"context_sha256": context_sha, "request_context": context,
                        "max_point_whole_chunk_upper": maxima,
                        "required_max_window_sites": required,
                        "effective_max_window_sites": effective})
    require(len(results) == 6 and len({row["context_sha256"] for row in results}) == 6,
            "six unique context limits required")
    receipt = {"schema": SCHEMA,
        "status": "COMPLETE_METADATA_BOUNDS_ALL_6_CONTEXTS_22_AUTOSOMES_BOTH_CHANNELS",
        "method": "MAX_WHOLE_INTERSECTING_CHUNK_ROWS_OVER_ALL_CHUNK_EDGE_POINT_CORES",
        "source_requests": {"path": str(Path(requests_path).resolve()), "sha256": requests_sha256},
        "stores": store_descriptors, "contexts": results,
        "channels": ["common", "rare"], "autosomes": 22,
        "genotype_chunks_opened": 0, "individual_genotypes_read": False,
        "targets_read": False, "evaluate_opened": False, "training_started": False}
    Path(output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True); parser.add_argument("--requests-sha256", required=True)
    parser.add_argument("--store-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = measure(args.requests, args.requests_sha256, args.store_root, args.output)
    print(json.dumps({"schema": receipt["schema"], "status": receipt["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
