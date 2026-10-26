#!/usr/bin/env python3
"""Build one chromosome's map-supported cell axis without reading genotypes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import shutil
import tempfile

REQUEST_SCHEMA = "r02_autosomal_cell_axis_request_v1"
RECEIPT_SCHEMA = "r02_autosomal_cell_axis_receipt_v1"
HEADER = ["cell_index", "chromosome", "position_bp", "query_cm", "left_edge_cm",
          "right_edge_cm", "score_weight_cm", "query_valid"]
HISTORICAL_HEADER = HEADER[1:]
BASE_FIELDS = {"schema", "chromosome", "map_sha256", "grid_implementation_sha256",
               "max_step_cm", "max_queries", "source_binding"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_request(request):
    require(isinstance(request, dict), "request must be a JSON object")
    require(set(request) in (BASE_FIELDS, BASE_FIELDS | {"reference_grid_sha256"}),
            "request fields differ")
    require(request["schema"] == REQUEST_SCHEMA, "request schema differs")
    require(isinstance(request["chromosome"], str) and request["chromosome"] in {str(i) for i in range(1, 23)},
            "chromosome must be a string from 1 through 22")
    require(_digest_ok(request["map_sha256"]), "map_sha256 differs")
    require(_digest_ok(request["grid_implementation_sha256"]), "grid implementation SHA-256 differs")
    step = request["max_step_cm"]
    require(type(step) in (int, float) and math.isfinite(step) and 0 < step <= .01,
            "max_step_cm must be finite, positive, and at most 0.01 cM")
    require(type(request["max_queries"]) is int and 1 < request["max_queries"] <= 100000,
            "max_queries must be an integer from 2 through 100000")
    require(isinstance(request["source_binding"], dict) and request["source_binding"],
            "source_binding must be a nonempty metadata object")
    if "reference_grid_sha256" in request:
        require(_digest_ok(request["reference_grid_sha256"]), "reference grid SHA-256 differs")
    return request


def read_map(path, chromosome):
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        require(next(reader, None) == ["chrom", "pos", "cm"], "map header must be chrom,pos,cm")
        points = []
        for line_number, row in enumerate(reader, 2):
            require(len(row) == 3, f"map row {line_number} has the wrong columns")
            require(row[0].removeprefix("chr") == chromosome, f"map row {line_number} chromosome differs")
            try:
                position, cm = int(row[1]), float(row[2])
            except ValueError as error:
                raise ValueError(f"map row {line_number} is not numeric") from error
            require(position > 0 and math.isfinite(cm), f"map row {line_number} is invalid")
            points.append((position, cm))
    require(len(points) > 1, "map needs at least two rows")
    require(all(a[0] < b[0] for a, b in zip(points, points[1:])), "map positions are not strictly increasing")
    require(all(a[1] <= b[1] for a, b in zip(points, points[1:])), "map cM values decrease")
    return points


def load_grid(path, expected_sha256):
    require(sha256(path) == expected_sha256, "grid implementation changed")
    spec = importlib.util.spec_from_file_location("r02_frozen_query_grid", path)
    require(spec is not None and spec.loader is not None, "grid implementation cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "grid", None)), "grid implementation lacks grid()")
    function_sha = hashlib.sha256(inspect.getsource(module.grid).encode()).hexdigest()
    return module.grid, function_sha


def cell_index(cm, edges):
    require(type(cm) in (int, float) and math.isfinite(cm), "cell coordinate must be finite")
    require(edges, "cell edges are empty")
    matches = []
    for index, edge in enumerate(edges):
        require(len(edge) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in edge),
                "cell edges must be finite pairs")
        left, right = edge
        require(left <= right, "cell edge order differs")
        if left <= cm < right or (index == len(edges) - 1 and left <= cm <= right):
            matches.append(index)
    require(len(matches) == 1, "coordinate is outside the axis or belongs to multiple cells")
    return matches[0]


def _read_cells(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames in (HEADER, HISTORICAL_HEADER), "reference grid header differs")
        reference_format = ("AUTOSOMAL_CELL_AXIS_WITH_CELL_INDEX"
                            if reader.fieldnames == HEADER else "HISTORICAL_QUERY_GRID_ORDER_INDEXED")
        return list(reader), reference_format


def compare_reference(path, rows):
    reference, reference_format = _read_cells(path)
    require(len(reference) == len(rows), "reference grid cell count differs")
    maximum = 0.0
    float_fields = ("query_cm", "left_edge_cm", "right_edge_cm", "score_weight_cm")
    for expected, observed in zip(reference, rows):
        expected_index = int(expected["cell_index"]) if "cell_index" in expected else observed[0]
        require(expected_index == observed[0] and expected["chromosome"] == observed[1]
                and int(expected["position_bp"]) == observed[2]
                and expected["query_valid"].lower() == str(observed[7]).lower(),
                "reference grid discrete values differ")
        for field, value in zip(float_fields, observed[3:7]):
            parsed = float(expected[field])
            require(math.isfinite(parsed), "reference grid has a nonfinite value")
            difference = abs(parsed - value); maximum = max(maximum, difference)
            require(difference <= 1e-12, "reference grid floating values differ")
    return maximum, reference_format


def _reuse(output, identities):
    require(output.is_dir(), "output path exists and is not a directory")
    require({p.name for p in output.iterdir()} == {"cells.tsv", "receipt.json"},
            "existing output is partial or conflicting")
    receipt = json.loads((output / "receipt.json").read_text())
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("status") == "COMPLETE",
            "existing receipt is incomplete")
    for key, value in identities.items():
        require(receipt.get(key) == value, f"existing receipt {key} differs")
    require(receipt.get("cells_tsv_sha256") == sha256(output / "cells.tsv"),
            "existing cells.tsv changed")
    return receipt


def build(request_path, map_path, implementation_path, output, reference_path=None):
    request_sha = sha256(request_path)
    request = validate_request(json.loads(Path(request_path).read_text()))
    require(sha256(map_path) == request["map_sha256"], "map changed")
    require((reference_path is None) == ("reference_grid_sha256" not in request),
            "reference grid and request binding differ")
    reference_sha = None
    if reference_path is not None:
        reference_sha = sha256(reference_path)
        require(reference_sha == request["reference_grid_sha256"], "reference grid changed")
    grid, function_sha = load_grid(implementation_path, request["grid_implementation_sha256"])
    code_sha = sha256(__file__)
    identities = {"request_sha256": request_sha, "map_sha256": request["map_sha256"],
                  "code_sha256": code_sha, "grid_implementation_sha256": request["grid_implementation_sha256"],
                  "grid_function_sha256": function_sha, "reference_grid_sha256": reference_sha}
    output = Path(output)
    if output.exists():
        return _reuse(output, identities), True
    points = read_map(map_path, request["chromosome"])
    raw = grid(points, points[0][0], points[-1][0], float(request["max_step_cm"]), request["max_queries"])
    rows = [(i, request["chromosome"], int(row[0]), *map(float, row[1:5]), bool(row[5]))
            for i, row in enumerate(raw)]
    maximum, reference_format = (compare_reference(reference_path, rows)
                                 if reference_path is not None else (None, None))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        cells = temporary / "cells.tsv"
        with cells.open("x", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(HEADER); writer.writerows(rows)
        receipt = {"schema": RECEIPT_SCHEMA, "status": "COMPLETE", **identities,
                   "cells_tsv_sha256": sha256(cells),
                   "chromosome": request["chromosome"], "cell_count": len(rows),
                   "map_interval_bp": [points[0][0], points[-1][0]],
                   "map_interval_cm": [points[0][1], points[-1][1]],
                   "max_step_cm": float(request["max_step_cm"]), "max_queries": request["max_queries"],
                   "source_binding": request["source_binding"], "reference_max_abs_difference": maximum,
                   "reference_grid_format": reference_format,
                   "scope": "GEOMETRY_ONLY_NO_GENOTYPES_NO_TRAINING", "truth_read": False,
                   "evaluate_read": False, "map_supported_means_callability": False,
                   "rare_alleles_or_phase_modified": False}
        (temporary / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        # Claim the destination without replacing even an empty directory created
        # concurrently. Moving the receipt last makes every partial state fail closed.
        output.mkdir(exist_ok=False)
        (temporary / "cells.tsv").replace(output / "cells.tsv")
        (temporary / "receipt.json").replace(output / "receipt.json")
        temporary.rmdir()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True); parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--grid-implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--reference-grid", type=Path)
    args = parser.parse_args()
    receipt, reused = build(args.request, args.map, args.grid_implementation, args.output_dir, args.reference_grid)
    print(json.dumps({"status": "REUSED" if reused else "CREATED", "receipt": str(args.output_dir / "receipt.json"),
                      "cells_tsv_sha256": receipt["cells_tsv_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
