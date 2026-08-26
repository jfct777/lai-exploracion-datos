#!/usr/bin/env python3
"""Convert FLARE probabilities and mosaic truth into separate M34 artifacts.

The ``f0`` command is truth-blind and writes only the probability baseline and
its marker genetic coordinates.  The ``truth`` command is separate, accepts
only FIT or VALID, and writes the exact label schema consumed by M34 training.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import m33_safe_bridge_core as core
import m34_generate_mosaics as mosaics


F0_MEMBERS = {
    "sample_key_sha256",
    "marker_chrom",
    "marker_pos",
    "marker_ref",
    "marker_alt",
    "F0",
}
TRUTH_MEMBERS = {"sample_key_sha256", "marker_pos", "labels"}
ANCESTRY_ALIASES = {
    "AFR": "AFR",
    "African": "AFR",
    "EUR": "EUR",
    "European": "EUR",
    "NAM": "NAM",
    "Native_American": "NAM",
}
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_output_directory(path: Path) -> None:
    require(not path.is_symlink(), "Output directory cannot be a symlink")
    require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
            "Output directory must be absent or empty")
    path.mkdir(parents=True, exist_ok=True)


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write uncompressed NPY members without buffering a full F0 copy."""
    require(not path.exists(), f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True
        ) as archive:
            for name in sorted(arrays):
                array = np.ascontiguousarray(arrays[name])
                require(array.dtype.kind != "O", "Object arrays are forbidden")
                member = zipfile.ZipInfo(f"{name}.npy", FIXED_ZIP_TIME)
                member.compress_type = zipfile.ZIP_STORED
                member.external_attr = 0o100444 << 16
                with archive.open(member, "w", force_zip64=True) as output:
                    np.lib.format.write_array(output, array, allow_pickle=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        os.chmod(path, 0o400)
    finally:
        if temporary.exists():
            temporary.unlink()


def reopen_npz(path: Path, expected: Mapping[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(expected), "NPZ member inventory differs")
        for name, wanted in expected.items():
            observed = archive[name]
            require(
                observed.dtype == wanted.dtype
                and observed.shape == wanted.shape
                and np.array_equal(observed, wanted),
                f"NPZ member differs after reopen: {name}",
            )


def parse_ancestry_order(value: str | Sequence[str]) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",")) if isinstance(value, str) else tuple(value)
    require(names == ("AFR", "EUR", "NAM"),
            "M34 ancestry order must be exactly AFR,EUR,NAM")
    return names


def parse_flare_id_map(value: str | Mapping[str, str]) -> dict[str, str]:
    if isinstance(value, str):
        parsed: dict[str, str] = {}
        for token in value.split(","):
            require("=" in token, "FLARE ID map must use 0=AFR,1=EUR,2=NAM")
            code, ancestry = (item.strip() for item in token.split("=", 1))
            require(code not in parsed, "FLARE ancestry ID is repeated")
            parsed[code] = ancestry
    else:
        parsed = {str(key): str(item) for key, item in value.items()}
    require(set(parsed.values()) == {"AFR", "EUR", "NAM"} and len(parsed) == 3,
            "FLARE ID map must cover AFR, EUR and NAM exactly once")
    try:
        numeric = sorted(int(code) for code in parsed)
    except ValueError:
        raise ValueError("FLARE ancestry IDs must be integers") from None
    require(numeric == list(range(len(parsed))),
            "FLARE ancestry IDs must be contiguous from zero")
    return parsed


def parse_ancestry_header(line: str) -> dict[str, str]:
    prefix = "##ANCESTRY=<"
    require(line.startswith(prefix) and line.rstrip().endswith(">"),
            "Malformed FLARE ancestry header")
    body = line.strip()[len(prefix):-1]
    result: dict[str, str] = {}
    for token in body.split(","):
        require(token.count("=") == 1, "Malformed FLARE ancestry header token")
        label, code = token.split("=", 1)
        require(code not in result and label, "Duplicate FLARE ancestry header ID")
        result[code] = label
    return result


def validate_header_mapping(header: Mapping[str, str], id_map: Mapping[str, str]) -> None:
    require(set(header) == set(id_map), "FLARE ancestry IDs differ from the declared map")
    for code, label in header.items():
        canonical = ANCESTRY_ALIASES.get(label)
        require(canonical is not None and canonical == id_map[code],
                f"FLARE ancestry label/ID mapping differs for ID {code}")


@dataclass(frozen=True)
class FlareAxis:
    samples: tuple[str, ...]
    loci: tuple[tuple[int, str, str], ...]
    ancestry_header: dict[str, str]


def scan_flare(path: Path, id_map: Mapping[str, str]) -> FlareAxis:
    samples: tuple[str, ...] | None = None
    loci: list[tuple[int, str, str]] = []
    ancestry_header: dict[str, str] | None = None
    format_headers: set[str] = set()
    with mosaics.open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##ANCESTRY=<"):
                require(ancestry_header is None, "Duplicate FLARE ancestry header")
                ancestry_header = parse_ancestry_header(line)
                continue
            if line.startswith("##FORMAT=<ID="):
                format_headers.add(line.split("ID=", 1)[1].split(",", 1)[0])
                continue
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#CHROM"):
                require(samples is None, "Duplicate FLARE #CHROM header")
                fields = line.rstrip("\n").split("\t")
                require(len(fields) > 9 and fields[8] == "FORMAT",
                        "FLARE VCF has no sample FORMAT axis")
                samples = tuple(fields[9:])
                require(all(samples) and len(samples) == len(set(samples)),
                        "FLARE sample axis is empty or duplicated")
                require({"AN1", "AN2", "ANP1", "ANP2"}.issubset(format_headers),
                        "FLARE ancestry FORMAT headers are incomplete")
                continue
            if line.startswith("#"):
                continue
            require(samples is not None, "FLARE record precedes #CHROM header")
            fields = line.rstrip("\n").split("\t", 9)
            require(len(fields) == 10, f"Malformed FLARE row {line_number}")
            require(mosaics.canonical_contig(fields[0]) == "22",
                    "M34 FLARE conversion accepts chr22 only")
            position = int(fields[1])
            ref, alt = fields[3].upper(), fields[4].upper()
            require(
                position > 0
                and len(ref) == len(alt) == 1
                and ref in "ACGT"
                and alt in "ACGT"
                and ref != alt,
                "FLARE marker is not a biallelic SNV",
            )
            require(not loci or position > loci[-1][0],
                    "FLARE marker positions are duplicated or out of order")
            fmt = fields[8].split(":")
            require(len(fmt) == len(set(fmt)) and
                    {"AN1", "AN2", "ANP1", "ANP2"}.issubset(fmt),
                    "FLARE record ancestry fields are incomplete or duplicated")
            loci.append((position, ref, alt))
    require(samples is not None and loci and ancestry_header is not None,
            "FLARE VCF lacks samples, markers or ancestry header")
    validate_header_mapping(ancestry_header, id_map)
    return FlareAxis(samples, tuple(loci), ancestry_header)


def parse_flare_probabilities(
    path: Path,
    axis: FlareAxis,
    ancestry_order: Sequence[str],
    id_map: Mapping[str, str],
) -> tuple[np.ndarray, dict[str, float | int]]:
    samples, markers, ancestries = len(axis.samples), len(axis.loci), len(ancestry_order)
    values = np.empty((samples, 2, markers, ancestries), dtype="<f4")
    output_index = {ancestry: index for index, ancestry in enumerate(ancestry_order)}
    raw_to_output = {
        int(code): output_index[ancestry] for code, ancestry in id_map.items()
    }
    raw_min, raw_max = math.inf, -math.inf
    vector_count = 0
    marker_index = 0
    with mosaics.open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            require(marker_index < markers and len(fields) == 9 + samples,
                    f"Malformed FLARE probability row {line_number}")
            require(
                (int(fields[1]), fields[3].upper(), fields[4].upper())
                == axis.loci[marker_index],
                "FLARE marker axis changed between passes",
            )
            fmt = fields[8].split(":")
            indexes = {name: fmt.index(name) for name in ("AN1", "AN2", "ANP1", "ANP2")}
            for sample_index, sample_field in enumerate(fields[9:]):
                tokens = sample_field.split(":")
                require(len(tokens) == len(fmt),
                        f"FORMAT/sample width differs at FLARE row {line_number}")
                for haplotype in (0, 1):
                    hard = tokens[indexes[f"AN{haplotype + 1}"]]
                    require(hard in id_map, "FLARE hard ancestry call is outside the ID map")
                    probability_tokens = tokens[indexes[f"ANP{haplotype + 1}"]].split(",")
                    require(len(probability_tokens) == ancestries,
                            "FLARE probability vector length differs")
                    try:
                        raw = np.asarray([float(item) for item in probability_tokens], dtype="<f8")
                    except ValueError:
                        raise ValueError("FLARE probability vector is non-numeric") from None
                    require(np.all(np.isfinite(raw)) and np.all(raw >= 0),
                            "FLARE probability vector is negative or non-finite")
                    total = float(raw.sum(dtype=np.float64))
                    require(0.98 <= total <= 1.02 and total > 0,
                            "Rounded FLARE probability mass lies outside [0.98,1.02]")
                    hard_index = int(hard)
                    require(abs(float(raw[hard_index]) - float(raw.max())) <= 1e-12,
                            "FLARE hard call is not an argmax of its probability vector")
                    normalized = raw / total
                    for raw_index, output_ancestry_index in raw_to_output.items():
                        values[sample_index, haplotype, marker_index, output_ancestry_index] = (
                            normalized[raw_index]
                        )
                    raw_min = min(raw_min, total)
                    raw_max = max(raw_max, total)
                    vector_count += 1
            marker_index += 1
    require(marker_index == markers, "FLARE marker axis is incomplete on the second pass")
    for start in range(0, markers, 4096):
        block = values[:, :, start:start + 4096, :]
        require(np.all(np.isfinite(block)) and np.all(block >= 0),
                "Normalized F0 contains invalid values")
        require(np.allclose(block.sum(axis=3), 1.0, rtol=0, atol=5e-6),
                "Normalized float32 F0 fails the simplex check")
    return np.ascontiguousarray(values), {
        "probability_vectors": vector_count,
        "raw_probability_sum_min": raw_min,
        "raw_probability_sum_max": raw_max,
    }


def output_descriptor(path: Path) -> dict[str, int | str]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def run_f0(args: argparse.Namespace) -> dict[str, Any]:
    ancestry_order = parse_ancestry_order(args.ancestry_order)
    id_map = parse_flare_id_map(args.flare_id_map)
    _prepare_output_directory(args.outdir)
    axis = scan_flare(args.flare_anc, id_map)
    genetic_map = mosaics.read_genetic_map(args.genetic_map, "22")
    require(axis.loci[0][0] >= genetic_map.start_bp and
            axis.loci[-1][0] <= genetic_map.end_bp,
            "Genetic map does not cover the complete FLARE marker axis")
    probabilities, probability_audit = parse_flare_probabilities(
        args.flare_anc, axis, ancestry_order, id_map
    )
    marker_positions = np.asarray([row[0] for row in axis.loci], dtype="<i8")
    marker_cm = np.asarray(
        [genetic_map.bp_to_cm(int(position)) for position in marker_positions], dtype="<f8"
    )
    require(np.all(np.isfinite(marker_cm)) and np.all(marker_cm[:-1] <= marker_cm[1:]),
            "Interpolated marker cM axis is invalid")
    arrays = {
        "sample_key_sha256": np.asarray(
            [core.sample_key(sample) for sample in axis.samples], dtype="|S64"
        ),
        "marker_chrom": np.full(len(axis.loci), 22, dtype="|u1"),
        "marker_pos": marker_positions,
        "marker_ref": np.asarray([row[1].encode("ascii") for row in axis.loci], dtype="|S1"),
        "marker_alt": np.asarray([row[2].encode("ascii") for row in axis.loci], dtype="|S1"),
        "F0": probabilities,
    }
    require(set(arrays) == F0_MEMBERS and arrays["F0"].shape ==
            (len(axis.samples), 2, len(axis.loci), len(ancestry_order)),
            "F0 schema or dimensions differ")
    marker_arrays = {"marker_cM": marker_cm}
    f0_path = args.outdir / "m34_f0.npz"
    marker_path = args.outdir / "marker_cM.npz"
    write_deterministic_npz(f0_path, arrays)
    write_deterministic_npz(marker_path, marker_arrays)
    reopen_npz(f0_path, arrays)
    reopen_npz(marker_path, marker_arrays)
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "M34_PARSE_FLARE_F0",
        "decision": "PASS_F0_TRUTH_BLIND",
        "ancestry_order": list(ancestry_order),
        "flare_id_map": id_map,
        "sample_count": len(axis.samples),
        "marker_count": len(axis.loci),
        "haplotype_count": 2,
        **probability_audit,
        "truth_opened": False,
        "contains_truth": False,
        "contains_raw_sample_ids": False,
        "inputs": {
            "flare_anc_sha256": sha256_file(args.flare_anc),
            "genetic_map_sha256": sha256_file(args.genetic_map),
        },
        "outputs": {
            f0_path.name: output_descriptor(f0_path),
            marker_path.name: output_descriptor(marker_path),
        },
    }
    receipt["semantic_sha256"] = canonical_json_sha256(receipt)
    receipt_path = args.outdir / "m34_f0.receipt.json"
    core.write_exclusive_json(receipt_path, receipt)
    return receipt


def load_f0_axes_without_probabilities(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), "F0 artifact path is invalid")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == F0_MEMBERS, "F0 member inventory differs")
        axes = {
            name: np.ascontiguousarray(archive[name])
            for name in ("sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref", "marker_alt")
        }
    samples = len(axes["sample_key_sha256"])
    markers = len(axes["marker_pos"])
    require(samples > 0 and markers > 0 and axes["sample_key_sha256"].dtype == np.dtype("|S64"),
            "F0 sample or marker axis is empty")
    require(len(set(axes["sample_key_sha256"].tolist())) == samples,
            "F0 sample axis is duplicated")
    require(all(axes[name].shape == (markers,)
                for name in ("marker_chrom", "marker_pos", "marker_ref", "marker_alt")),
            "F0 marker axes differ")
    require(np.all(axes["marker_chrom"] == 22) and
            np.all(axes["marker_pos"][:-1] < axes["marker_pos"][1:]),
            "F0 marker positions are invalid or out of order")
    return axes


@dataclass(frozen=True)
class TruthSegment:
    sample_id: str
    haplotype: int
    start_bp: int
    end_bp_exclusive: int
    ancestry: str


def load_truth_segments(path: Path, ancestry_order: Sequence[str]) -> dict[tuple[str, int], list[TruthSegment]]:
    expected = {"target_id", "haplotype", "chrom", "start_bp", "end_bp_exclusive", "ancestry"}
    grouped: dict[tuple[str, int], list[TruthSegment]] = {}
    with mosaics.open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and set(reader.fieldnames) == expected,
                "Truth segment member inventory differs")
        for line_number, row in enumerate(reader, 2):
            sample_id = row["target_id"]
            require(sample_id and sample_id == sample_id.strip(),
                    f"Invalid truth sample at row {line_number}")
            try:
                haplotype = int(row["haplotype"])
                start = int(row["start_bp"])
                end = int(row["end_bp_exclusive"])
            except ValueError:
                raise ValueError(f"Non-integer truth coordinate at row {line_number}") from None
            require(haplotype in (0, 1), "Truth haplotype must be zero or one")
            require(mosaics.canonical_contig(row["chrom"]) == "22",
                    "Truth contains a chromosome other than chr22")
            require(start > 0 and end > start, "Truth segment is empty or invalid")
            require(row["ancestry"] in ancestry_order, "Truth ancestry is outside the declared axis")
            segment = TruthSegment(sample_id, haplotype, start, end, row["ancestry"])
            key = (sample_id, haplotype)
            previous = grouped.setdefault(key, [])
            require(not previous or start >= previous[-1].start_bp,
                    "Truth segments are out of order within a haplotype")
            previous.append(segment)
    require(grouped, "Truth segment table is empty")
    return grouped


def align_truth(
    grouped: Mapping[tuple[str, int], list[TruthSegment]],
    sample_keys: np.ndarray,
    marker_positions: np.ndarray,
    ancestry_order: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    sample_by_key: dict[bytes, str] = {}
    for sample_id, _haplotype in grouped:
        key = core.sample_key(sample_id)
        require(key not in sample_by_key or sample_by_key[key] == sample_id,
                "Truth sample-key collision or duplicate identity")
        sample_by_key[key] = sample_id
    wanted_keys = [bytes(value) for value in sample_keys.tolist()]
    require(set(sample_by_key) == set(wanted_keys),
            "Truth and F0 sample axes are incomplete or different")
    ancestry_index = {name: index for index, name in enumerate(ancestry_order)}
    labels = np.empty((len(wanted_keys), 2, len(marker_positions)), dtype="|i1")
    common_extent: tuple[int, int] | None = None
    segment_count = 0
    for sample_index, sample_key in enumerate(wanted_keys):
        sample_id = sample_by_key[sample_key]
        for haplotype in (0, 1):
            segments = grouped.get((sample_id, haplotype))
            require(segments is not None and len(segments) > 0,
                    "Truth lacks one sample/haplotype axis")
            for left, right in zip(segments, segments[1:]):
                require(left.end_bp_exclusive == right.start_bp,
                        "Truth segments contain a gap or overlap")
            extent = (segments[0].start_bp, segments[-1].end_bp_exclusive)
            common_extent = extent if common_extent is None else common_extent
            require(extent == common_extent, "Truth haplotypes have different genomic extents")
            require(extent[0] <= int(marker_positions[0]) and
                    int(marker_positions[-1]) < extent[1],
                    "One or more F0 marker positions lie outside truth")
            pointer = 0
            for marker_index, position_value in enumerate(marker_positions):
                position = int(position_value)
                while pointer + 1 < len(segments) and position >= segments[pointer].end_bp_exclusive:
                    pointer += 1
                segment = segments[pointer]
                require(segment.start_bp <= position < segment.end_bp_exclusive,
                        "F0 marker position falls in a truth gap")
                labels[sample_index, haplotype, marker_index] = ancestry_index[segment.ancestry]
            segment_count += len(segments)
    return labels, {
        "truth_segments": segment_count,
        "truth_extent_start_bp": common_extent[0] if common_extent else 0,
        "truth_extent_end_bp_exclusive": common_extent[1] if common_extent else 0,
    }


def run_truth(args: argparse.Namespace) -> dict[str, Any]:
    ancestry_order = parse_ancestry_order(args.ancestry_order)
    require(args.role in {"FIT", "VALID"}, "Truth role must be FIT or VALID")
    truth_directory = args.outdir.resolve()
    f0_directory = args.f0.resolve().parent
    require(
        truth_directory != f0_directory
        and f0_directory not in truth_directory.parents
        and truth_directory not in f0_directory.parents,
        "Truth output must be physically separate from the F0 directory tree",
    )
    _prepare_output_directory(args.outdir)
    axes = load_f0_axes_without_probabilities(args.f0)
    with np.load(args.marker_cm, allow_pickle=False) as archive:
        require(set(archive.files) == {"marker_cM"}, "marker_cM member inventory differs")
        marker_cm = np.ascontiguousarray(archive["marker_cM"])
    require(marker_cm.dtype == np.dtype("<f8") and
            marker_cm.shape == axes["marker_pos"].shape and
            np.all(np.isfinite(marker_cm)) and np.all(marker_cm[:-1] <= marker_cm[1:]),
            "marker_cM axis differs from F0")
    grouped = load_truth_segments(args.truth_segments, ancestry_order)
    labels, truth_audit = align_truth(
        grouped, axes["sample_key_sha256"], axes["marker_pos"], ancestry_order
    )
    arrays = {
        "sample_key_sha256": np.ascontiguousarray(axes["sample_key_sha256"]),
        "marker_pos": np.ascontiguousarray(axes["marker_pos"], dtype="<i8"),
        "labels": np.ascontiguousarray(labels, dtype="|i1"),
    }
    require(set(arrays) == TRUTH_MEMBERS and arrays["labels"].shape ==
            (len(arrays["sample_key_sha256"]), 2, len(arrays["marker_pos"])),
            "Truth NPZ schema or dimensions differ")
    truth_path = args.outdir / "truth.npz"
    write_deterministic_npz(truth_path, arrays)
    reopen_npz(truth_path, arrays)
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "M34_ALIGN_MOSAIC_TRUTH",
        "decision": f"PASS_TRUTH_ALIGNED_{args.role}",
        "role": args.role,
        "ancestry_order": list(ancestry_order),
        "sample_count": len(arrays["sample_key_sha256"]),
        "marker_count": len(arrays["marker_pos"]),
        "haplotype_count": 2,
        **truth_audit,
        "contains_probabilities": False,
        "f0_probability_values_loaded": False,
        "test_role_permitted": False,
        "inputs": {
            "truth_segments_sha256": sha256_file(args.truth_segments),
            "f0_axes_source_sha256": sha256_file(args.f0),
            "marker_cM_sha256": sha256_file(args.marker_cm),
        },
        "outputs": {truth_path.name: output_descriptor(truth_path)},
    }
    receipt["semantic_sha256"] = canonical_json_sha256(receipt)
    core.write_exclusive_json(args.outdir / "truth.receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    f0 = subcommands.add_parser("f0", help="Create truth-blind F0 and marker cM artifacts")
    f0.add_argument("--flare-anc", type=Path, required=True)
    f0.add_argument("--genetic-map", type=Path, required=True)
    f0.add_argument("--ancestry-order", required=True)
    f0.add_argument("--flare-id-map", required=True)
    f0.add_argument("--outdir", type=Path, required=True)

    truth = subcommands.add_parser("truth", help="Align sealed FIT/VALID segment truth")
    truth.add_argument("--truth-segments", type=Path, required=True)
    truth.add_argument("--f0", type=Path, required=True)
    truth.add_argument("--marker-cm", type=Path, required=True)
    truth.add_argument("--ancestry-order", required=True)
    truth.add_argument("--role", required=True, choices=("FIT", "VALID"))
    truth.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = run_f0(args) if args.command == "f0" else run_truth(args)
    print(json.dumps({"decision": receipt["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
