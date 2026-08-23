#!/usr/bin/env python3
"""Create an authenticated probability-only FLARE F0 artifact for M33."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import m33_safe_bridge_core as core


SCHEMA_ID = "m33_m0_flare_f0_sanitized_v1"
RECEIPT_SCHEMA_ID = "m33_m0_flare_f0_sanitized_receipt_v1"
ANCESTRIES = ("AFR", "EUR", "ASIA")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_auth(path: Path, sources: Mapping[str, Path], git_commit: str) -> str:
    require(len(git_commit) == 40 and set(git_commit) <= set("0123456789abcdef"),
            "git commit is not exact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(set(payload) == {"schema_version", "stage", "status", "implementation_commit", "files"},
            "source-auth keys differ")
    require(payload["schema_version"] == "1.0.0" and
            payload["stage"] == "M33_F0_SANITIZE_SOURCE_AUTH" and
            payload["status"] == "AUTHORIZED_EXACT_SANITIZER_SOURCES",
            "source-auth identity differs")
    require(payload["implementation_commit"] == git_commit, "source-auth commit differs")
    require(set(payload["files"]) == set(sources), "source-auth file inventory differs")
    for relative, staged in sources.items():
        require(sha256_file(staged) == payload["files"][relative],
                f"source-auth hash differs: {relative}")
    return sha256_file(path)


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(
        "r", encoding="utf-8", newline=""
    )


def _vcf_rows(path: Path) -> tuple[tuple[str, ...], list[tuple[int, int, str, str]], np.ndarray, dict[str, Any]]:
    ancestry_codes: dict[str, str] = {}
    samples: tuple[str, ...] | None = None
    loci: list[tuple[int, int, str, str]] = []
    marker_rows: list[np.ndarray] = []
    raw_min = float("inf")
    raw_max = float("-inf")
    vector_count = 0
    format_headers: set[str] = set()

    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##ANCESTRY=<"):
                body = line.strip()[len("##ANCESTRY=<"):-1]
                for token in body.split(","):
                    ancestry, code = token.split("=", 1)
                    require(code not in ancestry_codes, "duplicate FLARE ancestry code")
                    ancestry_codes[code] = ancestry
                continue
            if line.startswith("##FORMAT=<ID="):
                format_headers.add(line.split("ID=", 1)[1].split(",", 1)[0])
                continue
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                require(len(fields) > 9, "FLARE VCF has no TARGET samples")
                samples = tuple(fields[9:])
                require(all(samples) and len(samples) == len(set(samples)), "FLARE sample axis is invalid")
                require({"ANP1", "ANP2"}.issubset(format_headers), "FLARE ANP1/ANP2 headers are missing")
                continue
            require(samples is not None, "FLARE record precedes #CHROM header")
            require(ancestry_codes == {"0": "AFR", "1": "EUR", "2": "ASIA"},
                    "FLARE ancestry order differs from AFR/EUR/ASIA")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples), f"malformed FLARE row {line_number}")
            chrom_text = fields[0].removeprefix("chr")
            require(chrom_text == "22", "SANITIZE_FLARE_F0 accepts chr22 only")
            pos = int(fields[1])
            ref, alt = fields[3], fields[4]
            require(pos > 0 and len(ref) == 1 and len(alt) == 1 and "," not in alt,
                    "FLARE locus is not a chr22 biallelic SNV")
            locus = (22, pos, ref, alt)
            require(not loci or (pos, ref, alt) > (loci[-1][1], loci[-1][2], loci[-1][3]),
                    "FLARE marker axis is not strictly ordered or is duplicated")
            fmt = fields[8].split(":")
            require(len(fmt) == len(set(fmt)) and "ANP1" in fmt and "ANP2" in fmt,
                    "FLARE row lacks unique ANP1/ANP2 fields")
            indexes = (fmt.index("ANP1"), fmt.index("ANP2"))
            marker = np.empty((len(samples), 2, 3), dtype="<f8")
            for sample_index, raw_sample in enumerate(fields[9:]):
                values = raw_sample.split(":")
                require(len(values) == len(fmt), f"FORMAT/sample width differs at row {line_number}")
                for haplotype, field_index in enumerate(indexes):
                    tokens = values[field_index].split(",")
                    require(len(tokens) == 3, "ANP vector does not have three ancestries")
                    probability = np.asarray([float(value) for value in tokens], dtype="<f8")
                    require(np.all(np.isfinite(probability)) and np.all(probability >= 0.0),
                            "ANP vector is non-finite or negative")
                    total = float(probability.sum(dtype=np.float64))
                    require(0.98 <= total <= 1.02, "ANP vector sum lies outside [0.98,1.02]")
                    marker[sample_index, haplotype] = probability
                    raw_min = min(raw_min, total)
                    raw_max = max(raw_max, total)
                    vector_count += 1
            loci.append(locus)
            marker_rows.append(marker)

    require(samples is not None and marker_rows, "FLARE VCF is empty")
    marker_major = np.asarray(marker_rows, dtype="<f8")
    probabilities = core.sanitize_f0(marker_major.transpose(1, 2, 0, 3))
    return samples, loci, probabilities, {
        "raw_probability_vector_count": vector_count,
        "raw_probability_sum_min": raw_min,
        "raw_probability_sum_max": raw_max,
    }


def load_target_axis(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) >= {"sample_key_sha256"}, "TARGET artifact lacks sample_key_sha256")
        axis = np.asarray(archive["sample_key_sha256"])
    require(axis.ndim == 1 and axis.dtype == np.dtype("|S64") and axis.size > 0,
            "TARGET sample-key axis is invalid")
    require(len(set(axis.tolist())) == axis.size, "TARGET sample-key axis is duplicated")
    return np.ascontiguousarray(axis)


def axis_sha256(names: tuple[str, ...], arrays: tuple[np.ndarray, ...]) -> str:
    require(len(names) == len(arrays), "axis names/arrays differ")
    digest = hashlib.sha256()
    for name, array in zip(names, arrays):
        value = np.ascontiguousarray(array)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    require(not temporary.exists(), "temporary receipt path already exists")
    encoded = (json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def reopen_validate(path: Path, expected: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(expected), "sanitized archive members differ")
        for name, value in expected.items():
            observed = np.asarray(archive[name])
            require(observed.dtype == value.dtype and observed.shape == value.shape and
                    np.array_equal(observed, value), f"sanitized archive drift: {name}")


def run(flare_anc: Path, target_artifact: Path, output_dir: Path, root_seed: int,
        source_auth_sha256: str) -> dict[str, Any]:
    require(type(root_seed) is int and root_seed >= 0, "root seed is invalid")
    require(len(source_auth_sha256) == 64 and set(source_auth_sha256) <= set("0123456789abcdef"),
            "source-auth SHA-256 is invalid")
    require(output_dir.is_dir() and not output_dir.is_symlink() and not any(output_dir.iterdir()),
            "output directory must be an existing empty directory")
    samples, loci, probabilities, audit = _vcf_rows(flare_anc)
    sample_keys = np.asarray([core.sample_key(sample) for sample in samples], dtype="|S64")
    target_axis = load_target_axis(target_artifact)
    require(np.array_equal(sample_keys, target_axis), "FLARE and TARGET ordered sample axes differ")

    arrays = {
        "sample_key_sha256": np.ascontiguousarray(sample_keys),
        "marker_chrom": np.full(len(loci), 22, dtype="|u1"),
        "marker_pos": np.asarray([row[1] for row in loci], dtype="<i8"),
        "marker_ref": np.asarray([row[2].encode() for row in loci], dtype="|S1"),
        "marker_alt": np.asarray([row[3].encode() for row in loci], dtype="|S1"),
        "F0": np.ascontiguousarray(probabilities, dtype="<f4"),
    }
    artifact = output_dir / "flare_f0_sanitized.npz"
    core.write_deterministic_npz(artifact, arrays)
    reopen_validate(artifact, arrays)
    semantic = core.semantic_arrays_sha256(SCHEMA_ID, arrays)
    sample_axis_hash = axis_sha256(("sample_key_sha256",), (arrays["sample_key_sha256"],))
    marker_axis_hash = axis_sha256(
        ("marker_chrom", "marker_pos", "marker_ref", "marker_alt"),
        (arrays["marker_chrom"], arrays["marker_pos"], arrays["marker_ref"], arrays["marker_alt"]),
    )
    receipt = {
        "stage": "M33_SANITIZE_FLARE_F0",
        "schema_id": RECEIPT_SCHEMA_ID,
        "status": "PASS_PROBABILITY_ONLY_REOPENED",
        "root_seed": root_seed,
        "sample_count": len(samples),
        "marker_count": len(loci),
        "ancestry_order": list(ANCESTRIES),
        "source_flare_sha256": sha256_file(flare_anc),
        "source_target_artifact_sha256": sha256_file(target_artifact),
        "source_auth_sha256": source_auth_sha256,
        "sample_axis_semantic_sha256": sample_axis_hash,
        "marker_axis_semantic_sha256": marker_axis_hash,
        "artifact_raw_sha256": sha256_file(artifact),
        "artifact_semantic_sha256": semantic,
        "raw_probability_vector_count": audit["raw_probability_vector_count"],
        "raw_probability_sum_min": audit["raw_probability_sum_min"],
        "raw_probability_sum_max": audit["raw_probability_sum_max"],
        "contains_raw_genotypes": False,
        "contains_hard_calls": False,
        "contains_truth": False,
        "append_only": True,
        "reopen_verified": True,
    }
    receipt_path = output_dir / "flare_f0_sanitized.receipt.json"
    atomic_json(receipt_path, receipt)
    reopened = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(reopened == receipt, "sanitized receipt changed after reopen")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flare-anc", type=Path, required=True)
    parser.add_argument("--target-rare-diploid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    sources: dict[str, Path] = {}
    for item in args.source:
        require("=" in item, "--source must be relative=staged")
        relative, staged = item.split("=", 1)
        require(relative and relative not in sources, "duplicate or empty --source")
        sources[relative] = Path(staged)
    source_auth_sha256 = validate_source_auth(args.source_auth, sources, args.git_commit)
    receipt = run(args.flare_anc, args.target_rare_diploid, args.output_dir,
                  args.root_seed, source_auth_sha256)
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
