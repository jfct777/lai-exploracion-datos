#!/usr/bin/env python3
"""Derive the exact M35D R1 truth subset selected by both paired predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m34_parse_flare_truth as deterministic


PREDICTION_MEMBERS = {
    "sample_key_sha256", "marker_pos", "marker_cM", "ancestry_names", "probabilities",
}
TRUTH_MEMBERS = {"sample_key_sha256", "marker_pos", "labels"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def axis_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_npz(path: Path, members: set[str]) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), f"invalid M35D input: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == members, f"M35D NPZ inventory differs: {path.name}")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def derive(args: argparse.Namespace) -> dict[str, object]:
    require(not args.output.exists() and not args.receipt.exists(),
            "refusing to overwrite M35D truth-subset outputs")
    direct = load_npz(args.direct_prediction, PREDICTION_MEMBERS)
    flare2 = load_npz(args.flare2_prediction, PREDICTION_MEMBERS)
    truth = load_npz(args.truth, TRUTH_MEMBERS)
    for name in ("sample_key_sha256", "marker_pos", "marker_cM", "ancestry_names"):
        require(np.array_equal(direct[name], flare2[name]),
                f"paired M35D prediction axis differs: {name}")
    require(np.array_equal(direct["sample_key_sha256"], truth["sample_key_sha256"]),
            "M35D prediction/truth sample axes differ")
    source_pos = truth["marker_pos"]
    retained_pos = direct["marker_pos"]
    require(source_pos.ndim == retained_pos.ndim == 1 and
            len(source_pos) == 42986 and len(retained_pos) == 42732 and
            np.all(np.diff(source_pos.astype(np.int64)) > 0) and
            np.all(np.diff(retained_pos.astype(np.int64)) > 0),
            "M35D R1 marker counts/order differ")
    indexes = np.searchsorted(source_pos, retained_pos)
    require(np.all(indexes < len(source_pos)) and
            np.array_equal(source_pos[indexes], retained_pos) and
            len(np.unique(indexes)) == len(indexes),
            "M35D prediction markers are not an exact ordered truth subset")
    excluded = np.ones(len(source_pos), dtype=bool)
    excluded[indexes] = False
    require(int(excluded.sum()) == 254, "M35D excluded marker count differs")
    subset = {
        "sample_key_sha256": truth["sample_key_sha256"],
        "marker_pos": source_pos[indexes],
        "labels": truth["labels"][:, :, indexes],
    }
    deterministic.write_deterministic_npz(args.output, subset)
    deterministic.reopen_npz(args.output, subset)
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M35D_R1_COMMON_AXIS_TRUTH_DERIVATION",
        "status": "PASS_M35D_R1_TRUTH_SUBSET_EXACT_COMMON_AXIS",
        "source_marker_count": int(len(source_pos)),
        "retained_marker_count": int(len(retained_pos)),
        "excluded_marker_count": int(excluded.sum()),
        "source_marker_axis_sha256": axis_sha256(source_pos),
        "retained_marker_axis_sha256": axis_sha256(retained_pos),
        "excluded_marker_axis_sha256": axis_sha256(source_pos[excluded]),
        "retained_index_axis_sha256": axis_sha256(indexes.astype("<u8")),
        "sample_axis_sha256": axis_sha256(truth["sample_key_sha256"]),
        "input_sha256": {
            "full_R1_truth": sha256_file(args.truth),
            "FLARE_F0_SAME_69_prediction": sha256_file(args.direct_prediction),
            "FLARE2_NATWGS_FINE_SAME_69_prediction": sha256_file(args.flare2_prediction),
        },
        "output_sha256": sha256_file(args.output),
        "comparison_policy": "score_both_methods_on_the_exact_same_retained_marker_axis",
        "canonical_M34_F0_policy": "full_42986_marker_context_only_no_cross_axis_delta",
        "R2_referenced": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-prediction", type=Path, required=True)
    parser.add_argument("--flare2-prediction", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = derive(parse_args())
    print(json.dumps({"status": result["status"]}, sort_keys=True))
