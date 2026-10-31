#!/usr/bin/env python3
"""Run the official FLARE 0.6.0 model builder with an explicit GMM seed."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import sklearn
import sklearn.mixture

from r02_flare2_model_io import parse_model, require, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--builder-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("n_ancestries", type=int)
    parser.add_argument("panels", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    require(0 <= args.seed < 2**32, "GMM seed must fit an unsigned 32-bit integer")
    require(args.n_ancestries > 1, "FLARE2 requires at least two ancestry clusters")
    require(args.builder.is_file() and sha256_file(args.builder) == args.builder_sha256,
            "official create_model_file.py hash differs")
    require(args.panels.is_file(), "FLARE panel-probability file is missing")
    require(not args.receipt.exists(), "refusing to overwrite builder receipt")

    original = sklearn.mixture.GaussianMixture
    fitted_models = []

    def seeded_gaussian_mixture(*positional, **keywords):
        require("random_state" not in keywords,
                "upstream builder unexpectedly supplies a GMM random_state")
        keywords["random_state"] = args.seed
        model = original(*positional, **keywords)
        fitted_models.append(model)
        return model

    np.random.seed(args.seed)
    sklearn.mixture.GaussianMixture = seeded_gaussian_mixture
    previous_argv = sys.argv
    namespace = {}
    try:
        sys.argv = [str(args.builder), str(args.n_ancestries), str(args.panels),
                    str(args.output_prefix)]
        namespace = runpy.run_path(str(args.builder), run_name="__main__")
    finally:
        for value in namespace.values():
            if isinstance(value, io.IOBase) and not value.closed:
                value.flush()
                value.close()
        sys.argv = previous_argv
        sklearn.mixture.GaussianMixture = original

    model_path = Path(str(args.output_prefix) + ".model")
    log_path = Path(str(args.output_prefix) + ".log")
    parsed = parse_model(model_path, args.n_ancestries)
    require(len(fitted_models) == 1 and hasattr(fitted_models[0], "converged_")
            and hasattr(fitted_models[0], "n_iter_"),
            "official builder did not expose one fitted GaussianMixture")
    receipt = {
        "schema": "r02_flare2_seeded_builder_receipt_v1",
        "status": "PASS_SEEDED_GMM_AND_NORMALIZED_CENTERS",
        "seed": args.seed,
        "upstream_builder_sha256": args.builder_sha256,
        "adapter_sha256": sha256_file(Path(__file__)),
        "panels_sha256": sha256_file(args.panels),
        "model_sha256": parsed["model_sha256"],
        "log_sha256": sha256_file(log_path),
        "p_row_sums": parsed["p_row_sums"],
        "normalization_tolerance": parsed["normalization_tolerance"],
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "target_truth_accessed": False,
        "gmm_fit": {"converged": bool(fitted_models[0].converged_),
                    "n_iter": int(fitted_models[0].n_iter_),
                    "metadata_status": "CAPTURED_FROM_FITTED_SKLEARN_OBJECT"},
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
