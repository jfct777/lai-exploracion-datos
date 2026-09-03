#!/usr/bin/env python3
"""Deterministic adapter for the vendored FLARE create_model_file.py builder."""

import argparse
import runpy
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--builder", required=True)
    parser.add_argument("n_ancestries", type=int)
    parser.add_argument("panels")
    parser.add_argument("output_prefix")
    args = parser.parse_args()
    if args.seed != 3401103:
        raise ValueError("M35 builder adapter seed must equal canonical M34 seed 3401103")
    np.random.seed(args.seed)
    sys.argv = [args.builder, str(args.n_ancestries), args.panels, args.output_prefix]
    runpy.run_path(args.builder, run_name="__main__")


if __name__ == "__main__":
    main()
