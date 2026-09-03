#!/usr/bin/env python3
"""Seed the FLARE2 Gaussian-mixture builder prospectively and run it unchanged."""

from __future__ import annotations

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
    if args.seed < 0:
        raise ValueError("M35B GMM seed must be non-negative")
    if args.n_ancestries != 3:
        raise ValueError("M35B is frozen to three ancestry clusters")
    np.random.seed(args.seed)
    sys.argv = [args.builder, str(args.n_ancestries), args.panels, args.output_prefix]
    runpy.run_path(args.builder, run_name="__main__")


if __name__ == "__main__":
    main()
