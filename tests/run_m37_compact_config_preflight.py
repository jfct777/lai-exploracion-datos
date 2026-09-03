#!/usr/bin/env python3
"""Verify the exact config order intended for the M37 Google Batch launch."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if not shutil.which("nextflow"):
        raise RuntimeError("Nextflow is required for the compact M37 config preflight")
    configs = ",".join(str(ROOT / "conf" / name) for name in (
        "m37_trace_compact_sweep.config",
        "m37_trace_gcp.config",
        "m37_r0_compact_sweep.config",
    ))
    completed = subprocess.run(
        ["nextflow", "-C", configs, "config", "-flat",
         str(ROOT / "workflows/m37_trace_compact_sweep.nf")],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    observed = completed.stdout
    required = (
        "process.executor = 'google-batch'",
        "process.resourceLabels.team = 'frank'",
        "params.m37_results_dir = 'gs://teams-usp/frank/lai-exploracion-datos/runs'",
        "workDir = 'gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/m37-r0-compact-sweep-20260903b'",
        "@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99",
    )
    missing = [value for value in required if value not in observed]
    if missing:
        raise AssertionError(f"M37 compact Google Batch config preflight differs: {missing}")
    print("PASS_M37_COMPACT_GCP_CONFIG")


if __name__ == "__main__":
    main()
