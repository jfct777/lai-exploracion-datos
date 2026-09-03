#!/usr/bin/env python3
"""Verify the exact config order intended for the M37 Google Batch launch."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "m37-r0-compact-sweep-20260903c"
WORK_DIR = f"gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/{RUN_ID}"
RECOMMENDED_CONFIG_ORDER = (
    "m37_trace_compact_sweep.config",
    "m37_trace_gcp.config",
    "m37_r0_compact_sweep.config",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract_binding(
    manifest_path: Path,
    parent_contract_path: Path,
    amendment_path: Path,
) -> None:
    """Reject a launch when the frozen manifest no longer binds exact contracts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    binding = manifest.get("contract_binding", {})
    observed_parent = _sha256(parent_contract_path)
    observed_amendment = _sha256(amendment_path)
    if (
        binding.get("parent_sha256") != observed_parent
        or binding.get("amendment_sha256") != observed_amendment
        or amendment.get("parent_contract_sha256") != observed_parent
    ):
        raise AssertionError(
            "M37 compact manifest/parent/amendment hash binding differs"
        )


def validate_effective_config(observed: str) -> None:
    """Reject an effective launch config with a null or displaced run namespace."""
    required = (
        "process.executor = 'google-batch'",
        "process.resourceLabels.team = 'frank'",
        "params.m37_results_dir = 'gs://teams-usp/frank/lai-exploracion-datos/runs'",
        f"params.m37_run_id = '{RUN_ID}'",
        f"workDir = '{WORK_DIR}'",
        "@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99",
    )
    missing = [value for value in required if value not in observed]
    null_namespace = any(
        value in observed
        for value in ("params.m37_run_id = null", "/work/nextflow/null")
    )
    if missing or null_namespace:
        raise AssertionError(
            "M37 compact Google Batch config preflight differs: "
            f"missing={missing}, null_namespace={null_namespace}; "
            "required config order is base,GCP,run-overlay"
        )


def main() -> None:
    validate_contract_binding(
        ROOT / "conf" / "m37_trace_compact_candidates.json",
        ROOT / "conf" / "m37_trace_sweep_contract.json",
        ROOT / "conf" / "m37_trace_compact_sweep_amendment.json",
    )
    if not shutil.which("nextflow"):
        raise RuntimeError("Nextflow is required for the compact M37 config preflight")
    configs = ",".join(
        str(ROOT / "conf" / name) for name in RECOMMENDED_CONFIG_ORDER
    )
    completed = subprocess.run(
        ["nextflow", "-C", configs, "config", "-flat",
         str(ROOT / "workflows/m37_trace_compact_sweep.nf")],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    validate_effective_config(completed.stdout)
    print("PASS_M37_COMPACT_GCP_CONFIG")


if __name__ == "__main__":
    main()
