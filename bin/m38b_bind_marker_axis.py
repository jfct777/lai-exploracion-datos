#!/usr/bin/env python3
"""Adapt the authenticated M38B alignment receipt to the M37 marker binder.

M38B deliberately reuses the historical, tested marker-axis implementation.
This adapter validates the new upstream receipt and creates the narrow joint
source receipt expected by M37; it does not broaden M37's accepted stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from m33_safe_bridge_core import write_exclusive_json
from m37_bind_marker_axis import bind_marker_axis
from m37_trace_core import require


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt_and_bind(f0: Path, marker_cm: Path, alignment_receipt: Path,
                   output: Path, adapter_receipt: Path,
                   expected_f0_sha256: str | None = None,
                   expected_marker_cm_sha256: str | None = None,
                   expected_alignment_receipt_sha256: str | None = None) -> dict[str, object]:
    require(expected_f0_sha256 is None or sha256(f0) == expected_f0_sha256,
            "M38B pinned F-minus hash differs")
    require(expected_marker_cm_sha256 is None or sha256(marker_cm) == expected_marker_cm_sha256,
            "M38B pinned marker-cM hash differs")
    require(expected_alignment_receipt_sha256 is None or
            sha256(alignment_receipt) == expected_alignment_receipt_sha256,
            "M38B pinned alignment receipt hash differs")
    receipt = json.loads(alignment_receipt.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == "M38B_ALIGN_FULL_AND_TRUTH_TO_F_MINUS_S660"
        and receipt.get("decision") == "PASS_EXACT_SHARED_F_MINUS_S660_SCORING_GRID"
        and receipt.get("scope", {}).get("target_partition") == "FIT"
        and receipt.get("scope", {}).get("valid_opened") is False
        and receipt.get("scope", {}).get("test_opened") is False,
        "M38B alignment receipt differs",
    )
    outputs = receipt.get("outputs", {})
    inputs = receipt.get("inputs", {})
    require(
        inputs.get("fminus_f0") == sha256(f0)
        and outputs.get(marker_cm.name, {}).get("sha256") == sha256(marker_cm),
        "M38B F-minus-S660/marker-cM hashes differ from alignment receipt",
    )
    marker_count = int(receipt.get("counts", {}).get("F_minus_S660", -1))
    require(marker_count == 42326, "M38B common marker count differs")
    adapter = {
        "schema_version": "1.0.0",
        "stage": "M37_MARKER_AXIS_SOURCE",
        "status": "PASS_M38B_TO_M37_MARKER_AXIS_ADAPTER",
        "marker_count": marker_count,
        "outputs": {
            f0.name: {"sha256": sha256(f0)},
            marker_cm.name: {"sha256": sha256(marker_cm)},
        },
        "m38b_alignment_receipt_sha256": sha256(alignment_receipt),
        "scope": {"chromosome": "22", "root": "R0", "partition": "FIT"},
    }
    write_exclusive_json(adapter_receipt, adapter)
    result = bind_marker_axis(f0, marker_cm, output, adapter_receipt)
    require(result.get("marker_count") == marker_count,
            "bound M38B marker count differs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--marker-cm", type=Path, required=True)
    parser.add_argument("--alignment-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--expected-f0-sha256", required=True)
    parser.add_argument("--expected-marker-cm-sha256", required=True)
    parser.add_argument("--expected-alignment-receipt-sha256", required=True)
    args = parser.parse_args()
    result = adapt_and_bind(
        args.f0, args.marker_cm, args.alignment_receipt,
        args.output, args.adapter_receipt, args.expected_f0_sha256,
        args.expected_marker_cm_sha256, args.expected_alignment_receipt_sha256,
    )
    print(json.dumps({"status": result["status"],
                      "marker_count": result["marker_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
