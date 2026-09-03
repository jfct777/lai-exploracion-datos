from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m37_trace_collect_metrics import ARMS, collect_metrics
import m37_trace_successive_halving as halving


def _paired_files(root: Path, omit: str | None = None) -> tuple[list[Path], list[Path]]:
    metrics: list[Path] = []
    receipts: list[Path] = []
    baseline = {"f1_boundary": {"0.2": .20}, "log_loss": .50}
    for index, arm in enumerate(ARMS):
        if arm == omit:
            continue
        metric = root / f"candidate.tcn.{arm}.metrics.json"
        payload = {
            "candidate_id": "candidate", "family": "tcn", "root": "R0", "arm": arm,
            "evaluation_split": "FIT_TUNE", "f1_boundary": {"0.2": .40 if arm == "RE" else .25 - index / 100},
            "log_loss": .30 if arm == "RE" else .35 + index / 100, "baseline": baseline,
        }
        metric.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        receipt = root / f"candidate.tcn.{arm}.metrics.receipt.json"
        receipt.write_text(json.dumps({
            "stage": "M37_TRACE_SCORE", "candidate_id": "candidate", "family": "tcn",
            "root": "R0", "arm": arm,
            "output_sha256": hashlib.sha256(metric.read_bytes()).hexdigest(),
        }, sort_keys=True) + "\n", encoding="utf-8")
        metrics.append(metric)
        receipts.append(receipt)
    return metrics, receipts


def test_collection_authenticates_complete_candidate_and_drives_promotion() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        metrics, receipts = _paired_files(root)
        rows, evidence = collect_metrics(list(reversed(metrics)), receipts, "R0", "FIT_TUNE")
        assert [row["arm"] for row in rows] == list(ARMS)
        assert len(evidence["metric_sha256"]) == len(ARMS)

        collection = root / "m37.R0.paired_metrics.json"
        collection.write_text(json.dumps({
            "schema_version": "1.0.0", "stage": "M37_TRACE_COLLECT_METRICS",
            "root": "R0", "evaluation_split": "FIT_TUNE", "rows": rows,
            "input_evidence": evidence,
        }, sort_keys=True) + "\n", encoding="utf-8")
        collection_receipt = root / "m37.R0.paired_metrics.receipt.json"
        collection_receipt.write_text(json.dumps({
            "stage": "M37_TRACE_COLLECT_METRICS", "root": "R0",
            "row_count": len(rows),
            "output_sha256": hashlib.sha256(collection.read_bytes()).hexdigest(),
        }, sort_keys=True) + "\n", encoding="utf-8")
        output = root / "m37.successive_halving.json"
        previous = sys.argv
        sys.argv = ["m37_trace_successive_halving.py", "--metrics-json", str(collection),
                    "--metrics-receipt", str(collection_receipt), "--keep-fraction", "1",
                    "--boundary-tolerance-cm", "0.2", "--minimum-f1-gain", "0",
                    "--maximum-log-loss-increase", "0", "--bootstrap-seed", "1103",
                    "--bootstrap-draws", "100", "--minimum-replication-roots", "3",
                    "--output", str(output)]
        try:
            halving.main()
        finally:
            sys.argv = previous
        plan = json.loads(output.read_text(encoding="utf-8"))
        plan_receipt = json.loads(output.with_suffix(".receipt.json").read_text(encoding="utf-8"))
        assert plan["root"] == "R0" and plan["ranked"][0]["promote"]
        assert plan_receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_collection_rejects_missing_arm_and_tampered_metric() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        metrics, receipts = _paired_files(root, omit="SHAM")
        try:
            collect_metrics(metrics, receipts, "R0", "FIT_TUNE")
        except ValueError as error:
            assert "complete paired arm" in str(error)
        else:
            raise AssertionError("an incomplete arm family must not be collected")

        metrics, receipts = _paired_files(root)
        metrics[0].write_text(metrics[0].read_text(encoding="utf-8") + " ", encoding="utf-8")
        try:
            collect_metrics(metrics, receipts, "R0", "FIT_TUNE")
        except ValueError as error:
            assert "hash differs" in str(error)
        else:
            raise AssertionError("a metric modified after scoring must be rejected")
