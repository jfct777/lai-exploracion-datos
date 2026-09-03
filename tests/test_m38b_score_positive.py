#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m38b_score_positive as subject  # noqa: E402
from m38b_score_oof import sha256_file  # noqa: E402


class M38BPositiveScorerTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, dict[str, Path], dict[str, Path]]:
        people = np.asarray([f"p{i:03d}" for i in range(96)], dtype="S64")
        folds = np.asarray([str(i % 3) for i in range(96)])
        pos = np.asarray([100, 200, 300, 400], dtype=np.int64)
        cm = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
        truth_labels = np.zeros((96, 4), dtype=np.uint8)
        truth = root / "truth.npz"
        np.savez(truth, sample_key_sha256=people, fold_ids=folds, marker_pos=pos,
                 marker_cM=cm, state_labels=truth_labels)
        provenance = {
            "model_contract_receipt_sha256": "c" * 64,
            "base_contract_sha256": "b" * 64,
            "amendment_sha256": "a" * 64,
            "amendment_2_sha256": "d" * 64,
            "folds_sha256": "f" * 64,
            "folds_receipt_sha256": "e" * 64,
        }
        truth_receipt = root / "truth.receipt.json"
        truth_receipt.write_text(json.dumps({
            "stage": "M38B_PACK_OOF_SCORE_TRUTH",
            "status": "PASS_TRUTH_SEPARATE_SCORING_BRANCH",
            "output_sha256": sha256_file(truth),
            **provenance,
        }), encoding="utf-8")
        predictions, receipts = {}, {}
        for logical_id, delta in subject.DELTA_IDS.items():
            confidence = 0.2 if delta == 0 else min(0.2 + delta * 0.2, 0.9)
            probability = np.full((96, 4, 6), (1.0 - confidence) / 5.0, dtype=np.float32)
            probability[:, :, 0] = confidence
            path = root / f"{logical_id}.npz"
            np.savez(path, probabilities=probability, sample_key_sha256=people,
                     fold_ids=folds, marker_pos=pos, marker_cM=cm,
                     state_names=np.asarray(["AA", "AE", "AN", "EE", "EN", "NN"]),
                     family=np.asarray(["tcn"]), arm=np.asarray(["POSITIVE"]),
                     seed_values=np.asarray([1103, 2207, 3301], dtype=np.int64),
                     positive_delta=np.asarray([delta], dtype=np.float64))
            receipt = root / f"{logical_id}.receipt.json"
            receipt.write_text(json.dumps({
                "stage": "M38B_COLLECT_DIAGNOSTIC_POSITIVE_OOF",
                "status": "PASS_DIAGNOSTIC_TRUTH_ENCODED_POSITIVE_CONTROL",
                "family": "tcn", "arm": "POSITIVE", "diagnostic_only": True,
                "positive_control_delta": delta, "seeds": [1103, 2207, 3301],
                "real_event_identity_sha256": "1" * 64,
                "real_event_masks_sha256": "2" * 64,
                "output_sha256": sha256_file(path), **provenance,
            }), encoding="utf-8")
            predictions[logical_id], receipts[logical_id] = path, receipt
        return truth, truth_receipt, predictions, receipts

    def test_string_fold_ids_are_scored_as_three_exact_folds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth, truth_receipt, predictions, receipts = self.make_fixture(root)
            output = root / "score.json"
            argv = ["m38b_score_positive.py", "--truth", str(truth),
                    "--truth-receipt", str(truth_receipt),
                    "--bootstrap-replicates", "100", "--output", str(output)]
            for logical_id in subject.DELTA_IDS:
                argv += ["--prediction", f"{logical_id}={predictions[logical_id]}",
                         "--prediction-receipt", f"{logical_id}={receipts[logical_id]}"]
            with mock.patch.object(sys, "argv", argv):
                subject.main()
            observed = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(observed["capacity_gate"]["pass"])
            self.assertTrue(all(len(row["fold_mean_deltas"]) == 3
                                for row in observed["contrasts"].values()))

    def test_physical_marker_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth, _truth_receipt, predictions, receipts = self.make_fixture(root)
            with np.load(predictions["POS_d1"], allow_pickle=False) as archive:
                payload = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
            payload["marker_pos"][-1] += 1
            predictions["POS_d1"].unlink()
            np.savez(predictions["POS_d1"], **payload)
            document = json.loads(receipts["POS_d1"].read_text(encoding="utf-8"))
            document["output_sha256"] = sha256_file(predictions["POS_d1"])
            receipts["POS_d1"].write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "identity differs"):
                subject.load_positive(
                    predictions["POS_d1"], receipts["POS_d1"], "POS_d1",
                    (np.asarray([f"p{i:03d}" for i in range(96)], dtype="S64"),
                     np.asarray([str(i % 3) for i in range(96)]),
                     np.asarray([0.0, 0.1, 0.2, 0.3]),
                     np.asarray([100, 200, 300, 400], dtype=np.int64)),
                )

    def test_event_identity_drift_across_positive_deltas_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth, truth_receipt, predictions, receipts = self.make_fixture(root)
            altered = json.loads(receipts["POS_d1"].read_text(encoding="utf-8"))
            altered["real_event_masks_sha256"] = "9" * 64
            receipts["POS_d1"].write_text(json.dumps(altered), encoding="utf-8")
            output = root / "score.json"
            argv = ["m38b_score_positive.py", "--truth", str(truth),
                    "--truth-receipt", str(truth_receipt),
                    "--bootstrap-replicates", "100", "--output", str(output)]
            for logical_id in subject.DELTA_IDS:
                argv += ["--prediction", f"{logical_id}={predictions[logical_id]}",
                         "--prediction-receipt", f"{logical_id}={receipts[logical_id]}"]
            with mock.patch.object(sys, "argv", argv), \
                    self.assertRaisesRegex(Exception, "does not share real event identity"):
                subject.main()


if __name__ == "__main__":
    unittest.main()
