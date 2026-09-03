from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m38b_decide as subject  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


PROVENANCE = {
    "model_contract_receipt_sha256": "c" * 64,
    "base_contract_sha256": "b" * 64,
    "amendment_sha256": "a" * 64,
    "amendment_2_sha256": "d" * 64,
    "folds_sha256": "f" * 64,
    "folds_receipt_sha256": "e" * 64,
}


class M38BDecisionTests(unittest.TestCase):
    def write_family(self, root: Path, family: str) -> tuple[Path, Path]:
        path = root / f"{family}.json"
        path.write_text(json.dumps({
            "stage": "M38B_OOF_SCORE", "status": "PASS_SCORED", "family": family,
            "candidate_incremental_gate": {"pass": True},
            "secondary_gates": {
                "weighted_uniform_no_sign_reversal": {"pass": True},
                "no_statistically_clear_harm": {"pass": True},
                "deploy_improvement_over_full_flare": {"pass": True},
                "no_statistically_clear_harm_vs_full": {"pass": True},
            },
        }), encoding="utf-8")
        receipt = root / f"{family}.receipt.json"
        receipt.write_text(json.dumps({
            "stage": "M38B_OOF_SCORE", "status": "PASS_SCORED", "family": family,
            "arms": ["RD", "RE", "SHAM", "full", "minus"],
            "output_sha256": digest(path), **PROVENANCE,
        }), encoding="utf-8")
        return path, receipt

    def run_decision(self, root: Path, capacity: bool) -> dict:
        analytic, analytic_receipt = self.write_family(root, "analytic")
        tcn, tcn_receipt = self.write_family(root, "tcn")
        positive = root / "positive.json"
        logical_ids = ["POS_d0", "POS_d0p25", "POS_d0p5", "POS_d1", "POS_d2"]
        positive.write_text(json.dumps({
            "stage": "M38B_SCORE_DIAGNOSTIC_POSITIVE_CONTROL", "family": "tcn",
            "logical_ids": logical_ids, "capacity_gate": {"pass": capacity},
        }), encoding="utf-8")
        positive_receipt = root / "positive.receipt.json"
        positive_receipt.write_text(json.dumps({
            "stage": "M38B_SCORE_DIAGNOSTIC_POSITIVE_CONTROL",
            "status": "PASS_DIAGNOSTIC_GRID_SCORED", "diagnostic_only": True,
            "family": "tcn", "logical_ids": logical_ids,
            "output_sha256": digest(positive), **PROVENANCE,
        }), encoding="utf-8")
        output = root / "decision.json"
        argv = ["m38b_decide.py", "--analytic", str(analytic),
                "--analytic-receipt", str(analytic_receipt), "--tcn", str(tcn),
                "--tcn-receipt", str(tcn_receipt), "--positive", str(positive),
                "--positive-receipt", str(positive_receipt), "--output", str(output)]
        with mock.patch.object(sys, "argv", argv):
            subject.main()
        return json.loads(output.read_text(encoding="utf-8"))

    def test_final_decision_combines_prespecified_gates_without_family_selection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = self.run_decision(Path(raw), True)
        self.assertTrue(result["families"]["analytic"]["incremental_information_supported"])
        self.assertTrue(result["families"]["tcn"]["incremental_information_supported"])
        self.assertFalse(result["families"]["tcn"]["family_selected"])

    def test_failed_positive_gate_makes_tcn_capacity_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = self.run_decision(Path(raw), False)
        self.assertEqual(result["families"]["tcn"]["status"], "CAPACITY_INCONCLUSIVE")
        self.assertFalse(result["families"]["tcn"]["improvement_over_full_flare_supported"])


if __name__ == "__main__":
    unittest.main()
