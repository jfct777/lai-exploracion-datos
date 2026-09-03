#!/usr/bin/env python3
"""Focused tests for the frozen M35D R1 experiment."""

from __future__ import annotations

import importlib.util
import gzip
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m35d", ROOT / "bin" / "m35d_natwgs_fine_r1.py")
M35D = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M35D)
CONTRACT = ROOT / "conf" / "m35d_natwgs_fine_r1_contract.json"


def write_screen(directory: Path, selection: int, granularity: str, gmm: int,
                 passed: bool) -> None:
    directory.mkdir()
    evidence = {
        "selection_seed": selection,
        "granularity": granularity,
        "gmm_seed": gmm,
        "status": ("PASS_M35D_CLUSTER_SEPARATION" if passed else
                   "NO_GO_M35D_CLUSTER_SEPARATION"),
        "NAM_support": 0.55 if passed else 0.49,
        "log_margin": 1.0,
        "target_truth_opened": False,
        "R2_referenced": False,
    }
    evidence_path = directory / "m35d.cluster_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    receipt = {
        "truth_input_present": False,
        "final_inference_performed": False,
        "evidence_sha256": M35D.sha256_file(evidence_path),
    }
    (directory / "m35d.screen_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8")


class M35DTests(unittest.TestCase):
    def test_contract_freezes_fine_primary_and_r2(self) -> None:
        contract = M35D.load_contract(CONTRACT)
        self.assertEqual(contract["cluster_screen"]["primary_granularity"], "fine")
        self.assertEqual(contract["preassigned_final_pair"]["selection_seed"], 350101)
        self.assertEqual(contract["scope"]["reserved_root_policy"],
                         "R2_must_not_be_read_referenced_or_scored")

    def _aggregate(self, failed_key: tuple[int, str, int] | None):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        directories = []
        for selection in M35D.SELECTION_SEEDS:
            for granularity in ("fine", "coarse"):
                for gmm in M35D.GMM_SEEDS:
                    key = (selection, granularity, gmm)
                    directory = root / ("_".join(map(str, key)))
                    write_screen(directory, *key, passed=(key != failed_key))
                    directories.append(directory)
        args = type("Args", (), {
            "contract": CONTRACT,
            "screen_dir": directories,
            "output": root / "gate.json",
            "go_token": root / "token.json",
        })()
        return temp, M35D.aggregate(args), args

    def test_all_nine_fine_required_for_promotion(self) -> None:
        temp, result, args = self._aggregate(None)
        self.addCleanup(temp.cleanup)
        self.assertEqual(result["status"],
                         "PASS_M35D_FINE_9_OF_9_GO_PREASSIGNED_R1_FINAL")
        token = json.loads(args.go_token.read_text())
        self.assertEqual(token["granularity"], "fine")
        self.assertFalse(token["R2_allowed"])

    def test_one_fine_failure_stops_before_truth(self) -> None:
        temp, result, args = self._aggregate((350303, "fine", 353301))
        self.addCleanup(temp.cleanup)
        self.assertEqual(result["status"],
                         "NO_GO_M35D_FINE_NOT_9_OF_9_STOP_BEFORE_R1_TRUTH")
        self.assertFalse(args.go_token.exists())
        self.assertFalse(result["truth_opened"])

    def test_coarse_failure_is_diagnostic_only(self) -> None:
        temp, result, args = self._aggregate((350303, "coarse", 353301))
        self.addCleanup(temp.cleanup)
        self.assertEqual(result["primary"]["passed"], 9)
        self.assertEqual(result["diagnostic"]["passed"], 8)
        self.assertTrue(args.go_token.exists())

    def test_screen_workflow_is_truth_free_and_personal(self) -> None:
        workflow = (ROOT / "workflows" / "m35d_natwgs_fine_r1_screen.nf").read_text()
        module = (ROOT / "modules" / "35D_NATWGS_FINE_R1_SCREEN.nf").read_text()
        config = (ROOT / "conf" / "m35d_natwgs_fine_r1_screen.config").read_text()
        for forbidden in ("truth_npz", "m35d_truth", "truth.npz"):
            self.assertNotIn(forbidden, workflow.lower())
            self.assertNotIn(forbidden, module.lower())
        self.assertNotIn("gs://teams-usp/frank/lai-exploracion-datos/runs/m34-nam-128-r2",
                         workflow.lower())
        self.assertIn("forbids every r2 input", workflow.lower())
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos/", config)
        self.assertIn("team: 'frank'", config)

    def test_final_workflow_links_common_axis_receipt_and_forbids_r2(self) -> None:
        workflow = (ROOT / "workflows" / "m35d_natwgs_fine_r1_final.nf").read_text()
        module = (ROOT / "modules" / "35D_NATWGS_FINE_R1_FINAL.nf").read_text()
        runner = (ROOT / "bin" / "m35d_natwgs_fine_r1.py").read_text()
        subset = (ROOT / "bin" / "m35d_subset_truth.py").read_text()
        self.assertIn("m35d_subset_truth.py", workflow)
        self.assertIn("m35d.r1_common_axis_truth.receipt.json", module)
        self.assertIn("cross_axis_delta_to_M34_full_reference", runner)
        self.assertNotIn("delta_FLARE_F0_same_69_minus_M34_full_reference", runner)
        self.assertIn("np.searchsorted(source_pos, retained_pos)", subset)
        self.assertIn('"excluded_marker_count": int(excluded.sum())', subset)
        self.assertIn("forbids every R2 input", workflow)

    def test_direct_flare_data_dependent_order_is_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direct.vcf.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("##fileformat=VCFv4.2\n")
                handle.write("##ANCESTRY=<NAM=0,AFR=1,EUR=2>\n")
            mapping = M35D.ancestry_header_mapping(path)
            self.assertEqual(mapping["cluster_to_ancestry"],
                             {"0": "NAM", "1": "AFR", "2": "EUR"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
