from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m36_cora_set", ROOT / "bin/m36_cora_set.py")
M36 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M36
SPEC.loader.exec_module(M36)


class M36CoraSetTests(unittest.TestCase):
    def fixture(self, name: str) -> Path:
        return ROOT / "tests/fixtures" / name

    def test_event_classes_keep_ac2_homalt_separate(self) -> None:
        rows = M36.read_tsv(self.fixture("m36_cora_events.tab"))
        by_event = {}
        for row in rows:
            by_event.setdefault(row["event_id"], []).append(row)
        self.assertEqual(M36.classify_event(by_event["evt_ac2_het"]), "AC2_HET")
        self.assertEqual(M36.classify_event(by_event["evt_ac2_homalt"]), "AC2_HOMALT")
        self.assertEqual(M36.classify_event(by_event["evt_mac4"]), "MAC3_10")

    def test_component_folds_do_not_split_components_and_exclude_cross_pairs(self) -> None:
        components = {"S1": "A", "S2": "A", "S3": "B", "S4": "B"}
        folds = M36.component_folds(components, 2)
        self.assertNotEqual(folds["A"], folds["B"])
        pairs = [
            {"sample_i": "S1", "sample_j": "S2"},
            {"sample_i": "S1", "sample_j": "S3"},
        ]
        observed = M36.pair_partition(pairs, components, folds)
        self.assertEqual(observed["assessment"], 1)
        self.assertEqual(observed["cross_fold_excluded"], 1)

    def test_successive_halving_is_finite_and_set_transformer_is_opt_in(self) -> None:
        plan = M36.successive_halving(("deep_sets",), (16, 64, 256), 2)
        self.assertEqual({row["family"] for row in plan}, {"deep_sets"})
        self.assertEqual([row["budget"] for row in plan if row["stage"] == 0], [16, 16])
        with_attention = M36.successive_halving(("deep_sets", "set_transformer"), (16, 64), 2)
        self.assertIn("set_transformer", {row["family"] for row in with_attention})

    def test_cli_smoke_rejects_same_chromosome_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "bad_targets.tsv"
            target.write_text(self.fixture("m36_cora_external_targets.tab").read_text().replace("chr21", "chr22"), encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "bin/m36_cora_set.py"),
                "--events", str(self.fixture("m36_cora_events.tab")),
                "--covariates", str(self.fixture("m36_cora_covariates.tab")),
                "--components", str(self.fixture("m36_cora_components.tab")),
                "--targets", str(target),
                "--contract", str(ROOT / "conf/m36_cora_set_preregistration.json"),
                "--feature-chrom", "chr22", "--model-families", "deep_sets",
                "--halving-budgets", "16,64", "--halving-eta", "2", "--n-folds", "2",
                "--smoke-only", "--outdir", str(Path(tmpdir) / "out"),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("target chromosome", result.stderr)

    def test_cli_smoke_writes_only_contract_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "out"
            command = [
                sys.executable, str(ROOT / "bin/m36_cora_set.py"),
                "--events", str(self.fixture("m36_cora_events.tab")),
                "--covariates", str(self.fixture("m36_cora_covariates.tab")),
                "--components", str(self.fixture("m36_cora_components.tab")),
                "--targets", str(self.fixture("m36_cora_external_targets.tab")),
                "--contract", str(ROOT / "conf/m36_cora_set_preregistration.json"),
                "--feature-chrom", "chr22", "--model-families", "deep_sets",
                "--halving-budgets", "16,64,256", "--halving-eta", "2", "--n-folds", "2",
                "--smoke-only", "--outdir", str(outdir),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((outdir / "m36_cora_smoke_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["event_class_counts"], {"AC2_HET": 1, "AC2_HOMALT": 1, "MAC3_10": 1})
            token_header = (outdir / "m36_cora_event_tokens.tsv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("mutation_context", token_header)
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["m14_as_truth"])


if __name__ == "__main__":
    unittest.main()
