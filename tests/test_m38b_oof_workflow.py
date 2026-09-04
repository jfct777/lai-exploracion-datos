from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class M38BOofWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = (ROOT / "workflows/m38b_r0_oof_models.nf").read_text()
        cls.modules = (ROOT / "modules/38B_OOF_MODELS.nf").read_text()
        cls.config = (ROOT / "conf/m38b_r0_oof_models.config").read_text()

    def test_duplicate_fold_keys_are_broadcast_not_joined(self) -> None:
        self.assertIn(".combine(M38B_PARTITION_TRUTH.out.bundle, by: 0)", self.workflow)
        self.assertNotIn(".join(M38B_PARTITION_TRUTH.out.bundle, by: 0)", self.workflow)

    def test_fanout_contract_is_exact(self) -> None:
        # 15 positive materializations; real RE/SHAM generate 6 partitions;
        # 6 analytic + 18 real TCN + 45 positive TCN = 69 fits.
        self.assertEqual(len(re.findall(r"tuple\([012], 'POS_d", self.workflow)), 15)
        self.assertIn("arm != 'RD'", self.workflow)
        self.assertEqual(self.workflow.count("[1103, 2207, 3301]"), 2)
        self.assertIn("M38B_FINAL_DECISION", self.workflow)
        trainer = (ROOT / "bin/m37_trace_train.py").read_text()
        self.assertIn("update + 1 >= train_event_count", trainer)
        self.assertIn("best_checkpoint_update >= train_event_count", trainer)
        self.assertIn("zero_support_updates == 0", trainer)

    def test_runtime_scope_and_primary_artifacts_are_frozen(self) -> None:
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos/runs", self.config)
        self.assertIn("--network none --user", self.config)
        self.assertIn("resourceLabels = [team: 'frank'", self.config)
        self.assertIn("m38b_oof_train_max_forks = 8", self.config)
        for digest in (
            "2768cfbfac7f31b8c2e094a507188275ffeb47ceeb6dc0583c77df064e42c2d0",
            "80ace3480cc97a8d1e9c990c6285971dc4094f375a0654137084d7a2d6e445df",
            "77f38d26c04bfe7f84be3dfa73e88e9625f1729c75980d12260a8cf840cac397",
            "a82ced908f16dfbe8e40edddeae32aedff02a357d4c35cbba1fb7afa08437079",
            "6b3ea98243cef084098d7675603154d496b0c07c2e742cf87a1028aa54517233",
            "4b92a83a7e68ae9b843533e56d4ff1b6c624005ca475aa2003a8f8901485786b",
        ):
            self.assertIn(digest, self.config)
        self.assertIn("VALID and TEST inputs are forbidden", self.workflow)
        self.assertIn("provenanceSources", self.workflow)
        self.assertIn("--source '${it}'", self.modules)

    def test_materialization_receipts_have_distinct_staging_names(self) -> None:
        """The same primary receipt can fill two roles without a staging collision."""
        factor_alias = "path(factorsReceipt, stageAs: 'receipts/factors.receipt.json')"
        reference_alias = (
            "path(referenceReceipt, stageAs: 'receipts/reference.receipt.json')"
        )
        self.assertIn(factor_alias, self.modules)
        self.assertIn(reference_alias, self.modules)
        self.assertNotIn(
            "path(factorsReceipt),\n          path(referenceReceipt)", self.modules
        )

    @unittest.skipUnless(shutil.which("nextflow"), "Nextflow is not installed")
    def test_nextflow_accepts_one_receipt_in_both_materialization_roles(self) -> None:
        """Exercise the production process, which source previews do not stage."""
        with tempfile.TemporaryDirectory(prefix="m38b-stage-smoke-") as tmp:
            root = Path(tmp)
            shared = root / "m38b_primary_factor_subset.receipt.json"
            other = root / "m38b_strict_sham.reference.receipt.json"
            inputs = {
                "selected": root / "selected.npz",
                "target": root / "target.npz",
                "reference": root / "reference.npz",
                "f0": root / "f0.npz",
                "axis": root / "axis.npz",
                "axis_receipt": root / "axis.receipt.json",
                "source": root / "source.py",
            }
            for path in (shared, other, *inputs.values()):
                path.write_text("{}\n", encoding="utf-8")
            workflow = root / "staging_smoke.nf"
            workflow.write_text(
                f"""nextflow.enable.dsl=2

include {{ M38B_MATERIALIZE_ARM }} from '{ROOT / 'modules/38B_OOF_MODELS.nf'}'

workflow {{
    shared = file(params.shared, checkIfExists: true)
    other = file(params.other, checkIfExists: true)
    selected = file(params.selected, checkIfExists: true)
    target = file(params.target, checkIfExists: true)
    reference = file(params.reference, checkIfExists: true)
    f0 = file(params.f0, checkIfExists: true)
    axis = file(params.axis, checkIfExists: true)
    axisReceipt = file(params.axis_receipt, checkIfExists: true)
    sources = Channel.value([file(params.source, checkIfExists: true)])
    rows = Channel.of(
        tuple('RE', selected, target, reference, shared, shared, f0, axis, axisReceipt),
        tuple('RD', selected, target, reference, shared, shared, f0, axis, axisReceipt),
        tuple('SHAM', selected, target, reference, shared, other, f0, axis, axisReceipt),
    )
    M38B_MATERIALIZE_ARM(rows, sources)
}}
""",
                encoding="utf-8",
            )
            arguments = [
                "nextflow", "run", str(workflow), "-stub-run", "-ansi-log", "false",
                "--shared", str(shared), "--other", str(other),
                "--m38b_oof_results_dir", str(root / "results"),
                "--m38b_oof_run_id", "staging-smoke",
                "--m38b_oof_materialize_max_forks", "3",
                "--m38b_oof_beta_prior_strength", "0.5",
            ]
            for name, path in inputs.items():
                arguments.extend([f"--{name}", str(path)])
            result = subprocess.run(
                arguments,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("input file name collision", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
