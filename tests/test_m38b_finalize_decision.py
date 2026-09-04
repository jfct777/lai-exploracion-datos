from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m38b_finalize_decision as subject  # noqa: E402


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


class M38BFinalizeDecisionTests(unittest.TestCase):
    def write_family(self, root: Path, family: str) -> tuple[Path, Path]:
        path = root / f"m38b.{family}.metrics.json"
        path.write_text(json.dumps({
            "stage": "M38B_OOF_SCORE", "status": "PASS_SCORED", "family": family,
            "candidate_incremental_gate": {"pass": False},
            "secondary_gates": {
                "weighted_uniform_no_sign_reversal": {"pass": True},
                "no_statistically_clear_harm": {"pass": True},
                "deploy_improvement_over_full_flare": {"pass": False},
                "no_statistically_clear_harm_vs_full": {"pass": True},
            },
        }), encoding="utf-8")
        receipt = root / f"m38b.{family}.metrics.receipt.json"
        receipt.write_text(json.dumps({
            "stage": "M38B_OOF_SCORE", "status": "PASS_SCORED", "family": family,
            "arms": ["RD", "RE", "SHAM", "full", "minus"],
            "output_sha256": digest(path), **PROVENANCE,
        }), encoding="utf-8")
        return path, receipt

    def make_fixture(self, root: Path) -> tuple[Path, list[str]]:
        analytic, analytic_receipt = self.write_family(root, "analytic")
        tcn, tcn_receipt = self.write_family(root, "tcn")
        positive = root / "m38b.positive.metrics.json"
        logical_ids = ["POS_d0", "POS_d0p25", "POS_d0p5", "POS_d1", "POS_d2"]
        positive.write_text(json.dumps({
            "stage": "M38B_SCORE_DIAGNOSTIC_POSITIVE_CONTROL", "family": "tcn",
            "logical_ids": logical_ids, "capacity_gate": {"pass": False},
        }), encoding="utf-8")
        positive_receipt = root / "m38b.positive.metrics.receipt.json"
        positive_receipt.write_text(json.dumps({
            "stage": "M38B_SCORE_DIAGNOSTIC_POSITIVE_CONTROL",
            "status": "PASS_DIAGNOSTIC_GRID_SCORED", "diagnostic_only": True,
            "family": "tcn", "logical_ids": logical_ids,
            "output_sha256": digest(positive), **PROVENANCE,
        }), encoding="utf-8")
        rows = [
            ("analytic_metrics", analytic), ("analytic_receipt", analytic_receipt),
            ("tcn_metrics", tcn), ("tcn_receipt", tcn_receipt),
            ("positive_metrics", positive), ("positive_receipt", positive_receipt),
        ]
        source_uri = (
            "gs://teams-usp/frank/lai-exploracion-datos/runs/"
            "m38b-r0-oof-models-20260903c"
        )
        manifest = root / "inputs.json"
        manifest.write_text(json.dumps({
            "schema_version": "1.0.0", "stage": "M38B_FINALIZER_INPUTS",
            "source_run_uri": source_uri,
            "artifacts": [{
                "logical_id": logical_id, "basename": path.name,
                "uri": f"{source_uri}/fixture/{path.name}", "sha256": digest(path),
            } for logical_id, path in rows],
            "decision_script": {
                "basename": "m38b_decide.py",
                "sha256": digest(ROOT / "bin/m38b_decide.py"),
            },
        }), encoding="utf-8")
        bindings = [f"{logical_id}={path}" for logical_id, path in rows]
        return manifest, bindings

    def run_fixture(self, root: Path, manifest: Path, bindings: list[str]) -> None:
        argv = [
            "m38b_finalize_decision.py", "--manifest", str(manifest),
            "--manifest-sha256", digest(manifest),
        ]
        for binding in bindings:
            argv.extend(["--artifact", binding])
        argv.extend([
            "--decision-script", str(ROOT / "bin/m38b_decide.py"),
            "--code-commit", "1" * 40,
            "--runtime-image", "registry/image@sha256:" + "2" * 64,
            "--provenance-source", str(ROOT / "modules/38B_FINALIZE_DECISION.nf"),
            "--output", str(root / "m38b.final_decision.json"),
            "--provenance-output", str(root / "m38b.finalizer.provenance.json"),
        ])
        with mock.patch.object(sys, "argv", argv):
            subject.main()

    def test_finalizes_exact_pinned_inputs_and_records_direct_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, bindings = self.make_fixture(root)
            self.run_fixture(root, manifest, bindings)
            provenance = json.loads((root / "m38b.finalizer.provenance.json").read_text())
            self.assertEqual(provenance["status"], "PASS_FINALIZED_IMMUTABLE_SCORES")
            self.assertEqual(len(provenance["artifacts"]), 6)
            self.assertEqual(
                provenance["finalizer_script_sha256"],
                digest(ROOT / "bin/m38b_finalize_decision.py"),
            )
            self.assertTrue((root / "m38b.final_decision.receipt.json").is_file())

    def test_mutated_score_is_rejected_before_decision(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, bindings = self.make_fixture(root)
            (root / "m38b.tcn.metrics.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(subject.M38BFinalizerError, "staged SHA-256 differs"):
                self.run_fixture(root, manifest, bindings)
            self.assertFalse((root / "m38b.final_decision.json").exists())

    def test_original_run_c_workflow_remains_byte_identical(self) -> None:
        self.assertEqual(
            digest(ROOT / "workflows/m38b_r0_oof_models.nf"),
            "7d87526e625ad26cc2b861bbfbf090394b3fee23b4b32c3e5af6a07a42bce27b",
        )

    @unittest.skipUnless(shutil.which("nextflow"), "Nextflow is not installed")
    def test_finalizer_process_accepts_exact_six_file_tuple(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m38b-finalizer-smoke-") as raw:
            root = Path(raw)
            paths = []
            for name in (
                "m38b.analytic.metrics.json", "m38b.analytic.metrics.receipt.json",
                "m38b.tcn.metrics.json", "m38b.tcn.metrics.receipt.json",
                "m38b.positive.metrics.json", "m38b.positive.metrics.receipt.json",
            ):
                path = root / name
                path.write_text("{}\n", encoding="utf-8")
                paths.append(path)
            for name in ("manifest.json", "m38b_decide.py", "m38b_finalize_decision.py",
                         "source.config"):
                (root / name).write_text("{}\n", encoding="utf-8")
            workflow = root / "smoke.nf"
            score_arguments = ", ".join(f'file("{path}")' for path in paths)
            workflow.write_text(f"""nextflow.enable.dsl=2
include {{ M38B_FINALIZE_DECISION }} from '{ROOT / 'modules/38B_FINALIZE_DECISION.nf'}'
workflow {{
  scores = Channel.value(tuple({score_arguments}))
  sources = Channel.value([file('{root / 'source.config'}')])
  M38B_FINALIZE_DECISION(scores, file('{root / 'manifest.json'}'), '{'a' * 64}',
    file('{root / 'm38b_decide.py'}'), file('{root / 'm38b_finalize_decision.py'}'),
    sources, '{'1' * 40}', 'registry/image@sha256:{'2' * 64}')
}}
""", encoding="utf-8")
            result = subprocess.run([
                "nextflow", "run", str(workflow), "-stub-run", "-ansi-log", "false",
                "--m38b_finalizer_results_dir", str(root / "results"),
                "--m38b_finalizer_source_run_id", "source",
                "--m38b_finalizer_run_id", "audit",
            ], cwd=root, text=True, capture_output=True, timeout=120, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("M38B_FINALIZE_DECISION", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
