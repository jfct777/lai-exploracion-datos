#!/usr/bin/env python3
"""Run M37 end to end with its production config and a tiny local overlay."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ("us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@"
         "sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99")
ARMS = ("RE", "RD", "POOLED", "SHAM", "GEOMETRY")


def quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "\\'") + "'"


def main() -> None:
    if not shutil.which("nextflow") or not shutil.which("docker"):
        raise RuntimeError("the M37 real-config test needs nextflow and docker")
    with tempfile.TemporaryDirectory(prefix="m37-real-config-") as raw:
        root = Path(raw)
        fixture = root / "fixture-fit"
        valid_fixture = root / "fixture-valid"
        for name, prefix in (("fixture-fit", "fit-person"), ("fixture-valid", "valid-person")):
            subprocess.run([
                "docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                "-v", f"{ROOT}:/workspace:ro", "-v", f"{root}:/test",
                "-w", "/workspace", IMAGE, "python3", "tests/make_m37_e2e_fixture.py",
                "--output-dir", f"/test/{name}", "--sample-prefix", prefix,
            ], check=True)

        candidate_rows = ",\n    ".join(
            "[candidate_id:'fixture_%s', family:'%s', arm:'%s', "
            "hazard_per_morgan:12.0, evidence_scale:1.0, hidden_dim:32, depth:2, "
            "kernel_size:3, dropout:0.0, seed:1103, learning_rate:0.0003, dilations:'1,2']" %
            (family, family, arm)
            for family in ("hmm", "tcn") for arm in ARMS
        )
        overlay = root / "e2e.config"
        overlay.write_text(f"""
params {{
  m37_run_id = 'real-config-e2e'
  m37_root = 'R0'
  m37_results_dir = {quote(root / 'results')}
  m37_run_overlay_config = {quote(overlay)}
  m37_run_overlay_uri = 'test://m37/real-config-e2e'
  m37_container_digest = {quote(IMAGE)}
  m37_fit_selected = {quote(fixture / 'selected.npz')}
  m37_fit_target = {quote(fixture / 'target.npz')}
  m37_fit_reference_folds = {quote(fixture / 'reference.npz')}
  m37_fit_f0 = {quote(fixture / 'f0.npz')}
  m37_fit_marker_cm = {quote(fixture / 'marker_axis_source.npz')}
  m37_fit_f0_receipt = {quote(fixture / 'f0.receipt.json')}
  m37_fit_truth = {quote(fixture / 'truth.npz')}
  m37_updates = 2
  m37_batch_people = 2
  m37_marker_shard = 13
  m37_validation_every = 1
  m37_early_stopping_patience = 1
  m37_train_max_forks = 5
  m37_materialize_cpus = 1
  m37_materialize_memory = '2 GB'
  m37_materialize_time = '5m'
  m37_train_cpus = 1
  m37_train_memory = '2 GB'
  m37_train_time = '5m'
  m37_candidates = [
    {candidate_rows}
  ]
}}
docker.enabled = true
docker.runOptions = '--user {os.getuid()}:{os.getgid()}'
process {{
  executor = 'local'
  container = params.m37_container_digest
  errorStrategy = 'terminate'
  maxRetries = 0
}}
""".strip() + "\n", encoding="utf-8")

        subprocess.run([
            "nextflow", "-C", f"{ROOT / 'conf/m37_trace_lai.config'},{overlay}",
            "run", str(ROOT / "workflows/m37_trace_lai.nf"),
            "-work-dir", str(root / "work"), "-ansi-log", "false",
        ], cwd=ROOT, check=True)

        tuning_expected = sorted(
            f"fixture_{family}.{family}.{arm}.READY.json"
            for family in ("hmm", "tcn") for arm in ARMS
        )
        frozen_expected = sorted(f"fixture_hmm.hmm.{arm}.READY.json" for arm in ARMS)
        def assert_ready(run_id: str, expected: list[str], expected_overlay: Path) -> list[str]:
            provenance = root / "results" / run_id / "provenance"
            observed = sorted(path.name for path in provenance.glob("*.READY.json"))
            if observed != expected:
                raise AssertionError(f"M37 READY set differs for {run_id}: {observed}")
            for ready_path in provenance.glob("*.READY.json"):
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
                manifest = ready_path.with_name(ready_path.name.replace(".READY.json", ".manifest.json"))
                if ready.get("status") != "READY" or not manifest.exists():
                    raise AssertionError(f"M37 provenance chain incomplete for {ready_path.name}")
                if ready.get("root") != "R0" or ready.get("run_overlay", {}).get("sha256") != hashlib.sha256(expected_overlay.read_bytes()).hexdigest():
                    raise AssertionError(f"M37 root/overlay provenance differs for {ready_path.name}")
            return observed

        observed = assert_ready("real-config-e2e", tuning_expected, overlay)
        promotion = root / "results" / "real-config-e2e" / "promotion"
        collected = promotion / "m37.R0.paired_metrics.json"
        plan = promotion / "m37.successive_halving.json"
        for artifact in (collected, collected.with_suffix(".receipt.json"),
                         plan, plan.with_suffix(".receipt.json")):
            if not artifact.exists():
                raise AssertionError(f"M37 authenticated promotion chain lacks {artifact.name}")
        collection_payload = json.loads(collected.read_text(encoding="utf-8"))
        if len(collection_payload.get("rows", [])) != 10 or {row.get("root") for row in collection_payload["rows"]} != {"R0"}:
            raise AssertionError("M37 metric collector did not emit ten root-authenticated arm rows")
        prediction_receipts = list((root / "results" / "real-config-e2e" / "predictions").glob("*.prediction.receipt.json"))
        if len(prediction_receipts) != 10:
            raise AssertionError("M37 training receipt set differs")
        for path in prediction_receipts:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            effective = receipt.get("effective_hyperparameters", {})
            if receipt.get("family") == "hmm" and set(effective) != {"hazard_per_morgan", "evidence_scale"}:
                raise AssertionError("M37 HMM effective hyperparameters are incomplete")
            if receipt.get("family") == "tcn" and not {"hidden_dim", "depth", "kernel_size", "dropout", "dilations", "learning_rate", "updates"}.issubset(effective):
                raise AssertionError("M37 TCN effective hyperparameters are incomplete")
        frozen_overlay = root / "frozen-e2e.config"
        frozen_overlay.write_text(overlay.read_text(encoding="utf-8") + f"""
params {{
  m37_run_id = 'frozen-real-config-e2e'
  m37_run_overlay_config = {quote(frozen_overlay)}
  m37_run_overlay_uri = 'test://m37/frozen-real-config-e2e'
  m37_valid_selected = {quote(valid_fixture / 'selected.npz')}
  m37_valid_target = {quote(valid_fixture / 'target.npz')}
  m37_valid_reference_folds = {quote(valid_fixture / 'reference.npz')}
  m37_valid_f0 = {quote(valid_fixture / 'f0.npz')}
  m37_valid_marker_cm = {quote(valid_fixture / 'marker_axis_source.npz')}
  m37_valid_f0_receipt = {quote(valid_fixture / 'f0.receipt.json')}
  m37_valid_truth = {quote(valid_fixture / 'truth.npz')}
  m37_frozen_candidate = [candidate_id:'fixture_hmm', family:'hmm',
    hazard_per_morgan:12.0, evidence_scale:1.0, hidden_dim:32, depth:2,
    kernel_size:3, dropout:0.0, seed:1103, learning_rate:0.0003, dilations:'1,2']
}}
""", encoding="utf-8")
        subprocess.run([
            "nextflow", "-C", f"{ROOT / 'conf/m37_trace_lai.config'},{frozen_overlay}",
            "run", str(ROOT / "workflows/m37_trace_frozen_eval.nf"),
            "-work-dir", str(root / "frozen-work"), "-ansi-log", "false",
        ], cwd=ROOT, check=True)
        frozen_observed = assert_ready("frozen-real-config-e2e", frozen_expected, frozen_overlay)
        for ready_path in (root / "results" / "frozen-real-config-e2e" / "provenance").glob("*.READY.json"):
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            metrics_path = root / "results" / "frozen-real-config-e2e" / "metrics" / ready_path.name.replace(".READY.json", ".metrics.json")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("evaluation_split") != "SEALED_VALID":
                raise AssertionError("frozen M37 evaluation did not use disjoint VALID")
        print(json.dumps({"status": "PASS_M37_REAL_CONFIG_E2E", "tuning_ready": observed,
                          "frozen_ready": frozen_observed}, sort_keys=True))


if __name__ == "__main__":
    main()
