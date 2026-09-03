from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import m37_trace_provenance as subject


def test_ready_preserves_dotted_arm_name_and_receipt_identity() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        artifact = root / "candidate.tcn.RE.metrics.json"
        artifact.write_text('{"metric": 1}\n', encoding="utf-8")
        receipt = root / "candidate.tcn.RE.metrics.receipt.json"
        receipt.write_text(json.dumps({
            "stage": "M37_TRACE_SCORE", "candidate_id": "candidate",
            "family": "tcn", "root": "R0", "arm": "RE",
            "output_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }), encoding="utf-8")
        overlay = root / "m37_r0_triage.config"
        overlay.write_text("params.m37_root = 'R0'\n", encoding="utf-8")
        prefix = root / "candidate.tcn.RE"
        previous = sys.argv
        sys.argv = ["m37_trace_provenance.py", "--artifact", str(artifact),
                    "--receipt", str(receipt), "--run-id", "fixture-run",
                    "--container-digest", "sha256:fixture", "--candidate-id", "candidate",
                    "--root", "R0", "--family", "tcn", "--arm", "RE",
                    "--run-overlay", str(overlay), "--run-overlay-uri", "repo://conf/m37_r0_triage.config",
                    "--auth-file", str(artifact),
                    "--output-prefix", str(prefix)]
        try:
            subject.main()
        finally:
            sys.argv = previous
        manifest = root / "candidate.tcn.RE.manifest.json"
        ready = root / "candidate.tcn.RE.READY.json"
        assert manifest.exists() and ready.exists()
        assert not (root / "candidate.tcn.manifest.json").exists()
        payload = json.loads(ready.read_text(encoding="utf-8"))
        assert payload["candidate_id"] == "candidate" and payload["root"] == "R0" and payload["arm"] == "RE"
        assert payload["run_overlay"] == {
            "uri": "repo://conf/m37_r0_triage.config",
            "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
        }
        assert payload["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
