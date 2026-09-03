from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compact_lane_is_nextflow_first_and_never_exports_predictions() -> None:
    module = (ROOT / "modules/37_TRACE_COMPACT_SWEEP.nf").read_text(encoding="utf-8")
    workflow = (ROOT / "workflows/m37_trace_compact_sweep.nf").read_text(encoding="utf-8")
    assert "M37_TRACE_COMPACT_SWEEP" in module
    assert "m37_trace_compact_sweep.py" in module
    assert "*.metrics.json" in module and "*.metrics.receipt.json" in module
    assert "prediction.npz" not in module and "checkpoint.pt" not in module
    assert "['hmm', 'tcn'].collect" in workflow
    assert "M37_TRACE_COMPACT_POSITIVE_CONTROL" in workflow
    assert "M37_TRACE_COLLECT_METRICS" in workflow
    assert "M37_TRACE_COMPACT_DECISION" in workflow
    assert "m37_valid_truth' is forbidden" not in workflow
    assert "is forbidden in the compact FIT/TUNE workflow" in workflow


def test_compact_cloud_overlay_is_personal_pinned_and_parallel() -> None:
    base = (ROOT / "conf/m37_trace_compact_sweep.config").read_text(encoding="utf-8")
    cloud = (ROOT / "conf/m37_trace_gcp.config").read_text(encoding="utf-8")
    overlay = (ROOT / "conf/m37_r0_compact_sweep.config").read_text(encoding="utf-8")
    assert "m37_compact_max_forks = 2" in base
    assert "executor = 'google-batch'" in cloud
    assert "team: 'frank'" in cloud
    assert "@sha256:" in cloud
    assert "gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/" in overlay
    assert "gs://projects-usp" not in overlay


def test_compact_manifest_has_complete_hmm_grid_and_three_calibrated_tcn_points() -> None:
    import json
    manifest = json.loads((ROOT / "conf/m37_trace_compact_candidates.json").read_text(encoding="utf-8"))
    hmm = [row for row in manifest["candidates"] if row["family"] == "hmm"]
    tcn = [row for row in manifest["candidates"] if row["family"] == "tcn"]
    assert len(hmm) == 20 and len(tcn) == 3
    assert {row["candidate_id"] for row in tcn} == {
        "tcn_anchor_pc", "tcn_mid_local", "tcn_wide_broad"
    }
    assert manifest["scope"] == {
        "root": "R0", "evaluation_split": "FIT_TUNE", "valid_access": "FORBIDDEN"
    }
    assert manifest["positive_control_status"]["tcn"]["status_at_1600_updates"].startswith("REQUIRED_RUNTIME")
