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
    assert "m37_fit_f0_receipt" in workflow
    assert "--f0-receipt" in module
    assert "m37_valid_truth' is forbidden" not in workflow
    assert "is forbidden in the compact FIT/TUNE workflow" in workflow


def test_compact_cloud_overlay_is_personal_pinned_and_parallel() -> None:
    base = (ROOT / "conf/m37_trace_compact_sweep.config").read_text(encoding="utf-8")
    cloud = (ROOT / "conf/m37_trace_gcp.config").read_text(encoding="utf-8")
    overlay = (ROOT / "conf/m37_r0_compact_sweep.config").read_text(encoding="utf-8")
    assert "m37_compact_max_forks = 2" in base
    assert "m37_compact_positive_control_updates = 200" in base
    assert "executor = 'google-batch'" in cloud
    assert "team: 'frank'" in cloud
    assert "@sha256:" in cloud
    assert "gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/" in overlay
    assert "gs://projects-usp" not in overlay
    assert "m37-r0-compact-sweep-20260903b" in overlay
    assert "m37-r0-compact-sweep-20260903a" not in overlay


def test_compact_manifest_has_space_filling_hmm_and_tcn_designs() -> None:
    import json
    manifest = json.loads((ROOT / "conf/m37_trace_compact_candidates.json").read_text(encoding="utf-8"))
    hmm = [row for row in manifest["candidates"] if row["family"] == "hmm"]
    tcn = [row for row in manifest["candidates"] if row["family"] == "tcn"]
    assert len(hmm) == 12 and len(tcn) == 6
    assert {row["candidate_id"] for row in tcn} == {
        "tcn_h32_d2_k3_do0_lr1e3_r02_l1",
        "tcn_h32_d4_k5_do02_lr1e4_r005_l2",
        "tcn_h64_d2_k5_do01_lr3e4_r01_l025",
        "tcn_h64_d3_k3_do02_lr1e3_r05_l05",
        "tcn_h96_d3_k5_do0_lr1e4_r02_l2",
        "tcn_h96_d4_k3_do01_lr3e4_r05_l05",
    }
    assert manifest["scope"] == {
        "root": "R0", "evaluation_split": "FIT_TUNE", "valid_access": "FORBIDDEN"
    }
    assert manifest["execution"]["updates"] == 200
    assert manifest["positive_control_status"]["tcn"]["status_at_200_updates"].startswith("REQUIRED_RUNTIME")
    assert manifest["equivalence"]["policy_by_family"] == {
        "hmm": "REQUIRE_CANONICAL_METRIC_REPLAY",
        "tcn": "NEWLY_FROZEN_RESIDUAL_OPERATOR_REFERENCE_ONLY",
    }


def test_event_radius_is_prospectively_frozen_without_changing_promotion_endpoint() -> None:
    import json
    amendment = json.loads(
        (ROOT / "conf/m37_trace_compact_sweep_amendment.json").read_text(encoding="utf-8")
    )
    radius = amendment["tcn"]["event_radius_amendment"]
    assert radius["timing"] == (
        "PROSPECTIVE_BEFORE_FIRST_CONSUMABLE_20260903B_CANDIDATE_SWEEP_EXECUTION"
    )
    assert amendment["tcn"]["event_radius_cM"] == [0.05, 0.1, 0.2, 0.5]
    assert "promotion remains fixed at 0.2 cM" in radius["metric_alignment"]
    assert amendment["candidate_count_by_family"]["tcn"] == 6
    assert amendment["superseded_execution"]["disposition"] == "SUPERSEDED_NO_CONSUMABLE_RESULTS"
    assert amendment["superseded_execution"]["result_use"] == "FORBIDDEN_FOR_CANDIDATE_SELECTION_OR_INFERENCE"
