from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_m37_lane_is_isolated_and_nextflow_first() -> None:
    module = (ROOT / "modules/37_TRACE_LAI.nf").read_text()
    workflow = (ROOT / "workflows/m37_trace_lai.nf").read_text()
    config = (ROOT / "conf/m37_trace_lai.config").read_text()
    assert "M37_TRACE_MATERIALIZE" in module
    assert "M37_TRACE_BIND_MARKER_AXIS" in module and "m37_bind_marker_axis.py" in workflow
    assert "M37_TRACE_TRAIN" in module and "M37_TRACE_SCORE" in module
    assert "M37_TRACE_COLLECT_METRICS" in module and "M37_TRACE_SUCCESSIVE_HALVING" in module
    assert "m37_trace_materialize.py" in workflow and "nextflow.enable.dsl=2" in workflow
    assert "m37_beta_prior_strength" in config and "m37_candidates" in config
    assert config.count("candidate_id:'hmm_r0'") == 5 and config.count("candidate_id:'tcn_r0'") == 5
    assert "m37_promotion_minimum_f1_gain" in config and "m37_promotion_maximum_log_loss_increase" in config
    assert ".combine(M37_TRACE_BIND_MARKER_AXIS.out.bundle, by: 0)" in workflow
    assert "M37_TRACE_COLLECT_METRICS.out.bundle" in workflow
    assert "m37_existing_metrics_json" not in workflow and "m37_existing_metrics_json" not in config
    assert "m37_run_overlay_config" in workflow and "m37_run_overlay_uri" in workflow
    assert "nextflow.config" not in workflow


def test_real_config_e2e_runner_covers_all_paired_arms() -> None:
    runner = (ROOT / "tests/run_m37_real_config_e2e.py").read_text()
    fixture = (ROOT / "tests/make_m37_e2e_fixture.py").read_text()
    assert "conf/m37_trace_lai.config" in runner and "workflows/m37_trace_lai.nf" in runner
    assert 'ARMS = ("RE", "RD", "POOLED", "SHAM", "GEOMETRY")' in runner
    assert 'for family in ("hmm", "tcn") for arm in ARMS' in runner
    assert "PASS_M37_REAL_CONFIG_E2E" in runner and "state_labels=truth" in fixture
