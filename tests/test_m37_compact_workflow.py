import json
import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_preflight_module():
    path = ROOT / "tests" / "run_m37_compact_config_preflight.py"
    spec = importlib.util.spec_from_file_location("m37_compact_preflight_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_preflight_rejects_stale_amendment_binding() -> None:
    preflight = _load_preflight_module()
    with tempfile.TemporaryDirectory(prefix="m37-binding-") as raw:
        root = Path(raw)
        parent = root / "parent.json"
        amendment = root / "amendment.json"
        manifest = root / "manifest.json"
        shutil.copyfile(ROOT / "conf" / "m37_trace_sweep_contract.json", parent)
        shutil.copyfile(ROOT / "conf" / "m37_trace_compact_sweep_amendment.json", amendment)
        shutil.copyfile(ROOT / "conf" / "m37_trace_compact_candidates.json", manifest)
        preflight.validate_contract_binding(manifest, parent, amendment)

        payload = json.loads(amendment.read_text(encoding="utf-8"))
        payload["status"] = "TAMPERED_AFTER_FREEZE"
        amendment.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            preflight.validate_contract_binding(manifest, parent, amendment)
        except AssertionError as error:
            assert "hash binding differs" in str(error)
        else:
            raise AssertionError("preflight accepted an altered amendment")


def test_compact_preflight_rejects_null_workdir() -> None:
    preflight = _load_preflight_module()
    assert preflight.RECOMMENDED_CONFIG_ORDER == (
        "m37_trace_compact_sweep.config",
        "m37_trace_gcp.config",
        "m37_r0_compact_sweep.config",
    )
    wrong = "\n".join((
        "process.executor = 'google-batch'",
        "process.resourceLabels.team = 'frank'",
        "params.m37_results_dir = 'gs://teams-usp/frank/lai-exploracion-datos/runs'",
        "params.m37_run_id = null",
        "workDir = 'gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/null'",
        "@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99",
    ))
    try:
        preflight.validate_effective_config(wrong)
    except AssertionError as error:
        assert "null_namespace=True" in str(error)
        assert "base,GCP,run-overlay" in str(error)
    else:
        raise AssertionError("preflight accepted a null M37 workDir")


def test_compact_nextflow_routing_e2e_for_tcn_pass_and_failure() -> None:
    """Exercise the real Nextflow/Groovy routing helpers in both TCN branches."""
    if shutil.which("nextflow") is None:
        return
    module = (ROOT / "modules/37_TRACE_COMPACT_SWEEP.nf").with_suffix("")
    with tempfile.TemporaryDirectory(prefix="m37-routing-") as raw:
        work = Path(raw)
        passed = work / "capacity-pass.json"
        failed = work / "capacity-fail.json"
        passed.write_text(json.dumps({
            "controls": {"tcn": {"eligible_candidate_ids": ["tcn_fixture"]}},
        }), encoding="utf-8")
        failed.write_text(json.dumps({
            "controls": {"tcn": {"eligible_candidate_ids": []}},
        }), encoding="utf-8")
        script = work / "routing_e2e.nf"
        script.write_text(f"""
nextflow.enable.dsl=2
include {{ m37_compact_capacity_families; m37_compact_decision_parts }} from '{module.as_posix()}'

workflow {{
    Channel.of(file(params.capacity_pass), file(params.capacity_fail))
        .flatMap {{ positive ->
            m37_compact_capacity_families(positive).collect {{ family ->
                tuple(positive.simpleName, family)
            }}
        }}
        .view {{ row -> "ROUTE:${{row[0]}}:${{row[1]}}" }}

    def audits = Channel.value([
        ['hmm.audit.json', 'hmm.audit.receipt.json'],
        ['tcn.audit.json', 'tcn.audit.receipt.json'],
    ])
    Channel.of(tuple('R0', 'metrics.json', 'metrics.receipt.json'))
        .combine(audits)
        .map {{ combined -> m37_compact_decision_parts(combined) }}
        .view {{ parts ->
            "DECISION:${{parts[0]}}:${{parts[3].join(',')}}:${{parts[4].join(',')}}"
        }}

    def hmmOnlyAudits = Channel.value([
        ['hmm-only.audit.json', 'hmm-only.audit.receipt.json'],
    ])
    Channel.of(tuple('R0', 'hmm.metrics.json', 'hmm.metrics.receipt.json'))
        .combine(hmmOnlyAudits)
        .map {{ combined -> m37_compact_decision_parts(combined) }}
        .view {{ parts ->
            "DECISION_HMM_ONLY:${{parts[0]}}:${{parts[3].join(',')}}:${{parts[4].join(',')}}"
        }}
}}
""", encoding="utf-8")
        environment = dict(os.environ)
        environment["NXF_OFFLINE"] = "true"
        result = subprocess.run(
            ["nextflow", "run", "-ansi-log", "false", str(script),
             "--capacity_pass", str(passed), "--capacity_fail", str(failed)],
            cwd=work, env=environment, text=True, capture_output=True,
            timeout=120, check=False,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        lines = set(result.stdout.splitlines())
        assert "ROUTE:capacity-pass:hmm" in lines
        assert "ROUTE:capacity-pass:tcn" in lines
        assert "ROUTE:capacity-fail:hmm" in lines
        assert "ROUTE:capacity-fail:tcn" not in lines
        assert (
            "DECISION:R0:hmm.audit.json,tcn.audit.json:"
            "hmm.audit.receipt.json,tcn.audit.receipt.json"
        ) in lines
        assert (
            "DECISION_HMM_ONLY:R0:hmm-only.audit.json:"
            "hmm-only.audit.receipt.json"
        ) in lines


def test_compact_lane_is_nextflow_first_and_never_exports_predictions() -> None:
    module = (ROOT / "modules/37_TRACE_COMPACT_SWEEP.nf").read_text(encoding="utf-8")
    workflow = (ROOT / "workflows/m37_trace_compact_sweep.nf").read_text(encoding="utf-8")
    assert "M37_TRACE_COMPACT_SWEEP" in module
    assert "m37_trace_compact_sweep.py" in module
    assert "*.metrics.json" in module and "*.metrics.receipt.json" in module
    assert "prediction.npz" not in module and "checkpoint.pt" not in module
    assert "m37_compact_capacity_families(positive)" in workflow
    assert "Files.newBufferedReader(positive_control_path)" in module
    assert "positive_control_path.toFile()" not in module
    assert "m37_compact_decision_parts(combined)" in workflow
    assert "M37_TRACE_COMPACT_CAPACITY_SCREEN" in workflow
    assert "M37_TRACE_COMPACT_CAPACITY_REPLICATION" in workflow
    assert "m37.capacity_selection.json" in module
    assert "--phase screen" in module and "--phase replicate" in module
    assert module.count("time '4h'") >= 2
    assert workflow.index("M37_TRACE_COMPACT_CAPACITY_REPLICATION") < workflow.index(
        "def familyInput"
    )
    assert workflow.index("m37_compact_capacity_families(positive)") < workflow.index("def featureFiles")
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
    assert "m37_compact_positive_control_updates" not in base
    assert "executor = 'google-batch'" in cloud
    assert "team: 'frank'" in cloud
    assert "@sha256:" in cloud
    assert "gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/" in overlay
    assert "gs://projects-usp" not in overlay
    assert "m37-r0-compact-sweep-20260903d" in overlay
    assert "m37-r0-compact-sweep-20260903c" not in overlay


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
    assert manifest["execution_budget_policy"]["updates_200_role"].startswith(
        "initial capacity rung"
    )
    assert manifest["positive_control_status"]["tcn"]["status_across_budget_ladder"] == (
        "REQUIRED_CANDIDATE_SPECIFIC_SYNTHETIC_CAPACITY_GATE"
    )
    assert manifest["positive_control_status"]["tcn"]["budget_ladder_updates"] == [
        200, 400, 800, 1600,
    ]
    assert manifest["positive_control_status"]["tcn"]["candidate_count"] == 6
    assert manifest["positive_control_status"]["tcn"]["rung_execution"] == (
        "DETERMINISTIC_RESTART_SAME_SEED_EACH_RUNG"
    )
    assert manifest["positive_control_status"]["tcn"]["rung_training_policy"] == (
        "EXACT_REQUESTED_UPDATES_FINAL_STATE_NO_EARLY_STOPPING_NO_BEST_CHECKPOINT_RESTORE"
    )
    assert "maximum_selected_candidates" not in manifest["positive_control_status"]["tcn"]
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
        "PROSPECTIVE_BEFORE_FIRST_CONSUMABLE_20260903D_CANDIDATE_SWEEP_EXECUTION"
    )
    assert amendment["tcn"]["event_radius_cM"] == [0.05, 0.1, 0.2, 0.5]
    assert "promotion remains fixed at 0.2 cM" in radius["metric_alignment"]
    assert amendment["candidate_count_by_family"]["tcn"] == 6
    assert amendment["capacity_control"]["screen_seed"] == 1103
    assert amendment["capacity_control"]["screen_candidate_count"] == 6
    assert amendment["capacity_control"]["budget_ladder_updates"] == [
        200, 400, 800, 1600,
    ]
    assert amendment["capacity_control"]["rung_training_policy"] == (
        "EXACT_REQUESTED_UPDATES_FINAL_STATE_NO_EARLY_STOPPING_NO_BEST_CHECKPOINT_RESTORE"
    )
    assert "completed 300 and restored checkpoint 200" in amendment[
        "capacity_control"
    ]["rung_training_policy_reason"]
    assert amendment["capacity_control"]["candidate_evaluation"] == (
        "ALL_DECLARED_CANDIDATES_ACROSS_ALL_FIXED_SEEDS_NO_RANKING"
    )
    assert amendment["capacity_control"]["replication_seeds"] == [1103, 2207, 3301]
    assert amendment["capacity_control"]["thresholds"][
        "hmm_additive_maximum_posterior_change"
    ] == 1e-4
    assert "no candidate is ranked" in amendment["capacity_control"][
        "screen_seed_reuse_interpretation"
    ].lower()
    assert amendment["capacity_control"]["valid_access"] == "FORBIDDEN"
    assert "maximum_selected_candidates" not in amendment["capacity_control"]
    assert amendment["superseded_execution"]["disposition"] == "SUPERSEDED_NO_CONSUMABLE_RESULTS"
    assert amendment["superseded_execution"]["result_use"] == "FORBIDDEN_FOR_CANDIDATE_SELECTION_OR_INFERENCE"
    run_b = amendment["superseded_execution_20260903b"]
    assert run_b["disposition"] == "SUPERSEDED_AFTER_LAUNCH_NONCONSUMABLE"
    assert "HMM, TCN and paired-metric collection completed" in run_b["execution"]
    assert run_b["result_use"] == "FORBIDDEN_FOR_CANDIDATE_SELECTION_OR_INFERENCE"
    run_c = amendment["superseded_execution_20260903c"]
    assert run_c["disposition"] == "SUPERSEDED_WRONG_WORKDIR_NONCONSUMABLE"
    assert run_c["resolved_work_dir"].endswith("/work/nextflow/null")
    assert run_c["submitted_stage"] == "synthetic capacity screen only"
    assert run_c["real_fit_tune_execution"] == (
        "NOT_CLAIMED_OR_INFERRED_FROM_THIS_LAUNCH_RECORD"
    )
    assert run_c["result_use"] == "FORBIDDEN_FOR_CANDIDATE_SELECTION_OR_INFERENCE"
