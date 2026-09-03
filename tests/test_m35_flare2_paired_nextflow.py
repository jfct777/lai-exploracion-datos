from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "35_FLARE2_PAIRED.nf"
WORKFLOW = ROOT / "workflows" / "m35_flare2_paired.nf"
CONFIG = ROOT / "conf" / "m35_flare2_paired.config"


def test_m35_isolated_workflow_is_pinned_and_preflight_by_default():
    module = MODULE.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    assert "M35_FLARE2_PAIRED_BASELINE" in module
    assert "m35_preflight_only = true" in config
    assert "@sha256:" in workflow
    assert "--network none" in module
    assert "overwrite: false" in module
    assert "/opt/flare/create_model_file.py" in module
    assert "executor = 'google-batch'" in config
    assert "gs://teams-usp/frank/lai-exploracion-datos" in config
    assert "main.nf" not in workflow


def test_m35_inference_process_has_no_scoring_or_label_inputs():
    module = MODULE.read_text(encoding="utf-8")
    process = module.split("process M35_FLARE2_PAIRED_BASELINE", 1)[1].split(
        "process M35_PACK_FLARE_PREDICTION", 1
    )[0]
    assert "scor" not in process.lower()
    assert "label" not in process.lower()
    assert "truth" not in process.lower()
