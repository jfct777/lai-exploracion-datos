import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))


def load_script(name: str):
    path = ROOT / "bin" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREPARE = load_script("m35b_prepare_balanced_reference")
AGGREGATE = load_script("m35b_aggregate_cluster_gate")
CONTRACT = ROOT / "conf" / "m35b_flare2_balanced_contract.json"


def row(ancestry: str, population: str) -> dict[str, str]:
    return {
        "ancestry": ancestry,
        "population": population,
        "canonical_population": f"{ancestry}|{population}",
        "role": "REF_TRAIN",
    }


def test_population_floor_hamilton_preserves_populations_and_exact_counts():
    reference = {}
    for index in range(7):
        reference[f"A1_{index}"] = row("African", "A1")
    for index in range(3):
        reference[f"A2_{index}"] = row("African", "A2")
    for index in range(6):
        reference[f"E1_{index}"] = row("European", "E1")
    for index in range(4):
        reference[f"E2_{index}"] = row("European", "E2")
    for index in range(5):
        reference[f"N1_{index}"] = row("Native_American", "N1")
    for index in range(5):
        reference[f"N2_{index}"] = row("Native_American", "N2")

    selected, receipt = PREPARE.deterministic_subset(reference, seed=350101, per_ancestry=5)
    assert len(selected) == 15
    assert receipt["selected_counts"] == {"AFR": 5, "EUR": 5, "NAM": 5}
    assert receipt["Hamilton_population_allocation"] == {
        "AFR": {"A1": 3, "A2": 2},
        "EUR": {"E1": 3, "E2": 2},
        "NAM": {"N1": 2, "N2": 3},
    }
    assert all(all(count >= 1 for count in populations.values())
               for populations in receipt["population_counts"].values())
    repeated, repeated_receipt = PREPARE.deterministic_subset(
        reference, seed=350101, per_ancestry=5)
    assert selected == repeated
    assert receipt == repeated_receipt


def test_hamilton_rejects_a_target_that_cannot_preserve_every_population():
    with unittest.TestCase().assertRaisesRegex(
            PREPARE.BalancedReferenceError, "lacks enough slots"):
        PREPARE._hamilton_with_population_floor(
            {"P1": ["a"], "P2": ["b"], "P3": ["c"]}, 2, 350101, "AFR")


def write_screen(directory: Path, contract: dict, selection: int, granularity: str,
                 gmm: int, passed: bool = True) -> None:
    directory.mkdir()
    support = 0.8 if passed else 0.49
    status = "PASS_M35B_CLUSTER_SEPARATION" if passed else "NO_GO_M35B_CLUSTER_SEPARATION"
    evidence = {
        "status": status,
        "selection_seed": selection,
        "granularity": granularity,
        "gmm_seed": gmm,
        "NAM_support": support,
        "log_margin": 0.5,
        "balanced_macro_counts": {"AFR": 25, "EUR": 25, "NAM": 25},
        "marker_axis_sha256": contract["scope"]["marker_axis_sha256"],
        "target_truth_opened": False,
    }
    evidence_path = directory / "m35b.cluster_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    receipt = {
        "truth_input_present": False,
        "final_inference_performed": False,
        "evidence_sha256": AGGREGATE.sha256_file(evidence_path),
    }
    (directory / "m35b.screen_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def make_grid(tmp_path: Path, failed_key=None):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    directories = []
    for selection in contract["reference_balance"]["selection_seeds"]:
        for granularity in ("coarse", "fine"):
            for gmm in contract["cluster_screen"]["gmm_seeds"]:
                key = selection, granularity, gmm
                directory = tmp_path / f"s{selection}_{granularity}_g{gmm}"
                write_screen(directory, contract, *key, passed=key != failed_key)
                directories.append(directory)
    return contract, directories


def test_primary_gate_requires_all_nine_coarse_combinations(tmp_path):
    _contract, directories = make_grid(tmp_path)
    output = tmp_path / "gate.json"
    token = tmp_path / "go.json"
    result = AGGREGATE.aggregate(CONTRACT, directories, output, token)
    assert result["status"] == "PASS_M35B_PRIMARY_9_OF_9_GO_PREASSIGNED_FINAL"
    assert result["primary"]["passed"] == result["primary"]["total"] == 9
    assert token.is_file()
    payload = json.loads(token.read_text(encoding="utf-8"))
    assert payload["selection_seed"] == 350101
    assert payload["gmm_seed"] == 351103
    assert payload["granularity"] == "coarse"


def test_no_go_does_not_choose_a_favorable_seed_or_emit_final_token(tmp_path):
    failed = (350202, "coarse", 352207)
    _contract, directories = make_grid(tmp_path, failed)
    output = tmp_path / "gate.json"
    token = tmp_path / "go.json"
    result = AGGREGATE.aggregate(CONTRACT, directories, output, token)
    assert result["status"] == "NO_GO_M35B_PRIMARY_NOT_9_OF_9_STOP_BEFORE_TRUTH"
    assert result["primary"]["passed"] == 8
    assert result["post_hoc_seed_selection"] is False
    assert not token.exists()


def test_fine_failures_are_reported_but_do_not_control_primary_gate(tmp_path):
    failed = (350303, "fine", 353301)
    _contract, directories = make_grid(tmp_path, failed)
    output = tmp_path / "gate.json"
    token = tmp_path / "go.json"
    result = AGGREGATE.aggregate(CONTRACT, directories, output, token)
    assert result["status"] == "PASS_M35B_PRIMARY_9_OF_9_GO_PREASSIGNED_FINAL"
    assert result["sensitivity"]["passed"] == 8
    assert token.is_file()


def test_nextflow_isolated_truth_blind_and_gcp_safe():
    workflow = (ROOT / "workflows" / "m35b_flare2_balanced.nf").read_text(encoding="utf-8")
    module = (ROOT / "modules" / "35B_FLARE2_BALANCED.nf").read_text(encoding="utf-8")
    config = (ROOT / "conf" / "m35b_flare2_balanced.config").read_text(encoding="utf-8")
    screen_process = module.split("process M35B_CLUSTER_SCREEN", 1)[1].split(
        "process M35B_AGGREGATE_CLUSTER_GATE", 1)[0]
    assert "truth" not in screen_process.lower()
    assert "m35b_truth_npz" not in workflow.split("if (runFinal)", 1)[0]
    assert "gs://teams-usp/frank/lai-exploracion-datos/" in workflow
    assert "resourceLabels = [team: 'frank']" in config
    assert "google-batch" in workflow and "@sha256:" in workflow
    assert "overwrite: false" in module
    assert "M27F" not in module


def test_contract_freezes_grid_primary_pair_and_population_stratification():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["reference_balance"]["selection_method"] == (
        "population_floor_then_capacity_weighted_Hamilton_with_sha256_ties")
    assert contract["reference_balance"]["selection_seeds"] == [350101, 350202, 350303]
    assert contract["cluster_screen"]["gmm_seeds"] == [351103, 352207, 353301]
    assert contract["cluster_screen"]["primary_gate"] == (
        "all_9_coarse_selection_by_gmm_combinations_must_pass")
    assert contract["primary_final_pair"] == {
        "selection_seed": 350101,
        "gmm_seed": 351103,
        "granularity": "coarse",
        "direct_flare_reference": "same_balanced_75_sample_subset",
        "paired_methods": ["FLARE_0_6_BALANCED", "FLARE2_BALANCED"],
        "launch_condition": "primary_cluster_gate_passes_9_of_9",
    }


if __name__ == "__main__":
    suite = unittest.TestSuite()
    without_tmp = [
        test_population_floor_hamilton_preserves_populations_and_exact_counts,
        test_hamilton_rejects_a_target_that_cannot_preserve_every_population,
        test_nextflow_isolated_truth_blind_and_gcp_safe,
        test_contract_freezes_grid_primary_pair_and_population_stratification,
    ]
    for function in without_tmp:
        suite.addTest(unittest.FunctionTestCase(function))
    with_tmp = [
        test_primary_gate_requires_all_nine_coarse_combinations,
        test_no_go_does_not_choose_a_favorable_seed_or_emit_final_token,
        test_fine_failures_are_reported_but_do_not_control_primary_gate,
    ]
    for function in with_tmp:
        def wrapped(current=function):
            with tempfile.TemporaryDirectory() as temporary:
                current(Path(temporary))
        suite.addTest(unittest.FunctionTestCase(wrapped))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
