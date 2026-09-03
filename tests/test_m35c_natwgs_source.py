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


PREPARE = load_script("m35c_prepare_source_comparison")
AGGREGATE = load_script("m35c_aggregate_source_gate")
CONTRACT = ROOT / "conf" / "m35c_natwgs_source_contract.json"


def make_natwgs_candidates():
    candidates = {}
    sizes = {
        "Brazil": ("BR", 17), "Peru": ("PE", 18), "Bolivia": ("BO", 16),
        "Mexico": ("MX", 8), "Argentina": ("AR", 7), "Ecuador": ("EC", 2),
        "Paraguay": ("PY", 1), "Colombia": ("CO", 1),
    }
    for country, (prefix, size) in sizes.items():
        for index in range(size):
            sample = f"{prefix}_{index:02d}"
            candidates[sample] = {
                "Country": country,
                "Population": f"{prefix}_POP_{index % max(1, min(size, 6))}",
            }
    return candidates


def test_natwgs_selection_is_exact_diverse_and_deterministic():
    candidates = make_natwgs_candidates()
    selected, receipt = PREPARE.select_natwgs(candidates, 350101, 23)
    repeated, repeated_receipt = PREPARE.select_natwgs(candidates, 350101, 23)
    assert selected == repeated
    assert receipt == repeated_receipt
    assert len(selected) == 23
    assert receipt["selected_country_count"] == 8
    assert receipt["selected_population_count"] >= 20
    assert sum(receipt["country_allocation"].values()) == 23
    assert all(value >= 1 for value in receipt["country_allocation"].values())


def test_natwgs_selection_seeds_are_not_collapsed_to_one_subset():
    candidates = make_natwgs_candidates()
    subsets = [PREPARE.select_natwgs(candidates, seed, 23)[0]
               for seed in (350101, 350202, 350303)]
    assert len({frozenset(subset) for subset in subsets}) == 3


def write_screen(directory: Path, arm: str, selection: int, granularity: str,
                 gmm: int, passed: bool = True, preparation_hash: str = "a" * 64):
    directory.mkdir()
    evidence = {
        "status": "PASS_M35C_CLUSTER_SEPARATION" if passed else "NO_GO_M35C_CLUSTER_SEPARATION",
        "arm": arm,
        "selection_seed": selection,
        "granularity": granularity,
        "gmm_seed": gmm,
        "NAM_support": 0.8 if passed else 0.49,
        "log_margin": 0.5,
        "balanced_macro_counts": {"AFR": 23, "EUR": 23, "NAM": 23},
        "marker_axis_sha256": "e82ef9b853283de33f5873b2fbdebebe79291969a9fe43deb4c8d685d4a71ea0",
        "target_truth_opened": False,
    }
    evidence_path = directory / "m35c.cluster_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    receipt = {
        "truth_input_present": False,
        "final_inference_performed": False,
        "evidence_sha256": AGGREGATE.sha256_file(evidence_path),
        "prepare_receipt_sha256": preparation_hash,
    }
    (directory / "m35c.screen_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def make_grid(base: Path, failed_key=None):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    directories = []
    for arm in ("EXTERNAL_NAM", "NATWGS"):
        for selection in contract["reference_design"]["selection_seeds"]:
            for granularity in ("coarse", "fine"):
                for gmm in contract["cluster_screen"]["gmm_seeds"]:
                    key = arm, selection, granularity, gmm
                    directory = base / "_".join(map(str, key))
                    write_screen(directory, *key, passed=key != failed_key,
                                 preparation_hash=f"{selection:064d}"[-64:])
                    directories.append(directory)
    return directories


def test_gate_requires_all_nine_natwgs_coarse_cases_only(tmp_path):
    directories = make_grid(tmp_path, failed_key=("EXTERNAL_NAM", 350101, "coarse", 351103))
    output = tmp_path / "gate.json"
    token = tmp_path / "go.json"
    result = AGGREGATE.aggregate(CONTRACT, directories, output, token)
    assert result["status"] == "PASS_M35C_NATWGS_PRIMARY_9_OF_9_GO_PREASSIGNED_POST_GATE"
    assert result["primary"]["passed"] == 9
    assert result["matched_comparator"]["passed"] == 8
    assert token.is_file()


def test_gate_stops_before_truth_on_any_natwgs_coarse_failure(tmp_path):
    directories = make_grid(tmp_path, failed_key=("NATWGS", 350202, "coarse", 352207))
    output = tmp_path / "gate.json"
    token = tmp_path / "go.json"
    result = AGGREGATE.aggregate(CONTRACT, directories, output, token)
    assert result["status"] == "NO_GO_M35C_NATWGS_PRIMARY_NOT_9_OF_9_STOP_BEFORE_TRUTH"
    assert result["primary"]["passed"] == 8
    assert result["truth_opened"] is False
    assert result["post_hoc_seed_selection"] is False
    assert not token.exists()


def test_workflow_is_truth_blind_nextflow_first_and_personal_bucket_only():
    workflow = (ROOT / "workflows" / "m35c_natwgs_source.nf").read_text(encoding="utf-8")
    module = (ROOT / "modules" / "35C_NATWGS_SOURCE.nf").read_text(encoding="utf-8")
    config = (ROOT / "conf" / "m35c_natwgs_source.config").read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert "truth" not in workflow.lower()
    assert "truth" not in module.lower()
    assert "gs://teams-usp/frank/lai-exploracion-datos/" in workflow
    assert "resourceLabels = [team: 'frank']" in config
    assert "overwrite: false" in module
    assert contract["relatedness_policy"]["method"] == "PC_Relate_without_KING"
    assert contract["reference_design"]["counts_per_arm"] == {"AFR": 23, "EUR": 23, "NAM": 23}
    assert contract["cluster_screen"]["primary_gate"] == (
        "all_9_NATWGS_coarse_combinations_must_pass"
    )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for function in (
        test_natwgs_selection_is_exact_diverse_and_deterministic,
        test_natwgs_selection_seeds_are_not_collapsed_to_one_subset,
        test_workflow_is_truth_blind_nextflow_first_and_personal_bucket_only,
    ):
        suite.addTest(unittest.FunctionTestCase(function))
    for function in (
        test_gate_requires_all_nine_natwgs_coarse_cases_only,
        test_gate_stops_before_truth_on_any_natwgs_coarse_failure,
    ):
        def wrapped(current=function):
            with tempfile.TemporaryDirectory() as temporary:
                current(Path(temporary))
        suite.addTest(unittest.FunctionTestCase(wrapped))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
