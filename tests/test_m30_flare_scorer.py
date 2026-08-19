import gzip
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
sys.path.insert(0, str(BIN))
SCRIPT = BIN / "m30_flare_scorer.py"
PREREG = ROOT / "conf" / "m30_flare_baseline_preregistration.json"
MODULE_NF = ROOT / "modules" / "30_FLARE_BASELINE.nf"
WORKFLOW_NF = ROOT / "workflows" / "m30_flare_baseline.nf"

SPEC = importlib.util.spec_from_file_location("m30_flare_scorer", SCRIPT)
M30 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M30
SPEC.loader.exec_module(M30)


def write_flare(path: Path, probability_a="0.33,0.33,0.33", hard_a="1"):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##ANCESTRY=<AFR=0,EUR=1,ASIA=2>\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT000\n")
        handle.write(
            "22\t10\t.\tA\tG\t.\tPASS\t.\tGT:AN1:AN2:ANP1:ANP2\t"
            f"0|1:{hard_a}:2:{probability_a}:0.00,0.00,1.00\n"
        )


def test_known_answers_cover_rules_alignment_phase_and_boundaries():
    checks = M30.run_known_answers()
    assert checks
    assert all(checks.values())
    assert checks["lineage_phase_not_swapped_post_truth"]
    assert checks["ordered_boundary_matching_is_one_to_one"]


def test_flare_two_decimal_probabilities_are_renormalized_and_ties_allowed(tmp_path):
    path = tmp_path / "flare.vcf.gz"
    write_flare(path)
    grid = M30.load_flare_grid(path)
    vector = grid.probabilities[0]["T000"][0]
    assert sum(vector) == pytest.approx(1.0)
    assert vector == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert grid.hard_labels[0]["T000"][0] == "EUR"


def test_flare_probability_sum_outside_rounding_interval_fails(tmp_path):
    path = tmp_path / "bad.vcf.gz"
    write_flare(path, probability_a="0.30,0.30,0.30", hard_a="0")
    with pytest.raises(M30.ScoringError, match=r"outside \[0.99,1.01\]"):
        M30.load_flare_grid(path)


def test_flare_hard_call_must_be_one_of_probability_maxima(tmp_path):
    path = tmp_path / "bad_hard.vcf.gz"
    write_flare(path, probability_a="0.80,0.10,0.10", hard_a="1")
    with pytest.raises(M30.ScoringError, match="not among tied probability maxima"):
        M30.load_flare_grid(path)


def test_alignment_rejects_ref_alt_mismatch():
    target = M30.TargetGrid((("22", 10, "A", "G"),), ("T000",))
    probabilities = ({"T000": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))},)
    labels = ({"T000": ("AFR", "EUR")},)
    gnomix = M30.PredictionGrid(target.loci, target.samples, probabilities, labels)
    flare = M30.PredictionGrid((("22", 10, "A", "T"),), target.samples, probabilities, labels)
    with pytest.raises(M30.ScoringError, match="FLARE locus grid"):
        M30.validate_alignment(target, gnomix, flare)


def test_macro_f1_reports_supported_classes_and_withholds_fixed_six_when_absent():
    summary = M30._macro_f1(Counter({("AA", "AA"): 1.0, ("AE", "AE"): 1.0}))
    assert summary["macro_f1_truth_supported"] == 1.0
    assert summary["macro_f1_fixed_six"] is None
    assert summary["truth_supported_classes"] == ["AA", "AE"]


def decision_fixture():
    root_delta = {
        "primary_macro_mae": -0.01,
        "mae_total": {ancestry: -0.01 for ancestry in M30.ANCESTRIES},
        "mae_truth_present": {ancestry: 0.0 for ancestry in M30.ANCESTRIES},
        "brier": -0.01,
        "macro_f1_truth_supported": 0.01,
        "boundaries": {
            "0.1": {"f1": 0.0, "false_transitions_per_cm": 0.0},
            "0.2": {"f1": 0.0, "false_transitions_per_cm": 0.0},
            "0.5": {"f1": 0.0, "false_transitions_per_cm": 0.0},
        },
    }
    roots = {root: json.loads(json.dumps(root_delta)) for root in M30.ROOTS}
    pooled = json.loads(json.dumps(root_delta))
    bootstrap = {"metrics": {"primary_macro_mae": {"lower": -0.02, "upper": -0.001}}}
    return roots, pooled, bootstrap


def test_decision_uses_exact_nonworsening_without_epsilon():
    roots, pooled, bootstrap = decision_fixture()
    assert M30.decide(roots, pooled, bootstrap)["label"] == "GO_FLARE_NEXT_DEV"
    pooled["mae_total"]["AFR"] = 1e-15
    assert M30.decide(roots, pooled, bootstrap)["label"] == "INCONCLUSIVE_TRADEOFF"


def test_interval_crossing_zero_is_inconclusive_not_keep():
    roots, pooled, bootstrap = decision_fixture()
    bootstrap["metrics"]["primary_macro_mae"]["upper"] = 0.0
    assert M30.decide(roots, pooled, bootstrap)["label"] == "INCONCLUSIVE_TRADEOFF"


def test_clear_direction_against_flare_keeps_gnomix():
    roots, pooled, bootstrap = decision_fixture()
    for root in M30.ROOTS:
        roots[root]["primary_macro_mae"] = 0.01
    assert M30.decide(roots, pooled, bootstrap)["label"] == "KEEP_GNOMIX"


def test_contract_freezes_bootstrap_and_truth_isolated_to_final_scorer():
    contract = json.loads(PREREG.read_text(encoding="utf-8"))
    scoring = contract["scoring_contract"]
    assert scoring["bootstrap"] == {
        "scheme": "Paired resampling of individuals with replacement inside each root; the same sampled indices are used for Gnomix and FLARE; root estimates are averaged with weight one half each.",
        "replicates": 10000,
        "seed": 3001702,
        "interval": "two-sided percentile 95%",
        "unit": "diploid individual",
    }
    assert "swap" not in scoring["phase_policy"].lower().replace("no global or local post-truth haplotype swap", "")
    module = MODULE_NF.read_text(encoding="utf-8")
    workflow = WORKFLOW_NF.read_text(encoding="utf-8")
    root17_process = module.split("process M30_RUN_FLARE_ROOT17", 1)[1].split("process M30_PREFLIGHT_ROOT18", 1)[0]
    assert "truth" not in root17_process.lower()
    assert workflow.index("M30_SCORER_KNOWN_ANSWERS(") < workflow.index("M30_RUN_FLARE_ROOT17(")
    assert workflow.index("M30_RUN_FLARE_ROOT18(") < workflow.index("M30_SCORE_FLARE_VS_GNOMIX(")
