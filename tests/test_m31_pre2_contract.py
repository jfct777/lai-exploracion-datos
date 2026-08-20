from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "conf" / "m31_ordered_linear_pre2_preregistration.json"
MODULE_PATH = ROOT / "bin" / "m31_pre2_contract.py"
SPEC = importlib.util.spec_from_file_location("m31_pre2_contract", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_payload() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def write_mutation(tmp_path: Path, path: tuple[str, ...], value: object) -> Path:
    payload = copy.deepcopy(load_payload())
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    output = tmp_path / ("-".join(path) + ".json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def test_canonical_contract_passes() -> None:
    payload = MODULE.validate_contract(CONTRACT)
    assert payload["implementation"]["real_run_authorized"] is False
    assert payload["roots"]["one_way_evaluation"]["label"] == "root18"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol_class",), "CONFIRMATORY"),
        (("scope",), "two_way"),
        (("prior_evidence", "pre1_is_immutable"), False),
        (("prior_evidence", "root18_used_by_prior_m29r_m30"), False),
        (("implementation", "real_run_authorized"), True),
        (("implementation", "container_digest"), "latest"),
        (("roots", "reciprocal_direction_forbidden"), False),
        (("arms", "scientific"), ["F0", "D"]),
        (("arms", "C_is_scientific_comparator"), True),
        (("selection", "tau"), 0.001),
        (("selection", "empty_guarded_set"), "BEST_AVAILABLE"),
        (("selection", "lexicographic_order"), ["-F1", "MAE"]),
        (("root18_open_gate", "if_any_requirement_fails"), "CONTINUE"),
        (("root18_decision", "failure_label"), "RETRY"),
        (("uncertainty", "role"), "significance_gate"),
        (("parallelism", "workers_screen"), [4]),
        (("claims_excluded",), ["DNABR_generalization"]),
    ],
)
def test_every_load_bearing_mutation_fails(tmp_path: Path, path: tuple[str, ...], value: object) -> None:
    mutated = write_mutation(tmp_path, path, value)
    with pytest.raises(MODULE.ContractError):
        MODULE.validate_payload(json.loads(mutated.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("estimand",), "D_vs_F0_only"),
        (("prior_evidence", "pre1_c_known_answer", "alpha"), 1.0),
        (("implementation", "implementation_basis_sha256", "bin/run_m31_ordered_linear.py"), "0" * 64),
        (("input_sha256", "root17.target"), "0" * 64),
        (("rare_universe", "minor_presence"), "I(state == ALT)"),
        (("rare_universe", "unsupported_in_REF_LAI"), "encode_zero"),
        (("rare_universe", "missing_call_policy"), "missing_is_zero"),
        (("arms", "ASIA_is_not_NAM"), False),
        (("selection", "development_minimum_delta_F1"), 1e-15),
        (("root18_open_gate", "scientific_requirements"), ["G_D_nonempty"]),
        (("root18_decision", "required"), ["F1_D>=F1_each_applicable_comparator+0.01"]),
        (("root18_decision", "success_if_G_L_empty"), "CANDIDATE_D_FOR_NEW_PROSPECTIVE_ROOTS"),
        (("claims_excluded",), ["confirmatory_validation"]),
    ],
)
def test_biological_and_decision_mutations_fail(path: tuple[str, ...], value: object) -> None:
    payload = copy.deepcopy(load_payload())
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(MODULE.ContractError, match="semantics drifted"):
        MODULE.validate_payload(payload)


def test_whitespace_drift_fails_file_hash_but_not_semantics(tmp_path: Path) -> None:
    payload = load_payload()
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    assert MODULE.validate_payload(payload)["experiment_id"] == "M31_ORDERED_LINEAR_DEV_PRE2"
    with pytest.raises(MODULE.ContractError, match="bytes differ"):
        MODULE.validate_contract(reformatted)


def test_nonfinite_json_fails(tmp_path: Path) -> None:
    text = CONTRACT.read_text(encoding="utf-8").replace('"tau": 1e-15', '"tau": NaN')
    mutated = tmp_path / "nan.json"
    mutated.write_text(text, encoding="utf-8")
    with pytest.raises(MODULE.ContractError):
        MODULE.validate_contract(mutated)
    payload = load_payload()
    payload["selection"]["tau"] = float("nan")
    with pytest.raises(MODULE.ContractError, match="must be finite"):
        MODULE.validate_payload(payload)


def test_cli_writes_contract_only_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "report.json"
    assert MODULE.main(["--contract", str(CONTRACT), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == {
        "schema_version": "2.0.0",
        "experiment_id": "M31_ORDERED_LINEAR_DEV_PRE2",
        "status": "PASS_CONTRACT_ONLY_NO_DATA",
        "contract_sha256": MODULE.EXPECTED_SHA256,
        "real_run_authorized": False,
    }
    assert "PASS_CONTRACT_ONLY_NO_DATA" in capsys.readouterr().out


def test_cli_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("keep\n", encoding="utf-8")
    with pytest.raises(MODULE.ContractError, match="refusing to overwrite"):
        MODULE.main(["--contract", str(CONTRACT), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "keep\n"
