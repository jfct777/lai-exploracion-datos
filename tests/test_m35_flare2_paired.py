import gzip
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
SCRIPT = ROOT / "bin" / "m35_flare2_paired.py"
CONTRACT = ROOT / "conf" / "m35_flare2_paired_contract.json"

SPEC = importlib.util.spec_from_file_location("m35_flare2_paired", SCRIPT)
M35 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M35)

CLOSER_SPEC = importlib.util.spec_from_file_location("m35_close_no_go", ROOT / "bin" / "m35_close_no_go.py")
CLOSER = importlib.util.module_from_spec(CLOSER_SPEC)
assert CLOSER_SPEC.loader is not None
CLOSER_SPEC.loader.exec_module(CLOSER)


def write_vcf(path: Path, samples: list[str], rows: list[list[str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" +
                     "\t".join(samples) + "\n")
        for offset, genotypes in enumerate(rows):
            handle.write(f"22\t{100 + offset * 100}\t.\tA\tG\t.\tPASS\t.\tGT\t" +
                         "\t".join(genotypes) + "\n")


def fixture(tmp_path: Path) -> Namespace:
    reference = tmp_path / "reference.vcf.gz"
    target = tmp_path / "target.vcf.gz"
    write_vcf(reference, ["AFR1", "EUR1", "NAM1"],
              [["0|1", "0|0", "1|0"], ["1|1", "0|1", "1|0"]])
    write_vcf(target, ["T1"], [["0|1"], ["1|0"]])
    reference_tbi = tmp_path / "reference.vcf.gz.tbi"
    target_tbi = tmp_path / "target.vcf.gz.tbi"
    reference_tbi.write_bytes(b"reference-index")
    target_tbi.write_bytes(b"target-index")
    panel = tmp_path / "sample-map.tsv"
    panel.write_text("AFR1\tAFR\nEUR1\tEUR\nNAM1\tNAM\n", encoding="utf-8")
    panel_macro = tmp_path / "panel-macro.tsv"
    panel_macro.write_text("AFR\tAFR\nEUR\tEUR\nNAM\tNAM\n", encoding="utf-8")
    genetic_map = tmp_path / "map.tsv"
    genetic_map.write_text("22\t50\t0\n22\t150\t0.1\n22\t250\t0.2\n", encoding="utf-8")
    jar = tmp_path / "flare.jar"
    jar.write_bytes(b"flare")
    builder = tmp_path / "create_model_file.py"
    builder.write_text("print('fixture')\n", encoding="utf-8")
    return Namespace(contract=CONTRACT, reference_vcf=reference, reference_tbi=reference_tbi,
                     target_vcf=target, target_tbi=target_tbi, sample_map=panel, panel_macro_map=panel_macro,
                     genetic_map=genetic_map, flare_jar=jar,
                     flare2_model_builder=builder, flare2_upstream_model_builder=builder,
                     outdir=tmp_path / "m35", preflight_only=True)


def test_contract_freezes_only_algorithmic_delta_on_a_shared_marker_spine():
    contract = M35.load_contract(CONTRACT)
    invariants = contract["comparison"]["fairness_invariants"]
    assert "marker_ref_alt_axis" in invariants
    assert "target_mosaics_and_target_phase" in invariants
    assert contract["methods"]["flare_0_6"]["parameters"]["panel-probs"] is False
    assert contract["methods"]["flare2"]["panel_probability_parameters"]["panel-probs"] is True
    assert contract["methods"]["flare2"]["final_parameters"]["update-p"] is True


def test_preflight_writes_delta_manifest_and_unmeasured_resource_plan(tmp_path):
    args = fixture(tmp_path)
    receipt = M35.run(args)
    delta = json.loads((args.outdir / "m35_paired.delta_manifest.json").read_text())
    resource = json.loads((args.outdir / "m35_paired.resource_estimate.json").read_text())
    assert receipt["status"] == "PASS_PREFLIGHT_ONLY"
    assert delta["shared_axes"]["marker_count"] == 2
    assert delta["shared_axes"]["target_phase_verified"] is True
    assert delta["method_delta"]["flare2"]["additional_information"] == "clustered reference-panel model only"
    assert resource["status"] == "UNMEASURED_PLANNING_ESTIMATE"
    assert receipt["label_input_present"] is False


def test_direct_and_flare2_commands_differ_only_in_model_construction_and_declared_parameters(tmp_path):
    args = fixture(tmp_path)
    contract = M35.load_contract(args.contract)
    prepared = M35.preflight(contract, {name: getattr(args, name) for name in M35.INPUT_NAMES}, args.outdir)
    direct = prepared["commands"]["direct"]
    panels = prepared["commands"]["panels"]
    final = prepared["commands"]["final"]
    assert "panel-probs=false" in direct and not any(item.startswith("model=") for item in direct)
    assert "panel-probs=true" in panels
    assert "update-p=true" in final and any(item.startswith("model=") for item in final)
    assert "ref=" + str(args.reference_vcf) in direct
    assert "gt=" + str(args.target_vcf) in final


def test_preflight_rejects_unphased_target(tmp_path):
    args = fixture(tmp_path)
    write_vcf(args.target_vcf, ["T1"], [["0/1"], ["1|0"]])
    with pytest.raises(M35.PairedContractError, match="unphased GT"):
        M35.run(args)


def test_cluster_assignment_writes_no_go_evidence_without_final_flare2(tmp_path):
    model = tmp_path / "model.model"
    model.write_text(
        "# list of reference panels\nAFR\tEUR\tNAM\n"
        "# p[i][j]: probability that a model state haplotype is in reference panel j\n"
        "#          when the model state ancestry is i\n"
        "0.96 0.03 0.01\n0.01 0.98 0.01\n0.10 0.60 0.30\n",
        encoding="utf-8",
    )
    evidence = M35.cluster_assignment_evidence_from_model(
        model, ["AFR", "EUR", "NAM"], {"AFR": "AFR", "EUR": "EUR", "NAM": "NAM"}, .5, .25,
    )
    assert evidence["status"] == "NO_GO_TRUTH_BLIND_CLUSTER_ASSIGNMENT"
    assert evidence["cluster_to_ancestry"] == {"0": "AFR", "1": "EUR", "2": "NAM"}
    assert evidence["selected_panel_probability"]["2"] == .30
    assert evidence["failure_reasons"] == ["insufficient_panel_support"]
