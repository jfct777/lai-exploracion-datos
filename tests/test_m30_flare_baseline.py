import gzip
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "m30_flare_baseline.py"
PREREG = ROOT / "conf" / "m30_flare_baseline_preregistration.json"
CONFIG = ROOT / "conf" / "m30_flare_baseline.config"
WORKFLOW = ROOT / "workflows" / "m30_flare_baseline.nf"
MODULE = ROOT / "modules" / "30_FLARE_BASELINE.nf"
DOCKERFILE = ROOT / "containers" / "m30-flare" / "Dockerfile"

SPEC = importlib.util.spec_from_file_location("m30_flare_baseline", SCRIPT)
M30 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M30)


def write_vcf(path: Path, samples: list[str], genotypes: list[list[str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(samples) + "\n")
        for offset, row in enumerate(genotypes):
            handle.write(f"22\t{100 + offset * 100}\t.\tA\tG\t.\tPASS\t.\tGT\t")
            handle.write("\t".join(row) + "\n")


def miniature_preflight_inputs(tmp_path: Path) -> Namespace:
    ref = tmp_path / "reference.vcf.gz"
    target = tmp_path / "target.vcf.gz"
    write_vcf(ref, ["AFR_1", "EUR_1", "ASIA_1"], [["0|1"] * 3, ["1|0"] * 3])
    write_vcf(target, ["TARGET_1"], [["0|1"], ["1|0"]])
    ref_tbi = tmp_path / "reference.vcf.gz.tbi"
    target_tbi = tmp_path / "target.vcf.gz.tbi"
    ref_tbi.write_bytes(b"ref-index")
    target_tbi.write_bytes(b"target-index")
    panel = tmp_path / "sample_map.tsv"
    panel.write_text("AFR_1\tAFR\nEUR_1\tEUR\nASIA_1\tASIA\n", encoding="utf-8")
    genetic_map = tmp_path / "map.tsv"
    genetic_map.write_text("22\t50\t0.1\n22\t200\t0.2\n22\t300\t0.3\n", encoding="utf-8")
    fb = tmp_path / "query_results.fb"
    msp = tmp_path / "query_results.msp"
    fb.write_text("frozen fb\n", encoding="utf-8")
    msp.write_text("frozen msp\n", encoding="utf-8")
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({
        "stage": "M29_AUTHENTICATED_B0_BINDING",
        "root_seed": 20260817,
        "sha256": {"fb": M30.sha256(fb), "msp": M30.sha256(msp)},
    }), encoding="utf-8")

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    prereg["input_contract"].update({
        "marker_count": 2,
        "reference_sample_count": 3,
        "target_sample_count": 1,
        "reference_panel_counts": {"AFR": 1, "EUR": 1, "ASIA": 1},
        "map_row_count": 3,
        "map_sha256": M30.sha256(genetic_map),
    })
    prereg["roots"]["root17"]["coordinate_range_bp"] = [100, 200]
    paths = {
        "reference_vcf": ref,
        "reference_tbi": ref_tbi,
        "target_vcf": target,
        "target_tbi": target_tbi,
        "sample_map": panel,
        "gnomix_binding": binding,
        "gnomix_fb": fb,
        "gnomix_msp": msp,
    }
    for label, path in paths.items():
        prereg["roots"]["root17"]["inputs"][label] = {
            "uri": f"fixture://{path.name}", "sha256": M30.sha256(path)
        }
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    return Namespace(
        root_label="root17",
        root_seed=20260817,
        preregistration=prereg_path,
        container_image="example.invalid/flare@sha256:test",
        container_digest="sha256:test",
        flare_jar_sha256=M30.EXPECTED_FLARE["jar_sha256"],
        reference_vcf=ref,
        reference_tbi=ref_tbi,
        target_vcf=target,
        target_tbi=target_tbi,
        sample_map=panel,
        genetic_map=genetic_map,
        gnomix_binding=binding,
        gnomix_fb=fb,
        gnomix_msp=msp,
        prior_root_audit=None,
        outdir=tmp_path / "out",
    )


def test_preregistration_freezes_direct_flare_and_defers_flare2():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    flare = prereg["methods"]["flare"]
    assert flare["version"] == "0.6.0"
    assert flare["reported_build"] == "616fcc9d4 03-Nov-2025"
    assert flare["jar_sha256"] == M30.EXPECTED_FLARE["jar_sha256"]
    assert "git_tag" not in flare and "git_commit" not in flare
    assert flare["source_head_is_not_asserted_equivalent_to_jar"] is True
    assert flare["parameters"] == {
        "array": False,
        "probs": True,
        "em": True,
        "min-mac": 1,
        "min-maf": 0.0,
        "gen": 10.0,
        "update-p": False,
        "panel-probs": False,
        "seed": 3001701,
        "nthreads": 4,
    }
    assert prereg["methods"]["flare2"]["status"] == "DEFERRED"
    assert prereg["truth_policy"]["truth_permitted_in_this_workflow"] is False


def test_preflight_builds_truth_blind_runtime_contract(tmp_path):
    args = miniature_preflight_inputs(tmp_path)
    M30.preflight(args)
    contract = json.loads((args.outdir / "root17.m30.run_contract.json").read_text())
    report = json.loads((args.outdir / "root17.m30.preflight.json").read_text())
    assert contract["status"] == "PREFLIGHT_PASS"
    assert contract["shape"]["markers"] == 2
    assert contract["shape"]["reference_panel_counts"] == {"AFR": 1, "EUR": 1, "ASIA": 1}
    assert contract["truth_accessed"] is False
    assert contract["scoring_implemented"] is False
    assert contract["flare2"]["status"] == "DEFERRED"
    assert report["status"] == "PASS"
    assert (args.outdir / "root17.flare.map").read_text().splitlines() == [
        "22\t22:50\t0.1\t50",
        "22\t22:200\t0.2\t200",
        "22\t22:300\t0.3\t300",
    ]


def test_preflight_fails_closed_on_hash_drift(tmp_path):
    args = miniature_preflight_inputs(tmp_path)
    args.target_tbi.write_bytes(b"changed")
    with pytest.raises(M30.ContractError, match="SHA-256 mismatch"):
        M30.preflight(args)


def test_scan_vcf_rejects_unphased_genotype(tmp_path):
    vcf = tmp_path / "unphased.vcf.gz"
    write_vcf(vcf, ["S1"], [["0/1"]])
    with pytest.raises(M30.ContractError, match="Unphased GT"):
        M30.scan_vcf(vcf, "22")


def test_flare_command_has_no_marker_filter_and_fixed_threads(tmp_path):
    command = M30.build_flare_command(
        "java",
        tmp_path / "flare.jar",
        tmp_path / "ref.vcf.gz",
        tmp_path / "gt.vcf.gz",
        tmp_path / "panel.tsv",
        tmp_path / "map.tsv",
        tmp_path / "out",
        M30.EXPECTED_PARAMS,
    )
    assert "array=false" in command
    assert "min-mac=1" in command
    assert "min-maf=0" in command
    assert "probs=true" in command
    assert "em=true" in command
    assert "gen=10" in command
    assert "update-p=false" in command
    assert "panel-probs=false" in command
    assert "nthreads=4" in command
    assert "seed=3001701" in command


def test_flare_model_and_log_audits_match_version_060_format(tmp_path):
    model = tmp_path / "run.model"
    model.write_text(
        "# list of ancestries\nAFR EUR ASIA\n\n"
        "# list of reference panels\nAFR EUR ASIA\n\n"
        "# T\n10.0\n# mu\n0.2 0.5 0.3\n"
        "# p\n1 0 0\n0 1 0\n0 0 1\n"
        "# theta\n0.01 0.02 0.03\n0.02 0.01 0.03\n0.03 0.02 0.01\n"
        "# rho\n2 3 4\n",
        encoding="utf-8",
    )
    model_audit = M30.audit_flare_model(model, ["AFR", "EUR", "ASIA"])
    assert model_audit["finite_parameters"] is True
    assert model_audit["ancestry_order"] == ["AFR", "EUR", "ASIA"]

    log = tmp_path / "run.log"
    lines = ["flare version 0.6.0 [616fcc9d4 03-Nov-2025]", "Parameters"]
    for name, value in {
        "array": "false", "min-maf": "0.0", "min-mac": "1",
        "probs": "true", "gen": "10.0", "em": "true",
        "update-p": "false", "nthreads": "4", "seed": "3001701",
    }.items():
        lines.append(f"  {name} : {value}")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_audit = M30.audit_flare_log(log, M30.EXPECTED_PARAMS)
    assert log_audit["effective_parameters"]["em"] == "true"


def test_nextflow_contract_is_sequential_and_has_no_truth_input():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    module = MODULE.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    assert workflow.index("M30_RUN_FLARE_ROOT17(") < workflow.index("M30_PREFLIGHT_ROOT18(")
    assert "M30_RUN_FLARE_ROOT17.out.audit" in workflow
    assert "--prior-root-audit" in module
    assert "resourceLabels = [team: 'frank']" in config
    assert "maxForks = 1" in config
    assert "executor.queueSize = 1" in config
    assert "def repositoryHead = headValue" in workflow
    assert "DNABR_GIT_COMMIT does not match repository HEAD" in workflow
    assert "def gitCommit = repositoryHead" in workflow
    assert "m30_root18_reference_tbi" in config
    assert "root18/ingest/m28c_b0_reference.vcf.gz.tbi" in config
    root17_inference = module.split("process M30_RUN_FLARE_ROOT17", 1)[1].split(
        "process M30_PREFLIGHT_ROOT18", 1
    )[0]
    root18_inference = module.split("process M30_RUN_FLARE_ROOT18", 1)[1].split(
        "process M30_SCORE_FLARE_VS_GNOMIX", 1
    )[0]
    assert "truth" not in root17_inference.lower()
    assert "truth" not in root18_inference.lower()
    assert "m28_lai_truth" in config


def test_runtime_container_recipe_is_pinned_and_does_not_vendor_jar():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    assert "eclipse-temurin:17-jre-jammy@sha256:1e38389e" in dockerfile
    assert M30.EXPECTED_FLARE["jar_sha256"] in dockerfile
    assert "COPY flare.jar /opt/flare/flare.jar" in dockerfile
    assert not (DOCKERFILE.parent / "flare.jar").exists()
    assert "m30-flare-runtime@sha256:86bf36c5" in config
    assert "software/flare/0.6.0/flare.jar" in config
