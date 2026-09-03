#!/usr/bin/env python3

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import m34_parse_flare_truth as m34  # noqa: E402
from m38b_build_flare_contract import (  # noqa: E402
    FIXED_PARAMETERS,
    INPUT_MEMBERS,
    M38BFlareContractError,
    build_contract,
)
from m38b_parse_flare import parse_f0  # noqa: E402
from m38b_project_baselines import (  # noqa: E402
    M38BBaselineAlignmentError,
    align_baselines,
)
from m38b_run_flare import load_contract  # noqa: E402


WORKFLOW = ROOT / "workflows/m38b_prepare_baselines.nf"
CONFIG = ROOT / "conf/m38b_prepare_baselines.config"
MODULES = [
    ROOT / "modules/38B_FLARE_CONTRACT.nf",
    ROOT / "modules/38B_FLARE_RUN.nf",
    ROOT / "modules/38B_PARSE_F0.nf",
    ROOT / "modules/38B_PROJECT_BASELINES.nf",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_experiment(source_artifacts: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "experiment_id": "M38B_S660_INCREMENTAL_LAI_CHR22_R0_FIT",
        "status": "PREREGISTERED_AMENDED_BEFORE_OUTCOME_ACCESS",
        "claim_scope": {
            "analysis_level": "EXPLORATORY",
            "chromosome": "22",
            "mosaic_root": "R0",
            "target_partition": "FIT_ONLY",
            "target_people": 96,
            "valid_opened": False,
            "test_opened": False,
        },
        "locus_universes": {
            "f_full_count": 42986,
            "s660_count": 660,
            "f_minus_s660_count": 42326,
            "f_minus_is_common_only": False,
            "variant_key": ["CHROM", "POS", "REF", "ALT"],
        },
        "flare_parameters": {
            "array": False,
            "probs": True,
            "em": True,
            "min_mac": 1,
            "min_maf": 0.0,
            "generations": 12.0,
            "update_p": False,
            "panel_probs": False,
            "seed": 3401103,
            "nthreads": 4,
        },
        "source_artifacts": source_artifacts,
    }


def write_selected(path: Path, keys: list[tuple[str, int, str, str]]) -> None:
    np.savez(
        path,
        alt=np.asarray([key[3].encode("ascii") for key in keys], dtype="|S1"),
        cM=np.arange(len(keys), dtype="<f8"),
        chrom=np.asarray([int(key[0]) for key in keys], dtype="|u1"),
        locus_id=np.arange(len(keys), dtype="<u8"),
        pos=np.asarray([key[1] for key in keys], dtype="<i8"),
        ref=np.asarray([key[2].encode("ascii") for key in keys], dtype="|S1"),
    )


def make_f0(
    path: Path,
    keys: list[tuple[str, int, str, str]],
    sample_keys: np.ndarray,
    offset: float = 0.0,
) -> None:
    probabilities = np.zeros((len(sample_keys), 2, len(keys), 3), dtype="<f4")
    probabilities[..., 0] = 0.6 - offset
    probabilities[..., 1] = 0.3 + offset
    probabilities[..., 2] = 0.1
    arrays = {
        "sample_key_sha256": np.asarray(sample_keys, dtype="|S64"),
        "marker_chrom": np.asarray([int(row[0]) for row in keys], dtype="|u1"),
        "marker_pos": np.asarray([row[1] for row in keys], dtype="<i8"),
        "marker_ref": np.asarray([row[2].encode("ascii") for row in keys], dtype="|S1"),
        "marker_alt": np.asarray([row[3].encode("ascii") for row in keys], dtype="|S1"),
        "F0": probabilities,
    }
    m34.write_deterministic_npz(path, arrays)


def make_marker_cm(path: Path, values: list[float]) -> None:
    m34.write_deterministic_npz(path, {"marker_cM": np.asarray(values, dtype="<f8")})


def make_truth(path: Path, positions: list[int], sample_keys: np.ndarray) -> None:
    labels = np.zeros((len(sample_keys), 2, len(positions)), dtype="|i1")
    labels[:, 1, :] = 1
    labels[:, :, -1] = 2
    m34.write_deterministic_npz(
        path,
        {
            "sample_key_sha256": np.asarray(sample_keys, dtype="|S64"),
            "marker_pos": np.asarray(positions, dtype="<i8"),
            "labels": labels,
        },
    )


class M38BContractTests(unittest.TestCase):
    def test_contract_freezes_exact_m34_flare_parameters_and_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for index, name in enumerate(INPUT_MEMBERS):
                path = root / name
                path.write_bytes(f"input-{index}\n".encode("ascii"))
                inputs[name] = path
            source = {
                "f_minus_reference_vcf_sha256": digest(inputs["reference_vcf"]),
                "f_minus_target_vcf_sha256": digest(inputs["target_vcf"]),
            }
            experiment = root / "experiment.json"
            write_json(experiment, base_experiment(source))
            hashes = {name: digest(path) for name, path in inputs.items()}
            contract = build_contract(
                experiment_path=experiment,
                inputs=inputs,
                expected_sha256=hashes,
            )
            self.assertEqual(contract["parameters"], FIXED_PARAMETERS)
            self.assertEqual(contract["expected_shape"]["marker_count"], 42326)
            self.assertEqual(contract["scope"]["target_partition"], "FIT")
            self.assertFalse(contract["scope"]["valid_opened"])
            self.assertFalse(contract["scope"]["truth_available_to_stage"])
            contract_path = root / "flare.contract.json"
            write_json(contract_path, contract)
            self.assertEqual(load_contract(contract_path)["parameters"], FIXED_PARAMETERS)

    def test_contract_accepts_authenticated_nextflow_symlink_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "sources"
            staged_dir = root / "staged"
            source_dir.mkdir()
            staged_dir.mkdir()
            inputs = {}
            for index, name in enumerate(INPUT_MEMBERS):
                source = source_dir / name
                source.write_bytes(f"input-{index}\n".encode("ascii"))
                staged = staged_dir / name
                staged.symlink_to(source)
                inputs[name] = staged
            source_hashes = {
                "f_minus_reference_vcf_sha256": digest(inputs["reference_vcf"]),
                "f_minus_target_vcf_sha256": digest(inputs["target_vcf"]),
            }
            experiment = source_dir / "experiment.json"
            write_json(experiment, base_experiment(source_hashes))
            staged_experiment = staged_dir / "experiment.json"
            staged_experiment.symlink_to(experiment)
            hashes = {name: digest(path) for name, path in inputs.items()}
            contract = build_contract(
                experiment_path=staged_experiment,
                inputs=inputs,
                expected_sha256=hashes,
            )
            self.assertEqual(contract["expected_sha256"], hashes)

    def test_contract_rejects_parameter_drift_and_wrong_vcf_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for index, name in enumerate(INPUT_MEMBERS):
                path = root / name
                path.write_bytes(f"input-{index}\n".encode("ascii"))
                inputs[name] = path
            source = {
                "f_minus_reference_vcf_sha256": digest(inputs["reference_vcf"]),
                "f_minus_target_vcf_sha256": digest(inputs["target_vcf"]),
            }
            experiment_payload = base_experiment(source)
            experiment_payload["flare_parameters"]["seed"] = 1
            experiment = root / "bad-experiment.json"
            write_json(experiment, experiment_payload)
            hashes = {name: digest(path) for name, path in inputs.items()}
            with self.assertRaisesRegex(M38BFlareContractError, "parameters"):
                build_contract(
                    experiment_path=experiment,
                    inputs=inputs,
                    expected_sha256=hashes,
                )
            experiment_payload["flare_parameters"]["seed"] = 3401103
            write_json(experiment, experiment_payload)
            hashes["target_vcf"] = "0" * 64
            with self.assertRaisesRegex(M38BFlareContractError, "target_vcf"):
                build_contract(
                    experiment_path=experiment,
                    inputs=inputs,
                    expected_sha256=hashes,
                )


class M38BParseTests(unittest.TestCase):
    def test_parse_reuses_flare_probabilities_without_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flare = root / "m38b_f_minus_s660.anc.vcf.gz"
            text = (
                "##fileformat=VCFv4.2\n"
                "##ANCESTRY=<AFR=0,EUR=1,NAM=2>\n"
                "##FORMAT=<ID=AN1,Number=1,Type=Integer,Description=\"A\">\n"
                "##FORMAT=<ID=AN2,Number=1,Type=Integer,Description=\"A\">\n"
                "##FORMAT=<ID=ANP1,Number=.,Type=Float,Description=\"P\">\n"
                "##FORMAT=<ID=ANP2,Number=.,Type=Float,Description=\"P\">\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT1\tT2\n"
                "22\t10\t.\tA\tG\t.\tPASS\t.\tAN1:AN2:ANP1:ANP2\t"
                "0:1:0.8,0.1,0.1:0.1,0.8,0.1\t1:2:0.1,0.8,0.1:0.1,0.1,0.8\n"
                "22\t30\t.\tG\tA\t.\tPASS\t.\tAN1:AN2:ANP1:ANP2\t"
                "1:1:0.1,0.8,0.1:0.1,0.8,0.1\t2:0:0.1,0.1,0.8:0.8,0.1,0.1\n"
            )
            with gzip.open(flare, "wt", encoding="utf-8") as handle:
                handle.write(text)
            genetic_map = root / "map.txt"
            genetic_map.write_text("22\t1\t0\n22\t100\t1\n", encoding="utf-8")
            receipt_path = root / "flare.receipt.json"
            write_json(
                receipt_path,
                {
                    "stage": "M38B_F_MINUS_S660_FLARE",
                    "status": "PASS_TRUTH_BLIND_FLARE_F_MINUS_S660_FIT",
                    "scope": {"target_partition": "FIT"},
                    "truth_accessed": False,
                    "scoring_performed": False,
                    "shape": {"marker_count": 2, "target_sample_count": 2},
                    "ancestry_vcf_audit": {"sha256": digest(flare)},
                },
            )
            outdir = root / "parsed"
            receipt = parse_f0(
                flare_anc=flare,
                flare_receipt=receipt_path,
                genetic_map=genetic_map,
                genetic_map_sha256=digest(genetic_map),
                outdir=outdir,
                expected_samples=2,
                expected_markers=2,
            )
            self.assertEqual(receipt["decision"], "PASS_M38B_F_MINUS_S660_F0_TRUTH_BLIND")
            self.assertFalse(receipt["truth_opened"])
            with np.load(outdir / "m38b_f_minus_s660_f0.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["F0"].shape, (2, 2, 2, 3))
                self.assertTrue(np.allclose(archive["F0"].sum(axis=3), 1.0))


class M38BProjectionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, object]:
        full_keys = [
            ("22", 10, "A", "G"),
            ("22", 20, "C", "T"),
            ("22", 30, "G", "A"),
            ("22", 40, "T", "C"),
        ]
        selected_keys = [full_keys[1], full_keys[3]]
        minus_keys = [full_keys[0], full_keys[2]]
        sample_keys = np.asarray([b"a" * 64, b"b" * 64], dtype="|S64")
        full_f0 = root / "full.npz"
        minus_f0 = root / "minus.npz"
        full_cm = root / "full-cm.npz"
        minus_cm = root / "minus-cm.npz"
        truth = root / "truth.npz"
        selected = root / "selected.npz"
        make_f0(full_f0, full_keys, sample_keys)
        make_f0(minus_f0, minus_keys, sample_keys, offset=0.05)
        make_marker_cm(full_cm, [0.0, 0.1, 0.2, 0.3])
        make_marker_cm(minus_cm, [0.0, 0.2])
        make_truth(truth, [10, 20, 30, 40], sample_keys)
        write_selected(selected, selected_keys)
        parse_receipt = root / "parse.receipt.json"
        write_json(
            parse_receipt,
            {
                "stage": "M38B_PARSE_F_MINUS_S660_FLARE_F0",
                "decision": "PASS_M38B_F_MINUS_S660_F0_TRUTH_BLIND",
                "sample_count": 2,
                "marker_count": 2,
                "truth_opened": False,
                "outputs": {
                    minus_f0.name: {"sha256": digest(minus_f0)},
                    minus_cm.name: {"sha256": digest(minus_cm)},
                },
            },
        )
        experiment = root / "experiment.json"
        source = {
            "f_full_npz_sha256": digest(full_f0),
            "f_full_marker_cm_sha256": digest(full_cm),
            "fit_truth_sha256": digest(truth),
            "s660_selected_sha256": digest(selected),
        }
        write_json(experiment, base_experiment(source))
        return {
            "experiment": experiment,
            "full_f0": full_f0,
            "full_marker_cm": full_cm,
            "full_truth": truth,
            "selected_loci": selected,
            "fminus_f0": minus_f0,
            "fminus_marker_cm": minus_cm,
            "fminus_receipt": parse_receipt,
            "expected_full_f0_sha256": digest(full_f0),
            "expected_full_marker_cm_sha256": digest(full_cm),
            "expected_full_truth_sha256": digest(truth),
            "expected_selected_loci_sha256": digest(selected),
            "expected_samples": 2,
            "expected_full_markers": 4,
            "expected_selected_markers": 2,
        }

    def test_exact_partition_projection_and_truth_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._fixture(root)
            receipt = align_baselines(**arguments, outdir=root / "out")
            self.assertEqual(receipt["decision"], "PASS_EXACT_SHARED_F_MINUS_S660_SCORING_GRID")
            self.assertEqual(receipt["counts"]["F_full"], 4)
            self.assertEqual(receipt["counts"]["F_minus_S660"], 2)
            self.assertEqual(receipt["counts"]["S660"], 2)
            self.assertEqual(receipt["counts"]["partition_overlap"], 0)
            self.assertFalse(receipt["identity"]["position_only_matching_used"])
            self.assertFalse(receipt["identity"]["f_minus_is_common_only"])
            self.assertTrue(receipt["alignment"]["sample_axes_identical"])
            self.assertFalse(receipt["alignment"]["F_full_probability_values_reestimated"])
            with np.load(
                root / "out/m38b_f_full_projected_to_f_minus_s660.npz",
                allow_pickle=False,
            ) as projected:
                self.assertEqual(projected["marker_pos"].tolist(), [10, 30])
                self.assertEqual(projected["F0"].shape, (2, 2, 2, 3))
            with np.load(
                root / "out/m38b_fit_truth_projected_to_f_minus_s660.npz",
                allow_pickle=False,
            ) as projected_truth:
                self.assertEqual(projected_truth["marker_pos"].tolist(), [10, 30])
                self.assertEqual(projected_truth["labels"].shape, (2, 2, 2))

    def test_allele_axis_drift_fails_even_when_positions_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._fixture(root)
            sample_keys = np.asarray([b"a" * 64, b"b" * 64], dtype="|S64")
            bad_minus = root / "bad-minus.npz"
            make_f0(
                bad_minus,
                [("22", 10, "A", "T"), ("22", 30, "G", "A")],
                sample_keys,
            )
            parse_receipt = root / "bad-parse.receipt.json"
            write_json(
                parse_receipt,
                {
                    "stage": "M38B_PARSE_F_MINUS_S660_FLARE_F0",
                    "decision": "PASS_M38B_F_MINUS_S660_F0_TRUTH_BLIND",
                    "sample_count": 2,
                    "marker_count": 2,
                    "truth_opened": False,
                    "outputs": {
                        bad_minus.name: {"sha256": digest(bad_minus)},
                        Path(arguments["fminus_marker_cm"]).name: {
                            "sha256": digest(Path(arguments["fminus_marker_cm"]))
                        },
                    },
                },
            )
            arguments["fminus_f0"] = bad_minus
            arguments["fminus_receipt"] = parse_receipt
            with self.assertRaisesRegex(M38BBaselineAlignmentError, "exact union|absent"):
                align_baselines(**arguments, outdir=root / "out")

    def test_marker_cm_and_sample_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._fixture(root)
            bad_cm = root / "bad-cm.npz"
            make_marker_cm(bad_cm, [0.0, 0.21])
            parse_receipt = root / "bad-cm.receipt.json"
            write_json(
                parse_receipt,
                {
                    "stage": "M38B_PARSE_F_MINUS_S660_FLARE_F0",
                    "decision": "PASS_M38B_F_MINUS_S660_F0_TRUTH_BLIND",
                    "sample_count": 2,
                    "marker_count": 2,
                    "truth_opened": False,
                    "outputs": {
                        Path(arguments["fminus_f0"]).name: {
                            "sha256": digest(Path(arguments["fminus_f0"]))
                        },
                        bad_cm.name: {"sha256": digest(bad_cm)},
                    },
                },
            )
            arguments["fminus_marker_cm"] = bad_cm
            arguments["fminus_receipt"] = parse_receipt
            with self.assertRaisesRegex(M38BBaselineAlignmentError, "genetic-coordinate"):
                align_baselines(**arguments, outdir=root / "cm-out")


class M38BWorkflowTests(unittest.TestCase):
    def test_workflow_is_fit_only_nextflow_first_and_personal_bucket_only(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        config = CONFIG.read_text(encoding="utf-8")
        self.assertIn("M38B_BUILD_FLARE_CONTRACT", workflow)
        self.assertIn("M38B_RUN_FLARE_F_MINUS_S660", workflow)
        self.assertIn("M38B_PARSE_F_MINUS_S660_F0", workflow)
        self.assertIn("M38B_PROJECT_FULL_AND_TRUTH", workflow)
        self.assertNotIn("m38b_prepare_valid", workflow.lower() + config.lower())
        self.assertNotIn("m38b_prepare_test", workflow.lower() + config.lower())
        self.assertIn("m38b_prepare_partition = 'FIT'", config)
        self.assertIn("team: 'frank'", config)
        self.assertIn("lane: 'm38b-baseline'", config)
        self.assertIn("maxRetries = 0", config)
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos", config)
        self.assertIn("M38B_RUN_FLARE_F_MINUS_S660", workflow)
        for module in MODULES:
            text = module.read_text(encoding="utf-8")
            self.assertIn("--network none", text)
            self.assertIn("overwrite: false", text)

    def test_config_pins_canonical_hashes_and_exact_counts(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        for value in (
            "ec7869ebf400f6ba920a710f01c986b1cfaecc6709e812a70b2af197e67cf356",
            "ddcbee7afbef48ea0764eeb1ef89c379a4bbee9409e4ce84c59af79ecb7a5c36",
            "bee39c39b61f5be47eedf2770ab03367070b57c6a9ea0abb5880fd2efb5dafed",
            "4a57c0350ce3a8f1213b14da2b63671335dbb55ab59faab1f45c361c751942eb",
            "78263df0fb674b72d53f4ddb6544ad4a2efd204fcd2ec739da29c55a0a9dc1f6",
            "acfe6c19b87a32fb8d98616b00fe5a3ccbfa93af5e69a2f2e161d796edea7ad1",
            "8c804341b555f302591b12cd72e870b1ca7849055d1dcd2b5cfa09b725bd9420",
        ):
            self.assertIn(value, text)
        self.assertIn("m38b_prepare_expected_full_loci = 42986", text)
        self.assertIn("m38b_prepare_expected_selected_loci = 660", text)
        self.assertIn("m38b_prepare_expected_fminus_loci = 42326", text)


if __name__ == "__main__":
    unittest.main()
